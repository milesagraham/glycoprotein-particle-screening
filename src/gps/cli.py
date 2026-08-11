import functools
import typer
import starfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing_extensions import Annotated
from typing import Any, List, Tuple

app = typer.Typer(help="GPS: Screen particle picks from a pytom/RELION-style STAR file by eye.")


def load_star_data(star_file: Path) -> Tuple[pd.DataFrame, Any, str]:
    """Loads a STAR file and returns the particles dataframe"""
    #if there are multiple data blocks it will get read into a dictionary
    df_dict = starfile.read(star_file)
    is_dict = isinstance(df_dict, dict)

    # Check if it's a dictionary. If so, extract the 'particles' block (or default to the first available block).
    # If it's not a dictionary, it will have just been read in as a single table.
    block_name = 'particles' if is_dict and 'particles' in df_dict else list(df_dict.keys())[0] if is_dict else None
    df = df_dict[block_name] if block_name else df_dict

    # Check we have found our coordinates and euler angles
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column in STAR file: {col}")

    return df, df_dict, block_name


def euler_to_rotation_matrix(rot: float, tilt: float, psi: float) -> np.ndarray:
    """Converts RELION-style Euler angles to a 3x3 orthonormal rotation matrix describing the
    particle's own local frame: column 0 and 1 are its in-plane x/y axes, column 2 is its pointing
    (z) axis. R = Rz(rot + 90deg) @ Ry(tilt) @ Rz(psi) - the +90deg offset on rot makes column 2
    exactly reproduce the particle-orientation vector this codebase's predecessor (GCA) validated
    against real data, while still giving a full, psi-dependent in-plane frame that a single vector
    can't provide."""
    rot_rad, tilt_rad, psi_rad = np.deg2rad([rot, tilt, psi])

    cr, sr = np.cos(rot_rad + np.pi / 2), np.sin(rot_rad + np.pi / 2)
    rz_rot = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    ct, st = np.cos(tilt_rad), np.sin(tilt_rad)
    ry_tilt = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

    cp, sp = np.cos(psi_rad), np.sin(psi_rad)
    rz_psi = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])

    return rz_rot @ ry_tilt @ rz_psi


def discover_tomogram_sets(data_dir: Path) -> List[dict]:
    """Finds tomogram/starfile pairs that share an exact filename stem across the tomograms/ and
    starfiles/ subdirectories of data_dir - each is one tomogram to prepare for review."""
    tomo_dir = data_dir / "tomograms"
    star_dir = data_dir / "starfiles"
    for d in (tomo_dir, star_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected subdirectory not found: {d}")

    tomo_files = {p.stem: p for p in tomo_dir.glob("*.mrc")}
    star_files = {p.stem: p for p in star_dir.glob("*.star")}

    common_stems = sorted(set(tomo_files) & set(star_files))
    all_stems = set(tomo_files) | set(star_files)
    skipped = sorted(all_stems - set(common_stems))
    if skipped:
        typer.echo(f"Note: skipping {len(skipped)} file(s) without a matching stem in both "
                   f"subdirectories: {skipped}")
    if not common_stems:
        raise ValueError(f"No matching tomogram/starfile pairs found under {data_dir}")

    return [dict(stem=s, tomogram=tomo_files[s], starfile=star_files[s]) for s in common_stems]


def _process_tomogram_for_prepare(
    tomo_set: dict, inputs_dir: Path, particles_apx: float, tomo_apx: float,
    box_size_angstrom: float, context_box_size_angstrom: float,
) -> dict:
    """Renders review images for every particle in one tomogram and writes them to
    inputs_dir/<stem>/. Returns a small summary dict for the CLI's own progress reporting."""
    #lazy import: gps.review imports back from this module at load time (for `app` and
    #load_star_data/euler_to_rotation_matrix), so importing it here rather than at module level
    #avoids relying on import order between the two circularly-dependent modules
    from gps.review import render_review_data_for_tomogram
    records = render_review_data_for_tomogram(
        tomo_set, inputs_dir, particles_apx, tomo_apx, box_size_angstrom, context_box_size_angstrom,
    )
    return dict(stem=tomo_set['stem'], n_particles=len(records))


@app.command()
def prepare(
    data_dir: Annotated[Path, typer.Argument(
        help="Directory containing tomograms/ and starfiles/ subdirectories - matching files "
             "(same stem) across them are treated as one tomogram")],
    particles_apx: Annotated[
        float, typer.Option("--particles_apx", help="Particle coordinate pixel size in Angstroms/pixel")],
    tomo_apx: Annotated[
        float, typer.Option("--tomo_apx", help="Tomogram density pixel size in Angstroms/pixel")],
    box_size_angstrom: Annotated[
        float, typer.Option("--box-size-angstrom",
                             help="Width/height in Angstroms of the close-up review panels")] = 300.0,
    context_box_size_angstrom: Annotated[
        float, typer.Option("--context-box-size-angstrom",
                             help="Width/height in Angstroms of the wider context panel")] = 2000.0,
    workers: Annotated[
        int, typer.Option("--workers", "-j",
                           help="Number of tomograms to render in parallel (they're fully "
                                "independent of each other). Each worker holds one whole "
                                "tomogram's density volume in memory at once, so raise this "
                                "cautiously if RAM is limited. 1 disables parallelism.")] = 4,
):
    """
    Renders, for every particle in every matched tomogram/starfile pair found under data_dir, a
    tomogram density image oriented to that particle's own orientation - no analysis or automatic
    accept/reject, every particle goes to manual review. Writes to data_dir/gps_review_inputs/.
    Run `gps review data_dir` afterwards to triage the results by eye.
    """
    if not data_dir.is_dir():
        typer.echo(f"Input Error: {data_dir} is not a directory", err=True)
        raise typer.Exit(code=1)
    data_dir = data_dir.resolve()

    if particles_apx <= 0:
        typer.echo(f"Input Error: --particles_apx must be a positive number, got {particles_apx}", err=True)
        raise typer.Exit(code=1)

    if tomo_apx <= 0:
        typer.echo(f"Input Error: --tomo_apx must be a positive number, got {tomo_apx}", err=True)
        raise typer.Exit(code=1)

    if box_size_angstrom <= 0:
        typer.echo(f"Input Error: --box-size-angstrom must be a positive number, got {box_size_angstrom}", err=True)
        raise typer.Exit(code=1)

    if context_box_size_angstrom <= 0:
        typer.echo(f"Input Error: --context-box-size-angstrom must be a positive number, "
                   f"got {context_box_size_angstrom}", err=True)
        raise typer.Exit(code=1)

    if workers < 1:
        typer.echo("Input Error: --workers must be at least 1", err=True)
        raise typer.Exit(code=1)

    try:
        tomo_sets = discover_tomogram_sets(data_dir)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Input Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(tomo_sets)} tomogram(s) to prepare: {[s['stem'] for s in tomo_sets]}")

    inputs_dir = data_dir / "gps_review_inputs"
    inputs_dir.mkdir(exist_ok=True)

    process_one = functools.partial(
        _process_tomogram_for_prepare, inputs_dir=inputs_dir, particles_apx=particles_apx,
        tomo_apx=tomo_apx, box_size_angstrom=box_size_angstrom,
        context_box_size_angstrom=context_box_size_angstrom,
    )
    if workers == 1 or len(tomo_sets) == 1:
        summaries = [process_one(tomo_set) for tomo_set in tomo_sets]
    else:
        typer.echo(f"Preparing up to {min(workers, len(tomo_sets))} tomogram(s) at a time...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(process_one, tomo_sets))

    total_particles = sum(s['n_particles'] for s in summaries)
    typer.echo(f"\nDone: {total_particles} particle(s) prepared for review across "
               f"{len(tomo_sets)} tomogram(s), written to {inputs_dir}.")
    typer.echo(f"Run `gps review {data_dir}` to triage them.")

#imported for its side effect of registering the `review` command on `app` above - placed here,
#after `app` is fully defined, to avoid a circular import at module load time
from gps import review  # noqa: E402,F401

if __name__ == "__main__":
    app()
