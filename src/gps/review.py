"""
gps review: the second-step web app for rapidly accepting/rejecting particle picks by eye, using
real tomogram density slices oriented to each particle's own orientation.

Design notes:
- All the expensive work (tomogram slicing, image rendering) happens in `gps prepare` (in
  gps.cli), not here - this module only ever reads data_dir/gps_review_inputs/ (written by that
  command) and serves it. It never computes anything itself; if gps_review_inputs/ isn't there
  yet, `gps review` refuses to start rather than silently computing it.
  render_review_data_for_tomogram() below is the one exception - it's the actual image-rendering
  function, but it's only ever called from `gps prepare`, never from the review command in this
  module.
- Unlike GCA (this tool's predecessor, which compared a particle's orientation against an
  independently-fitted membrane normal), there is no second vector to compare a particle's
  orientation against here - the review frame *is* the particle's own orientation. So the panels
  never *automatically* draw an orientation arrow (tried once, rejected - see git history): every
  particle is displayed "canonically", with nothing to draw since the frame's own axes are the
  vector. The only arrows ever shown are the manual correction annotations described below, which
  are a genuinely different thing - user-drawn, not derived from the particle's already-known
  orientation, and can point anywhere. Don't conflate the two or reintroduce the automatic kind.
- Manual orientation correction: all three close-up panels (top-down, side (x), side (y)) are
  click-annotatable in the UI. A reviewer clicks a base point then an apex point (membrane ->
  particle) in any subset of the three; each panel's click only pins 2 of the
  pointing vector's 3 lab-frame components (top-down: x & y; side (x): x & z; side (y): y & z), so
  a single panel leaves one component undetermined - a real ambiguity, not just imprecision.
  top-down is deliberately included even though it contributes nothing when the particle's current
  orientation is roughly right (it's edge-on to the very axis being corrected) - but for a badly
  misoriented particle, what's labeled "top-down" can visually end up looking like a side view, so
  it needs to be just as clickable as the other two. Whichever components end up with 2 independent
  estimates (from 2 panels sharing that axis) are averaged for robustness; components with only one
  or zero estimates use that value or 0. Single/partial-panel corrections are allowed and not
  flagged as lower-confidence, per explicit user direction ("if it's the best I can see, it's
  better than nothing"). Enter saves the correction (recomputing rot/tilt via
  cli.z_vector_to_rot_tilt; psi is never touched - nothing in this workflow constrains it) and also
  marks the particle accepted, then advances - correcting and accepting are the same motion, not
  separate steps, again per explicit user direction. Corrections persist to
  gps_review_inputs/decisions.json's sibling, corrections.json, and are redrawn (reprojected onto
  all three panels, regardless of which contributed to the original click) whenever that particle
  is revisited.
- There is no automatic accept/reject step - every particle in every matched tomogram/starfile pair
  goes into the manual review queue.
- Raw tomogram slices are noisy, which makes it hard to tell where a membrane actually is,
  especially in the top-down panel. `--slab-slices` (see gps.cli) averages several parallel slices
  stepped along each panel's own depth axis (the axis that panel is looking down) instead of a
  single slice - noise averages down while a real, only-mildly-curved feature like a membrane stays
  roughly in place across nearby depths, so this mostly boosts contrast rather than smearing
  structure. Off (a single ordinary slice) by default; tried at --slab-slices 10 and found not to
  make a clear enough difference to be worth it yet, so it's parked at the default rather than
  pursued further for now.
- There are two wide panels, not one: "context" is oriented the same way as the "side (x)"
  close-up (particle-frame axes), while "traditional" is the plain, un-rotated tomogram XY slice at
  the particle's own position - what a reviewer would see scrolling through the raw tomogram in a
  standard viewer, unaffected by this particle's orientation. Added because the context panel's
  oblique, per-particle-rotated cut made it hard to relate a pick back to the tomogram a reviewer
  already knows how to read.
- Optional background-suppressing thresholding (`--threshold` in gps.cli): a light denoising blur
  (`PREVIEW_GAUSSIAN_SIGMA`) followed by clipping the display range so only the upper
  `(100 - threshold)` percent of the (blurred) intensity range renders at all - everything below
  goes flat black instead of visible speckle. Two earlier, different approaches to this same goal
  (CLAHE, then an always-on second UI row with this same blur+clip) were each tried and removed
  after being judged not to help. What's here now is deliberately opt-in and single-row: off by
  default (plain percentile-stretch display), and when set, replaces the display windowing on every
  panel rather than adding a second row - `--preview-thresholds` renders a small gallery across
  10/20/.../90 percent on a few random particles first, so a value can be picked visually before
  committing to a full `--threshold`-flagged run.
- Re-running `gps prepare` skips any tomogram whose input files and parameters are unchanged since
  the last run (fingerprinted in gps_review_inputs/<stem>/fingerprint.txt - this includes
  PANEL_VERSION, bumped whenever the set of files/fields written per particle changes, so an older
  cache with a different output shape is never silently reused), so an interrupted batch job can be
  resubmitted without repeating already-finished tomograms.
- Decisions are saved continuously to gps_review_inputs/decisions.json, so a review session can be
  closed and resumed later without losing progress.
"""
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mrcfile
import starfile
import typer
from typing_extensions import Annotated
from scipy.ndimage import gaussian_filter, map_coordinates

from gps.cli import app, load_star_data, euler_to_rotation_matrix, z_vector_to_rot_tilt

#review image geometry - three orthogonal views in the particle's own local frame (xy, xz, yz)
BOX_PIXELS = 170

#wider single-slice context view, oriented the same way as the xz close-up panel, showing where on
#the wider tomogram (e.g. a virus/vesicle surface) a pick sits - sampled close to the tomogram's
#native pixel size rather than oversampled, since there's no benefit interpolating finer than the
#data at this scale. The traditional panel uses the same box/pixel size but the tomogram's own raw
#axes instead - the plain, un-rotated XY view a reviewer would get scrolling through the tomogram
#in a standard viewer, unaffected by anything about this particle's own orientation
CONTEXT_BOX_PIXELS = 220

LAB_X = np.array([1.0, 0.0, 0.0])
LAB_Y = np.array([0.0, 1.0, 0.0])
LAB_Z = np.array([0.0, 0.0, 1.0])

#bump whenever the set of files/fields render_review_data_for_tomogram writes per particle changes,
#so a cache written by an older version (same input files/params, different output shape) is
#correctly treated as stale instead of being silently reused missing the new fields
PANEL_VERSION = 4

#how long, as a fraction of box_size_angstrom, the redrawn correction arrow appears on each panel
#for a fully-aligned (direction-cosine 1.0) component - see build_review_app's /api/correct
ARROW_DISPLAY_FRACTION = 0.35

#denoising blur applied before thresholding (--threshold and --preview-thresholds) - fixed rather
#than exposed as its own CLI flag, since it's just there to stop the percentile cutoff from picking
#out individual noise spikes, not a creative parameter worth its own knob
PREVIEW_GAUSSIAN_SIGMA = 1.0

#the fixed sweep --preview-thresholds renders per sampled particle
PREVIEW_THRESHOLDS = list(range(10, 91, 10))


def _file_fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def _tomogram_fingerprint(tomo_set: dict, params: dict) -> str:
    """Identifies one tomogram's cached render - a fresh run only skips recomputation if neither
    input file (by size/mtime) nor any of the rendering parameters have changed since the cache
    was written, so a resumed prepare job is safe to trust blindly."""
    parts = [
        _file_fingerprint(tomo_set['tomogram']),
        _file_fingerprint(tomo_set['starfile']),
        json.dumps(params, sort_keys=True),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _extract_slice(tomo: np.ndarray, center: np.ndarray, axis1: np.ndarray, axis2: np.ndarray,
                    grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
    points = (center[None, None, :] + grid_a[..., None] * axis1[None, None, :]
              + grid_b[..., None] * axis2[None, None, :])
    coords = np.stack([points[..., 2], points[..., 1], points[..., 0]], axis=0)  # z,y,x order
    return map_coordinates(tomo, coords, order=1, mode='nearest')


def _extract_slab(tomo: np.ndarray, center: np.ndarray, axis1: np.ndarray, axis2: np.ndarray,
                   axis_normal: np.ndarray, grid_a: np.ndarray, grid_b: np.ndarray,
                   n_slices: int) -> np.ndarray:
    """Averages n_slices parallel single-voxel-spaced slices, stepped along axis_normal (the axis
    the panel is looking down) and centered on `center`, instead of returning just the one slice
    through `center` itself. Raw tomogram slices are dominated by shot noise, which averages down
    while a real, locally-planar feature like a membrane - only mildly curved over a few voxels -
    stays roughly in place across nearby depths, so this mainly boosts contrast rather than
    smearing genuine structure. n_slices <= 1 is a single ordinary slice."""
    if n_slices <= 1:
        return _extract_slice(tomo, center, axis1, axis2, grid_a, grid_b)
    offsets = np.arange(n_slices) - (n_slices - 1) / 2.0
    slices = [_extract_slice(tomo, center + offset * axis_normal, axis1, axis2, grid_a, grid_b)
              for offset in offsets]
    return np.mean(slices, axis=0)


def _apply_threshold(img: np.ndarray, gaussian_sigma: float, threshold_percentile: float) -> np.ndarray:
    """Suppresses background noise so particle-like density stands out: a Gaussian blur first
    smooths out pixel-level shot noise, then the display range is clipped so only the upper
    (100 - threshold_percentile) percent of the (blurred) intensity range is shown at all -
    background below that collapses to flat black instead of visible speckle, while what's left
    gets stretched to full contrast. Returns values in [0, 1]. Purely a rendering choice - never
    feeds back into which particles get shown or any accept/reject decision."""
    smoothed = gaussian_filter(img, sigma=gaussian_sigma) if gaussian_sigma > 0 else img
    lo, hi = np.percentile(smoothed, [threshold_percentile, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(smoothed)
    clipped = np.clip(smoothed, lo, hi)
    return (clipped - lo) / (hi - lo)


def _save_single_panel(path: Path, img: np.ndarray, box_a: float, pixels: int,
                        vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    """Saves one clean panel - no title, ticks, spines, or padding, just the image and the amber
    crosshair - filling the figure edge-to-edge at an exact pixel size, so a browser click on the
    resulting PNG maps back to an Angstrom position by simple linear interpolation (no tight-bbox
    cropping uncertainty to account for). Used for the top-down/side (x)/side (y) panels, which is
    what the review UI's click-to-correct-orientation workflow needs pixel-accurate coordinates
    from (see build_review_app's /api/correct). vmin/vmax are passed through explicitly (rather
    than computing a percentile stretch here) when rendering already-normalized thresholded output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(img, [1, 99])
    dpi = 100
    fig = plt.figure(figsize=(pixels / dpi, pixels / dpi), dpi=dpi, facecolor='#14191a')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-box_a / 2, box_a / 2, -box_a / 2, box_a / 2])
    ax.scatter(0, 0, c='#f9a825', s=70, marker='+', linewidth=2.0, zorder=5)
    ax.set_xlim(-box_a / 2, box_a / 2)
    ax.set_ylim(-box_a / 2, box_a / 2)
    ax.axis('off')
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_context_panel(path: Path, img: np.ndarray, box_a: float,
                         vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(img, [1, 99])
    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor='#14191a')
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-box_a / 2, box_a / 2, -box_a / 2, box_a / 2])
    ax.scatter(0, 0, c='#f9a825', s=140, marker='+', linewidth=2.6, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def _draw_sweep_cell(ax, img: np.ndarray, box_a: float, title: str,
                      vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-box_a / 2, box_a / 2, -box_a / 2, box_a / 2])
    ax.scatter(0, 0, c='#f9a825', s=60, marker='+', linewidth=1.8, zorder=5)
    ax.set_title(title, fontsize=10, color='#c7d0d1')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_threshold_sweep_panel(path: Path, raw_img: np.ndarray,
                                 threshold_imgs: List[tuple], box_a: float) -> None:
    """One row per particle for --preview-thresholds: the raw side (x) slice, then that same slice
    thresholded at each percentile in threshold_imgs (a list of (percentile, image) pairs), so a
    reviewer can compare them side by side and pick a --threshold value for the full run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 1 + len(threshold_imgs)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 2.6), facecolor='#14191a')
    _draw_sweep_cell(axes[0], raw_img, box_a, "raw")
    for ax, (pct, img) in zip(axes[1:], threshold_imgs):
        _draw_sweep_cell(ax, img, box_a, f"threshold={pct:.0f}", vmin=0.0, vmax=1.0)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def render_review_data_for_tomogram(
    tomo_set: dict, inputs_dir: Path, particles_apx: float, tomo_apx: float,
    box_size_angstrom: float, context_box_size_angstrom: float, slab_slices: int,
    threshold: Optional[float],
) -> List[dict]:
    """Called from `gps prepare` (in gps.cli), never directly by this module's own CLI command.
    Renders a review image for every particle in one tomogram's STAR file. Writes one
    self-contained records.json to inputs_dir/<stem>/ - the starfile path plus one record (metrics
    + image paths) per particle - which is all `gps review` needs later to serve that tomogram; it
    never touches the tomogram again. Returns the same records as a plain list, for `gps prepare`'s
    own progress reporting.

    Re-running with the same input files and parameters is skipped (fingerprint match), so a
    `gps prepare` batch job interrupted partway through a large tomogram set can be resubmitted
    without repeating already-finished tomograms."""
    stem = tomo_set['stem']
    stem_dir = inputs_dir / stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    records_path = stem_dir / "records.json"
    fingerprint_path = stem_dir / "fingerprint.txt"

    params = dict(particles_apx=particles_apx, tomo_apx=tomo_apx,
                  box_size_angstrom=box_size_angstrom,
                  context_box_size_angstrom=context_box_size_angstrom, slab_slices=slab_slices,
                  threshold=threshold, panel_version=PANEL_VERSION)
    fingerprint = _tomogram_fingerprint(tomo_set, params)
    if records_path.exists() and fingerprint_path.exists() and fingerprint_path.read_text().strip() == fingerprint:
        records = json.loads(records_path.read_text())["records"]
        typer.echo(f"[{stem}] inputs and parameters unchanged since last run - reusing "
                   f"{len(records)} already-prepared review image(s), skipping recomputation.")
        return records

    def _finish(records: List[dict]) -> List[dict]:
        records_path.write_text(json.dumps(
            dict(stem=stem, starfile=str(tomo_set['starfile']), records=records), indent=2))
        fingerprint_path.write_text(fingerprint)
        return records

    typer.echo(f"[{stem}] loading star file...")
    df, _, _ = load_star_data(tomo_set['starfile'])

    scaling_factor = particles_apx / tomo_apx
    particle_coords_vox = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy() * scaling_factor
    eulers = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    lccmax = df['rlnLCCmax'].to_numpy() if 'rlnLCCmax' in df.columns else np.full(len(df), np.nan)

    typer.echo(f"[{stem}] loading tomogram density and rendering review images for "
               f"{len(df)} particles...")
    with mrcfile.open(tomo_set['tomogram'], permissive=True) as mrc:
        tomo = np.asarray(mrc.data, dtype=np.float32)

    half_width_vox = (box_size_angstrom / 2) / tomo_apx
    grid_1d = np.linspace(-half_width_vox, half_width_vox, BOX_PIXELS)
    grid_b, grid_a = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    context_half_width_vox = (context_box_size_angstrom / 2) / tomo_apx
    context_grid_1d = np.linspace(-context_half_width_vox, context_half_width_vox, CONTEXT_BOX_PIXELS)
    context_grid_b, context_grid_a = np.meshgrid(context_grid_1d, context_grid_1d, indexing='ij')

    records = []
    for i in range(len(df)):
        pos = particle_coords_vox[i]
        rot, tilt, psi = eulers[i]
        frame = euler_to_rotation_matrix(rot, tilt, psi)
        x_axis, y_axis, z_axis = frame[:, 0], frame[:, 1], frame[:, 2]

        img_xy = _extract_slab(tomo, pos, x_axis, y_axis, z_axis, grid_a, grid_b, slab_slices)
        img_xz = _extract_slab(tomo, pos, x_axis, z_axis, y_axis, grid_a, grid_b, slab_slices)
        img_yz = _extract_slab(tomo, pos, y_axis, z_axis, x_axis, grid_a, grid_b, slab_slices)

        #when --threshold is set, every panel's *display* is background-suppressed (blur + clip,
        #see _apply_threshold) instead of the plain percentile stretch - vmin/vmax=(0,1) since
        #_apply_threshold's output is already normalized to that range
        def display(im: np.ndarray):
            if threshold is None:
                return im, None, None
            return _apply_threshold(im, PREVIEW_GAUSSIAN_SIGMA, threshold), 0.0, 1.0

        #saved as 3 separate clean-edge images, not one composite, so a click in the review UI's
        #top-down/side (x)/side (y) panel maps back to an exact Angstrom position - see
        #_save_single_panel and build_review_app's /api/correct
        topdown_filename = f"idx{i}_topdown.png"
        sidex_filename = f"idx{i}_sidex.png"
        sidey_filename = f"idx{i}_sidey.png"
        disp_xy, vmin, vmax = display(img_xy)
        disp_xz, _, _ = display(img_xz)
        disp_yz, _, _ = display(img_yz)
        _save_single_panel(stem_dir / topdown_filename, disp_xy, box_size_angstrom, BOX_PIXELS, vmin, vmax)
        _save_single_panel(stem_dir / sidex_filename, disp_xz, box_size_angstrom, BOX_PIXELS, vmin, vmax)
        _save_single_panel(stem_dir / sidey_filename, disp_yz, box_size_angstrom, BOX_PIXELS, vmin, vmax)

        context_img = _extract_slab(tomo, pos, x_axis, z_axis, y_axis, context_grid_a,
                                    context_grid_b, slab_slices)
        context_filename = f"idx{i}_context.png"
        disp_context, vmin, vmax = display(context_img)
        _save_context_panel(stem_dir / context_filename, disp_context, context_box_size_angstrom, vmin, vmax)

        traditional_img = _extract_slab(tomo, pos, LAB_X, LAB_Y, LAB_Z, context_grid_a,
                                        context_grid_b, slab_slices)
        traditional_filename = f"idx{i}_traditional.png"
        disp_traditional, vmin, vmax = display(traditional_img)
        _save_context_panel(stem_dir / traditional_filename, disp_traditional,
                            context_box_size_angstrom, vmin, vmax)

        records.append(dict(
            key=f"{stem}:{i}", stem=stem, idx=i,
            lcc=(None if np.isnan(lccmax[i]) else float(lccmax[i])),
            rot=round(float(rot), 2), tilt=round(float(tilt), 2), psi=round(float(psi), 2),
            box_size_angstrom=box_size_angstrom,
            image_topdown=f"/api/image/{stem}/{topdown_filename}",
            image_sidex=f"/api/image/{stem}/{sidex_filename}",
            image_sidey=f"/api/image/{stem}/{sidey_filename}",
            context_image=f"/api/image/{stem}/{context_filename}",
            traditional_image=f"/api/image/{stem}/{traditional_filename}",
        ))

    typer.echo(f"[{stem}] {len(records)} review images ready.")
    return _finish(records)


def render_threshold_preview(
    tomo_sets: List[dict], preview_dir: Path, particles_apx: float, tomo_apx: float,
    box_size_angstrom: float, slab_slices: int, n_particles: int, thresholds: List[float],
) -> List[dict]:
    """Called from `gps prepare --preview-thresholds` (in gps.cli). Randomly samples n_particles
    particles from across all of tomo_sets, and for each writes one PNG - its side (x) slice raw,
    next to itself thresholded at every percentile in `thresholds` - to preview_dir, so a --threshold
    value can be picked visually before committing to it for a full render. Uses the exact same
    extraction pipeline (_extract_slab, same box size/slab-slices) as the real render, so the
    preview is representative of what --threshold would actually produce; never touches
    gps_review_inputs/ or writes anything gps review reads."""
    preview_dir.mkdir(parents=True, exist_ok=True)

    pool = []
    star_cache: Dict[str, tuple] = {}
    for tomo_set in tomo_sets:
        df, _, _ = load_star_data(tomo_set['starfile'])
        star_cache[tomo_set['stem']] = df
        pool.extend((tomo_set, i) for i in range(len(df)))

    sampled = random.sample(pool, min(n_particles, len(pool)))

    scaling_factor = particles_apx / tomo_apx
    half_width_vox = (box_size_angstrom / 2) / tomo_apx
    grid_1d = np.linspace(-half_width_vox, half_width_vox, BOX_PIXELS)
    grid_b, grid_a = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    tomo_cache: Dict[str, np.ndarray] = {}
    written = []
    for tomo_set, i in sampled:
        stem = tomo_set['stem']
        if stem not in tomo_cache:
            typer.echo(f"[{stem}] loading tomogram density...")
            with mrcfile.open(tomo_set['tomogram'], permissive=True) as mrc:
                tomo_cache[stem] = np.asarray(mrc.data, dtype=np.float32)
        tomo = tomo_cache[stem]
        df = star_cache[stem]

        row = df.iloc[i]
        pos = np.array([row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']]) * scaling_factor
        frame = euler_to_rotation_matrix(row['rlnAngleRot'], row['rlnAngleTilt'], row['rlnAnglePsi'])
        x_axis, y_axis, z_axis = frame[:, 0], frame[:, 1], frame[:, 2]

        img_xz = _extract_slab(tomo, pos, x_axis, z_axis, y_axis, grid_a, grid_b, slab_slices)
        threshold_imgs = [(pct, _apply_threshold(img_xz, PREVIEW_GAUSSIAN_SIGMA, pct)) for pct in thresholds]

        out_path = preview_dir / f"{stem}_idx{i}_threshold_preview.png"
        _save_threshold_sweep_panel(out_path, img_xz, threshold_imgs, box_size_angstrom)
        typer.echo(f"[{stem}:{i}] wrote {out_path.name}")
        written.append(dict(stem=stem, idx=i, path=str(out_path)))

    return written


def build_review_app(all_records: List[dict], inputs_dir: Path, tomogram_starfiles: Dict[str, Path]):
    from flask import Flask, jsonify, request, send_from_directory, Response

    flask_app = Flask(__name__)
    records_by_key = {r["key"]: r for r in all_records}
    decisions_path = inputs_dir / "decisions.json"
    corrections_path = inputs_dir / "corrections.json"
    decisions: Dict[str, str] = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    corrections: Dict[str, dict] = json.loads(corrections_path.read_text()) if corrections_path.exists() else {}
    history: List[str] = []

    def save_decisions():
        tmp = decisions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(decisions, indent=2))
        tmp.replace(decisions_path)

    def save_corrections():
        tmp = corrections_path.with_suffix(".ctmp")
        tmp.write_text(json.dumps(corrections, indent=2))
        tmp.replace(corrections_path)

    @flask_app.route("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @flask_app.route("/api/queue")
    def queue():
        out = []
        for r in all_records:
            r2 = dict(r)
            r2["decision"] = decisions.get(r["key"])
            r2["correction"] = corrections.get(r["key"])
            out.append(r2)
        return jsonify(out)

    @flask_app.route("/api/image/<stem>/<filename>")
    def image(stem, filename):
        return send_from_directory(inputs_dir / stem, filename)

    @flask_app.route("/api/decision", methods=["POST"])
    def decision():
        data = request.get_json()
        key, value = data["key"], data.get("decision")
        if value is None:
            decisions.pop(key, None)
        else:
            decisions[key] = value
        history.append(key)
        save_decisions()
        return jsonify(_counts(decisions))

    @flask_app.route("/api/correct", methods=["POST"])
    def correct():
        data = request.get_json()
        key = data["key"]
        record = records_by_key.get(key)
        if record is None:
            return jsonify({"error": "unknown particle key"}), 404

        #topdown/sidex/sidey are each either null or a [du, dv] apex-minus-base Angstrom vector in
        #that panel's own 2D plane, computed client-side from two clicks - see the docstring above
        #and the JS handlePanelClick()/saveCorrection() for the full geometry. Any subset of the 3
        #can be provided - e.g. for a badly-misoriented particle, the "top-down" panel can end up
        #visually showing what's actually a side-on view, so it's just as clickable as the other two
        topdown, sidex, sidey = data.get("topdown"), data.get("sidex"), data.get("sidey")
        if topdown is None and sidex is None and sidey is None:
            return jsonify({"error": "no annotation provided"}), 400

        frame = euler_to_rotation_matrix(record["rot"], record["tilt"], record["psi"])
        x_axis, y_axis, z_axis = frame[:, 0], frame[:, 1], frame[:, 2]

        #each of the 3 lab-frame components can be independently estimated by up to 2 of the 3
        #panels (topdown measures x & y, sidex measures x & z, sidey measures y & z) - average
        #whichever estimates are actually available for a given component, rather than summing, so
        #the result isn't biased toward whichever axis happened to get measured twice
        x_estimates, y_estimates, z_estimates = [], [], []
        if topdown is not None:
            x_estimates.append(topdown[0]); y_estimates.append(topdown[1])
        if sidex is not None:
            x_estimates.append(sidex[0]); z_estimates.append(sidex[1])
        if sidey is not None:
            y_estimates.append(sidey[0]); z_estimates.append(sidey[1])

        x_comp = sum(x_estimates) / len(x_estimates) if x_estimates else 0.0
        y_comp = sum(y_estimates) / len(y_estimates) if y_estimates else 0.0
        z_comp = sum(z_estimates) / len(z_estimates) if z_estimates else 0.0

        new_z_lab = x_comp * x_axis + y_comp * y_axis + z_comp * z_axis
        norm = np.linalg.norm(new_z_lab)
        if norm < 1e-9:
            return jsonify({"error": "degenerate annotation (clicked the same point twice?)"}), 400
        new_z_lab /= norm

        new_rot, new_tilt = z_vector_to_rot_tilt(new_z_lab)
        #reprojected onto all 3 panels regardless of which one(s) contributed to the click, so
        #revisiting this particle later shows a consistent, complete picture of the new orientation.
        #new_z_lab is unit-length, so its raw dot products with the frame axes are direction
        #cosines (magnitude <=1) - scaled here by a fraction of the actual box size so the redrawn
        #arrow reads as a real, visible on-panel vector rather than ~1/box_size_angstrom of a pixel
        #(the client's angstromToFrac expects genuine Angstrom-scale values, matching what a live
        #two-click annotation produces)
        box_a = record["box_size_angstrom"]
        arrow_scale = ARROW_DISPLAY_FRACTION * box_a
        correction = dict(
            rot=round(new_rot, 2), tilt=round(new_tilt, 2),
            proj_topdown=[round(float(new_z_lab @ x_axis) * arrow_scale, 2),
                          round(float(new_z_lab @ y_axis) * arrow_scale, 2)],
            proj_sidex=[round(float(new_z_lab @ x_axis) * arrow_scale, 2),
                        round(float(new_z_lab @ z_axis) * arrow_scale, 2)],
            proj_sidey=[round(float(new_z_lab @ y_axis) * arrow_scale, 2),
                        round(float(new_z_lab @ z_axis) * arrow_scale, 2)],
        )
        corrections[key] = correction
        decisions[key] = "accept"
        history.append(key)
        save_corrections()
        save_decisions()
        return jsonify({**correction, **_counts(decisions)})

    @flask_app.route("/api/undo", methods=["POST"])
    def undo():
        if not history:
            return jsonify({"ok": False})
        key = history.pop()
        decisions.pop(key, None)
        corrections.pop(key, None)
        save_decisions()
        save_corrections()
        return jsonify({"ok": True, "key": key, **_counts(decisions)})

    @flask_app.route("/api/export", methods=["POST"])
    def export():
        written = []
        for stem, starfile_path in tomogram_starfiles.items():
            keep_idxs = {r["idx"] for r in all_records if r["stem"] == stem and decisions.get(r["key"]) == "accept"}
            df, df_dict, block_name = load_star_data(starfile_path)
            out_df = df[df.index.isin(keep_idxs)].copy()
            for r in all_records:
                if r["stem"] != stem or r["idx"] not in out_df.index:
                    continue
                c = corrections.get(r["key"])
                if c is not None:
                    out_df.loc[r["idx"], "rlnAngleRot"] = c["rot"]
                    out_df.loc[r["idx"], "rlnAngleTilt"] = c["tilt"]
            out_data = {**df_dict, block_name: out_df} if block_name is not None else out_df
            out_path = starfile_path.parent / f"{stem}_reviewed.star"
            starfile.write(out_data, out_path, overwrite=True)
            written.append(dict(stem=stem, path=str(out_path), n=len(out_df)))

        n_unreviewed = sum(1 for r in all_records if decisions.get(r["key"]) is None)
        n_corrected = sum(1 for r in all_records if decisions.get(r["key"]) == "accept" and r["key"] in corrections)
        return jsonify({"written": written, "unreviewed_excluded": n_unreviewed, "corrected": n_corrected})

    return flask_app


def _counts(decisions: Dict[str, str]) -> dict:
    return dict(accepted=sum(1 for v in decisions.values() if v == "accept"),
                rejected=sum(1 for v in decisions.values() if v == "reject"))


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>gps review</title>
<style>
  :root {
    --bg: #14191a; --card: #1d2426; --line: #2c3537; --ink: #e7ecec; --ink-soft: #94a3a5;
    --teal: #3fa79c; --crimson: #d1495b; --amber: #f2a541;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); font-family: -apple-system, "Segoe UI", sans-serif;
         margin: 0; height: 100vh; display: flex; flex-direction: column; }
  header { display: flex; justify-content: space-between; align-items: center;
           padding: 14px 24px; border-bottom: 1px solid var(--line); flex-shrink: 0; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; color: var(--ink-soft); letter-spacing: 0.02em; }
  #counts { font-family: ui-monospace, monospace; font-size: 13px; color: var(--ink-soft); }
  #counts b.acc { color: var(--teal); } #counts b.rej { color: var(--crimson); }
  main { flex: 1; display: flex; align-items: center; justify-content: center;
         gap: 28px; min-height: 0; padding: 16px; overflow-y: auto; flex-wrap: wrap; }
  .raw-block { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  .block-label { align-self: flex-start; font-size: 12px; color: var(--ink-soft);
                 letter-spacing: 0.06em; text-transform: uppercase; margin-left: 2px; }
  .raw-group { display: flex; gap: 4px; position: relative; border: 3px solid var(--line);
               border-radius: 10px; overflow: hidden; background: #14191a; transition: border-color 0.1s; }
  .raw-group.acc { border-color: var(--teal); } .raw-group.rej { border-color: var(--crimson); }
  .panel-cell { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 10px 8px 12px; }
  .panel-inner { position: relative; line-height: 0; }
  .panel-img { display: block; width: min(40vh, 17vw); aspect-ratio: 1 / 1; }
  .anno-svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
  #svg-topdown, #svg-sidex, #svg-sidey { cursor: crosshair; }
  .anno-line { stroke: #7ee787; stroke-width: 2.5; }
  .anno-dot { fill: #7ee787; }
  .anno-dot.base { fill: none; stroke: #7ee787; stroke-width: 2; }
  #badge { position: absolute; top: 10px; right: 10px; padding: 3px 10px; border-radius: 5px;
           font-size: 12px; font-weight: 600; letter-spacing: 0.03em; display: none; z-index: 2; }
  #badge.acc { display: block; background: var(--teal); color: #06201d; }
  #badge.rej { display: block; background: var(--crimson); color: #2a0508; }
  .side-col { display: flex; flex-direction: column; align-items: center; gap: 8px; flex-shrink: 0; }
  .side-wrap { border: 2px solid var(--line); border-radius: 8px; overflow: hidden; line-height: 0;
               background: #14191a; }
  .side-img { display: block; width: min(40vh, 17vw); aspect-ratio: 1 / 1; }
  .side-label { font-size: 12px; color: var(--ink-soft); letter-spacing: 0.05em; text-transform: uppercase; }
  #meta { width: 280px; font-size: 15px; line-height: 2.2; flex-shrink: 0; }
  #meta .row { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 3px 0; }
  #meta .label { color: var(--ink-soft); }
  #meta .val { font-family: ui-monospace, monospace; }
  #meta .val.corrected { color: var(--teal); }
  #meta h2 { font-size: 22px; margin: 0 0 16px; font-weight: 600; }
  #progress-bar { height: 4px; background: var(--line); flex-shrink: 0; }
  #progress-fill { height: 100%; background: var(--teal); width: 0%; transition: width 0.15s; }
  footer { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px;
           border-top: 1px solid var(--line); font-size: 12.5px; color: var(--ink-soft); flex-shrink: 0; }
  footer kbd { background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px;
               font-family: ui-monospace, monospace; color: var(--ink); margin: 0 2px; }
  button { background: var(--teal); color: #06201d; border: none; border-radius: 6px; padding: 8px 16px;
           font-size: 13px; font-weight: 600; cursor: pointer; }
  button:hover { opacity: 0.9; }
  #done { display: none; text-align: center; color: var(--teal); font-size: 15px; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h1>GPS REVIEW</h1>
  <div id="counts">accepted <b class="acc" id="c-acc">0</b> &middot; rejected <b class="rej" id="c-rej">0</b></div>
  <button onclick="doExport()">Export reviewed STAR files</button>
</header>
<div id="progress-bar"><div id="progress-fill"></div></div>
<main>
  <div class="raw-block">
    <div class="block-label">click base then apex on any panel(s) to annotate, enter to accept</div>
    <div class="raw-group" id="imgwrap">
      <div class="panel-cell">
        <div class="panel-inner">
          <img class="panel-img" id="img-topdown" src="">
          <svg class="anno-svg" id="svg-topdown" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
        </div>
        <div class="side-label">top-down (along particle axis)</div>
      </div>
      <div class="panel-cell">
        <div class="panel-inner">
          <img class="panel-img" id="img-sidex" src="">
          <svg class="anno-svg" id="svg-sidex" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
        </div>
        <div class="side-label">side (x)</div>
      </div>
      <div class="panel-cell">
        <div class="panel-inner">
          <img class="panel-img" id="img-sidey" src="">
          <svg class="anno-svg" id="svg-sidey" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
        </div>
        <div class="side-label">side (y)</div>
      </div>
      <div id="badge"></div>
    </div>
  </div>
  <div class="side-col">
    <div class="side-wrap"><img class="side-img" id="ctximg" src=""></div>
    <div class="side-label">context (particle-oriented)</div>
  </div>
  <div class="side-col">
    <div class="side-wrap"><img class="side-img" id="tradimg" src=""></div>
    <div class="side-label">traditional (tomogram xy)</div>
  </div>
  <div id="meta">
    <h2 id="m-title">-</h2>
    <div class="row"><span class="label">particle idx</span><span class="val" id="m-idx">-</span></div>
    <div class="row"><span class="label">LCCmax</span><span class="val" id="m-lcc">-</span></div>
    <div class="row"><span class="label">rot</span><span class="val" id="m-rot">-</span></div>
    <div class="row"><span class="label">tilt</span><span class="val" id="m-tilt">-</span></div>
    <div class="row"><span class="label">psi</span><span class="val" id="m-psi">-</span></div>
    <div class="row"><span class="label">corrected</span><span class="val" id="m-corrected">no</span></div>
    <div id="done">All particles reviewed.</div>
  </div>
</main>
<footer>
  <span><kbd>enter</kbd> accept (or save annotation, if any) &nbsp;
        <kbd>del</kbd>/<kbd>backspace</kbd> reject &nbsp;
        <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate &nbsp; <kbd>z</kbd> undo</span>
  <span id="progress-text">0 / 0</span>
</footer>
<script>
let queue = [];
let idx = 0;
let counts = {accepted: 0, rejected: 0};

async function init() {
  const res = await fetch('/api/queue');
  queue = await res.json();
  recomputeCounts();
  idx = queue.findIndex(r => !r.decision);
  if (idx === -1) idx = 0;
  render();
}

function recomputeCounts() {
  counts.accepted = queue.filter(r => r.decision === 'accept').length;
  counts.rejected = queue.filter(r => r.decision === 'reject').length;
  document.getElementById('c-acc').innerText = counts.accepted;
  document.getElementById('c-rej').innerText = counts.rejected;
}

//clicking any subset of the 3 raw panels is fine - see /api/correct's docstring for why this
//actually determines a well-posed 3D direction from whichever subset was clicked
const PANEL_AXES = ['topdown', 'sidex', 'sidey'];

//per-particle in-progress clicks, not yet saved: {topdown: {base:[fx,fy], apex:[fx,fy]|null}|null, ...}
//fx/fy are fractions (0-1) of the panel image, top-left origin, matching browser click coordinates
let pendingClicks = {topdown: null, sidex: null, sidey: null};

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgClear(axis) {
  document.getElementById('svg-' + axis).innerHTML = '';
}

function svgDrawVector(axis, bx, by, ax, ay) {
  const svg = document.getElementById('svg-' + axis);
  svg.innerHTML = '';
  const line = document.createElementNS(SVG_NS, 'line');
  line.setAttribute('x1', bx * 100); line.setAttribute('y1', by * 100);
  line.setAttribute('x2', ax * 100); line.setAttribute('y2', ay * 100);
  line.setAttribute('class', 'anno-line');
  svg.appendChild(line);
  svg.appendChild(svgDot(bx * 100, by * 100, true));
  svg.appendChild(svgDot(ax * 100, ay * 100, false));
}

function svgDrawBaseOnly(axis, bx, by) {
  const svg = document.getElementById('svg-' + axis);
  svg.innerHTML = '';
  svg.appendChild(svgDot(bx * 100, by * 100, true));
}

function svgDot(cx, cy, isBase) {
  const dot = document.createElementNS(SVG_NS, 'circle');
  dot.setAttribute('cx', cx); dot.setAttribute('cy', cy); dot.setAttribute('r', isBase ? 2.4 : 3);
  dot.setAttribute('class', 'anno-dot' + (isBase ? ' base' : ''));
  return dot;
}

//angstrom <-> fraction conversion matches how _save_single_panel wrote the image: extent
//[-box_a/2, box_a/2] on both axes, origin='lower' (so image-top = +box_a/2, image-left = -box_a/2)
function fracToAngstrom(pt, boxA) {
  return [(pt[0] - 0.5) * boxA, (0.5 - pt[1]) * boxA];
}

function angstromToFrac(u, v, boxA) {
  return [u / boxA + 0.5, 0.5 - v / boxA];
}

function handlePanelClick(axis, evt) {
  const svg = document.getElementById('svg-' + axis);
  const rect = svg.getBoundingClientRect();
  const fx = (evt.clientX - rect.left) / rect.width;
  const fy = (evt.clientY - rect.top) / rect.height;
  const pc = pendingClicks[axis];
  if (!pc || pc.apex) {
    pendingClicks[axis] = {base: [fx, fy], apex: null};
    svgDrawBaseOnly(axis, fx, fy);
  } else {
    pc.apex = [fx, fy];
    svgDrawVector(axis, pc.base[0], pc.base[1], fx, fy);
  }
}

function clickVectorAngstrom(axis, boxA) {
  const pc = pendingClicks[axis];
  if (!pc || !pc.apex) return null;
  const [bu, bv] = fracToAngstrom(pc.base, boxA);
  const [au, av] = fracToAngstrom(pc.apex, boxA);
  return [au - bu, av - bv];
}

function drawSavedCorrection(r) {
  pendingClicks = {topdown: null, sidex: null, sidey: null};
  PANEL_AXES.forEach(svgClear);
  if (!r.correction) return;
  PANEL_AXES.forEach((axis) => {
    const proj = r.correction['proj_' + axis];
    const [fx, fy] = angstromToFrac(proj[0], proj[1], r.box_size_angstrom);
    svgDrawVector(axis, 0.5, 0.5, fx, fy);
  });
}

async function saveCorrection() {
  if (queue.length === 0) return;
  const r = queue[idx];
  const [topdown, sidex, sidey] = PANEL_AXES.map((axis) => clickVectorAngstrom(axis, r.box_size_angstrom));
  if (!topdown && !sidex && !sidey) return;
  const res = await fetch('/api/correct', {method: 'POST', headers: {'Content-Type': 'application/json'},
                           body: JSON.stringify({key: r.key, topdown, sidex, sidey})});
  if (!res.ok) { const err = await res.json(); alert('Could not save annotation: ' + err.error); return; }
  const data = await res.json();
  r.correction = {rot: data.rot, tilt: data.tilt, proj_topdown: data.proj_topdown,
                  proj_sidex: data.proj_sidex, proj_sidey: data.proj_sidey};
  r.decision = 'accept';
  recomputeCounts();
  if (idx < queue.length - 1) idx++;
  render();
}

function render() {
  if (queue.length === 0) { document.getElementById('m-title').innerText = 'No particles to review.'; return; }
  const r = queue[idx];
  document.getElementById('img-topdown').src = r.image_topdown;
  document.getElementById('img-sidex').src = r.image_sidex;
  document.getElementById('img-sidey').src = r.image_sidey;
  document.getElementById('ctximg').src = r.context_image;
  document.getElementById('tradimg').src = r.traditional_image;
  drawSavedCorrection(r);
  document.getElementById('m-title').innerText = r.stem;
  document.getElementById('m-idx').innerText = r.idx;
  document.getElementById('m-lcc').innerText = (r.lcc === null ? '-' : r.lcc.toFixed(2));
  const rotVal = r.correction ? r.correction.rot : r.rot;
  const tiltVal = r.correction ? r.correction.tilt : r.tilt;
  document.getElementById('m-rot').innerText = rotVal.toFixed(1) + '°';
  document.getElementById('m-tilt').innerText = tiltVal.toFixed(1) + '°';
  document.getElementById('m-psi').innerText = r.psi.toFixed(1) + '°';
  const corrEl = document.getElementById('m-corrected');
  corrEl.innerText = r.correction ? 'yes' : 'no';
  corrEl.className = r.correction ? 'val corrected' : 'val';
  document.getElementById('progress-text').innerText = (idx + 1) + ' / ' + queue.length;
  document.getElementById('progress-fill').style.width = (100 * (idx + 1) / queue.length) + '%';

  const wrap = document.getElementById('imgwrap');
  const badge = document.getElementById('badge');
  const state = r.decision === 'accept' ? 'acc' : (r.decision === 'reject' ? 'rej' : '');
  wrap.className = 'raw-group' + (state ? ' ' + state : '');
  badge.className = state;
  badge.innerText = r.decision === 'accept' ? 'ACCEPTED' : (r.decision === 'reject' ? 'REJECTED' : '');

  document.getElementById('done').style.display = queue.every(r => r.decision) ? 'block' : 'none';
  prefetch();
}

function prefetch() {
  for (let k = 1; k <= 3; k++) {
    const nr = queue[idx + k];
    if (nr) {
      [nr.image_topdown, nr.image_sidex, nr.image_sidey, nr.context_image, nr.traditional_image
      ].forEach((src) => {
        const im = new Image(); im.src = src;
      });
    }
  }
}

async function decide(value) {
  if (queue.length === 0) return;
  const r = queue[idx];
  r.decision = value;
  recomputeCounts();
  fetch('/api/decision', {method: 'POST', headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({key: r.key, decision: value})});
  if (idx < queue.length - 1) idx++;
  render();
}

function navigate(delta) {
  idx = Math.max(0, Math.min(queue.length - 1, idx + delta));
  render();
}

async function undo() {
  const res = await fetch('/api/undo', {method: 'POST'});
  const data = await res.json();
  if (data.ok) {
    const r = queue.find(x => x.key === data.key);
    if (r) { r.decision = null; r.correction = null; }
    recomputeCounts();
    idx = queue.findIndex(x => x.key === data.key);
    render();
  }
}

async function doExport() {
  const res = await fetch('/api/export', {method: 'POST'});
  const data = await res.json();
  let msg = data.written.map(w => `${w.stem}: ${w.n} particles -> ${w.path}`).join('\n');
  if (data.corrected > 0) {
    msg += `\n\n${data.corrected} particle(s) had a manually corrected orientation.`;
  }
  if (data.unreviewed_excluded > 0) {
    msg += `\n\n${data.unreviewed_excluded} unreviewed particle(s) were NOT included.`;
  }
  alert(msg);
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Backspace' || e.code === 'Delete') { e.preventDefault(); decide('reject'); }
  else if (e.code === 'ArrowRight') navigate(1);
  else if (e.code === 'ArrowLeft') navigate(-1);
  else if (e.key === 'z' || e.key === 'Z') undo();
  else if (e.key === 'Enter') {
    e.preventDefault();
    const hasAnnotation = PANEL_AXES.some((axis) => pendingClicks[axis] && pendingClicks[axis].apex);
    if (hasAnnotation) saveCorrection(); else decide('accept');
  }
});

PANEL_AXES.forEach((axis) => {
  document.getElementById('svg-' + axis).addEventListener('click', (e) => handlePanelClick(axis, e));
});

init();
</script>
</body>
</html>"""


@app.command()
def review(
    data_dir: Annotated[Path, typer.Argument(
        help="Directory previously prepared with `gps prepare` - i.e. containing a "
             "gps_review_inputs/ subdirectory")],
    port: Annotated[int, typer.Option("--port", help="Local port to serve the review UI on")] = 5050,
):
    """
    Launch a local web app to rapidly accept/reject particle picks by eye, using the tomogram
    density images `gps prepare` already rendered. Only ever reads data_dir/gps_review_inputs/ and
    serves it - never computes anything itself, so run `gps prepare` first if you haven't.
    """
    if not data_dir.is_dir():
        typer.echo(f"Input Error: {data_dir} is not a directory", err=True)
        raise typer.Exit(code=1)

    #must be absolute: Flask's send_from_directory resolves a relative `directory` against the
    #app's root_path (the installed gps package location), not the process cwd
    data_dir = data_dir.resolve()

    inputs_dir = data_dir / "gps_review_inputs"
    stem_dirs = (sorted(p for p in inputs_dir.iterdir() if (p / "records.json").exists())
                 if inputs_dir.is_dir() else [])
    if not stem_dirs:
        typer.echo(f"Input Error: no prepared review data found under {inputs_dir}. Run "
                   f"`gps prepare {data_dir} --particles_apx ... --tomo_apx ...` first (see "
                   f"`gps prepare --help`).", err=True)
        raise typer.Exit(code=1)

    all_records: List[dict] = []
    tomogram_starfiles: Dict[str, Path] = {}
    for stem_dir in stem_dirs:
        data = json.loads((stem_dir / "records.json").read_text())
        all_records.extend(data["records"])
        tomogram_starfiles[data["stem"]] = Path(data["starfile"])

    if not all_records:
        typer.echo("No particles found in any prepared tomogram - nothing to review.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"{len(all_records)} particles ready for review across {len(stem_dirs)} tomogram(s).")

    flask_app = build_review_app(all_records, inputs_dir, tomogram_starfiles)

    typer.echo(f"\nStarting review server on port {port}.")
    typer.echo("If this is running on a remote cluster, from your local machine run:")
    typer.echo(f"  ssh -L {port}:localhost:{port} <user>@<cluster-host>")
    typer.echo(f"then open http://localhost:{port} in your browser. Otherwise just open that "
               f"address directly.\n")

    flask_app.run(host="127.0.0.1", port=port, debug=False)
