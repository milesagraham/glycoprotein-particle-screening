# GPS: Glycoprotein Particle Screening

A command-line tool for rapidly screening particle picks (e.g. from `pytom`) by eye, from a
RELION-style STAR file of particle coordinates/orientations and the tomogram density they were
picked from.

For each particle it renders a real tomogram density slice oriented to that particle's own
orientation (not a membrane normal - this tool does no membrane/segmentation analysis at all), so
picks can be compared side by side in a consistent frame regardless of where or how they sit in the
tomogram.

![Example: one particle rendered in its own frame - top-down along its own axis, and two side views 90 degrees apart](docs/example_idx0_panels.png)

## Installation

Create a conda environment and `pip install` the package into it:

```bash
conda create -n gps python=3.10
conda activate gps
pip install -e .
```

This installs the `gps` command and its dependencies (`typer`, `mrcfile`, `starfile`, `numpy`, `scipy`,
`pandas`, `flask`, `matplotlib`). Python 3.8+ works; 3.10 is just a safe default. Re-run `pip install -e .`
after pulling new changes, and `conda activate gps` again at the start of every new shell session.

## Usage

`gps` is a two-step, batch-first tool: `prepare` always renders a review image for every particle
found under a directory, and `review` is a strictly separate second step that only ever displays
images `prepare` already rendered - it never renders anything itself. Run `gps --help` to see both
commands.

There is no automatic accept/reject step anywhere in this tool - every particle from every matched
tomogram/starfile pair goes straight into the manual review queue.

### Directory layout

Both commands take one `data_dir` argument, expected to contain these subdirectories, with
matching files (same filename stem, e.g. `ts_028.mrc` / `ts_028.star`) across them treated as one
tomogram:

- `tomograms/` — one tomogram density `.mrc` per tomogram.
- `starfiles/` — one RELION-style particle `.star` file per tomogram, with at least
  `rlnCoordinateX/Y/Z` and `rlnAngleRot`/`rlnAngleTilt`/`rlnAnglePsi` columns. `rlnLCCmax`, if
  present (as in `pytom_match_pick` output), is shown during review but never used to filter.

No `segmentations/` directory is needed - this tool doesn't do any membrane analysis.

### Step 1: `gps prepare`

For each particle, builds its own local orientation frame from `rlnAngleRot`/`rlnAngleTilt`/`rlnAnglePsi`
and renders three orthogonal tomogram density slices in that frame, plus one wider context slice.
Runs across every matched tomogram under `data_dir`, in parallel:

```bash
gps prepare /path/to/data_dir --particles_apx 3.728 --tomo_apx 7.456
```

Pixel sizes are required since the tomogram and particle coordinates are often binned differently.

#### Key options

| Option | Default | What it does |
|---|---|---|
| `--box-size-angstrom` | 300 Å | Width/height of the three close-up panels. |
| `--context-box-size-angstrom` | 2000 Å | Width/height of the wider context panel. |
| `--workers` / `-j` | 4 | Number of tomograms rendered in parallel - they're fully independent of each other. `--workers 1` disables parallelism. |

Run `gps prepare --help` for the full list.

`--workers` parallelizes across tomograms using Python's `ProcessPoolExecutor`, which only spreads
work across CPU cores on the single machine the command is running on - it cannot reach across
nodes. On a cluster, submit `gps prepare` as a **single-node** job, sized to that node's core count;
requesting multiple SLURM nodes for one invocation will not speed it up, since every node but the
one actually running the process sits idle.

#### Output

One review image per particle is written to `data_dir/gps_review_inputs/<stem>/` (not into
`starfiles/` itself), together with a `records.json` per tomogram that `gps review` reads. This is
the expensive part of the whole tool (tomogram slicing, image rendering), so results are cached per
tomogram, fingerprinted on the input files and the options used - re-running only redoes tomograms
whose inputs or parameters actually changed, so an interrupted batch job can safely be resubmitted.

### Step 2: reviewing results with `gps review`

```bash
gps review /path/to/data_dir
```

Launches a local web app for fast, keyboard-driven manual triage. It only ever reads
`data_dir/gps_review_inputs/` and serves it - it never renders anything itself, and refuses to
start if `gps prepare` hasn't been run yet. Every particle from every prepared tomogram is combined
into a single review queue.

Each particle is shown as three panels in its own frame - a top-down view looking straight down the
particle's own pointing axis, and two side views 90 degrees apart around that axis - next to a
wider context view (same side-on orientation, zoomed out) showing where on the tomogram the pick
sits, e.g. relative to a virus/vesicle surface:

![Example: the same particle's wider context view](docs/example_idx0_context.png)

Unlike `gps`'s predecessor tool (GCA, which compares a particle's orientation against an
independently-fitted membrane normal), there's no second vector to compare a particle's orientation
against here, so the panels don't draw an orientation arrow - the frame itself *is* the particle's
orientation. The point of the three views is to let you pattern-match real particles vs. junk across
a consistent, comparable presentation.

#### Splitting compute from review on a cluster

Because `gps prepare` never starts a network server, and `gps review` never renders anything, they're
a natural fit for opposite ends of a cluster: run the parallel rendering as a batch job on a compute
node with a high `--workers`, then review on the login node:

```bash
# on a compute node, e.g. inside a Slurm job:
gps prepare /path/to/data_dir --particles_apx 3.728 --tomo_apx 7.456 --workers 32
```

```bash
# on the login node, once the batch job has finished:
gps review /path/to/data_dir
```

Once `gps review` is running:

- Open the printed URL directly, or - if running on a remote cluster - tunnel it first:
  ```bash
  ssh -L 5050:localhost:5050 user@cluster-host
  ```
  then open `http://localhost:5050` locally.
- `space` accepts the current particle and advances; `Backspace`/`Delete` rejects it (junk) and
  advances; `←`/`→` navigate without deciding; `z` undoes the last decision.
- Decisions are saved continuously to `data_dir/gps_review_inputs/decisions.json`, so a review
  session can be closed and resumed later without losing progress.
- The "Export reviewed STAR files" button writes one `<stem>_reviewed.star` per tomogram
  (accepted particles only) next to that tomogram's input STAR file.
