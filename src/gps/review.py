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
  orientation against here - the review frame *is* the particle's own orientation. So there are no
  orientation arrows to draw: every particle is already displayed "canonically", and the point of
  the three panels is just to let a reviewer visually pattern-match real particles vs junk across a
  consistent set of views.
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
- Re-running `gps prepare` skips any tomogram whose input files and parameters are unchanged since
  the last run (fingerprinted in gps_review_inputs/<stem>/fingerprint.txt), so an interrupted batch
  job can be resubmitted without repeating already-finished tomograms.
- Decisions are saved continuously to gps_review_inputs/decisions.json, so a review session can be
  closed and resumed later without losing progress.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import mrcfile
import starfile
import typer
from typing_extensions import Annotated
from scipy.ndimage import map_coordinates

from gps.cli import app, load_star_data, euler_to_rotation_matrix

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


def _draw_panel(ax, img: np.ndarray, box_a: float, title: str) -> None:
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-box_a / 2, box_a / 2, -box_a / 2, box_a / 2])
    ax.scatter(0, 0, c='#f9a825', s=90, marker='+', linewidth=2.2, zorder=5)
    ax.set_title(title, fontsize=10, color='#c7d0d1')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_review_panel(path: Path, img_xy: np.ndarray, img_xz: np.ndarray, img_yz: np.ndarray,
                        box_a: float) -> None:
    """Renders the three orthogonal views of the particle's own local frame: xy (looking down the
    particle's own pointing axis), and the two side views xz/yz, 90 degrees apart around it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor='#14191a')
    _draw_panel(axes[0], img_xy, box_a, "top-down (along particle axis)")
    _draw_panel(axes[1], img_xz, box_a, "side (x)")
    _draw_panel(axes[2], img_yz, box_a, "side (y)")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def _save_context_panel(path: Path, img: np.ndarray, box_a: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor='#14191a')
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-box_a / 2, box_a / 2, -box_a / 2, box_a / 2])
    ax.scatter(0, 0, c='#f9a825', s=140, marker='+', linewidth=2.6, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def render_review_data_for_tomogram(
    tomo_set: dict, inputs_dir: Path, particles_apx: float, tomo_apx: float,
    box_size_angstrom: float, context_box_size_angstrom: float, slab_slices: int,
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
                  context_box_size_angstrom=context_box_size_angstrom, slab_slices=slab_slices)
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

        filename = f"idx{i}.png"
        _save_review_panel(stem_dir / filename, img_xy, img_xz, img_yz, box_size_angstrom)

        context_img = _extract_slab(tomo, pos, x_axis, z_axis, y_axis, context_grid_a,
                                    context_grid_b, slab_slices)
        context_filename = f"idx{i}_context.png"
        _save_context_panel(stem_dir / context_filename, context_img, context_box_size_angstrom)

        traditional_img = _extract_slab(tomo, pos, LAB_X, LAB_Y, LAB_Z, context_grid_a,
                                        context_grid_b, slab_slices)
        traditional_filename = f"idx{i}_traditional.png"
        _save_context_panel(stem_dir / traditional_filename, traditional_img, context_box_size_angstrom)

        records.append(dict(
            key=f"{stem}:{i}", stem=stem, idx=i,
            lcc=(None if np.isnan(lccmax[i]) else float(lccmax[i])),
            rot=round(float(rot), 2), tilt=round(float(tilt), 2), psi=round(float(psi), 2),
            image=f"/api/image/{stem}/{filename}",
            context_image=f"/api/image/{stem}/{context_filename}",
            traditional_image=f"/api/image/{stem}/{traditional_filename}",
        ))

    typer.echo(f"[{stem}] {len(records)} review images ready.")
    return _finish(records)


def build_review_app(all_records: List[dict], inputs_dir: Path, tomogram_starfiles: Dict[str, Path]):
    from flask import Flask, jsonify, request, send_from_directory, Response

    flask_app = Flask(__name__)
    decisions_path = inputs_dir / "decisions.json"
    decisions: Dict[str, str] = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    history: List[str] = []

    def save_decisions():
        tmp = decisions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(decisions, indent=2))
        tmp.replace(decisions_path)

    @flask_app.route("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @flask_app.route("/api/queue")
    def queue():
        out = []
        for r in all_records:
            r2 = dict(r)
            r2["decision"] = decisions.get(r["key"])
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

    @flask_app.route("/api/undo", methods=["POST"])
    def undo():
        if not history:
            return jsonify({"ok": False})
        key = history.pop()
        decisions.pop(key, None)
        save_decisions()
        return jsonify({"ok": True, "key": key, **_counts(decisions)})

    @flask_app.route("/api/export", methods=["POST"])
    def export():
        written = []
        for stem, starfile_path in tomogram_starfiles.items():
            keep_idxs = {r["idx"] for r in all_records if r["stem"] == stem and decisions.get(r["key"]) == "accept"}
            df, df_dict, block_name = load_star_data(starfile_path)
            out_df = df[df.index.isin(keep_idxs)].copy()
            out_data = {**df_dict, block_name: out_df} if block_name is not None else out_df
            out_path = starfile_path.parent / f"{stem}_reviewed.star"
            starfile.write(out_data, out_path, overwrite=True)
            written.append(dict(stem=stem, path=str(out_path), n=len(out_df)))

        n_unreviewed = sum(1 for r in all_records if decisions.get(r["key"]) is None)
        return jsonify({"written": written, "unreviewed_excluded": n_unreviewed})

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
  main { flex: 1; display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
         min-height: 0; padding: 16px; gap: 14px; }
  #imgwrap { position: relative; border: 3px solid var(--line); border-radius: 10px; overflow: hidden;
             transition: border-color 0.1s; line-height: 0; background: #14191a; }
  #imgwrap.acc { border-color: var(--teal); } #imgwrap.rej { border-color: var(--crimson); }
  #img { max-height: 52vh; max-width: 34vw; display: block; }
  #badge { position: absolute; top: 10px; right: 10px; padding: 3px 10px; border-radius: 5px;
           font-size: 12px; font-weight: 600; letter-spacing: 0.03em; display: none; }
  #badge.acc { display: block; background: var(--teal); color: #06201d; }
  #badge.rej { display: block; background: var(--crimson); color: #2a0508; }
  .side-col { display: flex; flex-direction: column; align-items: center; gap: 8px; flex-shrink: 0; }
  .side-wrap { border: 2px solid var(--line); border-radius: 8px; overflow: hidden; line-height: 0;
               background: #14191a; }
  .side-img { max-height: 32vh; max-width: 14vw; display: block; }
  .side-label { font-size: 11px; color: var(--ink-soft); letter-spacing: 0.05em; text-transform: uppercase; }
  #meta { width: 260px; font-size: 14px; line-height: 2.1; flex-shrink: 0; }
  #meta .row { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 2px 0; }
  #meta .label { color: var(--ink-soft); }
  #meta .val { font-family: ui-monospace, monospace; }
  #meta h2 { font-size: 20px; margin: 0 0 14px; font-weight: 600; }
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
  <div id="imgwrap"><img id="img" src=""><div id="badge"></div></div>
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
    <div id="done">All particles reviewed.</div>
  </div>
</main>
<footer>
  <span><kbd>space</kbd> accept &nbsp; <kbd>del</kbd>/<kbd>backspace</kbd> reject &nbsp;
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

function render() {
  if (queue.length === 0) { document.getElementById('m-title').innerText = 'No particles to review.'; return; }
  const r = queue[idx];
  document.getElementById('img').src = r.image;
  document.getElementById('ctximg').src = r.context_image;
  document.getElementById('tradimg').src = r.traditional_image;
  document.getElementById('m-title').innerText = r.stem;
  document.getElementById('m-idx').innerText = r.idx;
  document.getElementById('m-lcc').innerText = (r.lcc === null ? '-' : r.lcc.toFixed(2));
  document.getElementById('m-rot').innerText = r.rot.toFixed(1) + '°';
  document.getElementById('m-tilt').innerText = r.tilt.toFixed(1) + '°';
  document.getElementById('m-psi').innerText = r.psi.toFixed(1) + '°';
  document.getElementById('progress-text').innerText = (idx + 1) + ' / ' + queue.length;
  document.getElementById('progress-fill').style.width = (100 * (idx + 1) / queue.length) + '%';

  const wrap = document.getElementById('imgwrap');
  const badge = document.getElementById('badge');
  wrap.className = r.decision === 'accept' ? 'acc' : (r.decision === 'reject' ? 'rej' : '');
  badge.className = wrap.className;
  badge.innerText = r.decision === 'accept' ? 'ACCEPTED' : (r.decision === 'reject' ? 'REJECTED' : '');

  document.getElementById('done').style.display = queue.every(r => r.decision) ? 'block' : 'none';
  prefetch();
}

function prefetch() {
  for (let k = 1; k <= 3; k++) {
    if (queue[idx + k]) {
      const im = new Image(); im.src = queue[idx + k].image;
      const ctxIm = new Image(); ctxIm.src = queue[idx + k].context_image;
      const tradIm = new Image(); tradIm.src = queue[idx + k].traditional_image;
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
    if (r) r.decision = null;
    recomputeCounts();
    idx = queue.findIndex(x => x.key === data.key);
    render();
  }
}

async function doExport() {
  const res = await fetch('/api/export', {method: 'POST'});
  const data = await res.json();
  let msg = data.written.map(w => `${w.stem}: ${w.n} particles -> ${w.path}`).join('\n');
  if (data.unreviewed_excluded > 0) {
    msg += `\n\n${data.unreviewed_excluded} unreviewed particle(s) were NOT included.`;
  }
  alert(msg);
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); decide('accept'); }
  else if (e.code === 'Backspace' || e.code === 'Delete') { e.preventDefault(); decide('reject'); }
  else if (e.code === 'ArrowRight') navigate(1);
  else if (e.code === 'ArrowLeft') navigate(-1);
  else if (e.key === 'z' || e.key === 'Z') undo();
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
