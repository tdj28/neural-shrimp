"""Movie: the attractor of the soft-token loop changing shape along a path through the steering-temperature plane (human,
18 Oct 2026: "a movie that scans from periodic points to chaotic points back to periodic points, showing the attractor
shape evolving, and a panel with the Lyapunov map and a red dot showing our position"; later "make the dots bigger" and
"also travel the vertical axis where alpha is 0.075").

Input: one or more sweep runs (tools/attractor_sweep.py) on the same principal axes, a plan that strings their frames
together in movie order, and the float64 map of the plane (cross-512 by default). Each frame shows (left) the recorded
states of all trajectories at that cell, coloured by the token the loop writes next (each state a dot with a soft halo,
so that a periodic orbit's few points stay visible); (right, top) the map, coloured as in the paper (chaotic cells by
their largest Lyapunov exponent, the others in gray by how strongly they contract), with the path dashed and a red dot
at the frame's cell; (right, bottom) the map's exponent along the path, legs marked, with the same position. The text
gives the cell, the frame's census (periods found among the trajectories, or none up to the sweep's maximum) and the
text one trajectory writes.

Usage: python tools/render_attractor_sweep.py --sweep H=llm-soft-loop/pod64/portraits/sweep \
         --sweep V=llm-soft-loop/pod64/portraits/sweep-col --plan H:0:359,V:153:0,V:0:299 \
         --legs "alpha up at log10 T -0.148|log10 T up at alpha 0.075|log10 T down at alpha 0.075" --out sweep.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
PALETTE = ["#ff6b4d", "#4db8ff", "#ffd93f", "#8cff73", "#d980ff", "#ff99cc", "#66fff2", "#ffb366", "#b3b3ff",
           "#c8ff4d"]
OTHER = "#8c8c99"


def map_image(scan, chaos_level=1e-3):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    d = np.load(DATA / "explore" / scan / "scan.npz")
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    chaos = (ly > chaos_level) & (per <= 0)
    hot = plt.get_cmap("plasma")(Normalize(0.0, float(ly.max()))(np.clip(ly, 0, None)))
    neg = np.clip(-ly, 0, None)
    cap = float(np.percentile(neg[~chaos], 99))
    grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
    img = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
    img[chaos] = hot[chaos]
    return lt, al, img, ly, per, chaos


def census_text(periods, max_period):
    vals, cnt = np.unique(periods, return_counts=True)
    parts = []
    for v, c in sorted(zip(vals, cnt), key=lambda t: -t[1]):
        parts.append(f"no period up to {max_period} in {c}" if v == 0 else f"period {v} in {c}")
    return ", ".join(parts) + f" of {len(periods)} trajectories"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sweep", action="append", required=True, help="NAME=RUN, or RUN (explore-relative); repeatable")
    ap.add_argument("--plan", default=None, help="NAME:FIRST:LAST,... frames in movie order, inclusive, either direction")
    ap.add_argument("--legs", default=None, help="labels of the plan's parts, separated by |")
    ap.add_argument("--scan", default="llm-soft-loop/pod64/cross-512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--hold", type=float, default=1.5, help="seconds to hold the first and last frames")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--keep-frames", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    runs = {}
    for k, sv in enumerate(args.sweep):
        name, run = sv.split("=", 1) if "=" in sv else (f"S{k}", sv)
        z = np.load(DATA / "explore" / run / "sweep.npz")
        rec = json.loads((DATA / "explore" / run / "sweep.json").read_text())
        runs[name] = {"alpha": z["alpha"], "log_t": np.broadcast_to(np.asarray(z["log_t"], dtype=float), z["alpha"].shape),
                      "coords": z["coords"], "tokens": z["tokens"], "periods": z["periods"], "args": rec["args"],
                      "basis": z["basis"]}
    first = next(iter(runs.values()))
    for nm, r in runs.items():
        if not np.allclose(r["basis"], first["basis"]):
            raise SystemExit(f"sweep {nm} uses other principal axes than {next(iter(runs))}; give it --basis-from")
    parts = []
    if args.plan:
        for part in args.plan.split(","):
            nm, a0, a1 = part.split(":")
            a0, a1 = int(a0), int(a1)
            parts.append([(nm, i) for i in (range(a0, a1 + 1) if a1 >= a0 else range(a0, a1 - 1, -1))])
    else:
        nm = next(iter(runs))
        parts.append([(nm, i) for i in range(len(runs[nm]["alpha"]))])
    seq = [x for p in parts for x in p]
    legs = args.legs.split("|") if args.legs else [None] * len(parts)
    A = np.array([runs[nm]["alpha"][i] for nm, i in seq])
    LT = np.array([runs[nm]["log_t"][i] for nm, i in seq])
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    used = np.concatenate([runs[nm]["tokens"][i].reshape(-1) for nm, i in seq])
    uniq, counts = np.unique(used, return_counts=True)
    top = uniq[np.argsort(-counts)][:len(PALETTE)]
    cmap = {int(t): PALETTE[k] for k, t in enumerate(top)}
    xy_all = np.concatenate([runs[nm]["coords"][i, :, :, :2].reshape(-1, 2) for nm, i in seq])
    lo, hi = np.percentile(xy_all, 0.3, axis=0), np.percentile(xy_all, 99.7, axis=0)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    lt, al, img, ly, per, chaos = map_image(args.scan)
    ii = np.array([int(np.argmin(np.abs(lt - x))) for x in LT])
    jj = np.array([int(np.argmin(np.abs(al - x))) for x in A])
    lam_path, chaos_path, per_path = ly[ii, jj], chaos[ii, jj], per[ii, jj]
    sa = first["args"]
    W, H = args.width, args.height
    fs = W / 1920
    plt.rcParams.update({"text.color": "w", "axes.labelcolor": "0.85", "xtick.color": "0.75", "ytick.color": "0.75",
                         "axes.edgecolor": "0.5"})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="k")
    fig.text(0.03, 0.955, "From repeating text to chaos and back: the attractor of a language model's feedback loop",
             fontsize=22 * fs)
    fig.text(0.03, 0.922, f"Qwen2.5-0.5B-Instruct, soft-token loop steered along 'philosophy' at layer 12, "
             f"{sa['dtype']}; each frame: {first['coords'].shape[1]} trajectories after {sa['transient']} steps, "
             f"{first['coords'].shape[2]} states each, on fixed principal axes", fontsize=13 * fs, color="0.75")
    x, y = 0.04, 0.893                               # the token colours, in rows above the attractor
    fig.text(x, y, "colour = next token:", fontsize=11 * fs, color="0.8")
    x += 0.085
    for k, t in enumerate(list(top) + [None]):
        lab = "other" if t is None else repr(tok.decode([int(t)]))
        w = 0.012 + 0.0058 * len(lab)
        if x + w > 0.54:
            x, y = 0.125, y - 0.03
        fig.text(x, y, "●", fontsize=12 * fs, color=OTHER if t is None else PALETTE[k])
        fig.text(x + 0.009, y, lab, fontsize=11 * fs, color="0.9", family="monospace")
        x += w
    axL = fig.add_axes([0.04, 0.17, 0.50, 0.65], facecolor="k")
    axL.set_xlim(lo[0], hi[0])
    axL.set_ylim(lo[1], hi[1])
    axL.set_xlabel("first principal coordinate of the state", fontsize=12 * fs)
    axL.set_ylabel("second principal coordinate", fontsize=12 * fs)
    axL.tick_params(labelsize=9 * fs)
    halo = axL.scatter(np.zeros(1), np.zeros(1), s=110.0 * fs, linewidths=0, alpha=0.10)   # periodic orbits stay visible
    scat = axL.scatter(np.zeros(1), np.zeros(1), s=9.0 * fs, linewidths=0, alpha=0.9)
    axM = fig.add_axes([0.60, 0.47, 0.37, 0.37])
    ext = [al[0], al[-1], lt[0], lt[-1]]
    axM.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    axM.plot(A, LT, "--", color="w", lw=0.9, alpha=0.75)
    dot, = axM.plot([A[0]], [LT[0]], "o", color="red", ms=11 * fs, mec="w", mew=1.5)
    axM.set_xlim(al[0], al[-1])
    axM.set_ylim(lt[0], lt[-1])
    axM.set_xlabel(r"steering strength $\alpha$", fontsize=12 * fs)
    axM.set_ylabel(r"$\log_{10}$ temperature", fontsize=12 * fs)
    axM.set_title("Lyapunov map of the plane (color: chaos; gray: repeating text)\n"
                  "red dot: the cell shown in this frame; dashed line: the path of the movie",
                  fontsize=11.5 * fs, color="0.9", linespacing=1.4)
    axM.tick_params(labelsize=9 * fs)
    axE = fig.add_axes([0.60, 0.17, 0.37, 0.21], facecolor="k")
    m = np.arange(len(seq))
    axE.plot(m, lam_path, color="0.85", lw=1.0)
    axE.fill_between(m, 0, np.where(chaos_path, lam_path, 0), color="#ff8c1a", alpha=0.8, lw=0)
    axE.axhline(0, color="0.6", lw=0.7)
    b = 0
    for p, lab in zip(parts, legs):
        if b:
            axE.axvline(b - 0.5, color="0.6", lw=0.8, ls=":")
        if lab:
            axE.text(b + len(p) / 2, 1.02, lab, transform=axE.get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=8.5 * fs, color="0.8")
        b += len(p)
    vline = axE.axvline(0, color="red", lw=1.4)
    edot, = axE.plot([0], [lam_path[0]], "o", color="red", ms=8 * fs, mec="w", mew=1.0)
    axE.set_xlim(-0.5, len(seq) - 0.5)
    axE.set_ylim(max(-1.6, float(np.percentile(lam_path, 1)) - 0.05), float(lam_path.max()) + 0.08)
    axE.set_xticks([])
    axE.set_xlabel("position along the path (red: this frame)", fontsize=12 * fs)
    axE.set_ylabel("largest Lyapunov\nexponent (map)", fontsize=11 * fs)
    axE.tick_params(labelsize=9 * fs)
    t_alpha = fig.text(0.04, 0.095, "", fontsize=17 * fs)
    t_census = fig.text(0.04, 0.062, "", fontsize=13 * fs, color="0.85")
    t_text = fig.text(0.04, 0.028, "", fontsize=13 * fs, color="#ffd93f", family="monospace")
    tmp = Path(args.keep_frames) if args.keep_frames else Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    tmp.mkdir(parents=True, exist_ok=True)
    hold = int(round(args.hold * args.fps))
    order = [0] * hold + list(range(len(seq))) + [len(seq) - 1] * hold
    rendered = {}
    for i, k in enumerate(order):
        if k in rendered:
            shutil.copyfile(rendered[k], tmp / f"f{i:05d}.png")
            continue
        nm, f = seq[k]
        r = runs[nm]
        xy = r["coords"][f, :, :, :2].reshape(-1, 2)
        tk = r["tokens"][f].reshape(-1)
        cols = [cmap.get(int(t), OTHER) for t in tk]
        for art in (halo, scat):
            art.set_offsets(xy)
            art.set_facecolors(cols)
        dot.set_data([A[k]], [LT[k]])
        vline.set_xdata([k, k])
        edot.set_data([k], [lam_path[k]])
        lab = "chaotic" if chaos_path[k] else (f"period {per_path[k]}" if per_path[k] > 0 else "no period, not chaotic")
        t_alpha.set_text(f"$\\alpha = {A[k]:.5f}$,  $\\log_{{10}}T = {LT[k]:.4f}$     map cell: {lab}, "
                         f"exponent {lam_path[k]:+.3f}")
        t_census.set_text("this frame: " + census_text(r["periods"][f], r["args"]["max_period"]))
        txt = tok.decode(r["tokens"][f, 0, -28:].tolist()).replace("\n", "⏎")
        t_text.set_text("text one trajectory writes: …" + txt[-95:])
        path = tmp / f"f{i:05d}.png"
        fig.savefig(path, facecolor="k")
        rendered[k] = path
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
                    args.out], check=True)
    if not args.keep_frames:
        shutil.rmtree(tmp)
    print(json.dumps({"out": args.out, "frames": len(order), "path_frames": len(seq),
                      "alpha_range": [float(A.min()), float(A.max())], "log_t_range": [float(LT.min()), float(LT.max())],
                      "tokens": [tok.decode([int(t)]) for t in top]}))


if __name__ == "__main__":
    main()
