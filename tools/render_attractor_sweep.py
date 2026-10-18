"""Movie: the attractor of the soft-token loop changing shape along a line of the steering-temperature plane (human,
18 Oct 2026: "a movie that scans from periodic points to chaotic points back to periodic points, showing the attractor
shape evolving, and a panel with the Lyapunov map and a red dot showing our position").

Input: a sweep run (tools/attractor_sweep.py) and the float64 map of the same plane (cross-512 by default). Each frame
shows (left) the recorded states of all trajectories at that steering strength on the sweep's fixed principal axes,
coloured by the token the loop writes next; (right, top) the map, coloured as in the paper (chaotic cells by their
largest Lyapunov exponent, the others in grey by how strongly they contract) with a red dot at the frame's cell; (right,
bottom) the map's exponent along the swept line with the same position marked. The text gives the frame's census
(periods found among the trajectories, or none up to the sweep's maximum) and the text one trajectory writes.

Usage: python tools/render_attractor_sweep.py --sweep llm-soft-loop/pod64/portraits/sweep --out sweep.mp4
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
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--scan", default="llm-soft-loop/pod64/cross-512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--hold", type=float, default=1.0, help="seconds to hold the first and last frames")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--keep-frames", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    z = np.load(DATA / "explore" / args.sweep / "sweep.npz")
    rec = json.loads((DATA / "explore" / args.sweep / "sweep.json").read_text())
    sa = rec["args"]
    alphas, coords, tokens, periods = z["alpha"], z["coords"], z["tokens"], z["periods"]
    log_t = float(z["log_t"])
    nF = len(alphas)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    uniq, counts = np.unique(tokens, return_counts=True)
    top = uniq[np.argsort(-counts)][:len(PALETTE)]
    cmap = {int(t): PALETTE[k] for k, t in enumerate(top)}
    xy_all = coords[..., :2].reshape(-1, 2)
    lo, hi = np.percentile(xy_all, 0.3, axis=0), np.percentile(xy_all, 99.7, axis=0)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    lt, al, img, ly, per, chaos = map_image(args.scan)
    row = int(np.argmin(np.abs(lt - log_t)))
    W, H = args.width, args.height
    fs = W / 1920
    plt.rcParams.update({"text.color": "w", "axes.labelcolor": "0.85", "xtick.color": "0.75", "ytick.color": "0.75",
                         "axes.edgecolor": "0.5"})
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="k")
    fig.text(0.03, 0.955, "From repeating text to chaos and back: the attractor of a language model's feedback loop",
             fontsize=22 * fs)
    fig.text(0.03, 0.922, f"Qwen2.5-0.5B-Instruct, soft-token loop steered along 'philosophy' at layer 12, "
             f"$\\log_{{10}}T={log_t}$, {sa['dtype']}; each frame: {coords.shape[1]} trajectories after {sa['transient']} "
             f"steps, {coords.shape[2]} states each, on fixed principal axes", fontsize=13 * fs, color="0.75")
    axL = fig.add_axes([0.04, 0.17, 0.50, 0.70], facecolor="k")
    axL.set_xlim(lo[0], hi[0])
    axL.set_ylim(lo[1], hi[1])
    axL.set_xlabel("first principal coordinate of the state", fontsize=12 * fs)
    axL.set_ylabel("second principal coordinate", fontsize=12 * fs)
    axL.tick_params(labelsize=9 * fs)
    scat = axL.scatter(np.zeros(1), np.zeros(1), s=4.0 * fs, linewidths=0, alpha=0.85)
    x = 0.04                                         # the token colours, as one line above the attractor
    fig.text(x, 0.893, "colour = next token:", fontsize=11 * fs, color="0.8")
    x += 0.085
    for k, t in enumerate(list(top) + [None]):
        lab = "other" if t is None else repr(tok.decode([int(t)]))
        fig.text(x, 0.893, "\u25cf", fontsize=12 * fs, color=OTHER if t is None else PALETTE[k])
        fig.text(x + 0.009, 0.893, lab, fontsize=11 * fs, color="0.9", family="monospace")
        x += 0.012 + 0.0058 * len(lab)
    axM = fig.add_axes([0.60, 0.47, 0.37, 0.42])
    ext = [al[0], al[-1], lt[0], lt[-1]]
    axM.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    axM.axhline(log_t, color="w", lw=0.8, ls="--", alpha=0.7)
    dot, = axM.plot([alphas[0]], [log_t], "o", color="red", ms=11 * fs, mec="w", mew=1.5)
    axM.set_xlabel(r"steering strength $\alpha$", fontsize=12 * fs)
    axM.set_ylabel(r"$\log_{10}$ temperature", fontsize=12 * fs)
    axM.set_title("where we are: the Lyapunov map of this plane (color: chaos, gray: repeating text)",
                  fontsize=11.5 * fs, color="0.9")
    axM.tick_params(labelsize=9 * fs)
    axE = fig.add_axes([0.60, 0.17, 0.37, 0.21], facecolor="k")
    lam = ly[row]
    axE.plot(al, lam, color="0.85", lw=1.0)
    axE.fill_between(al, 0, np.where(chaos[row], lam, 0), color="#ff8c1a", alpha=0.8, lw=0)
    axE.axhline(0, color="0.6", lw=0.7)
    vline = axE.axvline(alphas[0], color="red", lw=1.4)
    edot, = axE.plot([alphas[0]], [0], "o", color="red", ms=8 * fs, mec="w", mew=1.0)
    axE.set_xlim(al[0], al[-1])
    axE.set_ylim(max(-1.6, float(np.percentile(lam, 1)) - 0.05), float(lam.max()) + 0.05)
    axE.set_xlabel(r"steering strength $\alpha$ along the dashed line", fontsize=12 * fs)
    axE.set_ylabel("largest Lyapunov\nexponent (map)", fontsize=11 * fs)
    axE.tick_params(labelsize=9 * fs)
    t_alpha = fig.text(0.04, 0.095, "", fontsize=17 * fs)
    t_census = fig.text(0.04, 0.062, "", fontsize=13 * fs, color="0.85")
    t_text = fig.text(0.04, 0.028, "", fontsize=13 * fs, color="#ffd93f", family="monospace")
    tmp = Path(args.keep_frames) if args.keep_frames else Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    tmp.mkdir(parents=True, exist_ok=True)
    hold = int(round(args.hold * args.fps))
    order = [0] * hold + list(range(nF)) + [nF - 1] * hold
    rendered = {}
    for i, f in enumerate(order):
        if f in rendered:
            shutil.copyfile(rendered[f], tmp / f"f{i:05d}.png")
            continue
        xy = coords[f, :, :, :2].reshape(-1, 2)
        tk = tokens[f].reshape(-1)
        scat.set_offsets(xy)
        scat.set_facecolors([cmap.get(int(t), OTHER) for t in tk])
        a = float(alphas[f])
        dot.set_data([a], [log_t])
        j = int(np.argmin(np.abs(al - a)))
        vline.set_xdata([a, a])
        edot.set_data([a], [lam[j]])
        lab = "chaotic" if chaos[row, j] else (f"period {per[row, j]}" if per[row, j] > 0 else "no period, not chaotic")
        t_alpha.set_text(f"$\\alpha = {a:.5f}$     map cell: {lab}, exponent {lam[j]:+.3f}")
        t_census.set_text("this frame: " + census_text(periods[f], sa["max_period"]))
        ids = tokens[f, 0, -28:].tolist()
        txt = tok.decode(ids).replace("\n", "⏎")
        t_text.set_text("text one trajectory writes: …" + txt[-95:])
        path = tmp / f"f{i:05d}.png"
        fig.savefig(path, facecolor="k")
        rendered[f] = path
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
                    args.out], check=True)
    if not args.keep_frames:
        shutil.rmtree(tmp)
    print(json.dumps({"out": args.out, "frames": len(order), "sweep_frames": nF, "map_row_log_t": float(lt[row]),
                      "tokens": [tok.decode([int(t)]) for t in top]}))


if __name__ == "__main__":
    main()
