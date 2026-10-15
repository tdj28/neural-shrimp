"""Render the morph video (run V1): one 1920 x 1080 image per frame (left: largest Lyapunov exponent, right: period
of the attractor reached from the fixed start, both with fixed colour scales), the rotation angle from the first
direction and a progress bar; then an H.264 MP4 with ffmpeg, and a figure of sample frames for the paper.

Usage: python tools/render_morph.py FRAMES_DIR VECTORS_DIR OUT_DIR [--fps 24] [--samples 6] [--model LABEL] [--step 2]
                                     [--samples-only]   (only the figure of sample frames: no frame images, no video)
       (FRAMES_DIR holds frame-NNN/scan.npz; paths relative to the data folder's explore/)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def frames(folder, step=1):
    """Frame indices with a finished map (frame-NNN/scan.npz), sorted, and the planned indices on the grid of the
    given step between the first and last that have none (reported, and bridged in the video)."""
    found = sorted(folder.glob("frame-*/scan.npz"), key=lambda p: int(p.parent.name.split("-")[1]))
    if len(found) < 2:
        raise SystemExit(f"fewer than two frames in {folder}")
    idx = [int(p.parent.name.split("-")[1]) for p in found]
    missing = sorted(set(range(idx[0], idx[-1] + 1, step)) - set(idx))
    return idx, found, missing


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames")
    ap.add_argument("vectors")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--model", default="Qwen2.5-0.5B-Instruct")
    ap.add_argument("--step", type=int, default=1, help="planned spacing of the frame indices")
    ap.add_argument("--seconds-per-index", type=float, default=1 / 12,
                    help="video time per frame index: each map is held until the next one, so the rotation advances "
                         "at a constant rate and missing frames are bridged by a longer crossfade")
    ap.add_argument("--samples-only", action="store_true", help="only the figure of sample frames")
    args = ap.parse_args()
    base = ela.DATA / "explore"
    idx, paths, missing = frames(base / args.frames, args.step)
    print(json.dumps({"frames": len(idx), "missing": missing}), flush=True)
    info = json.loads((base / args.vectors / "vectors.json").read_text())
    angle_total = np.degrees(info["angle_rad"])
    out = Path(args.out)
    (out / "png").mkdir(parents=True, exist_ok=True)
    cmap_p = plt.get_cmap("tab20").copy()
    cmap_p.set_bad("white")
    shares = []
    for k, path in zip(idx, paths):
        z = np.load(path)
        L, P, al, lt = z["lyapunov"], z["period"], z["alpha"], z["log10_temperature"]
        s = info["frames_s"][k]
        chaotic = float(((L > 1e-3) & (P <= 0)).mean())
        shares.append((k, s, chaotic))
        if args.samples_only:
            continue
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        fig.patch.set_facecolor("white")
        ext = [al[0], al[-1], lt[0], lt[-1]]
        ax1 = fig.add_axes([0.05, 0.13, 0.40, 0.71])
        ax2 = fig.add_axes([0.53, 0.13, 0.40, 0.71])
        im1 = ax1.imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                         vmin=-0.5, vmax=0.5, interpolation="nearest")
        im2 = ax2.imshow(np.where(P > 0, P, np.nan), origin="lower", extent=ext, aspect="auto", cmap=cmap_p,
                         vmin=0.5, vmax=40.5, interpolation="nearest")
        for ax in (ax1, ax2):
            ax.set_xlabel("steering strength", fontsize=15)
            ax.set_ylabel("log10 temperature", fontsize=15)
            ax.tick_params(labelsize=12)
        ax1.set_title(f"largest Lyapunov exponent   (chaotic cells: {100 * chaotic:.1f} %)", fontsize=16)
        ax2.set_title("period of the attractor (white: none)", fontsize=16)
        c1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
        c2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
        c1.ax.tick_params(labelsize=11)
        c2.ax.tick_params(labelsize=11)
        fig.suptitle(f"{args.model}, soft-token feedback map: the steering direction rotates from one random "
                     f"direction to another  ({s * angle_total:5.1f} of {angle_total:.1f} degrees)", fontsize=19,
                     y=0.95)
        bar = fig.add_axes([0.05, 0.035, 0.88, 0.018])
        bar.barh([0], [s], color="0.25", height=1.0)
        bar.set_xlim(0, 1)
        bar.axis("off")
        fig.text(0.05, 0.065, f"float32 maps, {L.shape[0]} x {L.shape[1]} cells; each map continues from the previous "
                 f"map's final states (every tenth from the fixed start); constant push size", fontsize=12, color="0.35")
        fig.savefig(out / "png" / f"frame-{k:03d}.png", dpi=100)
        plt.close(fig)
    # concat list: each map held until the next available one (constant rotation speed), then 24 fps with linear
    # blending between consecutive maps (ffmpeg framerate filter), which bridges the missing frames
    if not args.samples_only:                                # the video
        lines = []
        for j, k in enumerate(idx):
            nxt = idx[j + 1] if j + 1 < len(idx) else k + args.step
            lines += [f"file '{(out / 'png' / f'frame-{k:03d}.png').resolve()}'", f"duration {(nxt - k) * args.seconds_per_index:.6f}"]
        lines.append(f"file '{(out / 'png' / f'frame-{idx[-1]:03d}.png').resolve()}'")
        (out / "frames.txt").write_text("\n".join(lines) + "\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(out / "frames.txt"),
                        "-vf", f"framerate=fps={args.fps}:interp_start=0:interp_end=255:scene=100",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
                        str(out / "morph.mp4")], check=True)
    pick = np.linspace(0, len(idx) - 1, args.samples).round().astype(int)
    fig, axes = plt.subplots(1, len(pick), figsize=(3.0 * len(pick), 3.2), squeeze=False)
    for ax, j in zip(axes[0], pick):
        z = np.load(paths[j])
        L, al, lt = z["lyapunov"], z["alpha"], z["log10_temperature"]
        im = ax.imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=[al[0], al[-1], lt[0], lt[-1]], aspect="auto",
                       cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
        ax.set_title(f"{shares[j][1] * angle_total:.0f} deg, chaotic {100 * shares[j][2]:.1f} %", fontsize=9)
        ax.set_xlabel("alpha", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("log10 T", fontsize=8)
    fig.tight_layout()
    cb = fig.colorbar(im, ax=list(axes[0]), fraction=0.012, pad=0.008, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cb.ax.set_yticklabels([r"$\leq -0.5$", "$-0.25$", "0", "0.25", r"$\geq 0.5$"], fontsize=7)
    cb.set_label("largest Lyapunov exponent per step", fontsize=8)
    fig.savefig(out / "morph-samples.png", dpi=250, bbox_inches="tight", pad_inches=0.05)    # keeps the bar's labels
    (out / "morph-shares.json").write_text(json.dumps({"missing_frames": missing, "frames": [
        {"frame": k, "s": s, "chaotic_share": c} for k, s, c in shares]}, indent=1))
    print(json.dumps({"frames": len(idx), "missing": len(missing), "mp4": str(out / "morph.mp4"),
                      "samples": str(out / "morph-samples.png")}))


if __name__ == "__main__":
    main()
