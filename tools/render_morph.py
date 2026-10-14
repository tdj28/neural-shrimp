"""Render the morph video (run V1): one 1920 x 1080 image per frame (left: largest Lyapunov exponent, right: period
of the attractor reached from the fixed start, both with fixed colour scales), the rotation angle from the first
direction and a progress bar; then an H.264 MP4 with ffmpeg, and a figure of sample frames for the paper.

Usage: python tools/render_morph.py FRAMES_DIR VECTORS_DIR OUT_DIR [--fps 24] [--samples 6] [--model LABEL]
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


def frames(folder):
    found = sorted(folder.glob("frame-*/scan.npz"))
    if not found:
        raise SystemExit(f"no frames in {folder}")
    idx = [int(p.parent.name.split("-")[1]) for p in found]
    if idx != list(range(idx[0], idx[0] + len(idx))):
        missing = sorted(set(range(idx[0], idx[-1] + 1)) - set(idx))
        raise SystemExit(f"frames missing: {missing[:20]}")
    return idx, found


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
    args = ap.parse_args()
    base = ela.DATA / "explore"
    idx, paths = frames(base / args.frames)
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
        fig.text(0.05, 0.065, f"float32 maps, {L.shape[0]} x {L.shape[1]} cells, every cell started from the same "
                 f"state; constant push size", fontsize=12, color="0.35")
        fig.savefig(out / "png" / f"frame-{k:03d}.png", dpi=100)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-start_number", str(idx[0]),
                    "-i", str(out / "png" / "frame-%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(out / "morph.mp4")], check=True)
    pick = np.linspace(0, len(idx) - 1, args.samples).round().astype(int)
    fig, axes = plt.subplots(1, len(pick), figsize=(3.0 * len(pick), 3.2), squeeze=False)
    for ax, j in zip(axes[0], pick):
        z = np.load(paths[j])
        L, al, lt = z["lyapunov"], z["alpha"], z["log10_temperature"]
        ax.imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=[al[0], al[-1], lt[0], lt[-1]], aspect="auto",
                  cmap="RdBu_r", vmin=-0.5, vmax=0.5, interpolation="nearest")
        ax.set_title(f"{shares[j][1] * angle_total:.0f} deg, chaotic {100 * shares[j][2]:.1f} %", fontsize=9)
        ax.set_xlabel("alpha", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("log10 T", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "morph-samples.png", dpi=250)
    (out / "morph-shares.json").write_text(json.dumps([{"frame": k, "s": s, "chaotic_share": c}
                                                       for k, s, c in shares], indent=1))
    print(json.dumps({"frames": len(idx), "mp4": str(out / "morph.mp4"), "samples": str(out / "morph-samples.png")}))


if __name__ == "__main__":
    main()
