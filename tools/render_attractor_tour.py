"""Movie: a three-dimensional tour of an attractor of the soft-token loop (human, 18 Oct 2026: "a 3d tour of the strange
attractor").

The recorded states of a portrait run (tools/attractor_portrait.py) are placed at their first three principal
coordinates and viewed by a camera that circles the attractor once (azimuth 0 to 360 degrees, elevation swinging
between --elev-min and --elev-max) in perspective. Each frame is a density image: every state adds its next token's
color to its pixel, nearer states more strongly (depth cue), and the summed color is tone-mapped on a black
background, with a faint glow. Frames are encoded with ffmpeg (H.264).

Usage: python tools/render_attractor_tour.py --run llm-soft-loop/pod64/portraits/f64-strong --out tour.mp4
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
PALETTE = [(1.00, 0.42, 0.30), (0.30, 0.72, 1.00), (1.00, 0.85, 0.25), (0.55, 1.00, 0.45), (0.85, 0.50, 1.00),
           (1.00, 0.60, 0.80), (0.40, 1.00, 0.95), (1.00, 0.70, 0.40)]
OTHER = (0.60, 0.60, 0.65)


def rotation(az, el):
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return Rx @ Rz


def render(P, col, az, el, W, H, scale, dist, glow):
    from scipy.ndimage import gaussian_filter
    Q = P @ rotation(az, el).T                     # camera frame: x right, y depth, z up
    depth = Q[:, 1]
    f = dist / (dist + depth)                       # perspective
    x = Q[:, 0] * f
    y = Q[:, 2] * f
    wdepth = np.exp(-0.8 * (depth - depth.min()) / (np.ptp(depth) + 1e-12))
    px = ((x * scale) + W / 2).astype(int)
    py = ((-y * scale) + H / 2).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    idx = py[ok] * W + px[ok]
    img = np.zeros((H * W, 3))
    for c in range(3):
        img[:, c] = np.bincount(idx, weights=col[ok, c] * wdepth[ok], minlength=H * W)
    img = gaussian_filter(img.reshape(H, W, 3), sigma=(0.7, 0.7, 0))      # splat each state over about a pixel
    lum = img.sum(2, keepdims=True)
    pos = lum[lum > 1e-9]
    l0, l1 = np.percentile(pos, 55), np.percentile(pos, 99.9)
    hue = img / np.maximum(lum, 1e-12)
    inten = np.clip(np.log1p(lum / l0) / np.log1p(l1 / l0), 0, 1)        # logarithmic exposure: faint filaments show
    out = hue * inten ** 0.8
    if glow > 0:
        out = out + glow * gaussian_filter(out, sigma=(3, 3, 0))
    return np.clip(out ** 0.85, 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--elev-min", type=float, default=8.0)
    ap.add_argument("--elev-max", type=float, default=38.0)
    ap.add_argument("--glow", type=float, default=0.6)
    ap.add_argument("--dims", default="0,1,2", help="principal coordinates for x, y, z")
    ap.add_argument("--title", default=None)
    ap.add_argument("--keep-frames", default=None, help="folder to keep the PNG frames in")
    ap.add_argument("--all-trajectories", action="store_true", help="keep trajectories that reach a periodic orbit (with "
                    "their transients: the approach to a periodic attractor)")
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    z = np.load(DATA / "explore" / args.run / "portrait.npz", allow_pickle=True)
    rec = json.loads((DATA / "explore" / args.run / "portrait.json").read_text())
    dims = [int(d) for d in args.dims.split(",")]
    C = z["coords"]
    T = z["tokens"]
    aper = (np.ones(len(z["periods"]), dtype=bool) if args.all_trajectories
            else z["periods"] == 0)                 # the chaotic attractor only: drop starts that reached a periodic orbit
    P = C[aper][:, :, dims].reshape(-1, 3).astype(np.float64)
    toks = T[aper].reshape(-1)
    P = P - np.median(P, axis=0)
    P = P / np.percentile(np.linalg.norm(P, axis=1), 99.0)                # frame the bulk, not the farthest excursions
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    uniq, counts = np.unique(toks, return_counts=True)
    top = uniq[np.argsort(-counts)][:len(PALETTE)]
    col = np.tile(np.array(OTHER), (len(toks), 1))
    for k, t in enumerate(top):
        col[toks == t] = PALETTE[k]
    labels = [repr(tok.decode([int(t)])).replace("\\n\\n", "\u23ce").replace("\\n", "\u23ce") for t in top]
    a = rec["args"]
    title = args.title or "A chaotic attractor of a language model's feedback loop (numerically observed)"
    sub = args.subtitle or (f"Qwen2.5-0.5B-Instruct, soft-token loop, steering $\\alpha={a['alpha']}$, "
                            f"$\\log_{{10}}T={a['log_t']}$, {a['dtype']}")
    W, H = args.width, args.height
    tmp = Path(args.keep_frames) if args.keep_frames else Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    tmp.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="k")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(np.zeros((H, W, 3)), interpolation="nearest")
    fs = W / 1920
    fig.text(0.02, 0.955, title, color="w", fontsize=24 * fs)
    fig.text(0.02, 0.922, sub, color="0.8", fontsize=15 * fs)
    when = f"each after {a['transient']} steps" if a["transient"] else "recorded from their start"
    fig.text(0.02, 0.025, f"{len(P):,} states of {int(aper.sum())} trajectories, {when}; "
             f"projection on principal coordinates {', '.join(str(d + 1) for d in dims)}; color: the most probable next token",
             color="0.6", fontsize=12 * fs)
    for k, lab in enumerate(labels):
        fig.text(0.02, 0.86 - 0.035 * k, "●", color=PALETTE[k], fontsize=16)
        fig.text(0.04, 0.862 - 0.035 * k, lab, color="0.9", fontsize=13, family="monospace")
    fig.text(0.02, 0.86 - 0.035 * len(labels), "●", color=OTHER, fontsize=16)
    fig.text(0.04, 0.862 - 0.035 * len(labels), "other tokens", color="0.9", fontsize=13)
    cam = fig.text(0.98, 0.03, "", color="0.6", fontsize=11, ha="right")
    scale = 0.62 * min(W, H)
    for i in range(args.frames):
        u = i / args.frames
        az = 2 * np.pi * u
        el = np.radians(args.elev_min + (args.elev_max - args.elev_min) * 0.5 * (1 - np.cos(2 * np.pi * u)))
        im.set_data(render(P, col, az, el, W, H, scale, 3.0, args.glow))
        cam.set_text(f"azimuth {np.degrees(az):5.1f}°, elevation {np.degrees(el):4.1f}°")
        fig.savefig(tmp / f"f{i:05d}.png", facecolor="k")
    plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow", "-movflags", "+faststart",
                    args.out], check=True)
    if not args.keep_frames:
        shutil.rmtree(tmp)
    print(json.dumps({"out": args.out, "frames": args.frames, "states": int(len(P)), "trajectories": int(aper.sum()),
                      "tokens": labels}))


if __name__ == "__main__":
    main()
