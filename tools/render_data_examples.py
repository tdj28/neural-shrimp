"""Video: closing slides of the educational cusp video, from the language model's own data (human, 18 Oct 2026: "end it
by showing actual examples ... from our actual data").

Slides (1280 x 720, slow zoom), cut from paper figures:
1. the period-5 shrimp of fig-real-shrimp, panel (a): followed domain, fold (black) and flip (red) curves, located cusps;
2. the junction of its upper leg, panel (d): two cusps that share a segment of the fold curve and the crossing of its
   other branches (a section of a swallowtail), with the orbit census;
3. a closing card: crossroad areas are verified in the models; in the model's own map two cusps of the node kind are the
   candidates and the classification is open.

Usage: python tools/render_data_examples.py --out examples.mp4
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "llm" / "figures" / "fig-real-shrimp.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", default="10,10,8", help="durations of the three slides")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    img = plt.imread(FIG)
    H, W = img.shape[:2]
    crops = {"a": img[: int(0.50 * H), : int(0.50 * W)], "d": img[int(0.50 * H):, int(0.50 * W):]}
    durs = [float(x) for x in args.seconds.split(",")]
    slides = [
        ("a", "The real thing: a shrimp in a language model's own map",
         "the period-5 window of Qwen2.5-0.5B-Instruct (gratitude steered at layer 4, hostility at layer 16), followed\n"
         "cell by cell; black: fold curves, red dashed: flip curves, gold stars: located cusps"),
        ("d", "Where the upper leg leaves the body: a section of a swallowtail",
         "two cusps share a segment of the fold curve and its other branches cross (magenta); at 13 points fixed in\n"
         "advance the period-5 orbit counts matched the swallowtail's prediction at 9 (green)"),
        (None, "What is verified, and what is open", None),
    ]
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, args.out, dpi=100):
        for (key, title, caption), dur in zip(slides, durs):
            n = int(round(dur * args.fps))
            for i in range(n):
                u = i / max(n - 1, 1)
                fig.clf()
                fig.text(0.04, 0.94, title, fontsize=19, weight="bold", va="top")
                if key is not None:
                    im = crops[key]
                    h, w = im.shape[:2]
                    z = 1.0 + 0.10 * (u * u * (3 - 2 * u))           # slow zoom towards the center
                    ax = fig.add_axes([0.04, 0.14, 0.92, 0.72])
                    ax.imshow(im, interpolation="lanczos")
                    cx, cy = w / 2, h / 2
                    ax.set_xlim(cx - w / (2 * z), cx + w / (2 * z))
                    ax.set_ylim(cy + h / (2 * z), cy - h / (2 * z))
                    ax.set_axis_off()
                    fig.text(0.04, 0.07, caption, fontsize=11.5, color="0.25", va="center", linespacing=1.4)
                else:
                    rows = [("verified in the models", "saddle, crossroad and spring areas (quartic map; Gallas's canonical cell)"),
                            ("verified in the model's map", "the shrimp's fold and flip curves; two swallowtail sections\n"
                             "(dovetails); located cusps of both kinds; 9 of 13 orbit counts as predicted"),
                            ("open", "whether the two node-kind cusps organize crossroad or spring areas:\n"
                             "the search is under way")]
                    for k, (head, body) in enumerate(rows):
                        y = 0.72 - 0.22 * k
                        fig.text(0.08, y, head, fontsize=17, weight="bold", color=["tab:blue", "tab:green", "tab:red"][k])
                        fig.text(0.08, y - 0.07, body, fontsize=14, va="top", linespacing=1.4)
                writer.grab_frame()
    print(json.dumps({"out": args.out, "slides": len(slides), "seconds": sum(durs),
                      "slide_starts": np.cumsum([0.0] + durs[:-1]).tolist()}))


if __name__ == "__main__":
    main()
