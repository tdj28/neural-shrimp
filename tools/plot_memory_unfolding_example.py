"""Illustrate the exact P2 counterexample; this is not neural scan data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(DATA.resolve()):
        parser.error("keep derived figures in external data storage")
    out.mkdir(parents=True, exist_ok=False)
    grid = np.linspace(-.36, .36, 801)
    X, Y = np.meshgrid(grid, grid)
    k = .2
    settings = [(.5, "Separated curves", "Positive saddle height"),
                (1., "A true trace crossing", "Zero saddle height"),
                (1.5, "Reconnected curves", "Negative saddle height")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 6.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.30, top=.76, wspace=.14)
    fig.suptitle("A crossing can return while coupling remains", x=.065,
                 ha="left", y=.96, fontsize=21, fontweight="bold", color="#152b3a")
    fig.text(.065, .87, "Exact constructed example  •  Two feedback paths change their balance",
             color="#4b6171", fontsize=12)
    records = []
    for ax, (r, title, subtitle) in zip(axes, settings):
        height = (1-r)*k*k
        trace = (Y*Y-X*X)/2 + height
        ax.contourf(X, Y, trace, levels=[-1, -.025, 0, .025, 1],
                    colors=["#dbeaf2", "#edf4f7", "#fcf1e5", "#f7e0c5"])
        for sign in (-1, 1):
            if height > 0:
                ax.plot(sign*np.sqrt(grid*grid+2*height), grid, color="#1c536c", lw=2.7)
            elif height < 0:
                ax.plot(grid, sign*np.sqrt(grid*grid-2*height), color="#1c536c", lw=2.7)
            else:
                ax.plot(grid, sign*grid, color="#1c536c", lw=2.7)
        ax.axhline(0, color="#b0bcc3", lw=.55, zorder=0)
        ax.axvline(0, color="#b0bcc3", lw=.55, zorder=0)
        ax.scatter([0], [0], s=37, c="#dd8434", edgecolors="white", linewidths=.8, zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(grid[0], grid[-1]); ax.set_ylim(grid[0], grid[-1])
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xticks([-.3, 0, .3]); ax.set_yticks([-.3, 0, .3])
        ax.set_xlabel("Rotated toy control U")
        ax.text(.5, -.27, subtitle + f"\nh = {height:+.2f};  r = {r:g}",
                transform=ax.transAxes, ha="center", va="top", fontsize=11)
        records.append({"r": r, "k": k, "height": height,
                        "determinant": -r*k**4, "saddle_multipliers": [-r*k*k, k*k]})
    axes[0].set_ylabel("Rotated toy control V")
    fig.text(.065, .065,
             "Dark curves: zero return trace. Orange dot: the saddle. At the crossing, multipliers are −0.04 and +0.04.",
             fontsize=10, color="#334b5b")
    fig.text(.065, .025,
             "This algebraic counterexample is not a fitted model or a measurement of the neural network.",
             fontsize=10, color="#334b5b")
    outputs = []
    for extension in ("png", "pdf"):
        path = out / ("coupling-counterexample." + extension)
        fig.savefig(path, dpi=190, facecolor="white")
        outputs.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size})
    plt.close(fig)
    source = Path(__file__)
    receipt = {"kind": "exact mathematical illustration, not experimental data",
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "trace": "(V^2-U^2)/2+(1-r)k^2", "coordinate_change": "u=(V+U)/sqrt(2),v=(V-U)/sqrt(2)",
               "settings": records, "outputs": outputs, "visual_review": "pending"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
