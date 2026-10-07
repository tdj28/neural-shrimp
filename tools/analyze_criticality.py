"""Analysis for the criticality stream (exploratory).

  boundary  for each Neimark-Sacker point inside a GPU rotation map's output range:
            the first divergent map cell above the boundary (in cells and decades),
            the labels just below and above, and the first Lyapunov coefficient d
  figure    d along the curve with the soft/hard classification of the fixed-start
            maps just beyond the curve (quasi-periodic, chaotic, divergent)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
CRIT = DATA / "explore" / "criticality"
MERGED = DATA / "explore/a3-ns-curve/s2-28011-merged/ns-curve.json"
MAPS = [DATA / "explore/a3-gpu-rotation/s2-28011-1024/rotation.npz",
        DATA / "explore/a3-gpu-rotation/s2-28011-right/rotation.npz"]
LABELS = {0: "divergent", 1: "fixed", 2: "periodic", 3: "chaotic", 4: "unresolved", 5: "drift"}


def curve_rows():
    out = {}
    for name in ("s2-28011-curve", "s2-28011-steep"):
        path = CRIT / name / "curve.json"
        if path.exists():
            for r in json.loads(path.read_text())["rows"]:
                if r.get("type") == "neimark_sacker":
                    out[round(r["log10_output"], 4)] = r
    return out


def merged_rows():
    rows = [r for r in json.loads(MERGED.read_text())["rows"] if r.get("type") == "neimark_sacker"]
    return sorted(rows, key=lambda r: r["log10_output"])


def classify_beyond(m, o, hb, offsets_cells=(2, 4, 9, 18, 36)):
    lh, lo, st, ftle = m["log10_hidden"], m["log10_output"], m["status"], m["ftle"]
    j = int(np.argmin(np.abs(lo - o)))
    i0 = int(np.searchsorted(lh, hb))
    column = st[:, j]
    above = np.flatnonzero(column[i0:] == 0)
    first_div = int(above[0]) if len(above) else None
    labels = []
    for k in offsets_cells:
        i = i0 + k
        if i >= len(lh):
            labels.append(None)
            continue
        s = int(st[i, j])
        if s in (4, 5) and np.isfinite(ftle[i, j]) and abs(ftle[i, j]) < 5e-3:
            labels.append("quasi-periodic")
        else:
            labels.append(LABELS[s])
    below = [LABELS[int(s)] for s in column[max(i0 - 3, 0):i0]]
    return {"first_divergent_cells_above": first_div,
            "first_divergent_decades_above": None if first_div is None else float(first_div * (lh[1] - lh[0])),
            "labels_beyond_cells": dict(zip(map(str, offsets_cells), labels)), "labels_just_below": below,
            "cell_decades": float(lh[1] - lh[0])}


def cmd_boundary(args):
    crit = curve_rows()
    out = []
    for r in merged_rows():
        o, hb = r["log10_output"], r["log10_hidden_boundary"]
        for path in MAPS:
            m = np.load(path)
            if m["log10_output"][0] <= o <= m["log10_output"][-1] and m["log10_hidden"][0] <= hb <= m["log10_hidden"][-1]:
                c = crit.get(round(o, 4))
                row = {"log10_output": o, "log10_hidden_boundary": hb, "map": path.parent.name,
                       "d": None if c is None else c["d"], **classify_beyond(m, o, hb)}
                out.append(row)
                break
    (CRIT / "boundary.json").write_text(json.dumps(out, indent=1))
    for row in out:
        d = "  n/a  " if row["d"] is None else f"{row['d']:+.3f}"
        print(f"{row['log10_output']:+.4f} d {d} first divergent cell above: {row['first_divergent_cells_above']}"
              f" | beyond (cells 2,4,9,18,36): {list(row['labels_beyond_cells'].values())} | below {row['labels_just_below']}")


def cmd_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((CRIT / "boundary.json").read_text())
    crit = sorted(curve_rows().values(), key=lambda r: r["log10_output"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, constrained_layout=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    o = np.array([r["log10_output"] for r in crit])
    d = np.array([r["d"] for r in crit])
    ax1.axhline(0, color="#898781", linewidth=0.8)
    ax1.plot(o, d, "-", color="#52514e", linewidth=1.0)
    ax1.scatter(o[d < 0], d[d < 0], s=14, color="#2a78d6", label="d < 0: supercritical (soft)", zorder=3)
    ax1.scatter(o[d > 0], d[d > 0], s=14, color="#eb6834", label="d > 0: subcritical (hard)", zorder=3)
    ax1.set_yscale("symlog", linthresh=0.05)
    ax1.set_ylabel("first Lyapunov coefficient d", fontsize=9)
    ax1.legend(fontsize=8, frameon=False, loc="lower left")
    ax1.set_title("Criticality of the period-2 Neimark-Sacker curve, S2-28011 (exploratory)", loc="left", fontsize=10)
    colors = {"quasi-periodic": "#1baf7a", "chaotic": "#a3a29b", "divergent": "#eb6834", "periodic": "#2a78d6",
              "fixed": "#52514e", "unresolved": "#e1e0d9", "drift": "#1baf7a", None: "#ffffff"}
    for r in rows:
        labels = list(r["labels_beyond_cells"].values())
        for k, lab in enumerate(labels):
            ax2.scatter(r["log10_output"], k, s=18, marker="s", color=colors.get(lab, "#ffffff"))
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(["+2 cells", "+4", "+9", "+18", "+36"], fontsize=8)
    ax2.set_ylabel("fixed-start map\nbeyond the curve", fontsize=9)
    ax2.set_xlabel("log10 output-layer learning rate (position along the curve)", fontsize=9)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=c, label=lab)
               for lab, c in (("quasi-periodic or drift", "#1baf7a"), ("chaotic", "#a3a29b"),
                              ("divergent", "#eb6834"), ("periodic", "#2a78d6"))]
    ax2.legend(handles=handles, fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    out = CRIT / "criticality.png"
    fig.savefig(out, dpi=170)
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("boundary")
    sub.add_parser("figure")
    args = parser.parse_args()
    {"boundary": cmd_boundary, "figure": cmd_figure}[args.command](args)


if __name__ == "__main__":
    main()
