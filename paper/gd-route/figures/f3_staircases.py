"""Figure 3: devil's staircases of the two-step rotation number along learning-rate slices.

Evidence: exploratory. Each slice varies the hidden-layer rate at a fixed
output-layer rate (tools/explore_a3_staircase.py): 2,000 cells, 40,000 updates,
rotation number of the two-step map from the last 4,000 states in each cell's
own principal plane; labels from EXP-028 screening of the first 20,000 updates.
Locked plateaus (periodic, period 2q, rotation m/q) are detected by
common.plateaus; quasi-periodic candidates are non-periodic cells with
|FTLE| < 0.005.
"""

import json

import numpy as np

import common as c

LABEL_MIN_CELLS = 8  # label merged plateaus (common.merged_plateaus) with at least this many locked cells
SLICES = [("S2-28011", c.EXPLORE / "a3-staircase/s2-28011-out-0.0856/staircase.json"),
          ("S1R-28002", c.EXPLORE / "a3-staircase/s1r-28002-out-0.9385/staircase.json")]


def main():
    c.style()
    fig, axes = c.plt.subplots(1, 2, figsize=(7.1, 3.3), layout="constrained")
    derived = {}
    for ax, letter, (name, path) in zip(axes, "AB", SLICES):
        data = json.loads(path.read_text())
        rows = data["rows"]
        merged = c.merged_plateaus(rows)
        derived[name] = {
            "fixed_log10_output": data["args"]["fixed"], "range_log10_hidden": data["args"]["range"],
            "cells": len(rows), "periodic_cells": sum(r["status"] == 2 for r in rows),
            "quasi_periodic_cells": sum(c.quasi_periodic(r) for r in rows),
            "chaotic_cells": sum(r["status"] == 3 for r in rows),
            "labeled_plateaus": [{k: p[k] for k in ("m", "q", "start", "end", "cells", "runs")}
                                 for p in merged if p["cells"] >= LABEL_MIN_CELLS],
        }
        groups = {
            "period 2 (decaying-spiral rotation)": [r for r in rows if r["status"] == 2 and r["period"] == 2
                                                          and r["rho"] is not None],
            "periodic, period >= 4": [r for r in rows if r["status"] == 2 and r["period"] >= 4 and r["rho"] is not None],
            "quasi-periodic candidate": [r for r in rows if c.quasi_periodic(r)],
            "chaotic candidate": [r for r in rows if r["status"] == 3 and r["rho"] is not None],
        }
        colors = list(zip(["#f5b99f", c.CLASS_COLORS[2], c.INK, c.CLASS_COLORS[3]], [3.0, 3.0, 2.0, 2.0]))
        style = dict(zip(groups, colors))
        for label, members in groups.items():
            if members:
                color, size = style[label]
                ax.plot([r["log10_varied"] for r in members], [r["rho"] for r in members], ".", markersize=size,
                        color=color, label=label, rasterized=True)
        previous = None
        spans = [p for p in c.merged_plateaus(rows) if p["cells"] >= LABEL_MIN_CELLS]
        best = {}
        for p in spans:
            if p["cells"] > best.get((p["m"], p["q"]), {"cells": 0})["cells"]:
                best[(p["m"], p["q"])] = p
        for p in sorted(best.values(), key=lambda p: p["start"]):
            mid = 0.5 * (p["start"] + p["end"])
            below = previous is not None and mid - previous < 0.02 * (data["args"]["range"][1] - data["args"]["range"][0]) * 5
            ax.annotate(f"{p['m']}/{p['q']}", (mid, p["m"] / p["q"]), textcoords="offset points",
                        xytext=(0, -10 if below else 5), ha="center", fontsize=6.5, color=c.INK)
            previous = None if below else mid
        regular = [r["rho"] for r in rows if r["rho"] is not None and r["rho"] > 0.02
                   and ((r["status"] == 2 and r["period"] >= 4) or c.quasi_periodic(r))]
        lo, hi = min(regular), max(regular)
        ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.set_xlim(data["args"]["range"])
        ax.set_xlabel(f"log10 hidden-layer rate (log10 output rate {data['args']['fixed']:.4f})")
        ax.set_ylabel("rotation number of the two-step map")
        ax.set_title(f"{letter}  {name}: rotation-number staircase")
        if letter == "A":
            fig.legend(loc="outside lower center", ncol=4, frameon=False, markerscale=3, fontsize=6.5,
                       columnspacing=1.2, handletextpad=0.3)
    parts = []
    for name, d in derived.items():
        labels = ", ".join(f"{p['m']}/{p['q']}" for p in d["labeled_plateaus"])
        parts.append(f"{name}: {d['quasi_periodic_cells']} quasi-periodic and {d['periodic_cells']} periodic of "
                     f"{d['cells']} cells, plateaus at {labels}")
    alt = ("Rotation number of the two-step map against the hidden-layer learning rate along one slice in each of two "
           "networks. Quasi-periodic cells trace a continuous curve; periodic cells sit on flat plateaus at rational "
           "values m/q whose period is 2q (or a doubling of it); chaotic cells scatter. " + "; ".join(parts) + ".")
    c.save(fig, "f3_staircases", [p for _, p in SLICES], __file__, "exploratory", alt, derived,
           "Plateaus: common.merged_plateaus (runs of the same m/q within 0.003 in log10 rate merged); labels need "
           f"at least {LABEL_MIN_CELLS} locked cells.")


if __name__ == "__main__":
    main()
