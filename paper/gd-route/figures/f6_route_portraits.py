"""Figure 6: what training looks like at four points of the route along one slice.

Evidence: exploratory. The four hidden-layer rates are chosen by a fixed rule
from the S2-28011 staircase receipt (output rate fixed at the 2/9 window's
column): (A) the period-2 cell closest to the onset of quasi-periodicity (its
two-step samples spiral into the 2-cycle); (B) the median quasi-periodic cell;
(C) the middle cell of the longest 2/9 plateau; (D) the median chaotic cell
between the 2/9 and 1/4 plateaus. Trajectories are recomputed here from the
shared initial parameters (S2-28011, 40,000 full-batch updates, float64 on
CPU); each panel projects one lobe of the two-step map on that cell's own
first two principal axes.
"""

import json
import sys

import numpy as np

import common as c

sys.path.insert(0, str(c.ROOT / "src"))

from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

STAIRCASE = c.EXPLORE / "a3-staircase/s2-28011-out-0.0856/staircase.json"
SOURCES = [c.ROOT / "src/neural_shrimp/gdmaps.py", c.ROOT / "src/neural_shrimp/trainability.py",
           c.ROOT / "src/neural_shrimp/models.py"]
STEPS = 40000


def select(rows):
    """Apply the fixed selection rule; returns {panel: row}."""
    p2 = [r for r in rows if r["status"] == 2 and r["period"] == 2 and r["rho"]]
    qp = [r for r in rows if c.quasi_periodic(r)]
    merged = c.merged_plateaus(rows)
    lock = max((p for p in merged if (p["m"], p["q"]) == (2, 9)), key=lambda p: p["cells"])
    quarter = max((p for p in merged if (p["m"], p["q"]) == (1, 4)), key=lambda p: p["cells"])
    locked = [r for r in rows if lock["start"] <= r["log10_varied"] <= lock["end"] and r["status"] == 2]
    chaotic = [r for r in rows if r["status"] == 3 and lock["end"] < r["log10_varied"] < quarter["start"]]
    return {"A": max(p2, key=lambda r: r["rho"]), "B": qp[len(qp) // 2], "C": locked[len(locked) // 2],
            "D": chaotic[len(chaotic) // 2]}


def main():
    c.style()
    data = json.loads(STAIRCASE.read_text())
    chosen = select(data["rows"])
    net, x, y, theta0 = SYSTEMS["S2"](28011)
    log_h = np.array([chosen[k]["log10_varied"] for k in "ABCD"])
    grid = GridMap(net, x, y, 10.0 ** log_h, np.full(4, 10.0 ** data["args"]["fixed"]))
    theta = np.repeat(theta0[None], 4, axis=0)
    early, late = [], []
    for step in range(STEPS):
        theta = grid.step(theta)
        if 2000 <= step < 20000 and step % 2 == 0:
            early.append(theta[0].copy())
        if step >= STEPS - 8000 and (step - (STEPS - 8000)) % 2 == 0:
            late.append(theta.copy())
    early, late = np.array(early), np.array(late)
    titles = {"A": "A  Period 2: inward spiral", "B": "B  Quasi-periodic circle",
              "C": "C  Locked 2/9, period 18", "D": "D  Chaotic: broken circle"}
    fig, axes = c.plt.subplots(1, 4, figsize=(7.1, 2.25), layout="constrained")
    derived = {}
    for i, (ax, key) in enumerate(zip(axes, "ABCD")):
        points = early if key == "A" else late[:, i, :]
        centered = points - points.mean(0)
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        planar = centered @ vt[:2].T
        share = float((singular[:2] ** 2).sum() / (singular ** 2).sum())
        if key == "A":
            from matplotlib.colors import LinearSegmentedColormap
            ramp = LinearSegmentedColormap.from_list("time", [c.PERIOD_RAMP[0], c.PERIOD_RAMP[-1]])
            ax.scatter(planar[:, 0], planar[:, 1], c=np.arange(len(planar)), cmap=ramp, s=0.6, linewidths=0,
                       rasterized=True)
            ax.text(0.02, 0.02, "light: early, dark: late", transform=ax.transAxes, fontsize=6, color=c.SECONDARY)
        else:
            size = 1.2 if key in "BD" else 9.0
            ax.plot(planar[:, 0], planar[:, 1], ".", markersize=size,
                    color=c.INK if key != "D" else c.CLASS_COLORS[3], rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[key], fontsize=7.5)
        r = chosen[key]
        ax.set_xlabel(f"log10 hidden rate {r['log10_varied']:.4f}\nrotation {r['rho']:.4f}, FTLE {r['ftle']:.4f}",
                      fontsize=6.5)
        derived[key] = {"log10_hidden": r["log10_varied"], "log10_output": data["args"]["fixed"],
                        "status": r["status"], "period": r["period"], "rho": r["rho"], "ftle": r["ftle"],
                        "planar_variance_share": share, "samples": int(len(points))}
    alt = ("Four phase portraits of full-batch training of the 13-parameter network along one learning-rate slice, "
           "each showing every second parameter vector projected on its own two principal axes. "
           f"A: period 2 near onset (rotation of the decaying spiral {derived['A']['rho']:.4f}); "
           f"B: a closed invariant curve (quasi-periodic, rotation {derived['B']['rho']:.4f}, FTLE "
           f"{derived['B']['ftle']:.4f}); C: nine points on the circle (locked at 2/9, period 18); "
           f"D: a thickened, broken circle (chaotic, FTLE {derived['D']['ftle']:.4f}).")
    c.save(fig, "f6_route_portraits", [STAIRCASE, *SOURCES], __file__, "exploratory", alt, derived,
           "Panel A uses updates 2,000 to 20,000 (the transient spiral); B to D use the last 8,000 updates.")


if __name__ == "__main__":
    main()
