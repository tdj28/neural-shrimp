"""Figure 9: the same crossing in a real-data network and in Hofmann's two-mode linear network.

Evidence: exploratory. Panel A (docs/explore-realdata-2026-10-07.md): 64-8-1 tanh
network on 1,024 UCI digit images, one global learning rate, weight decay 1e-3.
The numerically validated period-2 orbit loses stability at a complex-pair crossing whose
angle predicts the rotation of the newborn circle (star); the adiabatic sweep
from the numerically validated state measures the rotation beyond it (circles; gray where not
every angle step is forward, i.e. the loop is not star-shaped about its centroid
and the angle estimate is invalid).
Panels B and C (docs/explore-minimal-model-2026-10-07.md): the two-mode depth-two
linear network of Hofmann (2026) with weight decay 0.01, beyond its rotational
edge. B: rotation number against the distance beyond the edge for three
singular-value ratios (birth angle at the edge marked; locked cells in orange).
C: circle amplitude against the distance beyond the edge, with and without weight
decay; the exponents printed are slopes between the two smallest offsets.
"""

import json

import numpy as np

import common as c

RD = c.EXPLORE / "realdata/w8-n1024-halves-wd1e-3"
CONTINUE = RD / "continue-0.42/continue--0.42000.json"
SWEEP = RD / "sweep-ns/sweep.json"
HOF = c.EXPLORE / "minimal-model/hofmann"
STAIR = HOF / "staircase-lam0.01.json"
FINE = HOF / "staircase-lam0.01-fine08.json"
EDGE = HOF / "edge-lam0.01.json"
CIRCLE_WD = HOF / "circle-lam0.01-hysteresis.json"
CIRCLE_NOWD = HOF / "circle-lam0.0.json"
RATIO_STYLE = {0.7: ("o", c.FOLD), 0.8: ("s", c.NS), 0.9: ("^", c.SECONDARY)}


def valid(row):
    """The centroid-angle estimate is valid when every angle step is forward (a star-shaped loop)."""
    return row["finite"] and row["rho2"] is not None and row["rho2_forward"] >= 0.999


def small_offset_slope(rows):
    pos = sorted((r["delta"], r["amplitude"]) for r in rows if r["delta"] > 0 and r["amplitude"] > 0)[:2]
    (d1, a1), (d2, a2) = pos
    return float(np.log(a2 / a1) / np.log(d2 / d1))


def main():
    c.style()
    cont = json.loads(CONTINUE.read_text())
    sweep = json.loads(SWEEP.read_text())["rows"]
    fig, axes = c.plt.subplots(1, 3, figsize=(7.1, 2.8), layout="constrained")
    ax = axes[0]
    good = [r for r in sweep if valid(r)]
    bad = [r for r in sweep if r["finite"] and r["rho2"] is not None and not valid(r)]
    ax.plot([r["log10_eta"] for r in bad], [r["rho2"] for r in bad], "o", color="#c9c8c2", markersize=2.5,
            label="loop not star-shaped")
    ax.plot([r["log10_eta"] for r in good], [r["rho2"] for r in good], "o", color=c.FOLD, markersize=2.5,
            label="measured rotation")
    ax.plot(cont["log10_eta_boundary"], cont["theta_over_2pi"], "*", color=c.FLIP, markersize=9,
            markeredgecolor=c.INK, markeredgewidth=0.4, label="crossing angle (prediction)")
    ax.set_xlabel("log10 learning rate")
    ax.set_ylabel("rotation number (turns per two updates)")
    ax.set_title("A  Real data, one learning rate", fontsize=8, pad=28)
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 0.86), frameon=False, fontsize=5.8)
    nearest = min(good, key=lambda r: abs(r["log10_eta"] - cont["log10_eta_boundary"]))
    bx = axes[1]
    edges = {round(e["s2_over_s1"], 2): e for e in json.loads(EDGE.read_text())}
    stairs = json.loads(STAIR.read_text()) + json.loads(FINE.read_text())
    locked = {}
    for s in stairs:
        ratio = round(s["s2_over_s1"], 2)
        marker, color = RATIO_STYLE[ratio]
        rows = [r for r in s["rows"] if r["rotation"] is not None and np.isfinite(r["rotation"])]
        qp = [r for r in rows if r["period"] == 0]
        lk = [r for r in rows if r["period"] > 0]
        bx.plot([r["a"] - s["edge"] for r in qp], [r["rotation"] for r in qp], ".", color=color, markersize=1.6,
                rasterized=True)
        bx.plot([r["a"] - s["edge"] for r in lk], [r["rotation"] for r in lk], marker, color=c.FLIP, markersize=2.6,
                linestyle="")
        for r in lk:
            locked.setdefault(str(ratio), set()).add((r["period"], round(r["rotation"] * r["period"] / 2)))
        seen = {t.get_label() for t in bx.get_lines()}
        bx.plot(0, edges[ratio]["theta_over_2pi"], marker, color=color, markersize=5, markeredgecolor=c.INK,
                markeredgewidth=0.4, label=f"s2/s1 = {ratio:g}" if f"s2/s1 = {ratio:g}" not in seen else None)
    bx.set_xlabel("distance beyond the edge (learning rate)")
    bx.set_ylabel("rotation number")
    bx.set_title("B  Two-mode linear network", fontsize=8, pad=28)
    bx.legend(loc="upper right", frameon=False, fontsize=6)
    cx = axes[2]
    wd = json.loads(CIRCLE_WD.read_text())
    nowd = json.loads(CIRCLE_NOWD.read_text())
    slopes = {}
    for rows, fill, tag in ((wd, True, "weight decay 0.01"), (nowd, False, "no weight decay")):
        for ratio, (marker, color) in RATIO_STYLE.items():
            sub = [r for r in rows if round(r["s2_over_s1"], 2) == ratio and r["delta"] > 0]
            if not sub:
                continue
            slopes[f"{tag}, s2/s1 = {ratio:g}"] = small_offset_slope(sub)
            cx.plot([r["delta"] for r in sub], [r["amplitude"] for r in sub], marker + "-", color=color, markersize=3,
                    linewidth=0.7, markerfacecolor=color if fill else c.SURFACE)
    cx.set_xscale("log")
    cx.set_yscale("log")
    cx.set_xlabel("distance beyond the edge")
    cx.set_ylabel("circle amplitude")
    wd_s = [v for k, v in slopes.items() if k.startswith("weight")]
    nw_s = [v for k, v in slopes.items() if k.startswith("no")]
    cx.set_title("C  Circle amplitude", fontsize=8, pad=28)
    def span(values):
        lo, hi = f"{min(values):.2f}", f"{max(values):.2f}"
        return lo if lo == hi else f"{lo}-{hi}"

    cx.text(0.03, 1.03, f"filled: weight decay 0.01, slope {span(wd_s)}\nopen: no weight decay, slope {span(nw_s)}",
            transform=cx.transAxes, va="bottom", fontsize=6, color=c.INK)
    derived = {"realdata_crossing_log10_eta": cont["log10_eta_boundary"], "realdata_predicted": cont["theta_over_2pi"],
               "realdata_multiplier": cont["dominant"], "realdata_nearest_measured": nearest["rho2"],
               "realdata_nearest_log10_eta": nearest["log10_eta"], "sweep_valid_points": len(good),
               "hofmann_birth_angles": {str(k): v["theta_over_2pi"] for k, v in edges.items()},
               "hofmann_locked": {k: sorted([list(x) for x in v]) for k, v in locked.items()},
               "amplitude_slopes": slopes}
    alt = (f"Left: in a 529-parameter tanh network trained on digit images with one learning rate, the period-2 orbit "
           f"acquires complex multipliers at log10 learning rate {cont['log10_eta_boundary']:.4f}; the crossing angle "
           f"{cont['theta_over_2pi']:.7f} predicts the rotation measured on the circle just beyond, "
           f"{nearest['rho2']:.6f}. Middle: in the two-mode linear network of Hofmann (2026) with weight decay, the "
           f"rotation number beyond the edge starts at the crossing angle for three singular-value ratios, with locked "
           f"periodic cells at ratio 0.7 and 0.8. Right: the circle amplitude grows with exponent "
           f"{min(wd_s):.2f}-{max(wd_s):.2f} with weight decay (square-root onset) and "
           f"{min(nw_s):.2f}-{max(nw_s):.2f} without.")
    c.save(fig, "f9_generality", [CONTINUE, SWEEP, STAIR, FINE, EDGE, CIRCLE_WD, CIRCLE_NOWD], __file__, "exploratory",
           alt, derived, "Amplitude exponents are slopes between the two smallest offsets.")


if __name__ == "__main__":
    main()
