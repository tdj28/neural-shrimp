"""Figure 8: resonance ancestry of the rule-selected 1/5 window (S2-28011).

Evidence: exploratory (docs/explore-ancestry-2026-10-07.md), with frozen
acceptance rules. Six cross-sections beyond the predicted 1/5 resonance point
(right root) at detunings delta = 0.0128 / 4^k; on each, both fold edges of the
locked period-10 interval. Panels A and B: width and amplitude of the locked
interval against delta, with the resonant normal-form control's levels
(controls-002) and the theory slopes (q - 2) / 2 = 1.5 and 1/2 drawn through the
deepest level; the post-hoc slopes fitted over the three deepest levels are
printed. Panel C: centre offset of the locked interval against delta, with the
frozen estimator (linear fit over all six levels), its intercept and the frozen
tolerance (the frozen outcome is "alternative ancestry"), and a post-hoc fit over
the three deepest levels; the axes show the three deepest levels only.
"""

import json

import numpy as np

import common as c

LEVELS = c.EXPLORE / "ancestry/levels-right-attempt2/levels.json"
CONTROLS = c.EXPLORE / "ancestry/controls-002/controls.json"
CONNECT = c.EXPLORE / "ancestry/connect-right/connect.json"


def slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    c.style()
    levels = json.loads(LEVELS.read_text())
    controls = json.loads(CONTROLS.read_text())
    connect = json.loads(CONNECT.read_text())
    det = levels["details"]
    delta = np.array(levels["deltas"])
    width, amp, centre = (np.array(det[k]) for k in ("widths", "amplitudes", "centers"))
    pos = controls["positive"]
    pos_delta = np.array([lv["delta"] for lv in pos["levels"]])
    pos_width, pos_amp = np.array(pos["details"]["widths"]), np.array(pos["details"]["amplitudes"])
    deep = slice(len(delta) - 3, len(delta))
    w_slope, a_slope = slope(delta[deep], width[deep]), slope(delta[deep], amp[deep])
    fig, axes = c.plt.subplots(1, 3, figsize=(7.1, 2.75), layout="constrained")
    for ax, values, control, theory, fitted, title, ylabel in (
            (axes[0], width, pos_width, 1.5, w_slope, "A  Width of the locked interval", "width (decades)"),
            (axes[1], amp, pos_amp, 0.5, a_slope, "B  Amplitude relative to the 2-cycle", "amplitude")):
        ax.plot(delta, values, "o-", color=c.FOLD, markersize=4, linewidth=0.8, label="network, right 1/5 root")
        ax.plot(pos_delta, control, "s", color=c.MUTED, markersize=3.5, markerfacecolor="none",
                label="normal-form control")
        guide = values[-1] * (delta / delta[-1]) ** theory
        ax.plot(delta, guide, "--", color=c.SECONDARY, linewidth=0.8, label=f"slope {theory:g} (theory)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("detuning beyond the root (decades)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.text(0.04, 0.96, f"post hoc slope,\ndeepest 3 levels: {fitted:.3f}", transform=ax.transAxes,
                va="top", fontsize=6.3, color=c.INK)
    axes[0].legend(loc="lower right", frameon=False, fontsize=6)
    ax = axes[2]
    ax.plot(delta, centre, "o", color=c.FOLD, markersize=4)
    fit = np.polyfit(delta, centre, 1)
    deep_fit = np.polyfit(delta[deep], centre[deep], 1)
    grid = np.linspace(0, 3.2e-4, 200)
    ax.plot(grid, np.polyval(fit, grid), "-", color=c.SECONDARY, linewidth=0.9, label="frozen linear fit, all 6 levels")
    ax.plot(grid, np.polyval(deep_fit, grid), ":", color=c.FOLD, linewidth=0.9, label="post hoc fit, deepest 3 levels")
    ax.axhline(det["root_tolerance"], color=c.FLIP, linewidth=0.9, linestyle="--", label="frozen tolerance")
    ax.plot(0, det["extrapolated_center"], "x", color=c.INK, markersize=6, label="frozen intercept")
    ax.set_xlim(-1e-5, 3.2e-4)
    ax.set_ylim(-5e-5, 1.0e-3)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 3))
    ax.set_xlabel("detuning (decades)")
    ax.set_ylabel("centre offset from the root (decades)")
    ax.set_title(f"C  Frozen estimator; outcome:\n{levels['outcome'].replace('_', ' ')}", fontsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    derived = {"deltas": delta.tolist(), "widths": width.tolist(), "amplitudes": amp.tolist(),
               "centres": centre.tolist(), "frozen_outcome": levels["outcome"],
               "extrapolated_center": det["extrapolated_center"], "root_tolerance": det["root_tolerance"],
               "posthoc_width_slope": w_slope, "posthoc_amplitude_slope": a_slope,
               "controls": {k: controls[k]["outcome"] for k in ("positive", "specificity", "negative")},
               "connection_outcome": connect["outcome"]}
    alt = (f"Resonance ancestry of a rule-selected period-10 window (rotation 1/5). Locked intervals with both fold "
           f"edges were found on all {len(delta)} cross-sections approaching the predicted 1/5 resonance point; widths "
           f"and amplitudes shrink toward it. The frozen estimator, a linear fit of the interval centre over all "
           f"levels, extrapolates to {det['extrapolated_center']:.2e} decades, above the frozen tolerance of "
           f"{det['root_tolerance']:.0e}, so the frozen outcome is alternative ancestry. Post hoc, over the deepest "
           f"three levels the width scales with exponent {w_slope:.3f} (theory 1.5) and the amplitude with "
           f"{a_slope:.3f} (theory 0.5).")
    c.save(fig, "f8_ancestry", [LEVELS, CONTROLS, CONNECT], __file__, "exploratory (frozen acceptance rules)", alt,
           derived, "Slopes over the deepest three levels are post hoc and do not replace the frozen outcome.")


if __name__ == "__main__":
    main()
