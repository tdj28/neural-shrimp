#!/usr/bin/env python3
"""EXP-006 figures from immutable saved records; never execute neural dynamics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np

from hopfield_drift_design import summarize_trials
from hopfield_drift_metrics import periodic_interpolate
from plot_hopfield import exact_npz, sha, json_write

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "experiments/hopfield/EXP-006-plan.json"
ARMS = ["outward", "inward", "tangent_plus", "tangent_minus", "local_plus", "local_minus",
        "random_plus", "random_minus", "shuffled_outward", "shuffled_inward", "atlas_margin"]
COMPARATORS = ["local_symmetric", "random_symmetric", "shuffled_inward"]
COLORS = {"atlas_margin": "#0072B2", "local_symmetric": "#D55E00",
          "random_symmetric": "#CC79A7", "shuffled_inward": "#009E73", "inward": "#666666"}
LABELS = {"atlas_margin": "Atlas margin", "local_symmetric": "Local (mean of ±)",
          "random_symmetric": "Random (mean of ±)", "shuffled_inward": "Shuffled inward",
          "inward": "Inward diagnostic"}
GROUPS = ["lower_margin", "middle_margin", "upper_margin"]
GROUP_COLORS = ["#D55E00", "#CC79A7", "#0072B2"]
FAMILIES = ["neuralshrimp-drift-anchors", "neuralshrimp-drift-utility",
            "neuralshrimp-drift-continuation", "neuralshrimp-drift-waveforms"]
INDEX = "hopfield-drift-figure-provenance.json"
plt.rcParams.update({"svg.hashsalt": "neural-shrimp-exp006-v1"})


def close_identity(a, b, path="summary"):
    """Only arithmetic replay tolerances; no scientific decision gate changes."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b): raise ValueError(f"Schema drift at {path}")
        for key in a: close_identity(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b): raise ValueError(f"Census drift at {path}")
        for index, (x, y) in enumerate(zip(a, b)): close_identity(x, y, f"{path}[{index}]")
    elif isinstance(a, (float, np.floating)):
        if b is None or not np.isclose(a, b, rtol=1e-12, atol=1e-14, equal_nan=False):
            raise ValueError(f"Numerical identity drift at {path}")
    elif a != b:
        raise ValueError(f"Value drift at {path}")


def replay_array(actual, expected, name):
    """Portable arithmetic replay; file hashes still bind exact saved bytes."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Plotted array schema drift: {name}")
    if actual.dtype.kind in "fc":
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14,
                                   equal_nan=True, err_msg=f"Plotted array drift: {name}")
    elif not np.array_equal(actual, expected):
        raise ValueError(f"Plotted array drift: {name}")


def load(runroot):
    receipt = json.loads((runroot / "receipt.json").read_text())
    if receipt.get("status") != "complete": raise ValueError("Complete EXP-006 receipt required")
    p = json.loads((runroot / "manifest.json").read_text())
    if p != json.loads(PLAN.read_text()) or p["experiment_id"] != "EXP-006":
        raise ValueError("Frozen EXP-006 manifest mismatch")
    baselines = json.loads((runroot / "baselines.json").read_text())
    records = [json.loads(line) for line in (runroot / "records.jsonl").read_text().splitlines()]
    summary = json.loads((runroot / "summary.json").read_text())
    anchor_ids = [a["id"] for a in p["anchors"]]
    if set(baselines) != set(anchor_ids): raise ValueError("Baseline anchor census drift")
    eligible = [name for name in anchor_ids if baselines[name]["eligible"]]
    if eligible != receipt["eligible_anchors"]: raise ValueError("Baseline coverage differs from receipt")
    rebuilt = summarize_trials([r for r in records if r["anchor"] in eligible], anchor_ids, eligible,
        phases=tuple(p["phases"]), radii=tuple(p["radii"]), dt_values=tuple(p["dt"]),
        bootstrap_samples=p["bootstrap_samples"], seed=p["bootstrap_seed"])
    close_identity(rebuilt, summary)
    indexed = {}
    for row in records:
        key = (row["anchor"], row["phase"], row["radius"], row["arm"], row["dt"])
        if key in indexed: raise ValueError("Duplicate drift record")
        indexed[key] = row
    expected = {(a, phase, radius, arm, dt) for a in anchor_ids for phase in p["phases"]
                for radius in p["radii"] for arm in p["arms"] for dt in p["dt"]}
    expected |= {(a, phase, 0., "zero", dt) for a in anchor_ids for phase in p["phases"] for dt in p["dt"]}
    if set(indexed) != expected: raise ValueError("Full attempted trial census drift")
    run_sources = {"receipt.json", "manifest.json", "baselines.json", "records.jsonl", "summary.json"}
    branches = {}
    for seed in p["continuation"]["seeds"]:
        for direction in p["continuation"]["directions"]:
            rel = f"continuation/{seed}-direction-{direction:+d}.json"
            branches[(seed, direction)] = json.loads((runroot / rel).read_text())
            run_sources.add(rel)
    old_paths = [ROOT / item["path"] for item in p["source_artifacts"]
                 if "/focus/row-" in item["path"]]
    if not old_paths: raise ValueError("No frozen EXP-004 atlas sources")
    atlas = []
    for path in sorted(old_paths):
        with np.load(path, allow_pickle=False) as row:
            atlas.append((float(row["gamma"]), row["alpha"].copy(), row["period"].copy()))
    atlas.sort(key=lambda row: row[0])
    alpha = atlas[0][1]
    if any(not np.array_equal(row[1], alpha) for row in atlas): raise ValueError("Atlas alpha coordinate drift")
    gamma, periods = np.array([row[0] for row in atlas]), np.stack([row[2] for row in atlas])
    exemplar = next((a for a in p["anchors"] if a["group"] == "upper_margin" and a["id"] in eligible), None)
    example = {"anchor": None if exemplar is None else exemplar["id"], "phase": 0,
               "radius": 0.016, "dt": p["primary_dt"], "arms": {}, "selection":
               "First baseline-eligible upper-margin anchor in frozen order; phase 0, radius .016, primary dt"}
    if exemplar is not None:
        cert = baselines[exemplar["id"]]["certificate"]
        example["reference_cycle"] = np.asarray(cert["orbit_activations"][:-1])
        example["period"] = cert["period"]
        for arm in ["atlas_margin", "local_plus", "local_minus", "random_plus"]:
            row = indexed[(exemplar["id"], 0, 0.016, arm, p["primary_dt"])]
            item = {"status": row["status"]}
            if row["status"] == "complete":
                run_sources.add(row["raw"])
                with np.load(runroot / row["raw"], allow_pickle=False) as raw:
                    item.update(times=raw["times"].copy(), activations=np.tanh(raw["states"]),
                                phase_epoch=row["baseline_phase_epoch"])
            example["arms"][arm] = item
    receipt_sources = {item["path"]: item["sha256"] for item in receipt["artifacts"]}
    for rel in sorted(run_sources - {"receipt.json"}):
        if receipt_sources.get(rel) != sha(runroot / rel): raise ValueError(f"Saved data drift: {rel}")
    for item in p["source_artifacts"]:
        if sha(ROOT / item["path"]) != item["sha256"]: raise ValueError(f"Old source drift: {item['path']}")
    return {"plan": p, "receipt": receipt, "baselines": baselines, "records": records,
            "summary": rebuilt, "branches": branches, "alpha": alpha, "gamma": gamma,
            "periods": periods, "example": example,
            "run_sources": sorted(run_sources), "old_sources": p["source_artifacts"]}


def anchor_figure(data):
    p, baselines = data["plan"], data["baselines"]
    anchors = p["anchors"]
    coords = np.array([[a["alpha"], a["gamma"]] for a in anchors])
    margins = np.array([a["raw_margin"] for a in anchors])
    groups = np.array([GROUPS.index(a["group"]) for a in anchors])
    eligible = np.array([baselines[a["id"]]["eligible"] for a in anchors], bool)
    crop = np.array([a["crop_edge_lower_bound"] for a in anchors])
    arrays = dict(alpha=data["alpha"], gamma=data["gamma"], old_peak_period=data["periods"],
                  anchor_coordinates=coords, raw_margin=margins, crop_edge_lower_bound=crop,
                  group_index=groups, eligible=eligible)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 6.1), gridspec_kw={"width_ratios": [1.5, 1]})
    mask = np.where(data["periods"] == 9, 1, np.where(data["periods"] == 18, 2, 0))
    axes[0].pcolormesh(data["alpha"], data["gamma"], mask, shading="nearest",
                       cmap=ListedColormap(["#eeeeee", "#c6dbea", "#f1d4bb"]), vmin=0, vmax=2,
                       rasterized=True)
    for i, a in enumerate(anchors):
        color = GROUP_COLORS[groups[i]]
        axes[0].scatter(*coords[i], s=65, c=color if eligible[i] else "white", edgecolors=color,
                        marker="o", linewidths=1.6, zorder=3)
        if not eligible[i]: axes[0].plot(*coords[i], marker="x", color=color, ms=5, zorder=4)
        offset = (7, -11) if i == 9 else (4, 5)
        axes[0].annotate(f"{i:02d}", coords[i], xytext=offset, textcoords="offset points", fontsize=9,
                         bbox=dict(boxstyle="round,pad=.12", fc="white", ec="none", alpha=.8))
    axes[0].set(xlabel=r"Weight $\alpha$", ylabel=r"Weight $\gamma$", title="All 12 attempted anchors on the old numerical shrimp")
    bars = axes[1].barh(np.arange(12), margins, color=[GROUP_COLORS[g] for g in groups], alpha=.82)
    for i, bar in enumerate(bars):
        if not eligible[i]: bar.set_hatch("///"); bar.set_alpha(.4)
    axes[1].set(yticks=np.arange(12), yticklabels=[f"{i:02d}" for i in range(12)],
                xlabel="Old-atlas margin in raw weight units", ylabel="Attempted anchor", title=f"Baseline eligibility: {eligible.sum()}/12")
    axes[1].invert_yaxis(); axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    handles = [Line2D([], [], marker="o", ls="", color=color, label=group.replace("_", " ").title())
               for color, group in zip(GROUP_COLORS, GROUPS)]
    handles += [Line2D([], [], marker="o", ls="", color="black", label="Qualified baseline"),
                Line2D([], [], marker="x", ls="", color="black", label="Ineligible; retained")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, .065), fontsize=9)
    fig.text(.5, .018, "Background: old nine-peak candidates (blue), eighteen-peak candidates (tan), other outcomes (gray).\n"
             "Pixel margins describe the saved atlas, not exact orbit or basin boundaries. Anchor qualification is a new independent check.",
             ha="center", fontsize=9)
    for ax in axes: ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=.07, right=.98, bottom=.23, top=.91, wspace=.32)
    return fig, arrays, {"anchor_order": [a["id"] for a in anchors], "groups": GROUPS,
                         "eligible": [a["id"] for i, a in enumerate(anchors) if eligible[i]]}


def utility_figure(data):
    summary, p = data["summary"], data["plan"]
    radii, steps = np.asarray(p["radii"]), np.asarray(p["dt"])
    means = np.full((2, len(ARMS), 4), np.nan)
    effects = np.full((2, 3, 4, 3), np.nan)
    pooled = np.full((2, 3, 3), np.nan)
    complete = np.zeros((2, 4), bool)
    zero = np.full(2, np.nan)
    for i, panel in enumerate(summary["panels"]):
        for j, item in enumerate(panel["per_radius"]):
            complete[i, j] = item["complete"]
            if item["complete"]:
                means[i, :, j] = [item["arm_mean_rmse"][a] for a in ARMS]
            for k, name in enumerate(COMPARATORS):
                result = item["effects"][name]
                if result["defined"]: effects[i, k, j] = [result["mean_benefit"], *result["ci95"]]
        for k, name in enumerate(COMPARATORS):
            result = panel["equal_radius_average"][name]
            if result["defined"]: pooled[i, k] = [result["mean_benefit"], *result["ci95"]]
        zeros = [r for r in data["records"] if r["anchor"] in summary["eligible_anchors"] and
                 r["dt"] == panel["dt"] and r["arm"] == "zero"]
        if zeros and panel["zero_controls_complete"]:
            zero[i] = np.mean([r["wave_metrics"]["aligned_rms"]["aggregate"] for r in zeros])
    arrays = dict(radii=radii, steps=steps, arm_mean_rmse=means, contrast_effect_ci=effects,
                  equal_radius_average=pooled, defined=complete, zero_mean_rmse=zero)
    # 9-point labels remain above 7 points at a 9.8 by 7.2 inch landscape placement.
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 8.8), gridspec_kw={"height_ratios": [.92, 1.25, .92]})
    finite = means[np.isfinite(means)]
    vmax = max(.01, float(np.max(finite))) if finite.size else 1.
    effect_finite = effects[np.isfinite(effects)]
    effect_limits = ([min(0., float(effect_finite.min())), max(0., float(effect_finite.max()))]
                     if effect_finite.size else [-1., 1.])
    effect_padding = max((effect_limits[1] - effect_limits[0]) * .08, 1e-5)
    for i, dt in enumerate(steps):
        a = axes[0, i]
        curves = {"atlas_margin": means[i, ARMS.index("atlas_margin")],
                  "local_symmetric": means[i, [ARMS.index("local_plus"), ARMS.index("local_minus")]].mean(axis=0),
                  "random_symmetric": means[i, [ARMS.index("random_plus"), ARMS.index("random_minus")]].mean(axis=0),
                  "shuffled_inward": means[i, ARMS.index("shuffled_inward")],
                  "inward": means[i, ARMS.index("inward")]}
        for name, values in curves.items():
            a.plot(radii, values, "o", color=COLORS[name], label=LABELS[name], ms=4,
                    linestyle="--" if name == "inward" else "-")
        if np.isfinite(zero[i]): a.axhline(zero[i], color="#999999", ls=":", lw=1)
        a.set(xscale="log", xlabel="Raw weight perturbation norm", ylabel="Aligned activation RMSE",
              title=f"dt = {dt:g}" + (" (primary)" if dt == p["primary_dt"] else " (refinement check)"),
              ylim=(0, vmax * 1.08), xlim=(radii[0] * .85, radii[-1] * 1.15))
        a.grid(alpha=.15); a.legend(fontsize=9, loc="upper left", ncol=2, columnspacing=.7, handlelength=1.4)
        if not np.all(complete[i]):
            a.text(.98, .98, f"{(~complete[i]).sum()}/4 doses undefined", transform=a.transAxes,
                   ha="right", va="top", color="#b00020", fontsize=9)
        hm = axes[1, i]
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#dedede")
        im = hm.imshow(means[i], aspect="auto", vmin=0, vmax=vmax, cmap=cmap)
        hm.set(xticks=range(4), xticklabels=[f"{r:g}" for r in radii], yticks=range(11),
               yticklabels=[a.replace("_", " ") for a in ARMS], xlabel="Raw weight perturbation norm",
               title="Every arm, including both directional signs")
        for row in range(11):
            for col in range(4):
                value = means[i, row, col]
                hm.text(col, row, f"{value:.3f}" if np.isfinite(value) else "NA", ha="center", va="center",
                         fontsize=9, color="white" if np.isfinite(value) and value < .55 * vmax else "black")
        colorbar = fig.colorbar(im, ax=hm, fraction=.035, pad=.025, label="RMSE")
        colorbar.ax.tick_params(labelsize=9)
        benefit = axes[2, i]
        for k, name in enumerate(COMPARATORS):
            values = effects[i, k]
            benefit.plot(radii, values[:, 0], "o-", color=COLORS[name], label=LABELS[name], ms=4)
            benefit.fill_between(radii, values[:, 1], values[:, 2], alpha=.13, color=COLORS[name])
        benefit.axhline(0, color="black", lw=.8)
        benefit.set(xscale="log", xlabel="Raw weight perturbation norm",
                    ylabel="Comparator RMSE − atlas RMSE", title="Paired benefit and descriptive 95% intervals",
                    xlim=(radii[0] * .85, radii[-1] * 1.15),
                    ylim=(effect_limits[0] - effect_padding, effect_limits[1] + effect_padding))
        benefit.grid(alpha=.15)
        text = []
        for k, name in enumerate(COMPARATORS):
            value = pooled[i, k]
            text.append(f"{LABELS[name]}: {value[0]:+.4f} [{value[1]:+.4f}, {value[2]:+.4f}]"
                        if np.isfinite(value).all() else f"{LABELS[name]}: undefined")
        benefit.text(0, -.43, "Equal-radius averages:\n" + "\n".join(text), transform=benefit.transAxes,
                      fontsize=9, va="top")
    fig.suptitle(f"Does the saved shrimp atlas preserve a neural waveform?  {summary['eligible_anchor_count']}/12 anchors qualified", y=.98)
    fig.text(.5, .015, f"Equal anchor and phase weights; 10,000 anchor-bootstrap draws. Refinement steps are not independent replicates.\n"
             f"Eligible trial statuses: {summary['status_counts']}; missing={len(summary['missing_trials'])}, invalid={len(summary['invalid_trials'])}. "
             "Undefined comparisons remain NA.\nPositive benefit favors the atlas policy. One global phase offset is allowed; frequency is not fitted. Dotted line: zero-dose floor.",
             ha="center", fontsize=9)
    for ax in axes.ravel(): ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=.12, right=.93, top=.93, bottom=.22, hspace=.54, wspace=.4)
    return fig, arrays, {"arms": ARMS, "comparators": COMPARATORS, "summary": summary}


def continuation_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), sharex="col")
    arrays, conditions = {}, {}
    colors = {-1: "#0072B2", 1: "#D55E00"}
    for col, seed in enumerate(data["plan"]["continuation"]["seeds"]):
        notes = []
        for direction in [-1, 1]:
            branch = data["branches"][(seed, direction)]
            nodes = branch["nodes"]
            prefix = f"seed{col}_direction{'minus' if direction < 0 else 'plus'}"
            gamma = np.array([n.get("gamma", branch["gamma_seed"] + .41 * n["parameter"]) for n in nodes])
            periods = np.array([n["period"] for n in nodes])
            multipliers = np.array([[complex(v["real"], v["imag"]) for v in n["transverse_multipliers"]] for n in nodes])
            dominant = np.array([m[np.argmax(np.abs(m))] for m in multipliers])
            plotted = np.where(np.abs(dominant.imag) <= 1e-7, dominant.real, np.abs(dominant))
            stable = np.array([n["attracting"] for n in nodes], bool)
            unstable = np.array([n["unstable"] for n in nodes], bool)
            reversal = np.array([n.get("parameter_reversal", False) for n in nodes], bool)
            brackets = np.array([bool(n.get("bracket_flags")) for n in nodes])
            checkpoint_failed = np.zeros(len(nodes), bool)
            for cert in branch.get("checkpoint_certificates", []):
                checkpoint_failed[cert["continuation_node"]] = not cert.get("continued_orbit_agreement", False)
            for key, value in {"gamma": gamma, "period": periods, "transverse_multipliers": multipliers,
                               "dominant_plotted": plotted, "stable": stable, "unstable": unstable,
                               "reversal": reversal, "bracket": brackets,
                               "checkpoint_failed": checkpoint_failed,
                               "node_index": np.arange(len(nodes))}.items(): arrays[f"{prefix}_{key}"] = value
            color = colors[direction]
            for row, values in enumerate([periods, plotted]):
                ax = axes[row, col]
                ax.plot(gamma, values, color=color, lw=1, label=f"Initial gamma direction {direction:+d}")
                ax.scatter(gamma[stable], values[stable], s=15, c=color, zorder=3)
                ax.scatter(gamma[unstable], values[unstable], s=24, facecolors="white", edgecolors=color, zorder=3)
                undecided = ~(stable | unstable)
                ax.scatter(gamma[undecided], values[undecided], s=18, marker="s", c="#777777", zorder=3)
                ax.scatter(gamma[reversal], values[reversal], s=85, marker="*", c="black", zorder=5)
                ax.scatter(gamma[brackets], values[brackets], s=55, marker="D", facecolors="none", edgecolors="black", zorder=4)
                if row == 0:
                    ax.scatter(gamma[checkpoint_failed], values[checkpoint_failed], s=30, marker="x", c="#b00020", zorder=6)
                ax.grid(alpha=.15)
            failed = sum(not a["accepted"] for a in branch["attempts"])
            notes.append(f"{direction:+d}: {len(nodes)} nodes, {failed}/{len(branch['attempts'])} failed attempts; {branch['stop_reason']}")
            conditions[prefix] = {"stop_reason": branch["stop_reason"], "node_count": len(nodes),
                                  "failed_attempts": failed, "attempt_count": len(branch["attempts"]),
                                  "node_brackets": [n.get("bracket_flags", []) for n in nodes],
                                  "checkpoint_failures": int(checkpoint_failed.sum())}
        axes[0, col].set(title="Core seed" if seed == "core" else "Retained near-edge seed", ylabel="Physical orbit period T")
        axes[0, col].legend(fontsize=9)
        axes[1, col].axhline(1, color="black", ls="--", lw=.8)
        axes[1, col].axhline(-1, color="black", ls="--", lw=.8)
        axes[1, col].set_yscale("symlog", linthresh=2, linscale=1)
        transform = axes[1, col].yaxis.get_transform()
        bounds = transform.transform(np.array([min(-1., axes[1, col].dataLim.y0),
                                               max(1., axes[1, col].dataLim.y1)]))
        padding = .08 * (bounds[1] - bounds[0])
        axes[1, col].set_ylim(transform.inverted().transform(bounds + [-padding, padding]))
        lower, upper = axes[1, col].get_ylim()
        ticks = [value for value in [-100, -10, -1, 0, 1, 10, 100] if lower <= value <= upper]
        axes[1, col].set_yticks(ticks, labels=[str(value) for value in ticks])
        axes[1, col].set(xlabel=r"Weight $\gamma$ (alpha fixed)", ylabel="Dominant transverse multiplier")
        axes[1, col].text(0, -.27, "\n".join(notes), transform=axes[1, col].transAxes, fontsize=9, va="top")
    handles = [Line2D([], [], marker="o", color="black", ls="", label="Attracting"),
               Line2D([], [], marker="o", color="black", mfc="white", ls="", label="Unstable"),
               Line2D([], [], marker="s", color="#777777", ls="", label="Near-unit / undecided"),
               Line2D([], [], marker="*", color="black", ls="", label="Parameter reversal"),
               Line2D([], [], marker="D", color="black", mfc="white", ls="", label="Crossing bracket"),
               Line2D([], [], marker="x", color="#b00020", ls="", label="Checkpoint disagreement")]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, .065), fontsize=9)
    fig.suptitle("Continuation of the network’s nine-peak orbits", y=.985)
    fig.text(.5, .009, "Every accepted node is shown. A failed corrector is unresolved; brackets are numerical candidates.\n"
             "Multiplier axes: symmetric log, linear within ±2. Signed real multipliers are shown; complex dominant pairs use their modulus.",
             ha="center", fontsize=9)
    for ax in axes.ravel(): ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=.09, right=.98, bottom=.28, top=.92, hspace=.22, wspace=.28)
    return fig, arrays, conditions


def waveform_figure(data):
    example = data["example"]
    arrays, conditions = {}, {key: value for key, value in example.items()
                              if key not in ("arms", "reference_cycle")}
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 7.4), sharex=True, sharey=True)
    colors = {"atlas_margin": "#0072B2", "local_plus": "#D55E00", "local_minus": "#E69F00", "random_plus": "#CC79A7"}
    if example["anchor"] is None:
        for ax in axes.ravel(): ax.text(.5, .5, "No upper-margin anchor qualified", ha="center", transform=ax.transAxes)
    else:
        arrays["reference_cycle"] = example["reference_cycle"]
        for arm, item in example["arms"].items():
            conditions[arm] = item["status"]
            if item["status"] != "complete": continue
            arrays[f"{arm}_times"] = item["times"]
            arrays[f"{arm}_activations"] = item["activations"]
            reference = periodic_interpolate(example["reference_cycle"],
                       (item["times"] + item["phase_epoch"]) / example["period"])
            arrays[f"{arm}_synchronous_reference"] = reference
            col = 0 if arm == "atlas_margin" else 1
            for neuron in range(3):
                ax = axes[neuron, col]
                if arm in ("atlas_margin", "local_plus"):
                    ax.plot(item["times"], reference[:, neuron], color="#333333", lw=.6, alpha=.7, label="Original waveform")
                ax.plot(item["times"], item["activations"][:, neuron], color=colors[arm], lw=.65, alpha=.85,
                        label=arm.replace("_", " "))
        for col, arms in enumerate([["atlas_margin"], ["local_plus", "local_minus", "random_plus"]]):
            failed = [arm for arm in arms if example["arms"][arm]["status"] != "complete"]
            if failed: axes[0, col].text(.03, .93, "Failed: " + ", ".join(failed), transform=axes[0, col].transAxes, color="#b00020")
    for row in range(3):
        axes[row, 0].set_ylabel(f"Neuron {row+1} activation")
        for col in range(2): axes[row, col].set(ylim=(-1.08, 1.08)); axes[row, col].grid(alpha=.1)
    for ax in axes[-1]: ax.set_xlabel("Physical time after held weight perturbation")
    axes[0, 0].set_title("Atlas policy and original waveform")
    axes[0, 1].set_title("Local signs and random-plus diagnostic")
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.suptitle(f"Fixed waveform example: {example['anchor'] or 'no eligible anchor'}, phase 0, norm 0.016, dt {example['dt']:g}", y=.985)
    fig.text(.5, .018, "First eligible upper-margin anchor in the frozen order; all three neuron outputs and the full 300-time-unit records.\n"
             "These traces use the original time origin. The utility statistic separately permits one constant phase offset, without frequency fitting.",
             ha="center", fontsize=9)
    for ax in axes.ravel(): ax.tick_params(labelsize=9)
    fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.91, hspace=.2, wspace=.16)
    return fig, arrays, conditions


def build(runroot, destination):
    data = load(runroot)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    generators = [SCRIPT, ROOT / "tools/plot_hopfield.py", ROOT / "tools/hopfield_drift_design.py",
                  ROOT / "tools/hopfield_drift_metrics.py"]
    sources = [{"namespace": "run", "path": rel, "sha256": sha(runroot / rel)} for rel in data["run_sources"]]
    sources += [{"namespace": "repo", **item} for item in data["old_sources"]]
    for stem, plot in zip(FAMILIES, [anchor_figure, utility_figure, continuation_figure, waveform_figure]):
        fig, arrays, conditions = plot(data)
        exact_npz(destination / f"{stem}.plot-data.npz", arrays)
        for extension in ("pdf", "svg", "png"):
            metadata = ({"CreationDate": None, "ModDate": None, "Creator": "neural-shrimp saved evidence renderer"}
                        if extension == "pdf" else {"Date": None} if extension == "svg" else
                        {"Software": "neural-shrimp saved evidence renderer"})
            fig.savefig(destination / f"{stem}.{extension}", dpi=300, metadata=metadata)
        plt.close(fig)
        receipt = {"schema_version": 1, "experiment": "EXP-006", "stem": stem,
                   "generators": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in generators],
                   "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
                   "sources": sources, "conditions": conditions,
                   "plotted_arrays": {"path": f"{stem}.plot-data.npz", "sha256": sha(destination / f"{stem}.plot-data.npz")},
                   "outputs": [{"path": f"{stem}.{ext}", "sha256": sha(destination / f"{stem}.{ext}")}
                               for ext in ("pdf", "svg", "png")],
                   "scope": "Saved numerical evidence only; every declared arm and attempted anchor remains accounted for."}
        path = destination / f"{stem}.receipt.json"
        json_write(path, receipt); results.append({"path": path.name, "sha256": sha(path)})
    json_write(destination / INDEX, {"schema_version": 1, "experiment": "EXP-006", "figures": results})
    return [json.loads((destination / item["path"]).read_text()) for item in results]


def check(runroot, destination, verify=False):
    index = json.loads((destination / INDEX).read_text())
    if [item["path"] for item in index["figures"]] != [f"{stem}.receipt.json" for stem in FAMILIES]:
        raise ValueError("Figure family census drift")
    previous = []
    for item in index["figures"]:
        path = destination / item["path"]
        if sha(path) != item["sha256"]: raise ValueError("Figure receipt drift")
        receipt = json.loads(path.read_text()); previous.append(receipt)
        for source in receipt["generators"] + [receipt["plan"]]:
            if sha(ROOT / source["path"]) != source["sha256"]: raise ValueError(f"Generator/plan drift: {source['path']}")
        for source in receipt["sources"]:
            base = runroot if source["namespace"] == "run" else ROOT
            if sha(base / source["path"]) != source["sha256"]: raise ValueError(f"Input drift: {source['path']}")
        for output in receipt["outputs"] + [receipt["plotted_arrays"]]:
            if sha(destination / output["path"]) != output["sha256"]: raise ValueError(f"Figure output drift: {output['path']}")
    with tempfile.TemporaryDirectory(prefix="hopfield-drift-figures-") as temporary:
        rebuilt = build(runroot, Path(temporary))
        for before, after in zip(previous, rebuilt):
            close_identity(before["conditions"], after["conditions"], "plot_conditions")
            with np.load(destination / before["plotted_arrays"]["path"], allow_pickle=False) as a, np.load(Path(temporary) / after["plotted_arrays"]["path"], allow_pickle=False) as b:
                if a.files != b.files: raise ValueError("Plotted array schema drift")
                for key in a.files:
                    replay_array(a[key], b[key], key)
            if verify:
                for output in before["outputs"]:
                    if sha(Path(temporary) / output["path"]) != output["sha256"]:
                        raise ValueError(f"Nonidentical image regeneration: {output['path']}")
    print("PASS: EXP-006 sources, plotted arrays, all-arm census, coverage and summary replay." +
          (" Images regenerate byte-identically." if verify else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path); parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check", action="store_true"); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.check or args.verify: check(args.runroot.resolve(), args.destination.resolve(), args.verify)
    else: build(args.runroot.resolve(), args.destination.resolve()); print("Built four EXP-006 figure families.")


if __name__ == "__main__": main()
