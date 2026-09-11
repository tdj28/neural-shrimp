#!/usr/bin/env python3
"""Render frozen EXP-007 displays from saved evidence; never integrate dynamics."""
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
from matplotlib.patches import Circle
import numpy as np

from hopfield_drift_metrics import periodic_interpolate
from plot_hopfield import exact_npz, sha, json_write
from run_hopfield_holdout import summarize, validate, specs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "experiments/hopfield/EXP-007-plan.json"
ARMS = ["atlas", "local", "tangent", "axes", "uniform", "search", "hybrid"]
COMPARATORS = ["local", "tangent", "axes", "uniform", "search"]
EXAMPLE_ARMS = ["hybrid", "local", "uniform", "search"]
LABELS = {"atlas": "Atlas", "local": "Local ± preview", "tangent": "Tangent ± preview",
          "axes": "Four signed axes", "uniform": "Uniform 16", "search": "Augmented 21",
          "hybrid": "Frozen hybrid"}
COLORS = {"atlas": "#777777", "local": "#D55E00", "tangent": "#009E73", "axes": "#CC79A7",
          "uniform": "#E69F00", "search": "#56B4E9", "hybrid": "#0072B2"}
GROUPS = ["lower_margin", "middle_margin", "upper_margin"]
GROUP_COLORS = ["#D55E00", "#CC79A7", "#0072B2"]
FAMILIES = ["neuralshrimp-holdout-anchors", "neuralshrimp-holdout-utility",
            "neuralshrimp-holdout-waveforms"]
INDEX = "hopfield-holdout-figure-provenance.json"
plt.rcParams.update({"svg.hashsalt": "neural-shrimp-exp007-v1"})


def close_identity(a, b, path="value"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError(f"Schema drift at {path}")
        for key in a:
            close_identity(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise ValueError(f"Census drift at {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            close_identity(x, y, f"{path}[{index}]")
    elif isinstance(a, (float, np.floating)):
        if b is None or not np.isclose(a, b, rtol=1e-12, atol=1e-14, equal_nan=False):
            raise ValueError(f"Arithmetic drift at {path}")
    elif a != b:
        raise ValueError(f"Value drift at {path}")


def replay_array(actual, expected, name):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError(f"Plotted array schema drift: {name}")
    if actual.dtype.kind in "fc":
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14,
                                   equal_nan=True, err_msg=name)
    elif not np.array_equal(actual, expected):
        raise ValueError(f"Plotted array drift: {name}")


def number(value):
    return np.nan if value is None else float(value)


def load(runroot):
    receipt = json.loads((runroot / "receipt.json").read_text())
    p = json.loads((runroot / "manifest.json").read_text())
    validate(p)
    if receipt.get("status") != "complete" or p != json.loads(PLAN.read_text()):
        raise ValueError("A complete receipt and unchanged EXP-007 manifest are required")
    if receipt["manifest_sha256"] != sha(runroot / "manifest.json"):
        raise ValueError("Manifest byte drift")
    if (runroot / "failure.json").exists() or (runroot / "timeout.json").exists():
        raise ValueError("Attempt failure is present; no complete-result rendering")
    paths = [item["path"] for item in receipt["artifacts"]]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact path")
    # Bind every retained raw file, not merely the displayed exemplar.
    for item in receipt["artifacts"]:
        path = runroot / item["path"]
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Saved evidence drift: {item['path']}")
    for item in p["source_artifacts"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Old atlas/development source drift: {item['path']}")
    baselines = json.loads((runroot / "baselines.json").read_text())
    anchor_ids = [a["id"] for a in p["anchors"]]
    if set(baselines) != set(anchor_ids):
        raise ValueError("Baseline attempt census drift")
    eligible = [name for name in anchor_ids if baselines[name]["eligible"]]
    if eligible != receipt["eligible_anchors"]:
        raise ValueError("Eligibility differs from receipt")
    records = [json.loads(line) for line in (runroot / "records.jsonl").read_text().splitlines()]
    indexed = {(r["anchor"], r["phase"], r["radius"], r["arm"], r["dt"]): r for r in records}
    expected = {(s["anchor"], s["phase"], s["radius"], s["arm"], s["dt"]) for s in specs(p)}
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("Target census differs from all frozen attempts")
    summary = summarize(records, p, eligible)
    close_identity(summary, json.loads((runroot / "summary.json").read_text()), "summary")
    action_pairs = [(indexed[(aid, phase, radius, "tangent", p["primary_dt"])],
                     indexed[(aid, phase, radius, "axes", p["primary_dt"])])
                    for aid in eligible for phase in p["phases"] for radius in p["radii"]]
    action_equivalence = {"contexts": len(action_pairs), "identical_tangent_axes_directions": sum(
        left["status"] == right["status"] == "complete"
        and np.array_equal(left["direction"], right["direction"])
        for left, right in action_pairs)}
    old = []
    for item in p["source_artifacts"]:
        if "/focus/row-" in item["path"]:
            with np.load(ROOT / item["path"], allow_pickle=False) as raw:
                old.append((float(raw["gamma"]), raw["alpha"].copy(), raw["period"].copy()))
    old.sort(key=lambda row: row[0])
    if not old or any(not np.array_equal(row[1], old[0][1]) for row in old):
        raise ValueError("Old atlas coordinate drift")
    development_paths = [i["path"] for i in p["source_artifacts"]
                         if "EXP-006/" in i["path"] and i["path"].endswith("/manifest.json")]
    if len(development_paths) != 1:
        raise ValueError("Exactly one frozen development manifest required")
    development = json.loads((ROOT / development_paths[0]).read_text())["anchors"]
    preview = json.loads((runroot / "preview-census.json").read_text())
    sensitivity = [json.loads((runroot / rel).read_text()) for rel in paths
                   if rel.startswith("sensitivity/") and rel.endswith(".json")]
    costs = {"collection_wall_seconds": receipt["wall_seconds"],
             "preview_sum_concurrent_call_wall_seconds": preview["sum_concurrent_call_wall_seconds"],
             "sensitivity_sum_call_wall_seconds": sum(s["wall_seconds"] for s in sensitivity),
             "sensitivity_calls": len(sensitivity), "old_atlas_seconds": 140.54,
             "preview_census": preview}
    exemplar = next((a for a in p["anchors"] if a["id"] in eligible), None)
    example = {"anchor": None if exemplar is None else exemplar["id"], "phase": p["phases"][0],
               "radius": p["radii"][-1], "dt": p["primary_dt"], "arms": {},
               "selection": "First eligible anchor in frozen order; first frozen phase, largest radius, primary dt"}
    if exemplar is not None:
        cert = baselines[exemplar["id"]]["certificate"]
        example["reference_cycle"] = np.asarray(cert["orbit_activations"][:-1])
        example["period"] = cert["period"]
        for arm in EXAMPLE_ARMS:
            row = indexed[(exemplar["id"], example["phase"], example["radius"], arm, example["dt"])]
            item = {"status": row["status"]}
            if row["status"] == "complete":
                with np.load(runroot / row["raw"], allow_pickle=False) as raw:
                    times = raw["times"].copy()
                    activations = np.tanh(raw["states"])
                phase = row["tail_metrics"]["phase_offset_cycles"]
                sync = periodic_interpolate(example["reference_cycle"],
                                            (times + row["phase_epoch"]) / example["period"])
                aligned = periodic_interpolate(example["reference_cycle"],
                           (times + row["phase_epoch"]) / example["period"] + phase)
                errors = activations - aligned
                scored = times > p["preview_duration"]
                close_identity(float(np.sqrt(np.mean(errors[scored] ** 2))),
                               row["tail_metrics"]["aligned_rms"]["aggregate"], f"example.{arm}.rmse")
                item.update(times=times, activations=activations, synchronous_reference=sync,
                            aligned_reference=aligned, aligned_errors=errors,
                            phase_epoch=row["phase_epoch"], phase_offset_cycles=phase,
                            tail_rms=row["tail_metrics"]["aligned_rms"]["aggregate"])
            example["arms"][arm] = item
    return {"plan": p, "receipt": receipt, "baselines": baselines, "records": records,
            "summary": summary, "alpha": old[0][1], "gamma": np.array([r[0] for r in old]),
            "periods": np.stack([r[2] for r in old]), "development": development,
            "costs": costs, "example": example, "action_equivalence": action_equivalence,
            "run_sources": sorted(set(paths) | {"receipt.json"}),
            "old_sources": p["source_artifacts"]}


def anchor_figure(data):
    p = data["plan"]
    new = np.array([[a["alpha"], a["gamma"]] for a in p["anchors"]])
    old = np.array([[a["alpha"], a["gamma"]] for a in data["development"]])
    groups = np.array([GROUPS.index(a["group"]) for a in p["anchors"]])
    eligible = np.array([data["baselines"][a["id"]]["eligible"] for a in p["anchors"]], bool)
    origin = np.array([data["alpha"][0], data["gamma"][0]])
    widths = np.asarray(p["atlas_selection"]["normalization_widths"])
    halo = p["atlas_selection"]["exclusion_radius"]
    arrays = dict(alpha=data["alpha"], gamma=data["gamma"], old_peak_period=data["periods"],
                  new_coordinates=new, development_coordinates=old, origin=origin, widths=widths,
                  normalized_new=(new-origin)/widths, normalized_development=(old-origin)/widths,
                  group_index=groups, eligible=eligible, exclusion_radius=np.array(halo),
                  anchor_ids=np.array([a["id"] for a in p["anchors"]]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))
    mask = np.where(data["periods"] == 9, 1, np.where(data["periods"] == 18, 2, 0))
    for col, ax in enumerate(axes):
        normalized = col == 1
        aa = (data["alpha"]-origin[0])/widths[0] if normalized else data["alpha"]
        gg = (data["gamma"]-origin[1])/widths[1] if normalized else data["gamma"]
        n = (new-origin)/widths if normalized else new
        d = (old-origin)/widths if normalized else old
        ax.pcolormesh(aa, gg, mask, shading="nearest", rasterized=True,
                      cmap=ListedColormap(["#f1f1f1", "#c6dbea", "#f1d4bb"]), vmin=0, vmax=2)
        if normalized:
            for point in d:
                ax.add_patch(Circle(point, halo, facecolor="#777777", alpha=.13, edgecolor="#555555", lw=.8))
            ax.set_aspect("equal", adjustable="box")
        ax.scatter(d[:, 0], d[:, 1], s=45, marker="x", color="#444444", zorder=3)
        development_offsets = {0: (-2, 29), 1: (-5, 7), 2: (5, 8), 5: (4, -30), 9: (-4, -31)}
        for i, point in enumerate(d):
            offset = development_offsets.get(i, (-5, -11))
            ax.annotate(f"D{i:02d}", point, xytext=offset, textcoords="offset points",
                        fontsize=7, color="#444444", ha="right" if offset[0] < 0 else "left",
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=.5) if i in development_offsets else None)
        for i, point in enumerate(n):
            color = GROUP_COLORS[groups[i]]
            ax.scatter(*point, s=55, facecolor=color if eligible[i] else "white",
                       edgecolor=color, linewidth=1.4, zorder=4)
            if not eligible[i]:
                ax.scatter(*point, s=25, color=color, marker="x", zorder=5)
            offset = [(5, 6), (-4, 7), (4, -13), (5, 10)][i % 4]
            if i == 7: offset = (-8, -17)
            if i == 6: offset = (-8, -13)
            ax.annotate(f"H{i:02d}", point, xytext=offset, textcoords="offset points", fontsize=8,
                        ha="right" if offset[0] < 0 else "left", color=color,
                        bbox=dict(boxstyle="round,pad=.08", fc="white", ec="none", alpha=.75))
        ax.set(xlim=(aa[0], aa[-1]), ylim=(gg[0], gg[-1]),
               xlabel="Normalized alpha" if normalized else r"Weight $\alpha$",
               ylabel="Normalized gamma" if normalized else r"Weight $\gamma$",
               title="B  Normalized starting-point exclusions" if normalized else "A  Original weight coordinates")
    handles = [Line2D([], [], color=c, marker="o", ls="", label=g.replace("_", " "))
               for g, c in zip(GROUPS, GROUP_COLORS)]
    handles += [Line2D([], [], color="#444444", marker="x", ls="", label="D: EXP-006 start"),
                Line2D([], [], color="#444444", marker="o", mfc="white", ls="", label="Open/crossed: ineligible")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .06), ncol=3, fontsize=9)
    fig.suptitle("EXP-007: every attempted new start on the previously observed map", y=.99)
    fig.text(.5, .018, "H labels: new starts. Blue map: nine peaks; tan: eighteen; gray: other labels. "
             "Circles exclude normalized distance ≤0.04.\n"
             "Exclusions concern old starting points, not all previous intervention destinations; this is not an unseen-map test.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.97, bottom=.20, top=.90, wspace=.27)
    return fig, arrays, {"eligible": eligible.tolist(), "selection": p["atlas_selection"]["scope"],
                         "normalization": "Full old focus widths; origin at its lower-left corner", "halo": halo}


def utility_figure(data):
    p, summary = data["plan"], data["summary"]
    radii, steps = np.array(p["radii"]), np.array(p["dt"])
    primary = list(steps).index(p["primary_dt"])
    means = np.full((len(steps), len(ARMS), len(radii)), np.nan)
    effects = np.full((len(steps), len(COMPARATORS), len(radii)+1, 3), np.nan)
    for d, dt in enumerate(steps):
        panels = summary["steps"][str(dt)]["panels"]
        for j, radius in enumerate(radii):
            means[d, :, j] = [number(panels[str(radius)]["means"][a]) for a in ARMS]
        for j, radius in enumerate([*map(str, radii), "pooled"]):
            for k, comp in enumerate(COMPARATORS):
                c = panels[radius]["comparator_minus_hybrid"][comp]
                if c["status"] == "complete":
                    effects[d, k, j] = [c["benefit"], *c["ci95"]]
    anchor_ids = [a["id"] for a in p["anchors"]]
    eligible = summary["eligible_anchors"]
    largest = summary["steps"][str(p["primary_dt"])]["panels"][str(radii[-1])]
    per_anchor = np.full((len(anchor_ids), len(COMPARATORS)), np.nan)
    for i, aid in enumerate(anchor_ids):
        if aid not in eligible: continue
        e = eligible.index(aid)
        for k, comp in enumerate(COMPARATORS):
            v, h = largest["per_anchor"][comp][e], largest["per_anchor"]["hybrid"][e]
            if v is not None and h is not None: per_anchor[i, k] = v-h
    budgets = np.array([[0]*4, [2]*4, [2]*4, [4]*4, [16]*4, [21]*4, [2, 2, 2, 0]], int)
    arrays = dict(radii=radii, dt=steps, means=means, contrast_mean_ci=effects,
                  largest_per_anchor_contrast=per_anchor, preview_calls=budgets,
                  anchor_ids=np.array(anchor_ids), arms=np.array(ARMS), comparators=np.array(COMPARATORS),
                  eligible=np.array([a in eligible for a in anchor_ids]),
                  zero_mean=np.array([number(summary["steps"][str(dt)]["zero"]["mean"]) for dt in steps]))
    tangent_axes_overlap = np.array_equal(means[:, ARMS.index("tangent")],
                                          means[:, ARMS.index("axes")], equal_nan=False)
    equivalence = data.get("action_equivalence", {"contexts": 0, "identical_tangent_axes_directions": 0})
    same_actions = equivalence["contexts"] > 0 and equivalence["identical_tangent_axes_directions"] == equivalence["contexts"]
    short = {"atlas": "Atlas", "local": "Local", "tangent": "Tangent", "axes": "Axes",
             "uniform": "Uniform16", "search": "Aug.21", "hybrid": "Hybrid"}
    fig, display_axes = plt.subplots(2, 3, figsize=(12.6, 7.8))
    # Retain the existing logical references while displaying A--F row-major.
    axes = display_axes.ravel().reshape(3, 2)
    # Autoscale against both steps, retaining every finite value.
    finite = means[np.isfinite(means)]
    ymax = max(float(finite.max()) * 1.08, 1e-4) if finite.size else 1.
    for col, d in enumerate([primary, 1-primary]):
        ax = axes[0, col]
        for k, arm in enumerate(ARMS):
            marker = {"hybrid": "x", "tangent": "s", "axes": "o"}.get(arm, ".")
            ax.plot(radii, means[d, k], marker=marker, ms=5, color=COLORS[arm],
                    markerfacecolor="none" if arm == "axes" else COLORS[arm],
                    lw=2 if arm == "hybrid" else 1, ls="--" if arm in ("atlas", "axes") else "-", label=short[arm])
        floor = arrays["zero_mean"][d]
        if np.isfinite(floor): ax.axhline(floor, color="#999999", ls=":", lw=.8)
        ax.set(xscale="log", xlabel="Weight-change norm", ylabel="Tail RMSE",
               title=("A  Primary dose means" if col == 0 else "B  Coarse-step dose means") + f"\n(dt = {steps[d]:g}; same selections)",
               ylim=(0, ymax))
        ax.grid(alpha=.15)
        ax.legend(fontsize=9.5, ncol=2, loc="upper left", columnspacing=.65,
                  handlelength=1.25, handletextpad=.4, borderpad=.3, labelspacing=.25)
        missing = int(np.isnan(means[d]).sum())
        if missing: ax.text(.98, .98, f"{missing}/28 means undefined", transform=ax.transAxes,
                            ha="right", va="top", color="#a00020", fontsize=9.5)
    ax = axes[1, 0]
    for k, comp in enumerate(COMPARATORS):
        for j in range(4):
            m, lo, hi = effects[primary, k, j]
            x = j + (k-2)*.12
            if np.isfinite(m):
                ax.plot([x, x], [lo, hi], color=COLORS[comp], lw=1.3)
                ax.plot(x, m, "o", color=COLORS[comp], ms=4, label=LABELS[comp] if j == 0 else None)
            else: ax.annotate("×", (x, 0), color=COLORS[comp], ha="center")
    ax.axhline(0, color="black", lw=.7)
    ax.set(xticks=range(4), xticklabels=[f"{r:g}" for r in radii], xlabel="Weight-change norm",
           ylabel="Comparator − hybrid RMSE", title="C  Per-dose contrasts\n95% descriptive intervals")
    ax.grid(axis="y", alpha=.15)
    ax.legend(handles=[Line2D([], [], color=COLORS[c], marker="o", lw=1,
                             label=short[c]) for c in COMPARATORS],
              fontsize=9.5, ncol=2, loc="best", columnspacing=.65,
              handlelength=1.25, handletextpad=.4, borderpad=.3, labelspacing=.25)
    ax = axes[1, 1]
    for k, comp in enumerate(COMPARATORS):
        m, lo, hi = effects[primary, k, -1]
        if np.isfinite(m):
            ax.plot([lo, hi], [k, k], color=COLORS[comp], lw=2)
            ax.plot(m, k, "o", color=COLORS[comp], ms=5)
        else: ax.text(.02, k, "undefined", transform=ax.get_yaxis_transform(), color="#a00020")
    ax.axvline(0, color="black", lw=.7)
    ax.set(yticks=range(5), yticklabels=[short[c] for c in COMPARATORS],
           xlabel="Comparator − hybrid RMSE", title="D  Pooled contrasts\nEqual weights over four doses")
    ax.invert_yaxis(); ax.grid(axis="x", alpha=.15)
    ax = axes[2, 0]
    bound = max(float(np.nanmax(np.abs(per_anchor))), 1e-4) if np.any(np.isfinite(per_anchor)) else 1.
    cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#dedede")
    im = ax.imshow(per_anchor, aspect="auto", cmap=cmap, vmin=-bound, vmax=bound)
    ax.set(xticks=range(5), xticklabels=["Local", "Tan.", "Axes", "U16", "A21"],
           yticks=range(len(anchor_ids)), yticklabels=[f"H{i:02d}" for i in range(len(anchor_ids))],
           title=f"E  All starts, largest dose {radii[-1]:g}\nComparator − hybrid; blue favors hybrid")
    for i in range(len(anchor_ids)):
        for j in range(5):
            v = per_anchor[i, j]
            ax.text(j, i, f"{v:+.3f}" if np.isfinite(v) else "—", ha="center", va="center", fontsize=9.5,
                    color="white" if np.isfinite(v) and abs(v) > .6*bound else "black")
    colorbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    colorbar.ax.tick_params(labelsize=10)
    ax = axes[2, 1]
    ax.imshow(budgets, aspect="auto", cmap="Blues", vmin=0, vmax=21)
    ax.set(xticks=range(4), xticklabels=[f"{r:g}" for r in radii],
           yticks=range(7), yticklabels=["Atlas [M]", "Local [S]", "Tangent [M]", "Axes", "Uniform16", "A21 [S,M]", "Hybrid [S/M]"],
           xlabel="Weight-change norm", title="F  Preview budget\n75-unit calls per decision")
    for i in range(7):
        for j in range(4):
            ax.text(j, i, str(budgets[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if budgets[i, j] > 12 else "black")
    costs = data["costs"]
    fig.suptitle("EXP-007: all frozen waveform comparisons at new starting points", y=.99, fontsize=13)
    overlap_note = ("Tangent/axes select identical actions here." if same_actions else
                    "Tangent/axes means coincide." if tangent_axes_overlap else "")
    fig.text(.5, .015, f"Tail: 75<t≤300. Positive contrasts favor hybrid. {overlap_note}\n"
             "Intervals are descriptive for selected anchors; no population/multiplicity claim.\n"
             "H00–H11: holdout-00–11; Tan.=tangent; U16=uniform16; A21=augmented21. Gray/×: undefined; no success-only deletion.\n"
             "[S] sensitivity; [M] old map. Hybrid uses S for small doses, M for largest. All require baseline; previews shared in collection.\n"
             f"Collection {costs['collection_wall_seconds']:.2f} s; preview-call sum {costs['preview_sum_concurrent_call_wall_seconds']:.2f} s; "
             f"{costs['sensitivity_calls']} sensitivity calls {costs['sensitivity_sum_call_wall_seconds']:.2f} s. Concurrent sums ≠ elapsed time.\n"
             "Old atlas: 140.54 s separately. Preview counts exclude prerequisites. Total compute is not matched; no speedup established.",
             ha="center", fontsize=9.5, linespacing=1.12)
    for ax in axes.ravel():
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.title.set_fontsize(11)
    fig.subplots_adjust(left=.095, right=.975, bottom=.235, top=.865, hspace=.53, wspace=.47)
    return fig, arrays, {"status_counts": summary["status_counts"], "eligible": eligible,
                         "costs": costs, "endpoint": p["primary_endpoint"],
                         "tangent_axes_means_exactly_overlap": tangent_axes_overlap,
                         "tangent_axes_action_equivalence": equivalence,
                         "contrasts": "Five frozen comparators minus hybrid; no outcome-based selection",
                         "intervals": "10000 paired anchor bootstrap draws, percentile95, descriptive conditional on these starts"}


def waveform_figure(data):
    p, e = data["plan"], data["example"]
    arrays = {}
    conditions = {k: v for k, v in e.items() if k not in ("arms", "reference_cycle")}
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 8.4), sharex=True)
    if e["anchor"] is None:
        for ax in axes.ravel(): ax.text(.5, .5, "No eligible anchor", ha="center", transform=ax.transAxes)
    else:
        arrays["reference_cycle"] = e["reference_cycle"]
        reference_drawn = False
        uniform, search = e["arms"]["uniform"], e["arms"]["search"]
        same_trace = (uniform["status"] == search["status"] == "complete"
                      and np.array_equal(uniform["times"], search["times"])
                      and np.array_equal(uniform["activations"], search["activations"]))
        conditions["uniform_search_traces_identical"] = same_trace
        for arm in EXAMPLE_ARMS:
            item = e["arms"][arm]
            conditions[arm] = {"status": item["status"]}
            if item["status"] != "complete": continue
            for name in ["times", "activations", "synchronous_reference", "aligned_reference", "aligned_errors"]:
                arrays[f"{arm}_{name}"] = item[name]
            conditions[arm].update(phase_offset_cycles=item["phase_offset_cycles"], tail_rms=item["tail_rms"])
            for neuron in range(3):
                style = {"hybrid": "-", "local": "--", "uniform": "-.", "search": ":"}[arm]
                if not reference_drawn:
                    axes[neuron, 0].plot(item["times"], item["synchronous_reference"][:, neuron],
                                          color="#333333", lw=.7, label="Original synchronous reference")
                axes[neuron, 0].plot(item["times"], item["activations"][:, neuron], color=COLORS[arm],
                                      lw=.8, alpha=.85, ls=style, label=LABELS[arm])
                axes[neuron, 1].plot(item["times"], item["aligned_errors"][:, neuron], color=COLORS[arm],
                                      lw=.8, alpha=.85, ls=style, label=LABELS[arm])
            reference_drawn = True
        failed = [a for a in EXAMPLE_ARMS if e["arms"][a]["status"] != "complete"]
        if failed: axes[0, 0].text(.02, .98, "Undefined: " + ", ".join(failed),
                                  transform=axes[0, 0].transAxes, va="top", color="#a00020")
        if same_trace: axes[0, 1].text(.02, .05, "Uniform16 / augmented21: identical traces here",
                                      transform=axes[0, 1].transAxes, fontsize=8,
                                      bbox=dict(fc="white", ec="none", alpha=.85))
    for i, row in enumerate(axes):
        row[0].set(ylabel=f"Neuron {i+1} activation", ylim=(-1.08, 1.08))
        row[1].set(ylabel=f"Neuron {i+1} aligned error")
        row[1].axhline(0, color="#444444", lw=.5)
        for ax in row:
            ax.axvspan(0, p["preview_duration"], color="#777777", alpha=.12)
            ax.axvline(p["preview_duration"], color="#555555", ls=":", lw=.8)
            ax.set_xlim(0, p["probe_duration"]); ax.grid(alpha=.12)
    axes[0, 0].set_title("A  Candidate waveforms and original time origin")
    axes[0, 1].set_title("B  Errors after each policy's one saved tail-fit phase")
    for ax in axes[0]:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    for ax in axes[-1]: ax.set_xlabel("Physical time after fixed weight change")
    fig.suptitle(f"Frozen example: {e['anchor'] or 'none'}, phase index {e['phase']}, "
                 f"norm {e['radius']:g}, dt {e['dt']:g}", y=.995)
    fig.text(.5, .027, "First eligible anchor and first phase in frozen order; largest dose; all saved 0–300 samples. "
             "Gray interval: simulator preview 0–75.\n"
             "Tail-fit phase uses only 75<t≤300 and stays constant over the whole error trace; each policy has its own reference shift. "
             "No frequency fitting or time warp.\nThe synchronous reference on the left is not the phase-aligned reference used for the error statistic.",
             ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.08, right=.975, top=.91, bottom=.15, hspace=.23, wspace=.20)
    return fig, arrays, conditions


def build(runroot, destination):
    data = load(runroot)
    destination.mkdir(parents=True, exist_ok=True)
    generators = [SCRIPT, ROOT / "tools/plot_hopfield.py", ROOT / "tools/run_hopfield_holdout.py",
                  ROOT / "tools/hopfield_drift_metrics.py"]
    sources = [{"namespace": "run", "path": rel, "sha256": sha(runroot / rel)} for rel in data["run_sources"]]
    sources += [{"namespace": "repo", **item} for item in data["old_sources"]]
    results = []
    for stem, plot in zip(FAMILIES, [anchor_figure, utility_figure, waveform_figure]):
        fig, arrays, conditions = plot(data)
        exact_npz(destination / f"{stem}.plot-data.npz", arrays)
        for ext in ["pdf", "svg", "png"]:
            metadata = ({"CreationDate": None, "ModDate": None, "Creator": "neural-shrimp saved evidence renderer"}
                        if ext == "pdf" else {"Date": None} if ext == "svg" else
                        {"Software": "neural-shrimp saved evidence renderer"})
            fig.savefig(destination / f"{stem}.{ext}", dpi=300, metadata=metadata)
        plt.close(fig)
        receipt = {"schema_version": 1, "experiment": "EXP-007", "stem": stem,
                   "generators": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in generators],
                   "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
                   "sources": sources, "conditions": conditions,
                   "plotted_arrays": {"path": f"{stem}.plot-data.npz", "sha256": sha(destination / f"{stem}.plot-data.npz")},
                   "outputs": [{"path": f"{stem}.{ext}", "sha256": sha(destination / f"{stem}.{ext}")} for ext in ["pdf", "svg", "png"]],
                   "scope": "Frozen display contract; saved evidence only, no neural integrations; complete attempted census retained"}
        path = destination / f"{stem}.receipt.json"
        json_write(path, receipt)
        results.append({"path": path.name, "sha256": sha(path)})
    json_write(destination / INDEX, {"schema_version": 1, "experiment": "EXP-007", "figures": results})
    return [json.loads((destination / item["path"]).read_text()) for item in results]


def check(runroot, destination):
    index = json.loads((destination / INDEX).read_text())
    if [i["path"] for i in index["figures"]] != [f"{stem}.receipt.json" for stem in FAMILIES]:
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
            if sha(destination / output["path"]) != output["sha256"]: raise ValueError(f"Output byte drift: {output['path']}")
    with tempfile.TemporaryDirectory(prefix="hopfield-holdout-figures-") as temporary:
        rebuilt = build(runroot, Path(temporary))
        for before, after in zip(previous, rebuilt):
            close_identity(before["conditions"], after["conditions"], "plot conditions")
            with np.load(destination / before["plotted_arrays"]["path"], allow_pickle=False) as a, \
                 np.load(Path(temporary) / after["plotted_arrays"]["path"], allow_pickle=False) as b:
                if a.files != b.files: raise ValueError("Plotted array schema drift")
                for key in a.files: replay_array(a[key], b[key], key)
    print("PASS: EXP-007 complete input hashes, plotted arrays, census and summary replay; rendering bytes are not compared across platforms.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check: check(args.runroot.resolve(), args.destination.resolve())
    else:
        build(args.runroot.resolve(), args.destination.resolve())
        print("Built three saved-data EXP-007 figure families.")


if __name__ == "__main__":
    main()
