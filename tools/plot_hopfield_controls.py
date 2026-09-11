#!/usr/bin/env python3
"""Plot every frozen EXP-005 control from saved evidence; never integrate ODEs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_hopfield import exact_npz, sha, json_write, NEURON_COLORS

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
HELPER = ROOT / "tools/plot_hopfield.py"
PLAN = ROOT / "experiments/hopfield/EXP-005-plan.json"
STEM = "neuralshrimp-controls"
NAMES = ["core", "lower", "upper", "failed-near-edge"]
DISPLAY_NAMES = ["Core", "Lower endpoint", "Upper endpoint", "Near-edge (retained)"]
TITLE = "Independent controls link neural activity to perturbation sensitivity"
ALT = ("Twelve panels show every EXP-005 control: core, lower transect endpoint, upper endpoint, and the retained near-edge comparison. "
       "Each column reports both step-size finite-time Lyapunov estimates and its certificate or finite-horizon classification. "
       "Rows plot all three neuron potentials, all three actual tanh activations, and state plus activation distances between baseline and perturbed trajectories. "
       "Axes use common scales across each row. The complete 300-unit records are shown, without smoothing or excluded controls. "
       "Irregular classification is limited to the measured horizons; raw perturbation distance includes possible phase displacement.")


def load_results(runroot):
    receipt = json.loads((runroot / "receipt.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Incomplete EXP-005 attempt")
    for item in receipt["artifacts"]:
        if sha(runroot / item["path"]) != item["sha256"]:
            raise ValueError(f"Raw artifact drift: {item['path']}")
    plan = json.loads(PLAN.read_text())
    if plan["experiment_id"] != "EXP-005" or json.loads((runroot / "manifest.json").read_text()) != plan:
        raise ValueError("EXP-005 plan/manifest mismatch")
    summary = json.loads((runroot / "summary.json").read_text())
    if set(summary) != set(NAMES) or [p["name"] for p in plan["points"]] != NAMES:
        raise ValueError("Control census or order drift")
    values = {}
    for name, control in zip(NAMES, plan["points"]):
        item = summary[name]
        if item != json.loads((runroot / "points" / f"{name}.json").read_text()):
            raise ValueError(f"Summary differs from point receipt: {name}")
        if item["alpha"] != control["alpha"] or item["gamma"] != control["gamma"]:
            raise ValueError(f"Control coordinate drift: {name}")
        steps = item["step_results"]
        if [s["dt"] for s in steps] != plan["dt"]:
            raise ValueError(f"Step-size census drift: {name}")
        irregular = all(s["period"] == 0 and s["stats"][1] > plan["irregular_ftle_threshold"] for s in steps)
        if irregular != item["finite_horizon_irregular_support"]:
            raise ValueError(f"Irregular criterion mismatch: {name}")
        with np.load(runroot / "points" / f"{name}-behavior.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        times, states = arrays["times"], arrays["states"]
        if states.shape != (len(times), 3) or not np.all(np.diff(times) > 0):
            raise ValueError(f"Behavior shape or chronology: {name}")
        if times[0] != 0 or times[-1] != plan["behavior"]["duration"]:
            raise ValueError(f"Behavior horizon drift: {name}")
        for key, raw in [("activations", states), ("perturbed_activations", arrays["perturbed_states"])]:
            if not np.allclose(arrays[key], np.tanh(raw), rtol=1e-12, atol=1e-14):
                raise ValueError(f"Activation replay mismatch: {name} {key}")
        for key, first, second in [("separation", "states", "perturbed_states"), ("activation_separation", "activations", "perturbed_activations")]:
            replay = np.linalg.norm(arrays[first] - arrays[second], axis=1)
            if not np.allclose(replay, arrays[key], rtol=1e-12, atol=1e-14):
                raise ValueError(f"Perturbation distance replay mismatch: {name}")
        values[name] = (item, arrays)
    return receipt, values


def build(runroot, destination):
    receipt, values = load_results(runroot)
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, sharex=True, sharey="row", figsize=(11.8, 7.2))
    fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.77, wspace=.12, hspace=.25)
    plotted = {}; conditions = {}
    for col, (name, title) in enumerate(zip(NAMES, DISPLAY_NAMES)):
        record, data = values[name]; t = data["times"]; certificate = record["certificate"]
        if certificate.get("verified") and certificate.get("attracting"):
            classification = f"Certified {certificate['peak_count']}-peak cycle"
        elif record["finite_horizon_irregular_support"]:
            classification = "Finite-horizon irregular"
        else:
            classification = "Unresolved at this horizon"
        exponents = [float(step["stats"][1]) for step in record["step_results"]]
        heading = (f"{chr(65+col)}  {title}\n{classification}\n"
                   f"alpha={record['alpha']:.6g}, gamma={record['gamma']:.6g}\n"
                   f"FTLE: {exponents[0]:.3g} / {exponents[1]:.3g}")
        axes[0, col].set_title(heading, loc="left", fontsize=8, linespacing=1.35)
        for neuron, color in enumerate(NEURON_COLORS):
            style = ["-", "--", ":"][neuron]
            axes[0, col].plot(t, data["states"][:, neuron], color=color, ls=style, lw=.65)
            axes[1, col].plot(t, data["activations"][:, neuron], color=color, ls=style, lw=.65)
        for key, color, style in [("separation", "#233247", "-"), ("activation_separation", "#D55E00", "--")]:
            axes[2, col].plot(t, np.ma.masked_less_equal(data[key], 0), color=color, ls=style, lw=.9)
        axes[2, col].set_yscale("log")
        for ax in axes[:, col]:
            ax.set_xlim(t[0], t[-1]); ax.set_xticks([0, 150, 300]); ax.grid(alpha=.13)
        axes[2, col].set_xlabel("Elapsed time")
        plotted.update({f"{name}_{key}": data[key] for key in ["times", "states", "activations", "separation", "activation_separation"]})
        plotted[f"{name}_step_dt"] = np.asarray([step["dt"] for step in record["step_results"]])
        plotted[f"{name}_step_ftle"] = np.asarray(exponents)
        plotted[f"{name}_step_peak_period"] = np.asarray([step["period"] for step in record["step_results"]], dtype=int)
        plotted[f"{name}_coordinates"] = np.asarray([record["alpha"], record["gamma"]])
        conditions[name] = {"classification": classification, "certificate_verified": bool(certificate.get("verified")),
            "certificate_attracting": bool(certificate.get("attracting")),
            "finite_horizon_irregular_support": record["finite_horizon_irregular_support"],
            "step_periods": [int(s["period"]) for s in record["step_results"]],
            "final_state_distance": float(data["separation"][-1]),
            "final_activation_distance": float(data["activation_separation"][-1])}
    axes[0, 0].set_ylabel("Neuron potential $x_i$")
    axes[1, 0].set_ylabel(r"Activation $\tanh(x_i)$")
    axes[2, 0].set_ylabel("Nearby-state distance")
    handles = [Line2D([], [], color=color, ls=style, label=f"Neuron {i+1}") for i, (color, style) in enumerate(zip(NEURON_COLORS, ["-", "--", ":"]))]
    handles += [Line2D([], [], color="#233247", label="State distance"), Line2D([], [], color="#D55E00", ls="--", label="Activation distance")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .06), ncol=5, frameon=False, fontsize=8)
    fig.suptitle(TITLE, x=.075, y=.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(.075, .88, "All four controls retained; common initialization for each step-size run.\nHeader FTLE order: dt = 0.01 / 0.005; perturbation norm = 1e-6.", fontsize=9)
    fig.text(.075, .014, "Common scales expose the distance contrast. Finite-horizon irregular support is not a chaos proof;\nsmall same-time distances on a cycle can retain a neutral phase shift. No task-accuracy benefit is inferred.", fontsize=8)
    outputs = []
    for ext in ["pdf", "svg", "png"]:
        path = destination / f"{STEM}.{ext}"
        metadata = {"Creator": "tools/plot_hopfield_controls.py", "Title": TITLE}
        if ext == "pdf": metadata.update({"CreationDate": None, "ModDate": None, "Subject": ALT})
        else: metadata["Description"] = ALT
        if ext == "svg": metadata["Date"] = None
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        outputs.append({"path": path.name, "sha256": sha(path)})
    plt.close(fig)
    plot_path = destination / f"{STEM}.plot-data.npz"; exact_npz(plot_path, plotted)
    array_schema = {key: {"shape": list(value.shape), "dtype": str(value.dtype), "values_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()} for key, value in sorted(plotted.items())}
    figure_receipt = {"schema_version": 1, "figure_id": STEM, "study_id": "EXP-005", "run_id": runroot.name,
        "title": TITLE, "description": "Every independently restarted EXP-005 control and its full recorded parent-network behavior, with both refined finite-time exponents.",
        "alt_text": ALT, "generator": {"path": str(SCRIPT.relative_to(ROOT)), "sha256": sha(SCRIPT)},
        "supporting_generator": {"path": str(HELPER.relative_to(ROOT)), "sha256": sha(HELPER)},
        "generator_repository": "https://github.com/tdj28/neural-shrimp",
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
        "data_source": {"receipt": {"path": "receipt.json", "sha256": sha(runroot / "receipt.json")}, "source_commit": receipt["source_commit"],
            "artifacts": receipt["artifacts"], "fields": ["points/*-behavior.npz: times, states, activations, separation, activation_separation", "summary.json: coordinates, step_results[*].dt/period/stats[1], certificate, finite_horizon_irregular_support"]},
        "conditions": conditions, "outputs": outputs,
        "plotted_arrays": {"path": plot_path.name, "sha256": sha(plot_path), "arrays": array_schema},
        "transformations": ["All four frozen points in plan order; no omitted control.", "All recorded times, no smoothing, resampling or temporal selection.", "Roundoff-tolerant activation and Euclidean-distance identity checks before plotting (rtol=1e-12, atol=1e-14); raw hashes and plotted-array equality remain exact.", "Common y scales within each quantity; log distances mask exact zero only, which remains in the data archive.", "Finite-horizon irregular label requires both step-size criteria from the frozen plan; certified-cycle label requires verified and attracting."],
        "uncertainty": "Descriptive numerical trajectories and two-step finite-time estimates; no confidence intervals or asymptotic chaos proof.",
        "claim_scope": "Observable state/activation behavior and sensitivity in this fixed-weight recurrent network. No trained-task correctness or utility claim.",
        "accessibility": "Neuron lines use distinct dashes and colors; headings name the classifications; quantitative axes use common scales; full alt text is embedded in PDF Subject and SVG/PNG Description; exact plotted values are in the archive.",
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    path = destination / f"{STEM}.receipt.json"; json_write(path, figure_receipt)
    json_write(destination / "hopfield-controls-figure-provenance.json", {"schema_version": 1, "experiment": "EXP-005", "figures": [{"path": path.name, "sha256": sha(path)}]})
    return figure_receipt


def check(runroot, destination, verify=False):
    index = json.loads((destination / "hopfield-controls-figure-provenance.json").read_text())
    ref = index["figures"][0]; path = destination / ref["path"]
    if sha(path) != ref["sha256"]: raise ValueError("Figure receipt drift")
    previous = json.loads(path.read_text())
    for key in ["generator", "supporting_generator", "plan"]:
        item = previous[key]
        if sha(ROOT / item["path"]) != item["sha256"]: raise ValueError(f"Figure source drift: {key}")
    if sha(runroot / "receipt.json") != previous["data_source"]["receipt"]["sha256"]: raise ValueError("Experiment receipt drift")
    for item in previous["outputs"] + [previous["plotted_arrays"]]:
        if sha(destination / item["path"]) != item["sha256"]: raise ValueError(f"Output drift: {item['path']}")
    with tempfile.TemporaryDirectory(prefix="hopfield-controls-figure-") as tmp:
        out = Path(tmp); rebuilt = build(runroot, out)
        if previous["conditions"] != rebuilt["conditions"]: raise ValueError("Control classification or selection drift")
        with np.load(destination / previous["plotted_arrays"]["path"], allow_pickle=False) as a, np.load(out / rebuilt["plotted_arrays"]["path"], allow_pickle=False) as b:
            if a.files != b.files: raise ValueError("Plot schema drift")
            for key in a.files:
                if not np.array_equal(a[key], b[key], equal_nan=True): raise ValueError(f"Plotted array drift: {key}")
        if verify:
            for item in previous["outputs"]:
                if sha(out / item["path"]) != item["sha256"]: raise ValueError(f"Nonidentical image regeneration: {item['path']}")
    print("PASS: every EXP-005 source hash, all four controls, replay gates and exact plotted arrays." + (" Images regenerate byte-identically." if verify else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("runroot", type=Path)
    parser.add_argument("--destination", type=Path, required=True); parser.add_argument("--check", action="store_true"); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.check or args.verify: check(args.runroot.resolve(), args.destination.resolve(), args.verify)
    else: build(args.runroot.resolve(), args.destination.resolve()); print("Built EXP-005 four-control figure and provenance.")


if __name__ == "__main__": main()
