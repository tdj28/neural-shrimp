#!/usr/bin/env python3
"""Render EXP-004 saved results; this module never advances the neural ODE.

python tools/plot_hopfield.py RUNROOT --destination paper/figures
python tools/plot_hopfield.py RUNROOT --destination paper/figures --check

Each plot has a lossless plotted-array archive and a source/output hash receipt.
Missing rows remain visible. Period zero means no short period was detected;
it is never relabeled chaos. Candidate maps and orbit certificates are distinct.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "experiments/hopfield/EXP-004-plan.json"
NEURON_COLORS = ["#0072B2", "#D55E00", "#009E73"]
ROLE_MARKERS = {"core": "o", "outer": "s", "doubled": "D", "exterior": "X"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "neural-shrimp-hopfield-exp004-v1", "savefig.dpi": 300})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def exact_npz(path, arrays):
    """No timestamps or pickle: byte-identical array archives on the same NumPy."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, value in sorted(arrays.items()):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(f"Object array prohibited: {name}")
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def coordinates(record):
    if not isinstance(record, dict) or record.get("missing") or record.get("status") == "missing":
        return None
    value = record.get("coordinates")
    if isinstance(value, dict):
        return float(value["alpha"]), float(value["gamma"])
    if value is not None:
        return tuple(map(float, value))
    if "alpha" in record and "gamma" in record:
        return float(record["alpha"]), float(record["gamma"])
    return None


def selection_summary(selected):
    result = {}
    for role in ROLE_MARKERS:
        record = selected.get(role)
        point = coordinates(record)
        if point is None:
            result[role] = {"missing": True}
            continue
        certificate = record.get("certificate") or {}
        result[role] = {"coordinates": list(point), "candidate_period": record.get("candidate_period"),
            "certificate": {key: certificate.get(key) for key in ["verified", "attracting", "reason", "period", "peak_count", "relative_closure", "transverse_spectral_radius"]}}
    return result


def load_rows(runroot, name, alpha, gamma):
    """Align recorded rows to the declared coordinates, retaining absent cells."""
    shape = (len(gamma), len(alpha))
    result = {"alpha": np.asarray(alpha), "gamma": np.asarray(gamma),
        "period": np.full(shape, -1, dtype=int), "stats": np.full((*shape, 20), np.nan),
        "residual": np.full(shape, np.nan), "period_time": np.full(shape, np.nan),
        "present": np.zeros(shape, dtype=bool), "events": {}, "sources": []}
    occupied = set()
    for path in sorted((runroot / name).glob("row-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            row = {key: archive[key] for key in archive.files}
        if row["alpha"].shape != (len(alpha),) or not np.allclose(row["alpha"], alpha, rtol=0, atol=1e-12):
            raise ValueError(f"Wrong alpha axis: {path}")
        g = float(np.asarray(row["gamma"]).item())
        matches = np.flatnonzero(np.isclose(gamma, g, rtol=0, atol=1e-12))
        if len(matches) != 1 or int(matches[0]) in occupied:
            raise ValueError(f"Duplicate or undeclared gamma: {path}")
        i = int(matches[0]); occupied.add(i)
        expected = {"states": (len(alpha), 3), "stats": (len(alpha), 20),
            "period": (len(alpha),), "n_events": (len(alpha),),
            "residual": (len(alpha),), "period_time": (len(alpha),)}
        for key, wanted in expected.items():
            if row[key].shape != wanted:
                raise ValueError(f"Wrong {key} shape: {path}")
        if row["events"].ndim != 3 or row["events"].shape[0] != len(alpha) or row["events"].shape[2] != 4:
            raise ValueError(f"Wrong event shape: {path}")
        if not np.issubdtype(row["period"].dtype, np.integer) or np.any(row["period"] < 0):
            raise ValueError(f"Invalid peak period: {path}")
        for key in ["period", "stats", "residual", "period_time"]:
            result[key][i] = row[key]
        for j, count in enumerate(row["n_events"]):
            if int(count) != count or not 0 <= count <= row["events"].shape[1]:
                raise ValueError(f"Invalid event count: {path}")
            events = row["events"][j, :int(count)]
            if not np.all(np.isfinite(events)) or np.any(np.diff(events[:, 0]) <= 0):
                raise ValueError(f"Invalid chronological events: {path}")
            result["events"][i, j] = events
        result["present"][i] = True
        result["sources"].append(path)
    return result


def load(runroot):
    plan = json.loads(PLAN.read_text())
    manifest = json.loads((runroot / "manifest.json").read_text())
    selected = json.loads((runroot / "selected.json").read_text())
    if "selected" in selected:
        selected = selected["selected"]
    grids = {item["name"]: load_rows(runroot, item["name"],
        np.linspace(*item["alpha"]), np.linspace(*item["gamma"])) for item in plan["grids"]}
    transect = plan["transect"]
    grids["transect"] = load_rows(runroot, "transect", np.array([transect["alpha"]]), np.linspace(*transect["gamma"]))
    return plan, manifest, selected, grids


def period_style(grids):
    periods = sorted({int(p) for g in grids.values() for p in g["period"].ravel() if p > 0})
    values = [-1, 0] + periods
    palette = ["#DDDDDD", "#FCF9F2"]
    tab = plt.get_cmap("tab20")
    palette += ["#0072B2" if p == 9 else "#D55E00" if p == 18 else tab((i + 4) % 20) for i, p in enumerate(periods)]
    return values, ListedColormap(palette), BoundaryNorm(np.arange(len(values) + 1) - .5, len(values))


def period_map(ax, grid, style):
    values, cmap, norm = style
    encoded = np.full(grid["period"].shape, -1, dtype=int)
    for i, p in enumerate(values):
        encoded[grid["period"] == p] = i
    if np.any(encoded < 0):
        raise ValueError("Period palette omitted a recorded category")
    ax.pcolormesh(grid["alpha"], grid["gamma"], encoded, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
    format_map(ax, grid)


def format_map(ax, grid):
    ax.set(xlabel=r"Connection $\alpha$ (3 $\to$ 1)", ylabel=r"Connection $\gamma$ (1 $\to$ 3)",
        xlim=(grid["alpha"][0], grid["alpha"][-1]), ylim=(grid["gamma"][0], grid["gamma"][-1]))
    ax.ticklabel_format(useOffset=False, style="plain")


def annotate_points(ax, selected):
    for role, marker in ROLE_MARKERS.items():
        point = coordinates(selected.get(role))
        if point is not None:
            ax.scatter(*point, marker=marker, s=42, c="white", edgecolors="#151515", linewidths=1.0, zorder=5)


def period_legend(fig, grids, style, selected=False, y=.06):
    values, cmap, _ = style
    present = {int(p) for g in grids.values() for p in g["period"].ravel()}
    handles = [Patch(facecolor=cmap(i), edgecolor="#888888", label=("Not measured" if p == -1 else "No short period" if p == 0 else f"{p} peaks"))
        for i, p in enumerate(values) if p in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=min(len(handles), 5), frameon=False)
    if selected:
        handles = [Line2D([], [], marker=m, color="none", markerfacecolor="white", markeredgecolor="#151515", label=r.capitalize()) for r, m in ROLE_MARKERS.items()]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y-.05), ncol=4, frameon=False)


def grid_arrays(name, grid):
    return {f"{name}_{key}": grid[key] for key in ["alpha", "gamma", "period", "stats", "residual", "period_time", "present"]}


def save_figure(fig, destination, stem, title, description, alt, runroot, sources, arrays, transformations, selection):
    fig.suptitle(title, x=.065, ha="left", fontweight="bold", fontsize=13)
    outputs = []
    for ext in ["pdf", "svg", "png"]:
        path = destination / f"{stem}.{ext}"
        metadata = {"Creator": "tools/plot_hopfield.py", "Title": title}
        if ext == "pdf": metadata.update({"CreationDate": None, "ModDate": None, "Subject": alt})
        else: metadata["Description"] = alt
        if ext == "svg": metadata["Date"] = None
        fig.savefig(path, dpi=300, facecolor="white", metadata=metadata)
        outputs.append({"path": path.name, "sha256": sha(path)})
    plt.close(fig)
    archive_path = destination / f"{stem}.plot-data.npz"
    exact_npz(archive_path, arrays)
    schema = {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype),
        "values_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()} for key, value in sorted(arrays.items())}
    receipt = {"schema_version": 1, "figure_id": stem, "title": title,
        "description": description, "alt_text": alt,
        "study_id": "EXP-004", "run_id": runroot.name,
        "generator": {"path": "tools/plot_hopfield.py", "sha256": sha(SCRIPT)},
        "generator_repository": "https://github.com/tdj28/neural-shrimp",
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
        "source_root": ".", "sources": [{"path": str(p.relative_to(runroot)), "sha256": sha(p)} for p in sorted(set(sources))],
        "outputs": outputs, "plotted_arrays": {"path": archive_path.name, "sha256": sha(archive_path), "arrays": schema},
        "source_fields": {"row_arrays": ["alpha", "gamma", "period", "stats", "present (derived row census)", "events[:n_events,:]", "residual", "period_time"],
            "stats_abi": ["total_peaks", "finite_time_lyapunov_exponent", "ftle_measurement_time", "sample_time", "mean_x1", "mean_x2", "mean_x3", "std_x1", "std_x2", "std_x3", "mean_tanh_x1", "mean_tanh_x2", "mean_tanh_x3", "std_tanh_x1", "std_tanh_x2", "std_tanh_x3", "mean_squared_speed", "final_speed", "burn_steps", "sample_steps"],
            "selected_fields": ["alpha", "gamma", "candidate_period", "certificate", "behavior.times", "behavior.states", "behavior.activations", "behavior.separation", "behavior.activation_separation", "behavior.metrics"]},
        "transformations": transformations, "selection": selection,
        "uncertainty": "Descriptive numerical measurements; no sampling-confidence intervals. Step and transient checks are retained separately.",
        "claim_scope": "Peak-event maps screen candidate periodicity. Missing and no-short-period cells are retained. FTLE is finite-horizon. No learned-task performance or gradient-training claim.",
        "accessibility": "Axes label controls, time and quantities; categorical legend names every observed peak period; exact plotted values are in the accompanying archive; series use line styles and marker shapes; the complete alt text is embedded in PDF Subject and SVG/PNG Description metadata.",
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__}}
    receipt_path = destination / f"{stem}.receipt.json"
    json_write(receipt_path, receipt)
    return {"path": receipt_path.name, "sha256": sha(receipt_path)}


def maps(runroot, destination, selected, grids, common):
    overview, focus = grids["overview"], grids["focus"]
    style = period_style({"overview": overview, "focus": focus})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.1))
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.24, wspace=.28)
    for ax, grid, name in zip(axes, [overview, focus], ["A  Published zoom rectangle", "B  Focused shrimp rectangle"]):
        period_map(ax, grid, style); annotate_points(ax, selected)
        ax.set_title(name, loc="left")
    axes[0].add_patch(Rectangle((focus["alpha"][0], focus["gamma"][0]), np.ptp(focus["alpha"]), np.ptp(focus["gamma"]), fill=False, edgecolor="#252525", linewidth=.9, linestyle="--"))
    period_legend(fig, {"overview": overview, "focus": focus}, style, selected=True, y=.083)
    fig.text(.08, .015, "Colors count peaks in a repeating full-state event sequence; zero is unresolved, not a chaos label.", fontsize=8)
    return save_figure(fig, destination, "neuralshrimp-map", "A two-connection scan of recurrent neural dynamics",
        "Complete overview and focus grids show candidate peak-period organization at recorded connection weights.",
        "Two categorical maps show the declared control rectangles. Each cell uses its saved full-state peak-event period. A dashed rectangle locates the focus. Selected coordinates share markers between panels. Unmeasured cells are gray and no-short-period cells are pale.",
        runroot, common + overview["sources"] + focus["sources"], {**grid_arrays("overview", overview), **grid_arrays("focus", focus)},
        ["Align rows to the frozen coordinate grid; retain missing rows as category -1 and unresolved candidates as 0.", "Nearest-cell categorical display with actual sampled coordinates; no smoothing or image reconstruction.", "Period category means recurrence screen; independent certificates are separate."], selection_summary(selected))


def aligned(runroot, destination, selected, grids, common):
    grid = grids["focus"]; style = period_style({"focus": grid})
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.9))
    fig.subplots_adjust(left=.068, right=.985, top=.83, bottom=.33, wspace=.38)
    period_map(axes[0], grid, style); axes[0].set_title("A  Recurrent peak count", loc="left")
    for ax, index, title, cmap in [(axes[1], 1, "B  Finite-time instability", "coolwarm"), (axes[2], 15, "C  Activation variability", "viridis")]:
        data = np.ma.masked_invalid(grid["stats"][:, :, index])
        norm = None
        if index == 1 and data.count() and data.min() < 0 < data.max():
            norm = TwoSlopeNorm(vmin=float(data.min()), vcenter=0, vmax=float(data.max()))
        p = ax.pcolormesh(grid["alpha"], grid["gamma"], data, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
        format_map(ax, grid); ax.set_title(title, loc="left")
        box = ax.get_position()
        cax = fig.add_axes([box.x0, .185, box.width, .024])
        colorbar = fig.colorbar(p, cax=cax, orientation="horizontal",
            label="Finite-time Lyapunov exponent / time" if index == 1 else r"Standard deviation of $\tanh(x_3)$")
        colorbar.locator = matplotlib.ticker.MaxNLocator(5)
        colorbar.update_ticks()
    for ax in axes: annotate_points(ax, selected)
    period_legend(fig, {"focus": grid}, style, selected=False, y=.04)
    fig.text(.068, .012, "The same saved controls underlie every panel; finite-time exponent is not an asymptotic chaos certificate.", fontsize=8)
    return save_figure(fig, destination, "neuralshrimp-aligned-map", "Geometry alongside instability and the neuron's actual output",
        "Aligned maps compare candidate peak recurrence, measured finite-time exponent, and variability of the third neuron's tanh activation.",
        "Three maps use identical alpha and gamma coordinates. The first is categorical peak period, the second is finite-time Lyapunov exponent, and the third is standard deviation of tanh(x3). Their alignment exposes whether periodic geometry coincides with changes in measured neural activity. Missing measurements are masked.",
        runroot, common + grid["sources"], grid_arrays("focus", grid),
        ["No averaging across neighboring controls.", "FTLE is native stats column 1; activation standard deviation is native stats column 15.", "A diverging scale centers FTLE at zero only when both signs occur; quantitative color limits use all finite focus values."], selection_summary(selected))


def transect_plot(runroot, destination, selected, grids, common):
    grid = grids["transect"]; g = grid["gamma"]
    peak_gamma, peak_value = [], []
    for (i, j), events in sorted(grid["events"].items()):
        peak_gamma.extend([g[i]] * len(events)); peak_value.extend(events[:, 3])
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8.2, 7.6), gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    fig.subplots_adjust(left=.12, right=.97, top=.89, bottom=.10, hspace=.27)
    axes[0].scatter(peak_gamma, peak_value, s=.8, c="#233247", edgecolors="none", rasterized=True)
    axes[0].set(ylabel=r"Local maxima of $x_3$", title=f"A  Common initialization at alpha = {grid['alpha'][0]:.6g}")
    periods = np.where(grid["period"][:, 0] < 0, np.nan, grid["period"][:, 0])
    axes[1].plot(g, periods, ".-", color="#0072B2", markersize=2, linewidth=.7)
    axes[1].set(ylabel="Peak period", title="B  Full-state event recurrence (0: no short period)")
    axes[2].plot(g, grid["stats"][:, 0, 1], color="#AC3B40", lw=1)
    axes[2].axhline(0, color="#777777", ls="--", lw=.7)
    axes[2].set(ylabel="FTLE / time", title="C  Finite-horizon sensitivity")
    for j in range(3):
        axes[3].plot(g, grid["stats"][:, 0, 13+j], color=NEURON_COLORS[j], lw=1, ls=["-", "--", ":"][j], label=f"Neuron {j+1}")
    axes[3].set(ylabel="Activation std.", xlabel=r"Connection $\gamma$", title="D  Variability after the tanh readout")
    axes[3].legend(loc="best", ncol=3, frameon=False)
    for ax in axes:
        title = ax.get_title(); ax.set_title("")
        ax.set_xlim(g[0], g[-1]); ax.grid(alpha=.15); ax.set_title(title, loc="left", fontsize=9)
    fig.text(.12, .016, "All transect points start from the same prescribed state; these measurements do not inherit grid continuation.", fontsize=8)
    arrays = {**grid_arrays("transect", grid), "peak_gamma": np.asarray(peak_gamma), "peak_x3": np.asarray(peak_value)}
    return save_figure(fig, destination, "neuralshrimp-transect", "Network activity along an independent control transect",
        "An independent-initialization connection-weight transect aligns neuron-potential maxima with recurrence, finite-time instability, and activation variability.",
        "Four vertically aligned panels plot all retained x3 maxima, detected peak period, finite-time Lyapunov exponent and three neuron activation standard deviations against gamma. Zero period means no short period detected. Each point uses the same initial condition, independently of the two-dimensional continuation maps.",
        runroot, common + grid["sources"], arrays,
        ["Plot every valid retained maximum, without thinning or smoothing.", "Period zero is retained at zero; missing rows are gaps.", "Stats columns 13 through 15 are standard deviations of tanh activations; no potential-to-activation substitution."], {"alpha": float(grid["alpha"][0]), "selection": "Every declared transect point."})


def waveform(runroot, destination, selected, grids, common):
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(9, 8.0))
    fig.subplots_adjust(left=.095, right=.98, top=.855, bottom=.105, hspace=.30, wspace=.27)
    arrays = {}; included = {}
    for col, role in enumerate(["core", "exterior"]):
        record = selected.get(role); point = coordinates(record)
        behavior = record.get("behavior") if isinstance(record, dict) else None
        if point is None or not isinstance(behavior, dict) or "times" not in behavior:
            for ax in axes[:, col]:
                ax.text(.5, .5, f"{role.capitalize()} trace unavailable", transform=ax.transAxes, ha="center", va="center"); ax.set_axis_off()
            included[role] = {"missing": True}
            continue
        t = np.asarray(behavior["times"], dtype=float)
        states = np.asarray(behavior["states"], dtype=float)
        acts = np.asarray(behavior["activations"], dtype=float)
        if t.ndim != 1 or states.shape != (len(t), 3) or acts.shape != states.shape or not np.all(np.diff(t) > 0):
            raise ValueError(f"Malformed {role} behavior trace")
        if not np.allclose(acts, np.tanh(states), rtol=1e-12, atol=1e-14):
            raise ValueError(f"Activation readout replay failed for {role}")
        if "perturbed_states" in behavior:
            perturbed = np.asarray(behavior["perturbed_states"], dtype=float)
            if perturbed.shape != states.shape:
                raise ValueError(f"Malformed {role} perturbed states")
            replay = np.linalg.norm(perturbed - states, axis=1)
            if not np.allclose(replay, behavior["separation"], rtol=1e-12, atol=1e-14):
                raise ValueError(f"State separation replay failed for {role}")
            if "activation_separation" in behavior:
                replay = np.linalg.norm(np.tanh(perturbed) - acts, axis=1)
                if not np.allclose(replay, behavior["activation_separation"], rtol=1e-12, atol=1e-14):
                    raise ValueError(f"Activation separation replay failed for {role}")
        for j in range(3):
            axes[0, col].plot(t, states[:, j], color=NEURON_COLORS[j], ls=["-", "--", ":"][j], lw=.8, label=f"Neuron {j+1}")
            axes[1, col].plot(t, acts[:, j], color=NEURON_COLORS[j], ls=["-", "--", ":"][j], lw=.8)
        for key, label, color, ls in [("separation", "State distance", "#233247", "-"), ("activation_separation", "Activation distance", "#D55E00", "--")]:
            if key in behavior:
                values = np.asarray(behavior[key], dtype=float)
                if values.shape != t.shape: raise ValueError(f"Malformed {role} {key}")
                axes[2, col].plot(t, np.ma.masked_less_equal(values, 0), color=color, ls=ls, lw=1, label=label)
                arrays[f"{role}_{key}"] = values
        axes[2, col].set_yscale("log")
        certificate = record.get("certificate") or {}
        label = "certified cycle" if certificate.get("verified") and certificate.get("attracting") else "selected trajectory"
        axes[0, col].set_title(f"{role.capitalize()}: {label}\nalpha={point[0]:.6g}, gamma={point[1]:.6g}", loc="left")
        axes[0, col].set_ylabel("Neuron potential $x_i$")
        axes[1, col].set_ylabel(r"Activation $\tanh(x_i)$")
        axes[2, col].set(ylabel="Nearby-state distance", xlabel="Elapsed time")
        axes[0, col].legend(loc="upper right", ncol=3, fontsize=6.7, frameon=False)
        axes[2, col].legend(loc="best", fontsize=7, frameon=False)
        for ax in axes[:, col]: ax.set_xlim(t[0], t[-1]); ax.grid(alpha=.13)
        arrays.update({f"{role}_time": t, f"{role}_states": states, f"{role}_activations": acts})
        included[role] = {"coordinates": list(point), "candidate_period": record.get("candidate_period"),
            "certificate_verified": bool(certificate.get("verified")), "certificate_attracting": bool(certificate.get("attracting")), "metrics": behavior.get("metrics")}
    for row in range(3):
        active_axes = [ax for ax in axes[row] if ax.axison]
        if active_axes:
            limits = [ax.get_ylim() for ax in active_axes]
            for ax in active_axes:
                ax.set_ylim(min(pair[0] for pair in limits), max(pair[1] for pair in limits))
    fig.text(.095, .025, "A small state perturbation is applied at time zero. Raw separation includes neutral phase shifts;\nit is not a direct measure of transverse recovery or task correctness.", fontsize=8)
    return save_figure(fig, destination, "neuralshrimp-waveforms", "What the same parent network does inside and outside the window",
        "Recorded core and exterior trajectories compare all three internal neuron states, their actual tanh activations, and nearby-state separation.",
        "Two columns show the deterministically selected core and exterior. The first row plots all three neuron potentials, the second their tanh activations, and the third the logarithmic distance between base and perturbed trajectories. Failed or missing selections are stated rather than replaced. Raw separation can include a persistent phase shift of a stable cycle.",
        runroot, common, arrays,
        ["Plot all stored behavior sample times, with no simulation, smoothing or temporal selection; common y limits within each measured quantity.", "Independently replay tanh(states), and state/activation separation when perturbed states exist, at rtol1e-12,atol1e-14 before plotting.", "Exact zero distances masked on log axes but retained in plotted-array archive.", "Candidate labels upgraded to certified cycle only when certificate verified and attracting are both true."], included)


def build(runroot, destination):
    destination.mkdir(parents=True, exist_ok=True)
    plan, manifest, selected, grids = load(runroot)
    common = [runroot / "manifest.json", runroot / "selected.json"]
    receipts = [fn(runroot, destination, selected, grids, common) for fn in [maps, aligned, transect_plot, waveform]]
    index = {"schema_version": 1, "experiment": plan["experiment_id"], "generator_sha256": sha(SCRIPT),
        "manifest_sha256": sha(runroot / "manifest.json"), "figures": receipts,
        "measured_cells": {k: int(g["present"].sum()) for k, g in grids.items()},
        "missing_cells": {k: int((~g["present"]).sum()) for k, g in grids.items()},
        "scope": "Saved-result visualization only; no new target numerical trajectories."}
    json_write(destination / "hopfield-figure-provenance.json", index)
    return index


def check(runroot, destination, verify=False):
    index = json.loads((destination / "hopfield-figure-provenance.json").read_text())
    if index["generator_sha256"] != sha(SCRIPT) or index["manifest_sha256"] != sha(runroot / "manifest.json"):
        raise ValueError("Figure generator or source manifest drift")
    for ref in index["figures"]:
        receipt_path = destination / ref["path"]
        if sha(receipt_path) != ref["sha256"]: raise ValueError(f"Receipt drift: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if sha(ROOT / receipt["plan"]["path"]) != receipt["plan"]["sha256"]: raise ValueError("Plan drift")
        for source in receipt["sources"]:
            if sha(runroot / source["path"]) != source["sha256"]: raise ValueError(f"Source drift: {source['path']}")
        for output in receipt["outputs"] + [receipt["plotted_arrays"]]:
            if sha(destination / output["path"]) != output["sha256"]: raise ValueError(f"Output drift: {output['path']}")
    with tempfile.TemporaryDirectory(prefix="hopfield-figure-check-") as tmp:
        rebuilt = Path(tmp); new_index = build(runroot, rebuilt)
        for ref in index["figures"]:
            old = json.loads((destination / ref["path"]).read_text())
            new = json.loads((rebuilt / ref["path"]).read_text())
            with np.load(destination / old["plotted_arrays"]["path"], allow_pickle=False) as a, np.load(rebuilt / new["plotted_arrays"]["path"], allow_pickle=False) as b:
                if a.files != b.files: raise ValueError("Plotted array schema drift")
                for key in a.files:
                    if not np.array_equal(a[key], b[key], equal_nan=True): raise ValueError(f"Plotted array drift: {key}")
            if old["selection"] != new["selection"] or old["transformations"] != new["transformations"]:
                raise ValueError("Figure selection or transformation drift")
            if verify:
                for output in old["outputs"]:
                    if sha(rebuilt / output["path"]) != output["sha256"]: raise ValueError(f"Nonidentical regeneration: {output['path']}")
        if new_index["measured_cells"] != index["measured_cells"] or new_index["missing_cells"] != index["missing_cells"]:
            raise ValueError("Measurement census drift")
    print("PASS: four figure receipts, source/output hashes and exact plotted arrays re-derived." + (" Figure bytes regenerate identically." if verify else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runroot", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(); runroot = args.runroot.resolve(); destination = args.destination.resolve()
    if args.check or args.verify:
        check(runroot, destination, verify=args.verify)
    else:
        result = build(runroot, destination)
        print(json.dumps({"figures": len(result["figures"]), "measured_cells": result["measured_cells"], "missing_cells": result["missing_cells"]}, sort_keys=True))


if __name__ == "__main__":
    main()
