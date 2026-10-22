"""Plot the audited BSC6 pilot without model calls or inferred connections.

Conditional feature values and empirical error envelopes are distinct from
actual trace residuals and sampled coordinates. Failed and unattempted cases
remain visible. Rendering scientific evidence requires committed plot source.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
SOURCE = "tools/plot_baby_shrimp_mixed_junction.py"
FINISHER = "tools/baby_shrimp_mixed_junction_probe_finish.py"
DRIVER = "tools/baby_shrimp_mixed_junction_probe.py"
METADATA = ("id", "model_revision", "period", "start", "coordinate_scales",
            "anchor_directory", "anchor_hashes", "intervention")
SIDES = ("positive_tau_side", "negative_tau_side")
COLORS = ("#32759c", "#b76b2c", "#458365")
SCOPE = ("Conditional Hessian-feature estimates at numerically corrected intersections; the Hessian eigenvector need not be "
         "the geometric normal of its feature locus. Empirical envelopes are not confidence intervals "
         "or verified neural bounds. Actual trace residuals are retained as measured. No exact zero, ridge crossing, "
         "between-case connection, or superstability is established.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(path):
    require(isinstance(path, str) and path and not Path(path).is_absolute()
            and ".." not in Path(path).parts and Path(path).as_posix() == path, "invalid relative artifact path")
    return path


def git_bytes(commit, path):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "exact Git commit required")
    relative_path(path)
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def frozen_literal(source, name):
    nodes = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    require(len(nodes) == 1, "missing frozen dependency declaration: "+name)
    value = ast.literal_eval(nodes[0])
    require(isinstance(value, (list, set, tuple)) and value and all(isinstance(p, str) for p in value),
            "invalid frozen dependency declaration")
    return set(value)


def source_bindings(commit, sources, required):
    require(isinstance(sources, dict) and required <= set(sources), "source manifest incomplete")
    for path, sha in sources.items():
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
                and hashlib.sha256(git_bytes(commit, path)).hexdigest() == sha, "source hash differs: "+path)


def verify_raw_manifest(receipt):
    run = Path(receipt["raw_directory"])
    require(run.is_absolute() and run.is_dir() and not run.is_symlink()
            and run.resolve().is_relative_to((DATA/"explore").resolve()), "raw directory outside data store")
    run = run.resolve()
    entries = receipt["manifest"]
    require(isinstance(entries, list) and entries, "empty raw manifest")
    require(not any(p.is_symlink() for p in run.rglob("*")), "raw snapshot contains a symlink")
    names = set()
    for entry in entries:
        name = relative_path(entry["path"])
        require(name not in names, "duplicate raw artifact")
        names.add(name); path = run/name
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0
                and isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                and path.is_file() and path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                "raw artifact hash or size differs: "+name)
    require(names == {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()},
            "unmanifested or missing raw files")
    require("protocol.json" in names, "raw protocol missing")
    return run


def finite(value, shape=None):
    try:
        a = np.asarray(value)
        return a.dtype.kind in "iuf" and bool(np.isfinite(a).all()) and (shape is None or a.shape == shape)
    except (TypeError, ValueError):
        return False


def close(a, b, message, *, atol=1e-13, rtol=0):
    require(finite(a) and finite(b) and np.shape(a) == np.shape(b)
            and np.allclose(a, b, rtol=rtol, atol=atol), message)


def validate_plot_data(receipt, plan, raw_points, *, allow_partial=False):
    """Bind every displayed number/location to the independently audited rows.

    Scientific jet/envelope recomputation belongs to the bound finisher. This
    validates plotting relationships and replays finite stencil values from
    saved point records, without importing the worker or any model code.
    """
    require(receipt.get("id") == "BSC6-mixed-junction-offline-audit" and receipt.get("audit_finished") is True,
            "finished independent BSC6 audit required")
    require(type(receipt.get("complete")) is bool and type(receipt.get("execution_finished")) is bool,
            "missing scientific completion flags")
    require(receipt["complete"] or allow_partial, "partial pilot requires explicit --allow-partial")
    cfg = plan["mixed_junction"]; target = cfg["target"]
    require(plan["id"] == "BSC6-mixed-junction-probe" and plan["period"] == 18
            and cfg["beta_offsets"] == [0., -3e-9, 3e-9] and cfg["stencil_steps"] == [.0003, .0001]
            and cfg["primary_step"] == .0003, "frozen BSC6 design differs")
    require(finite(target["coordinate"], (2,)) and finite(target["beta"], ())
            and finite(plan["start"], (2,)) and finite(plan["coordinate_scales"], (2,)), "invalid coordinate metadata")
    points = {}
    for p in receipt["qualified_points"]:
        index = p["index"]
        require(type(index) is int and index >= 0 and index not in points and p.get("qualified") is True
                and p.get("closure_recomputed_from_saved_endpoint") is True, "invalid qualified point")
        require(index in raw_points and raw_points[index].get("qualified") is True, "raw qualified point absent")
        raw = raw_points[index]
        for key in ("index", "coordinate", "beta", "tau", "delta", "radius", "alpha", "log10_temperature"):
            require(raw[key] == p[key], "raw point/summary differs: "+key)
        require(finite(p["coordinate"], (2,)) and finite([p[k] for k in ("beta", "tau", "delta", "radius")])
                and p["delta"] < 0 and 0 < p["radius"] < 1, "invalid attracting opposite-sign point")
        physical = np.asarray(plan["start"])+np.asarray(plan["coordinate_scales"])*p["coordinate"]
        close([p["alpha"], p["log10_temperature"]], physical, "physical coordinate differs", atol=1e-15)
        phases = raw["phases"]
        require(p["phases"] in ([0], [0, 6, 12]) and [v["phase"] for v in phases] == p["phases"]
                and all(type(v) is int for v in p["phases"])
                and finite([v["tau"] for v in phases]), "invalid measured phases")
        require(phases[0]["tau"] == p["tau"], "phase-zero trace differs")
        points[index] = p
    require(set(raw_points) == set(points), "qualified raw point inventory differs")
    admitted, seen_admissions = set(), set()
    for row in receipt["point_admissions"]:
        index = row["point_index"]
        require(type(index) is int and index >= 0 and index not in seen_admissions, "invalid state-admission index")
        require(row["status"] in ("started", "passed", "failed", "failed orbit alignment"), "unknown state-admission status")
        seen_admissions.add(index)
        if row["status"] == "passed":
            require(index in points and row["passed"] is True, "state admission lacks qualified point")
            admitted.add(index)
    pending = receipt["unqualified_or_incomplete_points"]
    require(isinstance(pending, list) and type(receipt["attempted_points"]) is int
            and receipt["attempted_points"] == len(points)+len(pending), "incomplete point inventory differs")
    directories = {f"point-{i:04d}" for i in points}
    for p in pending:
        require(isinstance(p.get("directory"), str) and re.fullmatch(r"point-\d{4,}", p["directory"])
                and p["directory"] not in directories and isinstance(p.get("status"), str), "invalid pending point inventory")
        directories.add(p["directory"])
    require(directories == {f"point-{i:04d}" for i in range(receipt["attempted_points"])}, "point inventory has gaps")

    def point(index, center, beta):
        require(type(index) is int and index in admitted, "displayed target lacks admitted point")
        p = points[index]
        close(p["coordinate"], center, "target coordinate differs")
        require(p["beta"] == beta, "target beta differs")
        return p

    cases = receipt["cases"]
    require(isinstance(cases, list) and len(cases) <= 3, "too many target cases")
    rows = []
    for case_index, offset in enumerate(cfg["beta_offsets"]):
        beta = float(target["beta"]+offset)
        row = {"beta_offset": offset, "beta": beta, "case_index": case_index,
               "status": "not attempted", "admitted": False, "iterations": [], "feature": None,
               "feature_error": None, "feature_status": "unmeasured", "actual_traces": [], "sample_indices": []}
        rows.append(row)
        if case_index >= len(cases):
            continue
        case = cases[case_index]
        require(case["beta_offset"] == offset and case["beta"] == beta and type(case["complete"]) is bool,
                "case order, beta or completion differs")
        require(case["status"] in ("admitted", "running", "unresolved", "unresolved precision", "correction budget exhausted"),
                "unknown case status")
        require(case["complete"] or case_index == len(cases)-1, "unadmitted case has later targets")
        require(case_index == 0 or cases[0]["complete"], "outward case lacks admitted central case")
        base = target["coordinate"] if case_index == 0 else cases[0]["final_center"]
        predictor = np.asarray(base)+offset*np.asarray(target["moving_center_beta"])
        close(case["predicted_center"], predictor, "case predictor differs")
        row.update(status=case["status"], predictor=case["predicted_center"], admitted=case["complete"])
        require(len(case["iterations"]) <= cfg["maximum_corrections"]+1, "iteration budget differs")
        for number, iteration in enumerate(case["iterations"]):
            require(iteration["beta"] == beta and iteration["corrections_already_applied"] == number
                    and finite(iteration["center"], (2,)), "invalid iteration coordinate")
            display = {"number": number, "center": iteration["center"], "center_point_index": None}
            row["iterations"].append(display)
            for prefix, step in zip(("coarse", "fine"), cfg["stencil_steps"]):
                if prefix+"_jet" not in iteration:
                    continue
                jet = iteration[prefix+"_jet"]
                require(jet["step"] == step and jet["beta"] == beta and jet["center"] == iteration["center"], "matched jet placement differs")
                indices = np.asarray(jet["point_indices"])
                require(indices.shape == (3, 3) and indices.dtype.kind in "iu", "nine integer point IDs required")
                for iy, dy in enumerate((-1, 0, 1)):
                    for ix, dx in enumerate((-1, 0, 1)):
                        p = point(int(indices[iy, ix]), np.asarray(jet["center"])+step*np.array([dx, dy]), beta)
                        if p["index"] not in row["sample_indices"]:
                            row["sample_indices"].append(p["index"])
                center_point = points[int(indices[1, 1])]
                for field in ("tau", "delta"):
                    require(jet["fields"][field]["value"] == center_point[field], "jet center value differs")
                if prefix == "coarse":
                    display["center_point_index"] = center_point["index"]
                    display["tau"] = center_point["tau"]
            if "coarse_correction" in iteration:
                require("coarse_jet" in iteration and "coarse_geometry" in iteration, "feature lacks measured smooth jet")
                g, correction = iteration["coarse_geometry"], iteration["coarse_correction"]
                require(g["input_tau"] == iteration["coarse_jet"]["fields"]["tau"]["value"]
                        and g["projection"]["input_tau"] == g["input_tau"]
                        and g["projection"]["enforced_tau"] == 0. and g["projection"]["spatial_displacement_applied"] is False,
                        "conditional projection altered measured trace or coordinate")
                require(finite(correction["equations"], (2,)) and correction["equations"][0] == g["input_tau"], "corrector actual trace differs")
                for side in SIDES:
                    feature = g["branches"][side]["valley"]
                    require(feature["defined"] is True and finite(feature["feature"], ()), "invalid conditional feature")
                close(correction["equations"][1], g["branches"][SIDES[0]]["valley"]["feature"], "positive feature differs", atol=1e-18)
                close(correction["f_minus"], g["branches"][SIDES[1]]["valley"]["feature"], "negative feature differs", atol=1e-18)
        if case["iterations"]:
            last = case["iterations"][-1]
            if "coarse_correction" in last:
                row["feature"] = last["coarse_correction"]["f_minus"]
                row["feature_status"] = "diagnostic only; admission incomplete"
            if "coarse_jet" in last:
                row["actual_traces"].append({"source": "primary center", "phase": 0, "tau": last["coarse_jet"]["fields"]["tau"]["value"]})
            if "cyclic_point_index" in last:
                p = point(last["cyclic_point_index"], last["center"], beta)
                require(p["phases"] == [0, 6, 12], "missing cyclic phases")
                row["actual_traces"].extend({"source": "cyclic replay", "phase": v["phase"], "tau": v["tau"]}
                                            for v in raw_points[p["index"]]["phases"])
            if "uncertainty" in last:
                u = last["uncertainty"]
                error = u["intersection"]["f_minus_total_error"]
                require(finite(error, ()) and error >= 0 and row["feature"] is not None, "invalid empirical envelope")
                close(error, u["branch_envelopes"][SIDES[1]]["feature_error"]+u["intersection"]["f_minus_location_transfer"],
                      "total error omits location transfer", atol=1e-18)
                row["feature_error"] = error
            if "admission" in last:
                a = last["admission"]; fm = a["f_minus"]
                require(type(a["passed"]) is bool and a["checks"] and all(type(v) is bool for v in a["checks"].values())
                        and a["passed"] == all(a["checks"].values()), "admission checks differ")
                require(fm["value"] == row["feature"] and fm["empirical_total_error"] == row["feature_error"], "displayed admission value differs")
                v, e = row["feature"], row["feature_error"]
                status = ("resolved_positive" if v > cfg["signal_multiple"]*e else "resolved_negative" if v < -cfg["signal_multiple"]*e else
                          "residual_qualified_near_zero" if abs(v) <= cfg["f_minus_tolerance"] and e <= cfg["maximum_f_minus_total_uncertainty"] else "unresolved_sign")
                require(fm["status"] == (status if a["passed"] else "not_admitted"), "feature sign/admission label differs")
                row["feature_status"] = fm["status"]
            if case["complete"]:
                require(case["status"] == "admitted" and last.get("admission", {}).get("passed") is True
                        and case["final_point_index"] == last["cyclic_point_index"], "completed case lacks final admission")
                close(case["final_center"], last["center"], "final center differs")
                for key in ("value", "empirical_total_error"):
                    close(case["f_minus"][key], last["admission"]["f_minus"][key], "final feature differs", atol=1e-17, rtol=1e-8)
                require(case["f_minus"]["status"] == last["admission"]["f_minus"]["status"], "final feature label differs")
                require(row["feature_error"] <= cfg["maximum_f_minus_total_uncertainty"]
                        and max(abs(v["tau"]) for v in row["actual_traces"]) <= cfg["trace_tolerance"], "admitted precision cap failed")
                row["final_center"] = case["final_center"]
        elif case["complete"]:
            require(False, "completed case lacks measurements")
    # A timeout can leave valid points before an entire stencil was saved.
    # Display that evidence too; unadmitted state samples get separate marks.
    for row in rows:
        row["sample_indices"] = sorted(i for i, p in points.items() if p["beta"] == row["beta"] and i in admitted)
        row["unadmitted_sample_indices"] = sorted(i for i, p in points.items() if p["beta"] == row["beta"] and i not in admitted)
        require(row["status"] != "not attempted" or not row["sample_indices"] and not row["unadmitted_sample_indices"],
                "unattempted case contains target measurements")
    if receipt["complete"]:
        require(receipt["execution_finished"] and len(cases) == 3 and all(c["complete"] for c in cases)
                and not pending and receipt.get("terminal_failure") is None
                and not receipt["untested_or_incomplete_stages"], "complete receipt hides incomplete evidence")
    if receipt["resolved_f_minus_sign_bracket"] is not None:
        require(len(cases) == 3 and all(c["complete"] for c in cases), "sign bracket lacks admitted endpoints")
        labels = {rows[1]["feature_status"], rows[2]["feature_status"]}
        require(receipt["resolved_f_minus_sign_bracket"] is (labels == {"resolved_positive", "resolved_negative"}), "sign bracket label differs")
    return {"rows": rows, "points": points, "pending": pending, "plan": plan,
            "untested_or_incomplete_stages": receipt["untested_or_incomplete_stages"], "terminal_failure": receipt.get("terminal_failure")}


def load_receipt(path, expected, *, allow_partial=False):
    path = Path(path).resolve()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
            and path.is_file() and digest(path) == expected, "receipt hash differs")
    receipt = read(path)
    require(receipt.get("id") == "BSC6-mixed-junction-offline-audit" and receipt.get("audit_finished") is True,
            "finished independent BSC6 audit required")
    require(receipt.get("complete") is True or allow_partial, "partial pilot requires explicit --allow-partial")
    audit_source = git_bytes(receipt["analysis_commit"], FINISHER)
    required = frozen_literal(audit_source, "SOURCES")
    execution_required = frozen_literal(audit_source, "EXECUTION_SOURCES")
    require(FINISHER in required and DRIVER in execution_required and "tools/contraction_valley_geometry.py" in execution_required,
            "frozen dependency manifest incomplete")
    source_bindings(receipt["analysis_commit"], receipt["analysis_source_sha256"], required)
    source_bindings(receipt["execution_source_commit"], receipt["execution_source_sha256"], execution_required)
    blob = git_bytes(receipt["execution_source_commit"], receipt["plan_path"])
    require(hashlib.sha256(blob).hexdigest() == receipt["plan_sha256"], "plan hash differs")
    plan = json.loads(blob)
    require(receipt["scientific_metadata"] == {key: plan[key] for key in METADATA}, "scientific metadata differs")
    lifecycle = receipt["lifecycle"]; controller = lifecycle.get("controller")
    require(lifecycle.get("python_finalization") is True or isinstance(controller, dict)
            and controller.get("worker_reaped") is True, "audit lacks terminal execution evidence")
    if controller is not None:
        require(controller.get("worker_reaped") is True, "controller worker not reaped")
    if receipt["complete"]:
        require(lifecycle.get("python_complete") is True and lifecycle.get("python_finalization") is True
                and (controller is None or controller.get("execution_complete") is True and controller.get("timed_out") is False),
                "completed audit has incomplete lifecycle")
    run = verify_raw_manifest(receipt); protocol = read(run/"protocol.json")
    require(protocol["commit"] == receipt["execution_source_commit"] and protocol["source_sha256"] == receipt["execution_source_sha256"]
            and protocol["plan_sha256"] == receipt["plan_sha256"] and protocol["plan"] == plan
            and protocol["args"]["plan"] == receipt["plan_path"] and Path(protocol["args"]["out"]).resolve() == run,
            "raw execution protocol differs from audit")
    raw = {p["index"]: read(run/f"point-{p['index']:04d}"/"point.json") for p in receipt["qualified_points"]}
    return receipt, validate_plot_data(receipt, plan, raw, allow_partial=allow_partial)


def render(receipt, data, out, *, synthetic=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "savefig.facecolor": "white", "pdf.fonttype": 42})
    out = Path(out); cfg = data["plan"]["mixed_junction"]; rows = data["rows"]
    title = "Memory feature-intersection pilot"+(" — SYNTHETIC QA" if synthetic else "")
    status = "All three cases admitted" if receipt["complete"] else "PARTIAL / UNRESOLVED — available audited measurements only"
    def style(ax):
        ax.grid(alpha=.13); ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{name}.{ext}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.8))
    axes[0].set_title("A  Conditional Hessian feature f₋\nNegative-side limit at last attempted center", loc="left")
    axes[1].set_title("B  Actual signed trace residuals\nPrimary center and fresh cyclic replay, when available", loc="left")
    phase_markers = {0: "o", 6: "s", 12: "^"}
    for i, row in enumerate(rows):
        x = row["beta_offset"]*1e9; color = COLORS[i]
        if row["feature"] is not None:
            axes[0].errorbar(x, row["feature"]*1e7, yerr=None if row["feature_error"] is None else row["feature_error"]*1e7,
                            fmt="o" if row["admitted"] else "D", color=color, mfc=color if row["admitted"] else "white", capsize=5, ms=7, ls="none")
        else:
            axes[0].text(x, .08, "No feature\nestimate", transform=axes[0].get_xaxis_transform(), ha="center", fontsize=8, color=".45")
        for j, trace in enumerate(row["actual_traces"]):
            axes[1].scatter(x+(j-1.5)*.11, trace["tau"]*1e12, s=35, marker="x" if trace["source"] == "primary center" else phase_markers[trace["phase"]], color=color)
        if not row["actual_traces"]:
            axes[1].text(x, .08, "No center\ntrace", transform=axes[1].get_xaxis_transform(), ha="center", fontsize=8, color=".45")
    for ax in axes:
        style(ax); ax.axhline(0, color=".55", lw=.7); ax.set_xlim(-4.4, 4.4)
        ax.set_xticks([-3, 0, 3]); ax.set_xlabel("Memory offset β − βtarget (×10⁻⁹)")
    axes[0].set_ylabel("Conditional f₋ (×10⁻⁷)")
    if not any(row["feature"] is not None for row in rows):
        axes[0].set_yticks([])
    axes[1].set_ylabel("Measured τ (×10⁻¹²)")
    axes[1].axhspan(-cfg["trace_tolerance"]*1e12, cfg["trace_tolerance"]*1e12, color=".75", alpha=.16)
    axes[1].legend(handles=[Line2D([], [], marker=m, color=".4", ls="none", label=label)
        for m, label in (("x", "Primary"), ("o", "Phase 0"), ("s", "Phase 6"), ("^", "Phase 12"))], ncol=2, fontsize=8, frameon=False)
    fig.suptitle(title, x=.07, ha="left", fontsize=17)
    fig.text(.07, .915, f"{status}   |   βtarget = {cfg['target']['beta']:.15f}", fontsize=9, color=".35")
    for i, row in enumerate(sorted(rows, key=lambda r: r["beta_offset"])):
        label = ("small nominal residual" if row["feature_status"] == "residual_qualified_near_zero"
                 else row["feature_status"].replace("_", " "))
        fig.text(.07+i*.31, .245, f"β offset {row['beta_offset']*1e9:+g} ×10⁻⁹\n{row['status']}\n{label}", fontsize=8.5, va="top")
    fig.text(.07, .12, "Bars: empirical feature + location-transfer envelope; filled dots: admitted cases; open diamonds: unadmitted diagnostics.\n"
             "Trace panel: horizontal jitter separates phases only; shaded band is the frozen numerical trace gate.\n"
             "Hessian features need not define geometric-normal valleys; a small residual and these envelopes do not certify a zero.\n"
             "Points are not joined; ridge connections remain untested.", fontsize=8.8)
    fig.text(.07, .035, f"Retained: {len(data['pending'])} unqualified/incomplete point attempts; {len(data['untested_or_incomplete_stages'])} incomplete stages. "
             +("Terminal failure retained in figure receipt." if data["terminal_failure"] else "No terminal failure recorded."), fontsize=8, color=".4")
    fig.subplots_adjust(left=.075, right=.98, top=.80, bottom=.35, wspace=.30)
    save(fig, "mixed-junction-summary")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6.5))
    reference = np.asarray(cfg["target"]["coordinate"])
    all_xy = []
    for row in rows:
        if "predictor" in row:
            all_xy.append((np.asarray(row["predictor"])-reference)*1e4)
        all_xy.extend((np.asarray(data["points"][i]["coordinate"])-reference)*1e4
                      for i in row["sample_indices"]+row["unadmitted_sample_indices"])
        all_xy.extend((np.asarray(p["coordinate"])-reference)*1e4 for p in data["pending"]
                      if p.get("beta") == row["beta"] and finite(p.get("coordinate"), (2,)))
    extent = max(4., max((float(np.max(abs(q))) for q in all_xy), default=0.)*1.2)
    for i, (ax, row) in enumerate(zip(axes, rows)):
        color = COLORS[i]; style(ax)
        ax.set_title(f"β offset {row['beta_offset']*1e9:+g} ×10⁻⁹\n{row['status']}", loc="left", color=color)
        sample_ids = row["sample_indices"]
        if sample_ids:
            xy = (np.asarray([data["points"][j]["coordinate"] for j in sample_ids])-reference)*1e4
            ax.scatter(*xy.T, s=16, color=color, alpha=.35, linewidth=0)
        if row["unadmitted_sample_indices"]:
            xy = (np.asarray([data["points"][j]["coordinate"] for j in row["unadmitted_sample_indices"]])-reference)*1e4
            ax.scatter(*xy.T, marker="+", s=65, color="#b23c3c")
        if "predictor" in row:
            q = (np.asarray(row["predictor"])-reference)*1e4
            ax.scatter(*q, marker="x", color=".35", s=60, zorder=4)
        for iteration in row["iterations"]:
            if iteration["center_point_index"] is not None:
                q = (np.asarray(iteration["center"])-reference)*1e4
                ax.scatter(*q, marker="o", facecolor="white", edgecolor=color, s=70, zorder=5)
                ax.annotate(str(iteration["number"]), q, xytext=(6, 6), textcoords="offset points", color=color, fontsize=8)
        if row["admitted"]:
            q = (np.asarray(row["final_center"])-reference)*1e4
            ax.scatter(*q, marker="*", facecolor=color, edgecolor="black", linewidth=.6, s=120, zorder=6)
        for pending in data["pending"]:
            if pending.get("beta") == row["beta"] and finite(pending.get("coordinate"), (2,)):
                q = (np.asarray(pending["coordinate"])-reference)*1e4
                ax.scatter(*q, marker="+", color="#b23c3c", s=75, zorder=5)
        if row["status"] == "not attempted":
            ax.text(.5, .5, "Not attempted\nNo inferred location", ha="center", va="center", transform=ax.transAxes, color=".45")
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="Steering displacement (scaled units ×10⁻⁴)")
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel("Log-temperature displacement (scaled units ×10⁻⁴)")
    fig.suptitle(title+" — sampled locations", x=.055, ha="left", fontsize=17)
    fig.text(.055, .90, status+". Shared axes relative to the frozen conditional target; scaled Euclidean metric.", fontsize=9, color=".35")
    fig.legend(handles=[Line2D([], [], marker=m, ls="none", color=c, label=label, markerfacecolor=face)
        for m, c, face, label in (("o", ".5", ".5", "Measured stencil points"), ("x", ".35", ".35", "Conditional initial predictor"),
                                 ("o", ".3", "white", "Measured centers (iteration labels)"), ("*", ".3", ".3", "Admitted final center"),
                                 ("+", "#b23c3c", "#b23c3c", "Unadmitted/incomplete sample"))],
        loc="lower center", bbox_to_anchor=(.5, .12), ncol=3, frameon=False, fontsize=8.5)
    fig.text(.055, .04, "Only sampled locations are drawn; no lines, contours, projected points, or inferred curve connections.\n"
             "Predictors and samples without state admission are marked separately. Missing coordinates remain in the receipt inventory.", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.79, bottom=.26, wspace=.27)
    save(fig, "mixed-junction-locations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True); parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sha = digest(ROOT/SOURCE)
    require(hashlib.sha256(git_bytes(commit, SOURCE)).hexdigest() == sha, "commit plotter source before scientific rendering")
    receipt, data = load_receipt(args.receipt, args.receipt_sha256, allow_partial=args.allow_partial)
    out = args.out.resolve(); raw = Path(receipt["raw_directory"]).resolve()
    require(not out.exists() and not out.is_relative_to(ROOT.resolve()) and not out.is_relative_to(raw)
            and not raw.is_relative_to(out), "new output outside repository and raw run required")
    out.mkdir(parents=True); render(receipt, data, out)
    result = {"id": "BSC6-mixed-junction-figures", "receipt": str(args.receipt.resolve()), "receipt_sha256": args.receipt_sha256,
              "source": SOURCE, "source_commit": commit, "source_sha256": sha, "threads": 1,
              "scientific_complete": receipt["complete"], "partial_explicitly_allowed": args.allow_partial,
              "displayed_cases": data["rows"], "unqualified_or_incomplete_points": data["pending"],
              "point_admissions": receipt["point_admissions"],
              "untested_or_incomplete_stages": data["untested_or_incomplete_stages"], "terminal_failure": data["terminal_failure"],
              "scope": SCOPE, "visual_review": "required; successful rendering is not visual approval",
              "figures": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out/"receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()
