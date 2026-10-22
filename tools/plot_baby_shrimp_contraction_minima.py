"""Plot the independently audited BSC5 section-minima pilot, without model calls.

The optional trace-zero contour is a conditional local Taylor prediction. Actual
section observations are dots; independently searched minima are never joined.
Every receipt, frozen source, plan and raw-manifest hash is checked before use.
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
SOURCE = "tools/plot_baby_shrimp_contraction_minima.py"
FINISHER = "tools/baby_shrimp_contraction_minima_finish.py"
DRIVER = "tools/baby_shrimp_contraction_minima.py"
METADATA = ("id", "model_revision", "period", "start", "coordinate_scales",
            "anchor_directory", "anchor_hashes", "intervention")
COLORS = ("#206b9e", "#c6602d", "#43805b", "#8958a1")
STATUSES = {"converged", "no_bracket", "low_signal", "nonconverged",
            "budget_exhausted", "evaluation_failed", "interrupted before search returned"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_path(path):
    require(isinstance(path, str) and path and not Path(path).is_absolute()
            and ".." not in Path(path).parts and Path(path).as_posix() == path,
            "invalid relative artifact path")
    return path


def git_bytes(commit, path):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "exact Git commit required")
    relative_path(path)
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def frozen_literal(source, name):
    tree = ast.parse(source)
    nodes = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(nodes) == 1, "missing frozen dependency declaration: "+name)
    value = ast.literal_eval(nodes[0])
    require(isinstance(value, (list, set, tuple)) and value and all(isinstance(p, str) for p in value),
            "invalid frozen dependency declaration: "+name)
    return set(value)


def source_bindings(commit, sources, required):
    require(isinstance(sources, dict) and required <= set(sources), "source manifest incomplete")
    for path, sha in sources.items():
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
                and hashlib.sha256(git_bytes(commit, path)).hexdigest() == sha,
                "source hash differs: "+path)


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
        names.add(name)
        path = run/name
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0
                and isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                and path.is_file() and path.stat().st_size == entry["bytes"]
                and digest(path) == entry["sha256"], "raw artifact hash or size differs: "+name)
    require(names == {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()},
            "unmanifested or missing raw files")
    require("protocol.json" in names, "raw protocol missing")
    return run


def finite(value, shape=None):
    try:
        array = np.asarray(value, dtype=float)
        return bool(np.isfinite(array).all() and (shape is None or array.shape == shape))
    except (TypeError, ValueError):
        return False


def close(a, b, message, *, atol=1e-13):
    require(finite(a) and finite(b) and np.asarray(a).shape == np.asarray(b).shape
            and np.allclose(a, b, rtol=0, atol=atol), message)


def null_arms(hessian):
    """Predict directions from the smooth trace jet, never from radius fitting."""
    H = np.asarray(hessian, float)
    require(finite(H, (2, 2)) and np.allclose(H, H.T, rtol=0, atol=1e-13), "invalid trace Hessian")
    values, Q = np.linalg.eigh(H)
    require(values[0] < 0 < values[1], "trace jet is not a saddle")
    for i in range(2):
        if Q[np.argmax(abs(Q[:, i])), i] < 0:
            Q[:, i] *= -1
    arms = []
    for family, sign in enumerate((1., -1.)):
        tangent = Q @ np.array([np.sqrt(values[1]), sign*np.sqrt(-values[0])])
        tangent /= np.linalg.norm(tangent)
        for ray in (1., -1.):
            t = ray*tangent
            arms.append({"arm_id": f"line-{family}-ray-{int(ray):+d}", "tangent": t,
                         "normal": np.array([-t[1], t[0]])})
    return arms


def validate_plot_data(receipt, plan, *, allow_partial=False):
    """Cross-link plotted values, section geometry and all finite-offset minima."""
    require(receipt.get("id") == "BSC5-contraction-minima-offline-audit" and receipt.get("audit_finished") is True
            and type(receipt.get("complete")) is bool and type(receipt.get("execution_finished")) is bool,
            "independent BSC5 audit required")
    require(receipt["complete"] or allow_partial, "partial pilot requires explicit --allow-partial")
    require(plan["id"] == "BSC5-contraction-minima" and plan["period"] == 18
            and finite(plan["start"], (2,)) and finite(plan["coordinate_scales"], (2,))
            and min(plan["coordinate_scales"]) > 0, "invalid scientific coordinate plan")
    cfg = plan["contraction_minima"]
    mesh = cfg["section_mesh"]
    require(finite(mesh) and len(mesh) >= 3 and all(a < b for a, b in zip(mesh, mesh[1:])), "invalid section mesh")
    points = {}
    for p in receipt["qualified_points"]:
        require(type(p["index"]) is int and p["index"] >= 0 and p["index"] not in points
                and p.get("qualified") is True and finite(p["coordinate"], (2,))
                and finite([p["beta"], p["tau"], p["delta"], p["radius"]])
                and p["delta"] < 0 and 0 < p["radius"] < 1, "invalid or duplicate qualified point")
        physical = np.asarray(plan["start"])+np.asarray(plan["coordinate_scales"])*p["coordinate"]
        close([p["alpha"], p["log10_temperature"]], physical, "physical point coordinate differs", atol=1e-15)
        points[p["index"]] = p

    def point(index, coordinate, beta):
        require(type(index) is int and index in points, "unknown plotted point")
        p = points[index]
        close(p["coordinate"], coordinate, "plotted point coordinate differs")
        require(p["beta"] == beta, "plotted point beta differs")
        return p

    def values(row, p):
        require(all(row[k] == p[k] for k in ("tau", "delta", "radius")), "plotted measured value differs")

    sections = receipt["sections"]
    require(type(receipt.get("center_qualified")) is bool, "missing center admission status")
    if not receipt["center_qualified"]:
        require(not receipt["complete"] and not sections, "sections lack admitted center")
        return {"points": points, "sections": [], "arms": [], "center": None, "jet": None, "plan": plan}
    require(receipt["local_jets"] and receipt["center_point_index"] in points, "admitted center lacks measured jet")
    jet = receipt["local_jets"][-1]
    center, beta, step = np.asarray(jet["center"], float), jet["beta"], jet["step"]
    require(finite(center, (2,)) and finite(step) and step > 0, "invalid local jet center or step")
    central = point(receipt["center_point_index"], center, beta)
    require(central["phases"] == [0, 6, 12] and abs(central["tau"]) <= cfg["height_tolerance"],
            "center lacks cyclic admission")
    indices = jet["point_indices"]
    require(np.asarray(indices).shape == (3, 3), "local jet lacks nine measured points")
    t = np.array([[point(indices[iy][ix], center+step*np.array([dx, dy]), beta)["tau"]
                   for ix, dx in enumerate((-1, 0, 1))] for iy, dy in enumerate((-1, 0, 1))])
    g = np.array([(t[1, 2]-t[1, 0])/(2*step), (t[2, 1]-t[0, 1])/(2*step)])
    mixed = (t[2, 2]-t[0, 2]-t[2, 0]+t[0, 0])/(4*step*step)
    H = np.array([[(t[1, 2]-2*t[1, 1]+t[1, 0])/step**2, mixed],
                  [mixed, (t[2, 1]-2*t[1, 1]+t[0, 1])/step**2]])
    field = jet["fields"]["tau"]
    close(field["value"], t[1, 1], "local trace value differs", atol=1e-18)
    close(field["gradient"], g, "local trace gradient differs")
    close(field["hessian"], H, "local trace Hessian differs")
    arms = null_arms(H)
    floor = receipt["radius_floor"]
    require(finite(floor) and floor == max(cfg["radius_floor_minimum"], receipt["expanded_spectrum"]["variations"]["radius"])
            and floor <= 1e-11, "empirical radius floor differs")
    expected = [(distance, arm) for distance in cfg["longitudinal_steps"] for arm in arms]
    require(len(sections) <= len(expected), "extra section measurements")
    if receipt["complete"]:
        require(receipt["execution_finished"] and len(sections) == len(expected)
                and receipt.get("terminal_failure") is None, "complete pilot has missing scientific stages")
    checked = []
    for section, (distance, arm) in zip(sections, expected):
        require(section["arm_id"] == arm["arm_id"] and section["longitudinal"] == distance
                and section["beta"] == beta, "section identity/order/beta differs")
        for name in ("tangent", "normal"):
            close(section[name], arm[name], "section "+name+" differs")
        close(section["center"], center, "section center differs")
        origin = center+distance*arm["tangent"]
        close(section["origin"], origin, "section origin differs")
        observed = {}
        for row in section["predictor_checks"]:
            s = row["s"]
            require(finite(s) and mesh[0] <= s <= mesh[-1] and s not in observed, "invalid or duplicate profile coordinate")
            p = point(row["point_index"], origin+s*arm["normal"], beta)
            close(row["coordinate"], p["coordinate"], "profile point coordinate differs")
            values(row, p)
            alignment = row["orbit_alignment"]
            distances = alignment["all_alignment_maximum_relative_changes"]
            require(finite(distances, (18,)) and min(distances) >= 0
                    and alignment["reference_point_index"] == central["index"]
                    and type(alignment["best_right_roll"]) is int
                    and alignment["best_right_roll"] == int(np.argmin(distances)), "invalid orbit alignment")
            passed = alignment["best_right_roll"] == 0 and distances[0] < 1e-4
            require(row["orbit_alignment_passed"] is passed, "profile orbit admission differs")
            observed[s] = row
        attempts = section["callback_attempts"]
        require(len(attempts) <= cfg["section_maximum_evaluations"], "section callback budget exceeded")
        attempted, admitted = set(), {}
        for row in attempts:
            s = row["s"]
            require(finite(s) and mesh[0] <= s <= mesh[-1] and s not in attempted
                    and row["status"] in ("started", "failed", "complete"), "invalid section callback")
            attempted.add(s)
            if row["status"] == "complete":
                require(s in observed and observed[s]["orbit_alignment_passed"]
                        and observed[s]["point_index"] == row["index"] and observed[s]["radius"] == row["radius"],
                        "completed callback lacks admitted objective")
                admitted[s] = observed[s]
            else:
                require(row["radius"] is None and row["index"] is None, "unfinished callback invents objective")
        require(set(observed) <= attempted, "profile observation lacks callback")
        measured_mesh = []
        for s in mesh:
            if s not in admitted:
                break
            measured_mesh.append(admitted[s])
        brackets = [measured_mesh[i-1:i+2] for i in range(1, len(measured_mesh)-1)
                    if measured_mesh[i]["radius"] < min(measured_mesh[i-1]["radius"], measured_mesh[i+1]["radius"])]
        search = section["search"]
        require(search["status"] in STATUSES and type(search["complete"]) is bool
                and search["evaluation_count"] == len(attempts), "invalid section status or callback count")
        if receipt["complete"]:
            require(search["complete"], "completed pilot contains unresolved section")
        minima = search["minima"]
        if search["status"] == "interrupted before search returned":
            require(not search["complete"] and not minima, "interrupted search invents minima")
        else:
            settings = search["settings"]
            required_settings = {"mesh": mesh, "support": [mesh[0], mesh[-1]],
                "coordinate_tolerance": cfg["coordinate_tolerance"], "max_evaluations": cfg["section_maximum_evaluations"],
                "radius_floor": floor, "signal_multiple": cfg["signal_multiple"],
                "signal_threshold": floor*cfg["signal_multiple"], "validation_offsets": cfg["validation_offsets"]}
            require(all(settings.get(k) == v for k, v in required_settings.items()), "section settings differ from plan")
            counts = search["candidate_status_counts"]
            require(search["strict_mesh_brackets"] == len(brackets)
                    and all(k in STATUSES and type(v) is int and v > 0 for k, v in counts.items())
                    and sum(counts.values()) == len(brackets) and counts.get("converged", 0) == len(minima),
                    "competing section brackets or minima omitted")
            require(search["complete"] is (len(measured_mesh) == len(mesh)
                    and search["status"] in ("converged", "low_signal", "no_bracket")), "section completion differs")
            require(search["status"] != "no_bracket" or not brackets and not minima, "no-bracket section invents minimum")
        used_brackets = set()
        for minimum in minima:
            s = minimum["s"]
            require(s in admitted and minimum["point_index"] == admitted[s]["point_index"], "minimum is not an admitted sample")
            p = point(minimum["point_index"], origin+s*arm["normal"], beta)
            close(minimum["coordinate"], p["coordinate"], "minimum coordinate differs")
            values(minimum, p)
            require(minimum["beta"] == beta, "minimum beta differs")
            bracket = minimum["refined_bracket"]
            require(finite(bracket, (2,)) and bracket[0] < s < bracket[1]
                    and all(v in admitted for v in bracket)
                    and minimum["bracket_width"] == bracket[1]-bracket[0] <= cfg["coordinate_tolerance"],
                    "minimum lacks measured refinement width")
            matching = [i for i, b in enumerate(brackets) if minimum["mesh_bracket"] == [b[0]["s"], b[2]["s"]]
                        and b[0]["s"] < s < b[2]["s"]]
            require(len(matching) == 1 and matching[0] not in used_brackets, "minimum lacks unique strict mesh bracket")
            used_brackets.add(matching[0])
            mb = brackets[matching[0]]
            threshold = floor*cfg["signal_multiple"]
            require(min(mb[0]["radius"], mb[2]["radius"])-mb[1]["radius"] > threshold,
                    "minimum mesh contrast below empirical gate")
            validations = minimum["validation"]
            require([v["offset"] for v in validations] == cfg["validation_offsets"], "minimum missing two validation scales")
            for validation in validations:
                require(validation["status"] == "complete" and validation["resolved"] is True
                        and validation["lower_neighbor_resolved"] is False and len(validation["samples"]) == 2,
                        "minimum finite-offset gate failed")
                contrasts = []
                for sign, sample in zip((-1, 1), validation["samples"]):
                    ss = s+sign*validation["offset"]
                    require(sample["s"] == ss and ss in admitted and sample["index"] == admitted[ss]["point_index"]
                            and sample["radius"] == admitted[ss]["radius"], "minimum offset lacks actual admitted sample")
                    contrasts.append(sample["radius"]-p["radius"])
                require(validation["contrasts"] == contrasts and min(contrasts) > threshold,
                        "minimum finite-offset contrast differs")
        checked.append({"section": section, "admitted": admitted, "observed": observed, "brackets": brackets})
    return {"points": points, "sections": checked, "arms": arms, "center": central, "jet": jet, "plan": plan}


def load_receipt(path, expected, *, allow_partial=False):
    path = Path(path).resolve()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected)
            and path.is_file() and digest(path) == expected, "receipt hash differs")
    receipt = read(path)
    require(receipt.get("id") == "BSC5-contraction-minima-offline-audit" and receipt.get("audit_finished") is True,
            "finished independent BSC5 audit required")
    require(receipt.get("complete") is True or allow_partial, "partial pilot requires explicit --allow-partial")
    audit_source = git_bytes(receipt["analysis_commit"], FINISHER)
    audit_required = frozen_literal(audit_source, "SOURCES")
    execution_required = frozen_literal(audit_source, "EXECUTION_SOURCES")
    require(FINISHER in audit_required and DRIVER in execution_required
            and "tools/contraction_section_minima.py" in execution_required, "frozen dependency manifest incomplete")
    source_bindings(receipt["analysis_commit"], receipt["analysis_source_sha256"], audit_required)
    source_bindings(receipt["execution_source_commit"], receipt["execution_source_sha256"], execution_required)
    blob = git_bytes(receipt["execution_source_commit"], receipt["plan_path"])
    require(hashlib.sha256(blob).hexdigest() == receipt["plan_sha256"], "plan hash differs")
    plan = json.loads(blob)
    require(receipt["scientific_metadata"] == {key: plan[key] for key in METADATA}, "scientific metadata differs")
    lifecycle = receipt["lifecycle"]
    controller = lifecycle.get("controller")
    require(lifecycle.get("python_finalization") is True or isinstance(controller, dict)
            and controller.get("worker_reaped") is True, "audit lacks terminal execution evidence")
    if controller is not None:
        require(controller.get("worker_reaped") is True, "controller worker not reaped")
    if receipt["complete"]:
        require(lifecycle.get("python_complete") is True and lifecycle.get("python_finalization") is True
                and (controller is None or controller.get("execution_complete") is True
                     and controller.get("timed_out") is False), "completed audit has incomplete lifecycle")
    run = verify_raw_manifest(receipt)
    protocol = read(run/"protocol.json")
    require(protocol["commit"] == receipt["execution_source_commit"] and protocol["source_sha256"] == receipt["execution_source_sha256"]
            and protocol["plan_sha256"] == receipt["plan_sha256"] and protocol["plan"] == plan
            and protocol["args"]["plan"] == receipt["plan_path"] and Path(protocol["args"]["out"]).resolve() == run,
            "raw execution protocol differs from audit")
    return receipt, validate_plot_data(receipt, plan, allow_partial=allow_partial)


def render(receipt, data, out, *, synthetic=False):
    """Produce overview and full profile atlas, with no lines joining minima."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    out = Path(out)
    cfg = data["plan"]["contraction_minima"]
    status = "All planned section searches audited" if receipt["complete"] else "PARTIAL / UNRESOLVED — only audited available observations"
    heading = "Section-minima pilot"+(" — SYNTHETIC QA" if synthetic else "")
    color_of = {a["arm_id"]: COLORS[i] for i, a in enumerate(data["arms"])}
    def style(ax):
        ax.grid(alpha=.13)
        ax.spines[["top", "right"]].set_visible(False)
    def save(fig, stem):
        for ext in ("png", "pdf"):
            fig.savefig(out/f"{stem}.{ext}", dpi=190)
        plt.close(fig)
    def empty(ax, text):
        ax.text(.5, .5, text, ha="center", va="center", color=".4", transform=ax.transAxes)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.8))
    left, right = axes
    left.set_title("A  Local model prediction\nTrace-zero contour of the conditional quadratic jet", loc="left")
    right.set_title("B  Measured section geometry\nSeparate searches; minimum locations are not joined", loc="left")
    if data["center"] is not None:
        center = np.asarray(data["center"]["coordinate"])
        extent = 1.17*(max(cfg["longitudinal_steps"])+max(abs(s) for s in cfg["section_mesh"]))
        x = np.linspace(-extent, extent, 181)
        X, Y = np.meshgrid(x, x)
        field = data["jet"]["fields"]["tau"]
        H, g = np.asarray(field["hessian"]), np.asarray(field["gradient"])
        Z = field["value"]+g[0]*X+g[1]*Y+.5*(H[0, 0]*X*X+2*H[0, 1]*X*Y+H[1, 1]*Y*Y)
        if Z.min() < 0 < Z.max():
            left.contour(X*1e3, Y*1e3, Z, levels=[0], colors=["#abb8c4"], linestyles="--", linewidths=1.5)
        for i, arm in enumerate(data["arms"]):
            for distance in cfg["longitudinal_steps"]:
                origin = distance*arm["tangent"]
                endpoints = origin+np.array(cfg["section_mesh"])[[0, -1], None]*arm["normal"]
                left.plot(*(endpoints*1e3).T, color=COLORS[i], lw=1.2, alpha=.8)
            label_at = (max(cfg["longitudinal_steps"])+.00025)*arm["tangent"]*1e3
            left.text(*label_at, f"{i+1}", color=COLORS[i], ha="center", va="center", fontsize=11, weight="bold")
        for row in data["sections"]:
            section = row["section"]; color = color_of[section["arm_id"]]
            observed = list(row["admitted"].values())
            if observed:
                q = (np.array([p["coordinate"] for p in observed])-center)*1e3
                right.scatter(*q.T, s=13, color=color, alpha=.55, linewidths=0)
            rejected = [p for p in row["observed"].values() if not p["orbit_alignment_passed"]]
            if rejected:
                q = (np.array([p["coordinate"] for p in rejected])-center)*1e3
                right.scatter(*q.T, s=35, marker="x", color="#b23c3c")
            for minimum in section["search"]["minima"]:
                q = (np.array(minimum["coordinate"])-center)*1e3
                ends = (np.asarray(section["origin"])+np.array(minimum["refined_bracket"])[:, None]*np.asarray(section["normal"])-center)*1e3
                right.plot(*ends.T, color="black", lw=2.2, zorder=4)
                right.scatter(*q, marker="*", s=95, facecolor=color, edgecolor="black", linewidth=.55, zorder=5)
        for ax in axes:
            ax.scatter([0], [0], marker="+", color=".2", s=75, linewidth=1.3, zorder=6)
            ax.set(xlim=(-extent*1e3, extent*1e3), ylim=(-extent*1e3, extent*1e3))
        physical = np.array(data["plan"]["start"])+np.array(data["plan"]["coordinate_scales"])*center
        subtitle = f"{status}\nMemory β = {data['center']['beta']:.12g}; center α = {physical[0]:.12g}, log₁₀ T = {physical[1]:.12g}"
    else:
        for ax in axes:
            empty(ax, "Center admission not reached\nNo section measurements")
        subtitle = status
    for ax in axes:
        sx, sy = data["plan"]["coordinate_scales"]
        ax.set(xlabel=f"(α − αc) / {sx*1e-3:g}", ylabel=f"(log₁₀ T − log₁₀ Tc) / {sy*1e-3:g}")
        ax.set_aspect("equal", adjustable="box"); style(ax)
    if data["center"] is not None:
        left.legend(handles=[Line2D([], [], color="#abb8c4", ls="--", label="Conditional trace-jet zero"),
                             Line2D([], [], color=".35", lw=1, label="Planned transverse supports")],
                    loc="lower left", fontsize=8, frameon=False)
        right.legend(handles=[Line2D([], [], color=".45", marker="o", ls="none", ms=4, label="Admitted measured radius sample"),
                              Line2D([], [], color="black", marker="*", ls="-", ms=9, lw=2, label="Qualified minimum + refined bracket")],
                     loc="lower left", fontsize=8, frameon=False)
    else:
        for ax in axes:
            ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
    fig.suptitle(heading, x=.065, ha="left", fontsize=17)
    fig.text(.065, .91, subtitle, fontsize=9, color=".35")
    fig.text(.065, .05, "The trace jet predicts a frame; measured radius minima are searched without a trace-zero constraint.\n"
             "Black bars show the final sampled bracket width (often smaller than a marker), not statistical confidence intervals.\n"
             "Finite supported sections do not establish intrinsic ridges, between-section connections, or exact superstability.", fontsize=9)
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.19, wspace=.28)
    save(fig, "section-minima-geometry")

    distances = cfg["longitudinal_steps"]
    fig, axes = plt.subplots(len(distances), 4, figsize=(15, 3.25*len(distances)+2.4), squeeze=False)
    by_key = {(r["section"]["longitudinal"], r["section"]["arm_id"]): r for r in data["sections"]}
    arm_ids = [a["arm_id"] for a in data["arms"]] or [f"unadmitted-{i}" for i in range(4)]
    for iy, distance in enumerate(distances):
        for ix, arm_id in enumerate(arm_ids):
            ax = axes[iy, ix]; row = by_key.get((distance, arm_id)); color = COLORS[ix]
            ax.set_title(f"Arm {ix+1} · distance {distance:g}", loc="left", color=color)
            ax.set_xlim(np.array(cfg["section_mesh"])[[0, -1]]*1e4)
            ax.set_xlabel("Transverse offset s (×10⁻⁴)")
            if ix == 0:
                ax.set_ylabel("ρ − smallest sampled ρ (×10⁻⁹)")
            style(ax)
            if row is None:
                ax.set_yticks([])
                empty(ax, "Not attempted" if data["center"] else "Untested: center not admitted")
                continue
            section = row["section"]; search = section["search"]
            observed = list(row["admitted"].values())
            rejected = [p for p in row["observed"].values() if not p["orbit_alignment_passed"]]
            baseline = min((p["radius"] for p in observed), default=0.)
            for bracket in row["brackets"]:
                ax.axvspan(bracket[0]["s"]*1e4, bracket[2]["s"]*1e4, color=color, alpha=.065)
                p = bracket[1]
                ax.scatter(p["s"]*1e4, (p["radius"]-baseline)*1e9, s=65, facecolors="none", edgecolors=color, lw=1)
            if observed:
                ax.scatter([p["s"]*1e4 for p in observed], [(p["radius"]-baseline)*1e9 for p in observed],
                           s=15, color=color, linewidths=0, zorder=3)
            for p in rejected:
                if observed:
                    ax.scatter(p["s"]*1e4, (p["radius"]-baseline)*1e9, marker="x", color="#b23c3c", s=35)
            for attempt in section["callback_attempts"]:
                if attempt["status"] != "complete":
                    ax.plot(attempt["s"]*1e4, .025, "|", color="#b23c3c", ms=12, transform=ax.get_xaxis_transform())
            for minimum in search["minima"]:
                s = minimum["s"]; bracket = minimum["refined_bracket"]
                ax.errorbar(s*1e4, (minimum["radius"]-baseline)*1e9,
                            xerr=np.array([[s-bracket[0]], [bracket[1]-s]])*1e4,
                            fmt="*", ms=10, color="black", markerfacecolor=color, capsize=3, zorder=5)
            counts = search.get("candidate_status_counts", {})
            detail = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in counts.items())
            label = search["status"].replace("_", " ")
            if detail and detail != "1 "+label:
                label += "\n"+detail
            label += f"\n{len(observed)} admitted samples; {len(search['minima'])} qualified minima"
            if rejected:
                label += f"; {len(rejected)} alignment rejected"
            if not observed:
                ax.set_yticks([])
                empty(ax, "No admitted objective values")
            ax.text(.02, .97, label, transform=ax.transAxes, va="top", fontsize=7.7,
                    bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
            ax.margins(y=.27)
    fig.suptitle(heading+" — measured cross-sections", x=.055, ha="left", fontsize=17)
    fig.text(.055, .94, status+". Every planned arm/distance is displayed; dots are actual evaluations, with no connecting interpolation.", fontsize=10, color=".35")
    fig.text(.055, .035, "Open rings / pale spans: every strict mesh-minimum bracket. Stars / horizontal bars: qualified minima / final sampled bracket.\n"
             "Red crosses: alignment-rejected samples; red bottom ticks: failed or interrupted objective attempts (no inferred radius).\n"
             "Each panel subtracts its smallest admitted sampled radius for visibility; that reference is not a certified minimum.\n"
             "Two finite-offset checks and an empirical comparison floor resolve section candidates only; unseen minima and ridge connectivity remain untested.", fontsize=9)
    fig.subplots_adjust(left=.06, right=.985, top=.89, bottom=.14, hspace=.5, wspace=.3)
    save(fig, "section-minima-profiles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="Explicitly label a terminal audited incomplete pilot")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_sha = digest(ROOT/SOURCE)
    require(hashlib.sha256(git_bytes(commit, SOURCE)).hexdigest() == source_sha,
            "commit plotter source before rendering scientific evidence")
    receipt, data = load_receipt(args.receipt, args.receipt_sha256, allow_partial=args.allow_partial)
    out = args.out.resolve()
    require(not out.exists() and not out.is_relative_to(ROOT.resolve())
            and not out.is_relative_to(Path(receipt["raw_directory"]).resolve()), "new output outside repository and raw run required")
    out.mkdir(parents=True)
    render(receipt, data, out)
    result = {"id": "BSC5-section-minima-figures", "receipt": str(args.receipt.resolve()),
        "receipt_sha256": args.receipt_sha256, "scientific_complete": receipt["complete"],
        "partial_explicitly_allowed": args.allow_partial, "source": SOURCE, "source_commit": commit, "source_sha256": source_sha,
        "threads": 1, "figures": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
                                    for p in sorted(out.iterdir()) if p.is_file()],
        "visual_review": "required; not established by successful rendering",
        "scope": "Conditional local trace-jet reference versus actual finite-section measurements. No interpolation between minima or ridge topology claim."}
    (out/"receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()
