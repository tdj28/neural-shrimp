"""Plot only hash-pinned, independently audited BSC4 measurements; no model calls.

Receipt-only data access: Git supplies source/plan provenance, never raw arrays.
Line segments are guides between observations, not fitted contours or ridges.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
SOURCE = "tools/plot_baby_shrimp_memory_connection_probe.py"
FINISHER = "tools/baby_shrimp_memory_connection_probe_finish.py"
DRIVER = "tools/baby_shrimp_memory_connection_probe.py"
METADATA = ("id", "model_revision", "period", "start", "coordinate_scales",
            "anchor_directory", "anchor_hashes", "intervention")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def git_bytes(commit, path):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)
            and not Path(path).is_absolute() and ".." not in Path(path).parts, "invalid exact Git reference")
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def source_bindings(commit, sources, required):
    require(required <= set(sources), "source manifest incomplete")
    for path, sha in sources.items():
        require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
                and hashlib.sha256(git_bytes(commit, path)).hexdigest() == sha, "source hash differs: "+path)


def validate_plot_data(receipt, plan):
    """Cross-link every plotted observation to a qualified point and its geometry."""
    cfg = plan["connection_probe"]
    rows = receipt["qualified_points"]
    require(all(isinstance(p["index"], int) and not isinstance(p["index"], bool) for p in rows), "invalid point index")
    points = {p["index"]: p for p in rows}
    require(len(points) == len(rows), "duplicate qualified point index")
    for p in rows:
        require(p.get("qualified") is True and np.asarray(p["coordinate"]).shape == (2,)
                and np.isfinite([*p["coordinate"], p["beta"], p["tau"], p["delta"], p["radius"]]).all()
                and p["delta"] < 0 and 0 < p["radius"] < 1, "invalid qualified point")
    def point(index, beta, coordinate):
        require(isinstance(index, int) and not isinstance(index, bool) and index in points, "unknown plotted point")
        p = points[index]
        require(p["beta"] == beta and np.allclose(p["coordinate"], coordinate, rtol=0, atol=1e-13),
                "plotted point coordinate or beta differs")
        return p
    def unit(vector):
        vector = np.asarray(vector, float)
        require(vector.shape == (2,) and np.isfinite(vector).all()
                and abs(np.linalg.norm(vector)-1) < 1e-12, "invalid plotted direction")
        return vector
    plot = receipt["plot_data"]
    require(plot["coordinate_metric"] == "Euclidean in the plan's scaled coordinates", "coordinate metric differs")
    tilt = plot["candidate_tilt"]
    if tilt is not None:
        center = np.asarray(tilt["center"]); direction = unit(tilt["direction"])
        central = point(tilt["center_point_index"], tilt["beta"], center)
        summary = receipt["measurements"]["candidate_tilt"]
        require(summary["center_point_index"] == central["index"], "candidate center link differs")
        offsets = [s["offset"] for s in tilt["samples"]]
        require(offsets == cfg["candidate_tilt_offsets"][:len(offsets)], "candidate offsets differ")
        for s in tilt["samples"]:
            p = point(s["point_index"], tilt["beta"], center+s["offset"]*direction)
            require(abs(s["radius_difference"]-(p["radius"]-central["radius"])) <= 1e-18,
                    "plotted radius difference differs")
    cases = plot["axes"]
    require(len(cases) <= 2 and len(cases) == len(receipt["measurements"]["axes"]), "axis case count differs")
    for case, beta, summary in zip(cases, (cfg["before_beta"], cfg["after_beta"]), receipt["measurements"]["axes"]):
        require(case["beta"] == beta == summary["beta"] and len(case["axes"]) == 2, "axis beta/order differs")
        values = case["hessian_eigenvalues"]
        require(len(values) == 2 and np.isfinite(values).all() and values[0] < 0 < values[1], "invalid axis Hessian signs")
        directions = []
        for axis, label, key in zip(case["axes"], ("negative curvature", "positive curvature"),
                                    ("negative_axis_brackets", "positive_axis_brackets")):
            direction = unit(axis["direction"]); directions.append(direction)
            require(axis["label"] == label and [s["offset"] for s in axis["samples"]] == sorted([0., *cfg["axis_offsets"]]),
                    "axis sample label/order differs")
            sampled = [point(s["point_index"], beta, np.asarray(case["center"])+s["offset"]*direction) for s in axis["samples"]]
            require(sum(a["tau"]*b["tau"] < 0 for a, b in zip(sampled, sampled[1:])) == summary[key],
                    "axis sign bracket count differs")
        require(abs(np.dot(*directions)) < 1e-12, "axis directions not orthogonal")
    roots = plot["roots"]
    require(len(roots) == len(receipt["measurements"]["roots"]), "root count differs")
    if roots:
        H = np.asarray(receipt["measurements"]["matched_stencils"][-1]["hessian"])
        values, vectors = np.linalg.eigh(H)
        require(values[0] < 0 < values[1], "root lacks hyperbolic stencil")
        for column in range(2):
            if vectors[np.argmax(abs(vectors[:, column])), column] < 0:
                vectors[:, column] *= -1
        basis = vectors[:, [1, 0]]
        for root, summary in zip(roots, receipt["measurements"]["roots"]):
            require(np.allclose(root["slice_basis"], basis, rtol=0, atol=1e-13)
                    and root["beta"] == cfg["after_beta"], "root basis/beta differs")
            require(all(root[k] == summary.get(k) for k in ("along", "transverse", "side", "status",
                        "primary_point_index", "cyclic_point_index")), "root measurement link differs")
            location = np.asarray(root["center"])+basis@np.array([root["along"], root["transverse"]])
            primary = point(root["primary_point_index"], root["beta"], location)
            require(abs(primary["tau"]) < cfg["root_trace_tolerance"], "plotted root residual exceeds gate")
            if root["cyclic_point_index"] is not None:
                cyclic = point(root["cyclic_point_index"], root["beta"], location)
                require(cyclic.get("phases") == [0, 6, 12], "cyclic root label requires three checked phases")
                require(abs(cyclic["tau"]) < cfg["root_trace_tolerance"], "cyclic root residual exceeds gate")
    require(tilt is not None or cases or roots, "no audited observations to plot")
    return points


def load_receipt(path, expected):
    path = Path(path).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected) and path.is_file() and digest(path) == expected, "receipt hash differs")
    receipt = read(path)
    require(receipt["id"] == "BSC4-direct-connection-offline-audit" and isinstance(receipt["complete"], bool), "BSC4 audit required")
    require(not receipt["complete"] or receipt["measurements"]["all_measurement_stages_complete"],
            "completed receipt contains incomplete measurements")
    source_bindings(receipt["analysis_commit"], receipt["analysis_source_sha256"], {FINISHER,
        "tools/baby_shrimp_memory_finish.py", "tools/baby_shrimp_memory_orbit_alignment.py",
        "tools/baby_shrimp_memory_determinant_gradient.py"})
    # Obtain the frozen driver's literal dependency list without importing it.
    tree = ast.parse(git_bytes(receipt["execution_source_commit"], DRIVER))
    lists = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets)]
    require(len(lists) == 1 and DRIVER in lists[0], "frozen driver dependency declaration missing")
    source_bindings(receipt["execution_source_commit"], receipt["execution_source_sha256"], set(lists[0]))
    blob = git_bytes(receipt["execution_source_commit"], receipt["plan_path"])
    require(hashlib.sha256(blob).hexdigest() == receipt["plan_sha256"], "plan hash differs")
    plan = json.loads(blob)
    require(plan["id"] == "BSC4-direct-connection" and plan["period"] == 18
            and receipt["scientific_metadata"] == {key: plan[key] for key in METADATA}, "scientific metadata differs")
    return receipt, validate_plot_data(receipt, plan)


def render(receipt, points, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    plot = receipt["plot_data"]
    blue, orange = "#206b9e", "#c6602d"
    def empty(ax, text):
        ax.text(.5, .5, text, ha="center", va="center", transform=ax.transAxes, color=".45")
    ax = axes[0, 0]; tilt = plot["candidate_tilt"]
    ax.set_title("Candidate: measured contraction changes", loc="left")
    if tilt:
        samples = sorted([(0., 0.), *[(s["offset"], s["radius_difference"]) for s in tilt["samples"]]])
        x, y = np.array(samples).T
        ax.plot(x*1e4, y*1e9, "o--", color=blue, lw=1, ms=5, label="Observed points; dashed guide")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Candidate: measured contraction changes\nMemory = {100*tilt['beta']:.6f}%", loc="left")
    else:
        empty(ax, "Candidate tilt not yet tested")
    ax.set(xlabel="Displacement along −∇δ / ‖∇δ‖, scaled (×10⁻⁴)", ylabel="Pair spectral radius − center value (×10⁻⁹)")
    for ax, label, case in zip((axes[0, 1], axes[1, 0]), ("Before", "After"),
                               [*plot["axes"], *([None]*(2-len(plot["axes"]))) ]):
        ax.set_title(label+": measured trace along saddle axes", loc="left")
        if case:
            for axis, color in zip(case["axes"], (blue, orange)):
                ax.plot([s["offset"] for s in axis["samples"]],
                        [points[s["point_index"]]["tau"]*1e5 for s in axis["samples"]],
                        "o--", color=color, lw=1, ms=5, label=axis["label"])
            ax.legend(frameon=False, fontsize=8)
            ax.set_title(f"{label}: measured trace along saddle axes\nMemory = {100*case['beta']:.3f}%", loc="left")
        else:
            empty(ax, "Axis case not yet tested")
        ax.set(xlabel="Signed displacement along each scaled eigenaxis", ylabel="Leading-pair trace (×10⁻⁵)")
    heights = [points[s["point_index"]]["tau"]*1e5 for case in plot["axes"]
               for axis in case["axes"] for s in axis["samples"]]
    if heights:
        low, high = min(0., *heights), max(0., *heights)
        margin = max(.08*(high-low), 1e-8)
        for ax in (axes[0, 1], axes[1, 0]):
            ax.set_ylim(low-margin, high+margin)
    ax = axes[1, 1]
    ax.set_title("Higher memory: evaluated trace-zero points", loc="left")
    for cyclic, marker, label in ((False, "o", "Primary phase only"), (True, "s", "Three cyclic phases checked")):
        rows = [r for r in plot["roots"] if (r["cyclic_point_index"] is not None) == cyclic]
        if rows:
            ax.scatter([r["along"] for r in rows], [r["transverse"] for r in rows], marker=marker,
                       facecolor=orange if cyclic else "none", edgecolor=orange, s=65, label=label)
    if plot["roots"]:
        ax.scatter([0], [0], color=".35", marker="+", s=60, label="Continued center")
        ax.legend(frameon=False, fontsize=8)
    else:
        empty(ax, "Trace-zero searches not yet completed")
    ax.set(xlabel="Positive-curvature coordinate (fresh fine stencil)",
           ylabel="Negative-curvature coordinate (fresh fine stencil)")
    ax.set_aspect("equal", adjustable="datalim")
    for ax in axes.flat:
        ax.axhline(0, color=".55", lw=.6, zorder=0); ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    status = "Completed audit" if receipt["complete"] else "Partial attempt: only audited available observations shown"
    fig.suptitle("Direct probes of trace balance and contraction", x=.065, ha="left", fontsize=16)
    fig.text(.065, .91, status, color=".35", fontsize=10)
    fig.text(.065, .035, "Lower pair radius means stronger contraction. Dots are evaluated orbits; dashed segments only guide the eye.\n"
             "Root markers are numerical trace zeros of nonzero Ritz multipliers, with spectral isolation conditional on the computed subspace.\n"
             "Saved-data numerical audit: these finite samples establish neither exact superstability nor the connectivity of contraction ridges.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.17, hspace=.5, wspace=.3)
    for extension in ("png", "pdf"):
        fig.savefig(out/f"connection-probe.{extension}", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt, points = load_receipt(args.receipt, args.sha256)
    out = args.out.resolve()
    require(out.is_relative_to(DATA.resolve()) and not out.exists()
            and not out.is_relative_to(Path(receipt["raw_directory"]).resolve()), "fresh external output outside raw required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_sha = digest(ROOT/SOURCE)
    require(hashlib.sha256(git_bytes(commit, SOURCE)).hexdigest() == source_sha, "commit plotter before rendering")
    out.mkdir(parents=True)
    render(receipt, points, out)
    record = {"id": "BSC4-audited-connection-probe-figure", "receipt": str(args.receipt.resolve()),
              "receipt_sha256": args.sha256, "analysis_commit": commit, "source_sha256": {SOURCE: source_sha},
              "execution_source_commit": receipt["execution_source_commit"], "plan_sha256": receipt["plan_sha256"],
              "attempt_complete": receipt["complete"], "visual_review": "pending",
              "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(out.iterdir())]}
    (out/"receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"out": str(out), "receipt_sha256": digest(out/"receipt.json")}))


if __name__ == "__main__":
    main()
