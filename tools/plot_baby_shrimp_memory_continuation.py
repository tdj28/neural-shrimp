"""Plot accepted BSC3 continuation samples from explicitly hash-pinned audits.

Example: --receipt /external/audit/receipt.json SHA256 --receipt /external/next/receipt.json SHA256
No model, provider API, raw-data mutation, extrapolation, or new scientific audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
METADATA = ("id", "model_revision", "period", "start", "coordinate_scales", "anchor_directory", "anchor_hashes", "intervention")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_bytes(commit, name):
    require(re.fullmatch(r"[0-9a-f]{40}", commit) and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "invalid exact source reference")
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)


def bound_json(run, manifest, name):
    require(not Path(name).is_absolute() and ".." not in Path(name).parts, "invalid evidence path")
    matches = [row for row in manifest if row["path"] == name]
    path = run / name
    require(len(matches) == 1 and path.is_file() and not path.is_symlink(), "missing unique regular evidence: " + name)
    require(path.stat().st_size == matches[0]["bytes"] and digest(path) == matches[0]["sha256"], "evidence hash changed: " + name)
    return json.loads(path.read_text())


def load_receipt(path, expected):
    path = Path(path).resolve()
    require(re.fullmatch(r"[0-9a-f]{64}", expected) and digest(path) == expected, "receipt hash differs")
    receipt = json.loads(path.read_text())
    require(receipt["id"] == "BSC3-memory-offline-audit" and receipt["mode"] == "continue", "continuation audit required")
    analysis = git_bytes(receipt["analysis_commit"], "tools/baby_shrimp_memory_finish.py")
    require(hashlib.sha256(analysis).hexdigest() == receipt["analysis_source_sha256"], "audit analysis source differs")
    run, manifest = Path(receipt["raw_directory"]), receipt["manifest"]
    protocol = bound_json(run, manifest, "protocol.json")
    require(protocol["args"]["mode"] == "continue" and protocol["source_sha256"]
            and protocol["commit"] == receipt["execution_source_commit"], "execution source differs")
    plan = git_bytes(protocol["commit"], protocol["args"]["plan"])
    require(hashlib.sha256(plan).hexdigest() == receipt["plan_sha256"] == protocol["plan_sha256"]
            and json.loads(plan) == protocol["plan"], "committed plan binding differs")
    for source, sha in protocol["source_sha256"].items():
        require(hashlib.sha256(git_bytes(protocol["commit"], source)).hexdigest() == sha, "execution source hash differs")
    centers = receipt["accepted_centers"]
    require(centers, "receipt contains no accepted centers")
    for row in centers:
        require(row.get("converged") and row["kind"] == "continued saddle", "fully corrected accepted center required")
        require(np.asarray(row["center"]).shape == (2,) and np.isfinite([row["parameter"], row["tau"], *row["center"]]).all(), "invalid accepted center")
    root = receipt.get("height_root")
    roots = []
    if root and root.get("complete"):
        require(receipt["complete"] and root["result"]["complete"] and root["saddle_artifacts"], "incomplete numerical root")
        require(bound_json(run, manifest, "height-root.json") == root, "saved numerical root receipt differs")
        stored = bound_json(run, manifest, root["saddle_artifacts"][-1])
        require(root["result"]["saddle"] == {k: v for k, v in stored.items() if k not in ("stage", "cyclic_endpoint")}, "root source artifact differs")
        roots = [row for row in centers if all(row[k] == stored[k] for k in ("parameter", "center", "tau"))
                 and row["cyclic_point_index"] == stored["cyclic_endpoint"]["index"]]
        require(len(roots) == 1 and abs(roots[0]["tau"]) <= root["height_tolerance"], "root is not linked to a qualified accepted center")
    metadata = {key: protocol["plan"][key] for key in METADATA}
    return centers, roots, metadata, {"path": str(path), "sha256": expected, "execution_source_commit": protocol["commit"],
                                      "analysis_commit": receipt["analysis_commit"], "plan_sha256": receipt["plan_sha256"], "attempt_complete": receipt["complete"]}


def combine(inputs):
    points, roots, sources, common = {}, {}, [], None
    for path, sha in inputs:
        centers, highlighted, metadata, source = load_receipt(path, sha)
        require(common is None or common == metadata, "receipts use different scientific coordinates or intervention")
        common = metadata
        sources.append(source)
        for row in centers:
            beta = row["parameter"]
            if beta in points:
                require(all(points[beta][key] == row[key] for key in ("center", "tau", "delta", "radius")), "inconsistent duplicate beta")
            points[beta] = row
        roots.update({row["parameter"]: row for row in highlighted})
    require(points, "no accepted continuation samples")
    return [points[key] for key in sorted(points)], [roots[key] for key in sorted(roots)], sources, common


def render(points, roots, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    beta = 100 * np.array([p["parameter"] for p in points])
    tau, xy = np.array([p["tau"] for p in points]), np.array([p["center"] for p in points])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    fig.subplots_adjust(left=.09, right=.96, top=.79, bottom=.23, wspace=.32)
    fig.suptitle("Following the child's trace saddle", x=.09, ha="left", fontsize=23, fontweight="bold", y=.95)
    axes[0].axhline(0, color="#666666", lw=1, linestyle="--", label="Zero trace")
    axes[0].plot(beta, tau, "o-", color="#175d8d", lw=1.4, ms=5, label="Accepted centers")
    axes[0].set(xlabel="Feedback memory (%)", ylabel="Leading-pair trace at corrected saddle", title="Measured saddle height")
    axes[0].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3), useOffset=False)
    axes[1].plot(xy[:, 0], xy[:, 1], color="#b0b5ba", lw=1, zorder=1)
    scatter = axes[1].scatter(xy[:, 0], xy[:, 1], c=beta, cmap="viridis", s=42, edgecolors="white", linewidths=.5, zorder=2)
    axes[1].set(xlabel="Scaled steering displacement", ylabel="Scaled log-temperature displacement", title="The moving center")
    fig.colorbar(scatter, ax=axes[1], label="Memory (%)", fraction=.055, pad=.035)
    for index, row in enumerate(roots):
        style = dict(marker="*", color="#e68424", markeredgecolor="#713b08", markersize=13, linestyle="none", zorder=4)
        axes[0].plot(100 * row["parameter"], row["tau"], label="Audited numerical root" if index == 0 else None, **style)
        axes[1].plot(*row["center"], **style)
    for ax in axes:
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9, loc="best")
    fig.text(.09, .08, "Only audited accepted samples are shown; lines join samples and never extrapolate.\n"
             "Empirical numerical evidence, not an exact root, interval certificate, or proof of superstability.", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"memory-continuation-combined.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", nargs=2, action="append", required=True, metavar=("PATH", "SHA256"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    points, roots, sources, metadata = combine(args.receipt)
    out = args.out.resolve()
    require(out.is_relative_to(DATA.resolve()), "figures must remain in external data storage")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = "tools/plot_baby_shrimp_memory_continuation.py"
    require(hashlib.sha256(git_bytes(commit, source)).hexdigest() == digest(__file__), "commit plot source before execution")
    out.mkdir(parents=True, exist_ok=False)
    render(points, roots, out)
    record = {"id": "BSC3-combined-continuation-figure", "source_commit": commit, "source_sha256": digest(__file__),
              "receipts": sources, "scientific_metadata": metadata, "accepted_samples": len(points),
              "numerical_root_parameters": [r["parameter"] for r in roots], "visual_review": "pending",
              "figures": [{"name": p.name, "sha256": digest(p)} for p in sorted(out.iterdir())]}
    (out / "receipt.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "accepted_samples": len(points), "highlighted_roots": len(roots)}))


if __name__ == "__main__":
    main()
