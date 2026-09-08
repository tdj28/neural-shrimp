"""Receipt-backed descriptive analysis; no post hoc significance claims."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .cli import verify, save_json, sha256


def analyze(output, destination):
    output, destination = Path(output), Path(destination)
    rows = verify(output)
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((output / "manifest.json").read_text())
    summary = {"experiment_id": config["experiment_id"], "status": "exploratory_pilot",
               "receipt_sha256": sha256(output / "receipt.json"), "n_trials": len(rows),
               "status_counts": dict(Counter(r["status"] for r in rows)),
               "total_trial_seconds": sum(r["total_trial_seconds"] for r in rows),
               "claim_boundary": "Descriptive finite pilot. No shrimp geometry or practical advantage has been established by this summary."}
    lines = [f"# {config['experiment_id']} exploratory pilot", "", f"Receipt SHA256: `{summary['receipt_sha256']}`", "", f"Complete trial census: {len(rows)}.", ""]
    if config["kind"] == "utility":
        policy_rows = {}
        lines += ["Fixed gradient-update budgets; wall time includes policy diagnostics. This is not an equal-total-compute comparison and does not test incremental shrimp geometry.", "", "| Policy | Divergent / total | Triggered runs | Mean terminal test MSE (only if all runs finite) | Wall seconds |", "|---|---:|---:|---:|---:|"]
        for policy in config["policies"]:
            group = [r for r in rows if r["policy"] == policy]
            failures = sum(r["status"] == "divergent" for r in group)
            # Refuse to hide missing/divergent trajectories in a successful-only mean.
            mean = None if failures else float(np.mean([r["terminal_test_mse"] for r in group]))
            record = {"n": len(group), "divergent": failures, "triggered_runs": sum(bool(r["events"]) for r in group),
                      "mean_terminal_test_mse_all_finite": mean, "wall_seconds": sum(r["wall_seconds"] for r in group),
                      "diagnostic_seconds": sum(r["diagnostic_seconds"] for r in group),
                      "per_seed": {str(seed): {"failures": sum(r["status"] == "divergent" for r in group if r["seed"] == seed),
                            "test_mse": [r["terminal_test_mse"] for r in group if r["seed"] == seed]} for seed in config["seeds"]}}
            policy_rows[policy] = record
            shown = "undefined (divergence retained)" if mean is None else f"{mean:.6g}"
            lines.append(f"| {policy} | {failures}/{len(group)} | {record['triggered_runs']} | {shown} | {record['wall_seconds']:.3f} |")
        summary["policies"] = policy_rows
    else:
        summary["verified_attracting_nontrivial_cycles"] = sum(bool(r.get("cycle_certificate", {}).get("verified")) and bool(r.get("cycle_certificate", {}).get("attracting")) and r.get("diagnostic", {}).get("period", 0) > 1 for r in rows)
        lines += ["| Classification | Count |", "|---|---:|"] + [f"| {key} | {count} |" for key, count in sorted(summary["status_counts"].items())]
        lines += ["", f"Verified attracting nontrivial cycle roots: {summary['verified_attracting_nontrivial_cycles']}.", "", "A finite tail is a candidate generator. Certification checks orbit closure, primitive period and local multipliers. This coarse scan does not continue saddle-node/flip boundaries, establish shrimp topology, or certify chaos."]
        _plot_scan(rows, config, destination / "parameter-map.png")
    save_json(destination / "summary.json", summary)
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    save_json(destination / "provenance.json", {"source_receipt_sha256": summary["receipt_sha256"], "generated_files": [{"path": p.name, "sha256": sha256(p)} for p in sorted(destination.iterdir()) if p.is_file()]})
    print(json.dumps(summary, indent=2))


def _plot_scan(rows, config, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    if config["kind"] == "henon_scan":
        xkey, ykey, xv, yv = "a", "b", config["a_values"], config["b_values"]
    else:
        xkey, ykey, xv, yv = "lr_hidden", "lr_output", config["lr_hidden_values"], config["lr_output_values"]
    fig, axes = plt.subplots(1, len(config["seeds"]), figsize=(6 * len(config["seeds"]), 4.6), squeeze=False, constrained_layout=True)
    cmap = ListedColormap(["#d9dce1", "#353b47", "#377eb8", "#f28e2b"])
    for axis, seed in zip(axes[0], config["seeds"]):
        grid = np.zeros((len(yv), len(xv)))
        for r in rows:
            if r["seed"] != seed:
                continue
            period = r.get("diagnostic", {}).get("period")
            certificate = r.get("cycle_certificate", {})
            value = 1 if r["status"] == "divergent" else 0
            if period == 1:
                value = 2
            if period and period > 1 and certificate.get("verified") and certificate.get("attracting"):
                value = 3
            grid[yv.index(r[ykey]), xv.index(r[xkey])] = value
        axis.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=3, aspect="auto")
        axis.set_xticks(range(len(xv)), [f"{x:g}" for x in xv], rotation=60)
        axis.set_yticks(range(len(yv)), [f"{y:g}" for y in yv])
        axis.set(xlabel=xkey + " (sampled grid indices)", ylabel=ykey + " (sampled grid indices)", title=f"{config['experiment_id']} · seed {seed}")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(cmap.colors, ["Unresolved / uncertified", "Divergent", "Fixed-point candidate", "Verified attracting cycle"])], loc="outside lower center", ncol=2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
