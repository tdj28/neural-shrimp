"""Score every ARP1 row; calibration settings alone determine layouts and rules."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from metrics import dimensions, layout, predict, recurrence, reference, roughness, threshold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def analyze(source, out):
    tic = time.monotonic()
    source, out = source.resolve(), out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError("Raw analysis and figures must be outside Git")
    out.mkdir(parents=True, exist_ok=False)
    run = json.loads((source / "run.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if run["status"] != "complete" or run["mode"] != "collect":
        raise ValueError("Cannot analyze an incomplete acquisition")
    for name, item in manifest["files"].items():
        if sha(source / name) != item["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    plan = run["plan"]
    acts = np.load(source / "activations.npy", mmap_mode="r")
    states = np.load(source / "states.npy", mmap_mode="r")
    growth = np.load(source / "growth.npy", mmap_mode="r")
    nf = plan["feature_steps"]
    rows = []
    for i, config in enumerate(plan["rows"]):
        if not np.isfinite(acts[i]).all() or not np.isfinite(states[:, i]).all():
            raise ValueError(f"Nonfinite state or activation in {config['id']}")
        try:
            ref = reference(states[nf:, i], growth[nf:, i], plan["max_period"], plan["period_tolerance"])
        except ValueError as exc:
            ref = {"label": "unresolved", "reason": str(exc), "recurrence": recurrence(states[nf:, i])}
        rows.append({**config, "reference": ref, "features": {}})
    write(out / "reference.json", rows)
    cal = [i for i, r in enumerate(rows) if r["split"] == "calibration"]
    cp = [i for i in cal if rows[i]["reference"]["label"] == "periodic"]
    cc = [i for i in cal if rows[i]["reference"]["label"] == "finite-time chaotic"]
    if len(cp) != 1 or len(cc) != 1:
        write(out / "failure.json", {"reason": "Calibration requires one resolved trajectory of each class"})
        raise ValueError("Calibration label gate failed; all reference rows retained")
    orders = np.array([layout(np.concatenate([acts[i, :, layer] for i in cal])) for layer in range(24)])
    np.save(out / "calibration-layout.npy", orders)
    for i, row in enumerate(rows):
        for window in plan["windows"]:
            layer_scores = []
            for layer in range(24):
                x = acts[i, :window, layer]
                d = dimensions(x)
                d["roughness"] = roughness(x, orders[layer])["median"]
                layer_scores.append(d)
            scores = {name: float(np.median([s[name] for s in layer_scores]))
                      for name in ["raw_pr", "standardized_pr", "roughness"]}
            scores["layers"] = layer_scores
            scores["recurrence"] = recurrence(states[:window, i], plan["max_period"], plan["baseline_tolerance"])
            scores["recurrence_two_cycles"] = recurrence(states[:window, i], plan["max_period"], plan["baseline_tolerance"], min_cycles=2)
            row["features"][str(window)] = scores
        print(json.dumps({"scored": row["id"], "reference": row["reference"]["label"],
                          "seconds": round(time.monotonic() - tic, 2)}), flush=True)
    rules, evaluations = {}, []
    for window in plan["windows"]:
        key = str(window)
        rules[key] = {}
        for metric in ["raw_pr", "standardized_pr", "roughness"]:
            rule = threshold(rows[cp[0]]["features"][key][metric], rows[cc[0]]["features"][key][metric])
            rules[key][metric] = rule
            for row in rows:
                row["features"][key][f"prediction_{metric}"] = predict(row["features"][key][metric], rule)
        for row in rows:
            detected = row["features"][key]["recurrence"]["period"]
            row["features"][key]["recurrence_output"] = "periodic candidate" if detected else "unresolved"
            row["features"][key]["prediction_recurrence_closed_set"] = "periodic" if detected else "finite-time chaotic"
            detected2 = row["features"][key]["recurrence_two_cycles"]["period"]
            row["features"][key]["recurrence_two_cycles_output"] = "periodic candidate" if detected2 else "unresolved"
            row["features"][key]["prediction_recurrence_two_cycles_closed_set"] = "periodic" if detected2 else "finite-time chaotic"
        for metric in ["raw_pr", "standardized_pr", "roughness", "recurrence_closed_set", "recurrence_two_cycles_closed_set"]:
            resolved = [r for r in rows if r["split"] == "test" and r["reference"]["label"] != "unresolved"]
            correct = [r["features"][key][f"prediction_{metric}"] == r["reference"]["label"] for r in resolved]
            per_class = {}
            for label in ["periodic", "finite-time chaotic"]:
                group = [r for r in resolved if r["reference"]["label"] == label]
                hit = sum(r["features"][key][f"prediction_{metric}"] == label for r in group)
                per_class[label] = {"correct": hit, "total": len(group)}
            rates = [v["correct"] / v["total"] for v in per_class.values() if v["total"]]
            evaluations.append({"window": window, "metric": metric, "correct": sum(correct), "total": len(correct),
                                "accuracy": sum(correct) / len(correct) if correct else None,
                                "balanced_accuracy": float(np.mean(rates)) if len(rates) == 2 else None,
                                "per_class": per_class,
                                "unresolved_test_rows": [r["id"] for r in rows if r["split"] == "test" and r["reference"]["label"] == "unresolved"]})
    result = {"status": "exploratory", "source": str(source), "input_manifest_sha256": sha(source / "manifest.json"),
              "acquisition_source_commit": run["source_commit"], "analysis_source_sha256": sha(__file__),
              "analysis_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "amendment_sha256": sha(HERE / "amendment-01.md"),
              "metrics_source_sha256": sha(HERE / "metrics.py"), "rows": rows, "rules": rules, "evaluations": evaluations,
              "note": "Binary predictions are candidate labels; no recurrence is not by itself evidence of chaos.",
              "seconds": time.monotonic() - tic}
    write(out / "results.json", result)
    render(result, out)
    write(out / "manifest.json", {"files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
                                             for p in sorted(out.iterdir()) if p.is_file()}})
    print(json.dumps({"complete": True, "out": str(out), "evaluations": evaluations}), flush=True)


def render(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = {"periodic": "#287e8e", "finite-time chaotic": "#c46c2c", "unresolved": "#777777"}
    endpoints = []
    for row in result["rows"]:
        label = row["reference"]["label"]
        ws = [int(w) for w in row["features"]]
        vals = [row["features"][str(w)]["standardized_pr"] for w in ws]
        ax[0].plot(ws, vals, "o--" if row["split"] == "calibration" else "o-", color=colors[label], alpha=.8)
        endpoints.append((vals[-1], row["id"].replace("test-", ""), colors[label]))
    previous = -1.
    for value, name, color in sorted(endpoints):
        position = max(value, previous + .28)
        ax[0].annotate(name, (128, value), xytext=(134, position), fontsize=8, color=color,
                       va="center", arrowprops={"arrowstyle": "-", "color": color, "lw": .6})
        previous = position
    ax[0].set(xlabel="Activation observations", ylabel="Effective dimensionality (median over layers)",
              title="Do periodic trajectories use fewer activation patterns?", xlim=(12, 185), ylim=(-.2, 9.2))
    ax[0].set_xticks([16, 32, 64, 128])
    ax[0].text(.02, .98, "Teal: periodic • Orange: positive finite-time exponent\nDashed: calibration settings", transform=ax[0].transAxes,
               va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    tests = [r for r in result["rows"] if r["split"] == "test"]
    metrics = ["raw_pr", "standardized_pr", "roughness", "recurrence_closed_set", "recurrence_two_cycles_closed_set"]
    arr = np.array([[0 if r["reference"]["label"] == "unresolved" else
                    1 if r["features"]["128"][f"prediction_{m}"] == r["reference"]["label"] else -1
                    for m in metrics] for r in tests])
    ax[1].imshow(arr, cmap=ListedColormap(["#d67568", "#d9dde2", "#80c0ad"]), vmin=-1, vmax=1, aspect="auto")
    for i in range(len(tests)):
        for j in range(5):
            ax[1].text(j, i, "?" if arr[i, j] == 0 else "Yes" if arr[i, j] == 1 else "No", ha="center", va="center")
    ax[1].set_yticks(range(len(tests)), [r["id"].replace("test-", "") + " (" +
                      {"periodic": "P", "finite-time chaotic": "C", "unresolved": "?"}[r["reference"]["label"]] + ")" for r in tests])
    ax[1].set_xticks(range(5), ["Raw\ndimension", "Scaled\ndimension", "Visual\nroughness", "Return\n3 cycles", "Return\n2 cycles"])
    ax[1].set_title("Correct at 128 observations? Held-out settings")
    fig.suptitle("Activation-based regime diagnosis • Exploratory pilot", fontsize=17, weight="bold")
    fig.text(.5, .015, "P = numerical periodicity; C = positive finite-time exponent. Selected settings, one start each.\n"
             "Recurrence baseline is scored as a closed-set guess; its operational output is unresolved when no repeat is found.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, .94])
    fig.savefig(out / "overview.png", dpi=160)
    fig.savefig(out / "overview.pdf")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    analyze(args.source, args.out)
