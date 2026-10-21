"""Local independent numerical cross-check of saved ARP1 outputs (no model calls)."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def close(a, b, label):
    if not np.isclose(a, b, rtol=1e-8, atol=1e-10):
        raise AssertionError(f"{label}: {a} != {b}")


def independent_pr(x):
    x = np.asarray(x, dtype=np.float64)
    y = x - x.mean(axis=0)
    if np.sum(y ** 2) <= 1e-14 * max(float(np.sum(x ** 2)), 1e-300):
        return 0., 0.
    singular = np.linalg.svd(y, compute_uv=False)
    raw = np.sum(singular ** 2) ** 2 / np.sum(singular ** 4)
    scale = np.std(y, axis=0)
    keep = scale > max(scale.max() * 1e-8, 1e-15)
    singular = np.linalg.svd(y[:, keep] / scale[keep], compute_uv=False)
    return float(raw), float(np.sum(singular ** 2) ** 2 / np.sum(singular ** 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Audit receipt already exists")
    result = json.loads((args.analysis / "results.json").read_text())
    source = Path(result["source"])
    verified = 0
    for folder in [source, args.analysis]:
        m = json.loads((folder / "manifest.json").read_text())
        for name, item in m["files"].items():
            assert sha(folder / name) == item["sha256"], name
            assert (folder / name).stat().st_size == item["bytes"], name
            verified += 1
    states = np.load(source / "states.npy", mmap_mode="r")
    acts = np.load(source / "activations.npy", mmap_mode="r")
    growth = np.load(source / "growth.npy", mmap_mode="r")
    orders = np.load(args.analysis / "calibration-layout.npy")
    run = json.loads((source / "run.json").read_text())
    assert len(result["rows"]) == len(run["plan"]["rows"]) == 10
    row_checks = []
    for i, row in enumerate(result["rows"]):
        assert row["id"] == run["plan"]["rows"][i]["id"]
        x = np.asarray(states[-384:, i])
        closure = []
        for lag in range(1, 65):
            errors = [np.linalg.norm(x[t + lag] - x[t]) / max(np.linalg.norm(x[t]), 1e-15)
                      for t in range(len(x) - lag)]
            closure.append(max(errors))
        candidates = [j + 1 for j, e in enumerate(closure) if e <= 1e-9]
        p = candidates[0] if candidates else 0
        assert p == row["reference"]["recurrence"]["period"]
        g = np.asarray(growth[128:, i])
        if np.isfinite(g).all():
            means = np.array([sum(g[:, j]) / 512 for j in range(2)])
            blocks = np.array([[np.mean(g[b:b + 128, j]) for j in range(2)] for b in range(0, 512, 128)])
            agreement = abs(means[0] - means[1]) <= max(.03, .2 * max(abs(means)))
            label = "periodic" if p and max(means) < -.001 else "finite-time chaotic" if (
                not p and min(means) > .01 and blocks.min() > 0 and agreement) else "unresolved"
            assert label == row["reference"]["label"]
            for j in range(2):
                close(means[j], row["reference"]["exponents"][j], "exponent")
        else:
            assert row["reference"]["label"] == "unresolved"
        for window in [16, 128]:
            for layer in [0, 23]:
                x = np.asarray(acts[i, :window, layer], dtype=np.float64)
                raw, standard = independent_pr(x)
                score = row["features"][str(window)]["layers"][layer]
                close(raw, score["raw_pr"], "raw participation ratio")
                close(standard, score["standardized_pr"], "standardized participation ratio")
                order = orders[layer]
                # Enumerate undirected grid edges, independently of np.diff implementation.
                left = np.concatenate([order[:-1].ravel(), order[:, :-1].ravel()])
                right = np.concatenate([order[1:].ravel(), order[:, 1:].ravel()])
                rough = np.median(np.mean((x[:, left] - x[:, right]) ** 2, axis=1) / (2 * np.var(x, axis=1)))
                close(rough, score["roughness"], "neighbor roughness")
        row_checks.append({"id": row["id"], "reference": row["reference"]["label"], "period": p,
                           "passed": True, "svd_checks": 4, "roughness_checks": 4})
    tests = [r for r in result["rows"] if r["split"] == "test" and r["reference"]["label"] != "unresolved"]
    cal_p = [r for r in result["rows"] if r["split"] == "calibration" and r["reference"]["label"] == "periodic"]
    cal_c = [r for r in result["rows"] if r["split"] == "calibration" and r["reference"]["label"] == "finite-time chaotic"]
    assert len(cal_p) == len(cal_c) == 1
    for key, rules in result["rules"].items():
        for metric, rule in rules.items():
            p = cal_p[0]["features"][key][metric]
            c = cal_c[0]["features"][key][metric]
            close(rule["value"], (p + c) / 2, "calibration midpoint")
            assert rule["chaotic_above"] == (c > p)
            for row in result["rows"]:
                scores = row["features"][key]
                close(scores[metric], np.median([r[metric] for r in scores["layers"]]), "all-layer median")
                above = scores[metric] > (p + c) / 2
                expected = "finite-time chaotic" if above == (c > p) else "periodic"
                assert scores["prediction_" + metric] == expected
    for e in result["evaluations"]:
        hits = sum(r["features"][str(e["window"])]["prediction_" + e["metric"]] == r["reference"]["label"] for r in tests)
        assert hits == e["correct"] and len(tests) == e["total"]
    receipt = {"passed": True, "kind": "local numerical cross-check, not an external review",
               "results_sha256": sha(args.analysis / "results.json"), "audit_source_sha256": sha(__file__),
               "verified_files": verified, "rows": row_checks, "evaluation_rows_checked": len(result["evaluations"])}
    with args.out.open("x") as f:
        json.dump(receipt, f, indent=2); f.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
