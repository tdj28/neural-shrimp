"""Result-free manifest validation, immutable pilot execution, and analysis."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from .controls import HenonMap
from .diagnostics import iterate_map, classify_tail, certify_cycle
from .models import RegressionDataset, TanhRegressor, make_teacher_regression
from .utility import POLICIES, run_policy


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_manifest(config):
    if config.get("status") != "exploratory_pilot":
        raise ValueError("This runner only authorizes explicitly exploratory pilots")
    if config.get("kind") not in ("henon_scan", "training_scan", "utility"):
        raise ValueError("Unknown experiment kind")
    if not config.get("experiment_id") or not config.get("question"):
        raise ValueError("Question and experiment ID are required")
    for key in ("steps", "tail", "max_period"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if config["tail"] < 7 * config["max_period"] or config["tail"] > config["steps"]:
        raise ValueError("Tail must cover at least seven maximum-period blocks and fit the run")
    seeds = config["seeds"]
    if not seeds or len(seeds) != len(set(seeds)) or not all(isinstance(s, int) and s >= 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for key in ("atol", "rtol", "divergence_threshold"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if config["kind"] != "henon_scan":
        for key in ("d", "width", "n_train", "n_validation", "n_test"):
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"Invalid {key}")
        if not np.isfinite(config["ridge"]) or config["ridge"] < 0:
            raise ValueError("Invalid ridge")
        if set(config.get("input_files", {})) != set(map(str, seeds)):
            raise ValueError("Generate and hash all frozen input files with prepare-inputs before validation")
        for item in config["input_files"].values():
            if len(item.get("sha256", "")) != 64 or not item.get("path"):
                raise ValueError("Frozen input path/hash missing")
    if config["kind"] == "henon_scan":
        grids = [config["a_values"], config["b_values"]]
    elif config["kind"] == "training_scan":
        grids = [config["lr_hidden_values"], config["lr_output_values"]]
    else:
        grids = []
        pairs = config["learning_rate_pairs"]
        if not pairs or any(len(p) != 2 for p in pairs) or len(set(map(tuple, pairs))) != len(pairs):
            raise ValueError("Learning-rate pairs must be distinct pairs")
        if not np.isfinite(pairs).all() or np.min(pairs) <= 0:
            raise ValueError("Learning rates must be finite and positive")
        if sorted(config["policies"]) != sorted(POLICIES):
            raise ValueError("Utility requires fixed, scheduled, stagnation, and recurrence controls exactly once")
        if not 0 < config["decay_factor"] < 1:
            raise ValueError("Decay factor must lie strictly between zero and one")
        for key in ("check_every", "check_after", "max_decays"):
            if not isinstance(config[key], int) or config[key] <= 0:
                raise ValueError(f"Invalid {key}")
        if config["check_after"] < 2 * config["check_every"] or config["check_after"] >= config["steps"]:
            raise ValueError("Invalid diagnostic horizon")
        if not 0 <= config["stagnation_relative_improvement"] < 1:
            raise ValueError("Invalid stagnation criterion")
    for grid in grids:
        if not grid or len(set(grid)) != len(grid) or not np.isfinite(grid).all():
            raise ValueError("Grid values must be nonempty, unique, and finite")
        if config["kind"] == "training_scan" and min(grid) <= 0:
            raise ValueError("Learning rates must be positive")
    return len(trial_plan(config))


def trial_plan(config):
    kind = config["kind"]
    rows = []
    for seed in config["seeds"]:
        if kind == "henon_scan":
            panel = ({"a": a, "b": b} for a, b in itertools.product(config["a_values"], config["b_values"]))
        elif kind == "training_scan":
            panel = ({"lr_hidden": h, "lr_output": o} for h, o in itertools.product(config["lr_hidden_values"], config["lr_output_values"]))
        else:
            panel = ({"lr_hidden": h, "lr_output": o, "policy": p} for (h, o), p in itertools.product(config["learning_rate_pairs"], config["policies"]))
        for condition in panel:
            identity = {"experiment_id": config["experiment_id"], "seed": seed, **condition}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
            rows.append({"trial_id": config["experiment_id"] + "-" + digest, **identity})
    return rows


def source_receipt():
    root = Path(__file__).resolve().parents[2]
    archive_receipt = root / "runpod-bundle.json"
    if archive_receipt.exists():
        receipt = json.loads(archive_receipt.read_text())
        for relative, item in receipt["files"].items():
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or not source.is_file() or sha256(source) != item["sha256"]:
                raise ValueError("Archive source hash mismatch")
        return {"source_commit": receipt["source_commit"], "source_mode": "verified_archive", "source_receipt_sha256": sha256(archive_receipt), "files": [{"path": p, "sha256": item["sha256"]} for p, item in receipt["files"].items()]}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise ValueError("Commit and push the complete plan/code before execution; worktree is dirty")
    remote_refs = subprocess.check_output(["git", "for-each-ref", "--contains", commit, "--format=%(refname)", "refs/remotes/origin"], cwd=root, text=True).splitlines()
    if not remote_refs:
        raise ValueError("Push this source commit to origin before execution; no remote tracking ref contains it")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    return {"source_commit": commit, "source_mode": "clean_git", "remote_tracking_refs_containing_source": remote_refs, "files": [{"path": p, "sha256": sha256(root / p)} for p in tracked if p]}


def prepare_inputs(manifest):
    """Generate result-free teacher/data/initial-state artifacts before freeze."""
    manifest = Path(manifest).resolve()
    config = json.loads(manifest.read_text())
    if config["kind"] == "henon_scan" or "input_files" in config:
        raise ValueError("Inputs already prepared or not required; do not overwrite a frozen plan")
    root = Path(__file__).resolve().parents[2]
    destination = root / "experiments" / "inputs" / config["experiment_id"]
    destination.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for seed in config["seeds"]:
        data = make_teacher_regression(seed=seed, d=config["d"], n_train=config["n_train"], n_validation=config["n_validation"], n_test=config["n_test"])
        model = TanhRegressor(config["d"], config["width"], ridge=config["ridge"])
        path = destination / f"seed-{seed}.npz"
        np.savez_compressed(path, initial=model.initialize(seed + 100000), metadata=json.dumps(data.metadata),
                            **{k: getattr(data, k) for k in ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y")})
        inputs[str(seed)] = {"path": str(path.relative_to(root)), "sha256": sha256(path)}
    config["input_files"] = inputs
    validate_manifest(config)
    save_json(manifest, config)


def load_inputs(config, seed):
    root = Path(__file__).resolve().parents[2]
    item = config["input_files"][str(seed)]
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root) or sha256(path) != item["sha256"]:
        raise ValueError("Frozen input artifact hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        fields = {k: arrays[k] for k in ("train_x", "train_y", "validation_x", "validation_y", "test_x", "test_y")}
        if any(not np.isfinite(a).all() for a in fields.values()):
            raise ValueError("Nonfinite frozen input")
        data = RegressionDataset(**fields, metadata=json.loads(str(arrays["metadata"])))
        initial = arrays["initial"]
    model = TanhRegressor(config["d"], config["width"], ridge=config["ridge"])
    if initial.shape != (model.n_parameters,) or not np.isfinite(initial).all():
        raise ValueError("Frozen initial state incompatible with model")
    for split, count in (("train", config["n_train"]), ("validation", config["n_validation"]), ("test", config["n_test"])):
        if getattr(data, split + "_x").shape != (count, config["d"]) or getattr(data, split + "_y").shape != (count,):
            raise ValueError("Frozen split shape mismatch")
    return data, model, initial, path


def run(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    config = json.loads(manifest.read_text())
    expected_count = validate_manifest(config)
    provenance = source_receipt()
    source_root = Path(__file__).resolve().parents[2]
    if not manifest.is_relative_to(source_root):
        raise ValueError("Runtime manifest must be inside the frozen source tree")
    relative_manifest = str(manifest.relative_to(source_root))
    bound_files = {item["path"]: item["sha256"] for item in provenance["files"]}
    if bound_files.get(relative_manifest) != sha256(manifest):
        raise ValueError("Runtime manifest must be committed and match its source receipt")
    output.mkdir(parents=True, exist_ok=False)
    (output / "trials").mkdir()
    (output / "inputs").mkdir()
    save_json(output / "manifest.json", config)
    rows = trial_plan(config)
    save_json(output / "trial_plan.json", rows)
    metadata = {**provenance, "experiment_id": config["experiment_id"], "status": "running", "design_status": "exploratory_pilot",
                "manifest_sha256": sha256(manifest), "manifest_copy_sha256": sha256(output / "manifest.json"),
                "expected_trials": expected_count, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "python": platform.python_version(), "platform": platform.platform(), "dtype": "float64", "device": "cpu",
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
                "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "paid_compute_usd": None, "cost_note": "Billing is recorded in the separate task resource ledger; execution time is not a provider invoice.", "paid_review": "not_requested"}
    save_json(output / "started.json", metadata)
    completed = 0
    dataset_cache = {}
    try:
        for row in rows:
            started = time.perf_counter()
            kind, seed = config["kind"], row["seed"]
            if kind != "henon_scan" and seed not in dataset_cache:
                data, model, initial, input_path = load_inputs(config, seed)
                dataset_cache[seed] = data, model, initial
                shutil.copyfile(input_path, output / "inputs" / input_path.name)
            if kind == "utility":
                data, model, initial = dataset_cache[seed]
                result, raw = run_policy(model, data, initial, (row["lr_hidden"], row["lr_output"]), row["policy"], config)
            else:
                if kind == "henon_scan":
                    mapping, initial = HenonMap(row["a"], row["b"]), np.array([0.1, 0.1])
                else:
                    data, model, initial = dataset_cache[seed]
                    mapping = model.training_map(data.train_x, data.train_y, row["lr_hidden"], row["lr_output"])
                trajectory = iterate_map(mapping, initial, config["steps"], tail=config["tail"], divergence_threshold=config["divergence_threshold"])
                raw = {"states": trajectory.states}
                if trajectory.status == "divergent":
                    diagnostic = {"classification": "divergent", "period": None}
                else:
                    diagnostic = classify_tail(trajectory.states, max_period=config["max_period"], atol=config["atol"], rtol=config["rtol"])
                result = {"status": diagnostic["classification"], "diagnostic": diagnostic, "gradient_updates": trajectory.steps_completed}
                period = diagnostic.get("period")
                if period:
                    result["cycle_certificate"] = certify_cycle(mapping, trajectory.states[-1], period, jacobian=mapping.jacobian, atol=config["atol"], rtol=config["rtol"])
                if kind == "training_scan" and trajectory.status != "divergent":
                    last = trajectory.states[-1]
                    result["terminal_test_mse"] = float(np.mean((model.predict(last, data.test_x) - data.test_y) ** 2))
            result = {**row, **result, "total_trial_seconds": time.perf_counter() - started, "manifest_sha256": metadata["manifest_sha256"]}
            raw_path = output / "trials" / (row["trial_id"] + ".npz")
            np.savez_compressed(raw_path, **raw)
            result["raw_sha256"] = sha256(raw_path)
            save_json(raw_path.with_suffix(".json"), result)
            completed += 1
            if completed % 10 == 0 or completed == expected_count:
                print(json.dumps({"completed": completed, "expected": expected_count}), flush=True)
    except BaseException as error:
        save_json(output / "failed.json", {"completed_trials": completed, "error_type": type(error).__name__, "error": str(error)})
        raise
    entries = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output / "receipt.json", {"status": "completed", "source_commit": provenance["source_commit"], "expected_trials": expected_count, "completed_trials": completed, "files": entries})
    print(json.dumps({"output": str(output), "receipt_sha256": sha256(output / "receipt.json")}))


def verify(output):
    output = Path(output)
    receipt = json.loads((output / "receipt.json").read_text())
    for item in receipt["files"]:
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != item["sha256"]:
            raise ValueError(f"Hash mismatch: {item['path']}")
    config = json.loads((output / "manifest.json").read_text())
    expected = {r["trial_id"] for r in trial_plan(config)}
    results = [json.loads(p.read_text()) for p in sorted((output / "trials").glob("*.json"))]
    ids = [r["trial_id"] for r in results]
    if set(ids) != expected or len(ids) != len(expected) or receipt["completed_trials"] != len(expected):
        raise ValueError("Missing, duplicate, or unexpected results")
    for result in results:
        if sha256(output / "trials" / (result["trial_id"] + ".npz")) != result["raw_sha256"]:
            raise ValueError("Trial raw data mismatch")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate")
    val.add_argument("manifests", nargs="+")
    plan = sub.add_parser("plan")
    plan.add_argument("manifest")
    inputs = sub.add_parser("prepare-inputs")
    inputs.add_argument("manifest")
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", required=True)
    execute.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("output")
    report = sub.add_parser("analyze")
    report.add_argument("output")
    report.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        for path in args.manifests:
            config = json.loads(Path(path).read_text())
            count = validate_manifest(config)
            if config["kind"] != "henon_scan":
                for seed in config["seeds"]:
                    load_inputs(config, seed)
            print(json.dumps({"manifest": path, "trials": count}))
    elif args.command == "plan":
        config = json.loads(Path(args.manifest).read_text())
        validate_manifest(config)
        print(json.dumps(trial_plan(config), indent=2))
    elif args.command == "prepare-inputs":
        prepare_inputs(args.manifest)
    elif args.command == "run":
        run(args.manifest, args.output)
    elif args.command == "verify":
        print(json.dumps({"verified_trials": len(verify(args.output))}))
    else:
        from .report import analyze
        analyze(args.output, args.destination)


if __name__ == "__main__":
    main()
