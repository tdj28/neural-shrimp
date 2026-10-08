"""Prepare exact inputs and run the bounded local-CPU C1 certificate pilot."""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import flint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from neural_shrimp.rigorous_orbits import canonical_bytes, check_orbit, dyadic, sha256

SOURCES = ["src/neural_shrimp/rigorous_orbits.py", "tools/certify_c1.py", "tools/verify_certificates.py"]


def sources():
    return {p: sha256((ROOT / p).read_bytes()) for p in SOURCES}


def prepare(certificate_path, plan_path, index):
    from neural_shrimp.trainability import SYSTEMS, array_sha256
    original = json.loads(certificate_path.read_bytes())
    row = original[index]
    spec = next(m for m in json.loads(plan_path.read_bytes())["maps"]
                if m["map_id"] == row["map_id"].removesuffix("-zoom"))
    if 29000 <= spec["seed"] <= 29999:
        raise ValueError("reserved confirmatory seed")
    net, x, y, theta = SYSTEMS[spec["system"]](spec["seed"])
    realized = {"x": array_sha256(x), "y": array_sha256(y), "theta0": array_sha256(theta)}
    if realized != spec["data_sha256"] or asdict(net) != spec["network"]:
        raise ValueError("regenerated EXP-028 model/data do not match frozen plan")
    network = asdict(net)
    for key in ("in_scale", "out_scale", "loss_scale", "ridge"):
        network[key] = dyadic(network[key])
    return {"schema": "neural-shrimp-c1-input-v1", "network": network,
            "x": [[dyadic(v) for v in r] for r in x], "y": [dyadic(v) for v in y],
            "lr_hidden": dyadic(row["lr_hidden"]), "lr_output": dyadic(row["lr_output"]),
            "period": row["period"], "approximation": [dyadic(v) for v in row["certificate"]["orbit"][0]],
            "provenance": {"certificate_file": str(certificate_path),
                "certificate_sha256": sha256(certificate_path.read_bytes()), "row_index": index,
                "plan_file": str(plan_path), "plan_sha256": sha256(plan_path.read_bytes()),
                "map_id": row["map_id"], "component": row["component"],
                "data_sha256": realized,
                "selection": "best conditioned embedded S2 period-5/10 candidate by existing float condition(M-I); exploratory",
                "semantics": "all binary64 saved inputs converted to exact dyadic rationals; tanh and all GD arithmetic then exact real"}}


def run(inputs, output, settings):
    output.mkdir(parents=True, exist_ok=False)
    input_bytes = canonical_bytes(inputs)
    (output / "inputs.json").write_bytes(input_bytes)
    start = time.monotonic()
    metadata = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "source_sha256": sources(), "python": platform.python_version(),
                "python_flint": flint.__version__, "flint": flint.__FLINT_VERSION__,
                "platform": platform.platform(), "pid": os.getpid(), "cpu_threads": 1,
                "nice": os.getpriority(os.PRIO_PROCESS, 0), "gpu": "not used", "paid_compute": False,
                "start_utc": datetime.now(timezone.utc).isoformat()}
    (output / "runtime-start.json").write_bytes(canonical_bytes(metadata))
    try:
        result = check_orbit(inputs, **settings)
    except Exception as error:
        failure = {"passed": False, "evidence_rung": 0, "error": type(error).__name__ + ": " + str(error),
                   "settings": settings}
        (output / "failure.json").write_bytes(canonical_bytes(failure))
        raise
    envelope = {"schema": "neural-shrimp-c1-certificate-v1", "inputs_sha256": sha256(input_bytes),
                "runtime": metadata, "result": result}
    certificate = canonical_bytes(envelope)
    (output / "certificate.json").write_bytes(certificate)
    metadata.update(elapsed_seconds=time.monotonic() - start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (output / "runtime-end.json").write_bytes(canonical_bytes(metadata))
    inventory = {p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size}
                 for p in sorted(output.iterdir()) if p.is_file()}
    receipt = {"schema": "neural-shrimp-c1-receipt-v1", "external_root": str(output.resolve()),
               "source_commit": metadata["source_commit"], "source_sha256": metadata["source_sha256"],
               "files": inventory, "passed": result["passed"], "evidence_rung": result["evidence_rung"],
               "research_status": result["research_status"], "statement": result["exact_statement"],
               "contraction_norm_upper": result["contraction_norm_upper"],
               "attraction_witness_power": result["attraction_witness_power"],
               "attraction_norm_upper": result["power_bounds"][-1]["norm_inf_upper"],
               "elapsed_seconds": metadata["elapsed_seconds"], "max_rss_kib": metadata["max_rss_kib"]}
    (output / "receipt.json").write_bytes(canonical_bytes(receipt))
    print(json.dumps(receipt, indent=2))
    return 0 if result["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--certificates", type=Path, required=True)
    p.add_argument("--plan", type=Path, default=ROOT / "experiments/trainability/EXP-028-plan.json")
    p.add_argument("--index", type=int, default=121)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--precision", type=int, default=256)
    p.add_argument("--radius-exponent", type=int, default=-80)
    p.add_argument("--newton-steps", type=int, default=5)
    p.add_argument("--max-power", type=int, default=512)
    args = parser.parse_args()
    if args.action == "prepare":
        payload = prepare(args.certificates, args.plan, args.index)
        with args.out.open("xb") as handle:
            handle.write(canonical_bytes(payload))
        print(json.dumps({"path": str(args.out), "sha256": sha256(args.out.read_bytes())}))
        return 0
    return run(json.loads(args.input.read_bytes()), args.out,
               {k: getattr(args, k) for k in ("precision", "radius_exponent", "newton_steps", "max_power")})


if __name__ == "__main__":
    raise SystemExit(main())
