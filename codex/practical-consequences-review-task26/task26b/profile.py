"""One manufactured full-shape cost check; never opens original REPRO inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as driver

OUTPUT = "/data2/neural-shrimp-data/codex/practical-consequences-review-task26/task26b/profile-001"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    plan = driver.read_json(args.plan)
    if plan.get("schema") != "task26b-synthetic-profile-v1" or plan.get("admission") != "reviewed":
        raise ValueError("prepared synthetic profile refuses execution")
    if plan["resources"] != driver.CAPS or plan["output"] != OUTPUT or str(Path(args.output).absolute()) != OUTPUT:
        raise ValueError("fixed synthetic envelope/output")
    if set(plan["sources"]) != {"core.py", "run.py", "profile.py"}:
        raise ValueError("profile source set")
    for name, sha in plan["sources"].items():
        if driver.digest(HERE/name) != sha:
            raise ValueError("profile source identity: " + name)
    driver.establish_limits()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    written = 0
    def save(name, payload, closing=False):
        nonlocal written
        data = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if written + len(data) > driver.CAPS["output_bytes"] - (0 if closing else 1024**2):
            raise ValueError("synthetic serialization exceeds fixed output budget")
        with (out/name).open("xb") as f:
            f.write(data)
        written += len(data)
        return len(data)
    try:
        import numpy as np
        import core
        if np.__version__ != plan["numpy_version"]:
            raise ValueError("NumPy version")
        steps, n = 1500, 448
        t = np.arange(steps, dtype=float)[:, None]
        j = np.arange(n, dtype=float)[None, :]
        base = np.zeros((steps, n), dtype="<f4")
        # Deterministic manufactured shapes include long attained-max plateaus.
        base[:, :192] = np.minimum(2., 1e-14 * np.exp((.02 + np.arange(192)/19200)[None, :] * t))
        base[:, 192:384] = 1e-13 * (1. + .25*np.sin(t/7 + np.arange(192)[None, :]))
        base[:, 192:208] = 0.       # 16 exact-zero series: all steps attain max.
        base[:, 208:224] = 1e-13   # 16 constant positive series: max and min ties.
        base[:, 384:] = 1e-7      # 64 more constant positive series.
        labels = ["chaotic"]*192 + ["periodic"]*192 + ["other"]*64
        specification = {"cells": [[i/1000., (i+1)/10000.] for i in range(n)],
            "source": ["synthetic-map-A" if i%2 == 0 else "synthetic-map-B" for i in range(n)],
            "label": labels, "map_exponent": [.02+i/19200 if i<192 else -.01 for i in range(n)],
            "map_period": [0]*192 + [2]*192 + [0]*64}
        arrays, originals = {}, {}
        for k, comp in enumerate(core.COMPARISONS):
            d = (base * np.float32((1, 5, 10, 50)[k])).astype("<f4")
            agreement = ((t+j+k) % 11 > 2)
            agreement[:, 192:384] = True
            agreement[1000:1001, 192:384:13] = False
            agreement[:, 384:] = False
            arrays[comp+"_dist"], arrays[comp+"_agree"] = d, agreement
            # Original-summary fabrication is itself independent own code, not a
            # producer invocation; formula correctness was tested separately.
            originals[comp] = [{key: value for key, value in core.cell_summary(d[:, i], agreement[:, i], 500).items()
                                if key in core.OLD_FIELDS} for i in range(n)]
        saved = {"cells": specification["cells"], "labels": labels,
                 "args": {"steps": steps, "tail": 500, "compare": list(core.COMPARISONS)},
                 "comparisons": originals}
        generated = time.perf_counter()
        result = core.calculate(arrays, saved, specification, expected_counts={"chaotic":192,"periodic":192,"other":64})
        calculated = time.perf_counter()
        result["resource_profile_only"] = True
        result["synthetic_fixture"] = plan["fixture"]
        result["sources"] = plan["sources"]
        result["target_or_producer_payload_read"] = False
        cells = {"schema": "task26b-cell-results-v1", "cells": result.pop("cells"),
                 "figure_summaries": result.pop("figure_summaries")}
        cell_bytes = save("synthetic-cell-results.json", cells)
        result["cell_results"] = {"path": "synthetic-cell-results.json", "bytes": cell_bytes,
                                  "sha256": driver.digest(out/"synthetic-cell-results.json")}
        report_bytes = save("synthetic-report.json", result)
        serialized = time.perf_counter()
        for name, sha in plan["sources"].items():
            if driver.digest(HERE/name) != sha:
                raise ValueError("profile source changed during run")
        save("status.json", {"status": "resource_profile_completed", "computational_completed": True,
             "resource_profile_only": True, "scientific_claims_accepted": False,
             "generated_array_payload_bytes": sum(a.nbytes for a in arrays.values()),
             "synthetic_generation_and_import_seconds": generated-start,
             "own_reduction_seconds": calculated-generated,
             "serialization_and_write_seconds": serialized-calculated,
             "worker_seconds_through_serialization": serialized-start,
             "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             "serialized_scientific_bytes": cell_bytes+report_bytes,
             "numpy": np.__version__, "python": sys.version, "nice": os.getpriority(os.PRIO_PROCESS, 0),
             "as_limits": resource.getrlimit(resource.RLIMIT_AS), "sources": plan["sources"]}, closing=True)
        return 0
    except Exception as exc:
        save("failure.json", {"computational_completed": False, "resource_profile_only": True,
             "error_type": type(exc).__name__, "error": str(exc),
             "seconds_before_failure_serialization": time.perf_counter()-start}, closing=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
