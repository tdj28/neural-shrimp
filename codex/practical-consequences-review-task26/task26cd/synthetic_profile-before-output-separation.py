"""One prospective full-shape synthetic profile, no original inputs or producer imports."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent


def own(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    S, M, fixtures = own("statistics"), own("map_evidence"), own("test_statistics")
    output = HERE / "synthetic-profile-001"
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    rng = np.random.default_rng(260315)
    alpha, bias, summaries = [], [], {}
    for name, tgrid, width, chunk in (("amap-a", [-1., -.35, 27], 101, 1364),
                                     ("amap-b", [-.325, .3, 26], 101, 1313),
                                     ("bmap-a", [-1., -.35, 27], 129, 1161),
                                     ("bmap-b", [-.325, .3, 26], 129, 1118)):
        args = {"tgrid": tgrid, "alpha": [-.12, .12, width] if name[0] == "a" else [0., 0., 1],
                "beta": [0., 0., 1] if name[0] == "a" else [-8., 8., width], "record": 1500,
                "chunk": chunk, "eps": [1e-6, 1e-9]}
        positions = M.grid(args)
        arrays = {key: rng.normal(size=(1000, len(positions))).astype(np.float32) for key in M.NAMES}
        arrays["cells"] = positions
        cells = []
        for i, (a, b, t) in enumerate(positions):
            chaotic = i % width % 3 == 0
            means = {"mean_" + key: float(arrays[key][:, i].mean(dtype=np.float64)) + .001
                     for key in M.NAMES}
            cells.append({"alpha": float(a), "beta": float(b), "log10_temperature": float(t),
                          "lyapunov": [.2, .21] if chaotic else [-.2, -.21], "eps": args["eps"][:],
                          "period": 0 if chaotic else 4, "chaotic": chaotic, "boundary_term": .03,
                          "frac_gap_below_1": .3, "frac_gap_below_2": .5,
                          "max_rayleigh_change": float(arrays["rayleigh_change"][:, i].max()), **means})
        progress = []
        for s in range(0, len(cells), chunk):
            progress.extend({"recorded": k, "of": 1500, "seconds": 1.} for k in (500, 1000, 1500))
            end = min(s + chunk, len(cells))
            progress.append({"done": end, "of": len(cells), "chaotic": sum(c["chaotic"] for c in cells[s:end]), "seconds": 3.})
        data = {"args": args, "cells": cells, "bias": None, "seconds": 3.}
        with (output / (name + "-cells.jsonl")).open("x") as stream:
            summaries[name] = M.audit_map(data, arrays, args, progress,
                                          lambda record: stream.write(json.dumps(record, allow_nan=False) + "\n"))
        (alpha if name[0] == "a" else bias).extend(cells)
        del arrays, data, cells
        gc.collect()
    map_seconds = time.perf_counter() - start
    xo, selection = fixtures.crossover()
    statistics_start = time.perf_counter()
    result = S.combined_statistics(alpha, bias, xo, selection, 10000)
    statistics_seconds = time.perf_counter() - statistics_start
    serialized = json.dumps({"maps": summaries, "statistics": result}, allow_nan=False)
    (output / "synthetic-statistics.json").write_text(serialized + "\n")
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(output.iterdir()) if p.is_file()]
    S.require(sum(p["bytes"] for p in files) <= 128 * 1024 * 1024, "synthetic retained output cap")
    receipt = {"scope": "Synthetic full shape, all 53 rows eligible for all eight contrasts; no target byte decoding/model execution",
               "map_cells": [len(alpha), len(bias)], "XO_cells": len(xo["cells"]), "draws": 10000,
               "map_generation_and_ledger_seconds": map_seconds, "statistics_seconds": statistics_seconds,
               "elapsed_through_serialization_and_output_hash_seconds": time.perf_counter() - start,
               "peak_rss_observed_before_final_receipt_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "files": files, "total_retained_bytes_before_receipt": sum(p["bytes"] for p in files),
               "I_O_limit": "Includes synthetic ledger/statistics writing, not target archive hashing or compressed NPZ decompression; original metadata IO separately measured."}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
