"""Stitch two_block_map.py bands into one map (run HP5, 19 Oct 2026: a 512 x 512 close-up of the period-5 window split
across pods by rows of alpha2).

Each band folder holds tb-t<log10T>.npz (alpha1, alpha2, log10_temperature, lyapunov, period, chaotic) and its .json.
The stitch refuses unless every band has the same alpha1 grid and log10 T, the bands' alpha2 rows follow each other
in order, and all alpha2 steps are equal to within --rtol. It writes the stacked arrays and a record of every band
(path, sha256, rows, seconds).

Usage: python tools/stitch_tb_bands.py --bands llm-soft-loop/pod64/tb/hp5-c00 ... --out llm-soft-loop/pod64/hp5
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bands", nargs="+", required=True, help="explore-relative band folders in row order")
    ap.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance on equal alpha2 steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parts, record = [], []
    for b in args.bands:
        files = sorted((DATA / "explore" / b).glob("tb-t*.npz"))
        if len(files) != 1:
            raise SystemExit(f"{b}: expected one tb-t*.npz, found {len(files)}")
        f = files[0]
        z = dict(np.load(f))
        meta = json.loads(f.with_suffix(".json").read_text())
        parts.append(z)
        record.append({"band": b, "file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                       "rows": int(len(z["alpha2"])), "seconds": meta.get("seconds")})
    a1 = parts[0]["alpha1"]
    t = float(parts[0]["log10_temperature"])
    for p, r in zip(parts, record):
        if not np.array_equal(p["alpha1"], a1):
            raise SystemExit(f"{r['band']}: alpha1 grid differs from the first band's")
        if float(p["log10_temperature"]) != t:
            raise SystemExit(f"{r['band']}: log10 T {float(p['log10_temperature'])} differs from {t}")
    a2 = np.concatenate([p["alpha2"] for p in parts])
    steps = np.diff(a2)
    if not (np.all(steps > 0) and np.allclose(steps, steps[0], rtol=args.rtol, atol=0)):
        raise SystemExit(f"alpha2 rows are not ordered with equal steps (steps {steps.min():.6g} to {steps.max():.6g})")
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"alpha1": a1, "alpha2": a2, "log10_temperature": np.array(t)}
    for k in ("lyapunov", "period", "chaotic"):
        arrays[k] = np.concatenate([p[k] for p in parts], axis=0)
    path = out / "map.npz"
    np.savez_compressed(path, **arrays)
    summary = {"bands": record, "shape": list(arrays["lyapunov"].shape), "alpha1": [float(a1[0]), float(a1[-1])],
               "alpha2": [float(a2[0]), float(a2[-1])], "log10_temperature": t,
               "alpha2_step_range": [float(steps.min()), float(steps.max())],
               "chaotic_share": float(arrays["chaotic"].mean()),
               "map_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "map.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("shape", "alpha1", "alpha2", "chaotic_share", "map_sha256")}))


if __name__ == "__main__":
    main()
