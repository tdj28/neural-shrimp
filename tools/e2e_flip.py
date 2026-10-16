"""Run E2E: locate a flip between two records of a saved branch (tools/e2e_branches.py output) by the secant
iteration of tools/e2e_band.refine_flip (real crossing multiplier only; AD multipliers on both sides).

Usage: python tools/e2e_flip.py --branch p2-tcont-a0.070-down --i 4 --j 5 --period 2 --fixed alpha --value 0.070 --out NAME
       (indices into the branch's records; record i stable with mu > -1, record j with mu < -1)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_band as ebd  # noqa: E402
import e2e_model as em  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--branch2", default=None, help="record j from this branch instead")
    ap.add_argument("--i", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--fixed", choices=("alpha", "log_t"), required=True)
    ap.add_argument("--value", type=float, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = em.Model(args.threads)
    b1 = em.OUT / args.branch
    b2 = em.OUT / (args.branch2 or args.branch)
    r1 = json.loads((b1 / "branch.json").read_text())["records"][args.i]
    r2 = json.loads((b2 / "branch.json").read_text())["records"][args.j]
    s1 = np.load(b1 / "branch-states.npy")[args.i]
    s2 = np.load(b2 / "branch-states.npy")[args.j]
    res = ebd.refine_flip(model, args.fixed, args.value, args.period, r1, s1, r2, s2, iters=8, tol=1e-7)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "flip.json").write_text(json.dumps({"args": vars(args), **res}, indent=1))
    print(json.dumps({"flip": res["lambda"], "sides": [s["classified"] for s in res["sides"]]}))


if __name__ == "__main__":
    main()
