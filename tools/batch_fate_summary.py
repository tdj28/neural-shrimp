"""Summary of a batch_fate.py run (run BF): per layout the captures, the agreement between layouts start by start,
the capture-step distributions, how far the layouts' trajectories have separated at each recorded step, and the
determinism controls (reruns whose recorded states must equal the main run's bitwise).

Usage: python tools/batch_fate_summary.py --run llm-soft-loop/batch-fate/bf1 \
           --control llm-soft-loop/batch-fate/bf1-repeat64 --control llm-soft-loop/batch-fate/bf1-repeat1 \
           --out docs/batch-fate/bf1-receipt.json
"""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_soft_loop as esl  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(run):
    d = esl.DATA / "explore" / run
    rec = json.loads((d / "fate.json").read_text())
    z = np.load(d / "fate.npz", allow_pickle=True)
    lays = {}
    for lab in rec["labels"]:
        cyc_tok = json.loads(str(z[f"{lab}_cyc_tok"][0]))
        states = {int(k.rsplit("_", 1)[1]): z[k] for k in z.files if k.startswith(f"{lab}_cyc_state_")}
        lays[lab] = {"cap": z[f"{lab}_cap"], "per": z[f"{lab}_per"], "rec": z[f"{lab}_rec"],
                     "mind": z[f"{lab}_mind"], "cyc_tok": {int(k): v for k, v in cyc_tok.items()}, "cyc": states}
    return d, rec, z, lays


def orbit_ids(lays, tol=1e-6):
    """Cluster every captured cycle by period and point set (same orbit at any phase)."""
    reps, ids = [], {}
    for lab, L in lays.items():
        for s, pts in sorted(L["cyc"].items()):
            for k, (q, rp) in enumerate(reps):
                if q == len(pts):
                    dist = np.min(np.linalg.norm(rp[:, None] - pts[None], axis=-1)) / np.linalg.norm(rp[0])
                    if dist < tol:
                        ids[(lab, s)] = k
                        break
            else:
                reps.append((len(pts), pts))
                ids[(lab, s)] = len(reps) - 1
    return reps, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--control", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d, rec, z, lays = load(args.run)
    n = len(z["starts"])
    steps = rec["args"]["steps"]
    every = rec["args"]["record_every"]
    reps, ids = orbit_ids(lays)
    orbits = [{"orbit": k, "period": int(q), "tokens_first_capture": next(
        (lays[lab]["cyc_tok"][s] for (lab, s), kk in ids.items() if kk == k), None)} for k, (q, _) in enumerate(reps)]
    per_layout = {}
    for lab, L in lays.items():
        cap = L["cap"]
        caught = cap >= 0
        surv = {str(t): int(np.sum(~caught | (cap > t))) for t in (100, 300, 500, 1000, 2000, 3000, 4000, steps)}
        per_layout[lab] = {
            "captured": int(caught.sum()), "not_captured": int((~caught).sum()),
            "by_orbit": {str(k): int(sum(1 for s in range(n) if ids.get((lab, s)) == k)) for k in range(len(reps))},
            "capture_steps": {str(s): int(cap[s]) for s in np.nonzero(caught)[0]},
            "uncaptured_after_steps": surv,
            "uncaptured_min_recurrence_distance": (float(np.nanmin(L["mind"][~caught])) if (~caught).any() else None)}
    agreement = {}
    for a, b in combinations(lays, 2):
        ca, cb = lays[a]["cap"] >= 0, lays[b]["cap"] >= 0
        both = np.nonzero(ca & cb)[0]
        same_orbit = int(sum(1 for s in both if ids[(a, s)] == ids[(b, s)]))
        same_step = int(sum(1 for s in both if lays[a]["cap"][s] == lays[b]["cap"][s]))
        agreement[f"{a}~{b}"] = {"both": int(len(both)), "both_same_orbit": same_orbit, "both_same_step": same_step,
                                 f"only_{a}": int(np.sum(ca & ~cb)), f"only_{b}": int(np.sum(~ca & cb)),
                                 "neither": int(np.sum(~ca & ~cb)),
                                 "starts_only_one": [int(s) for s in np.nonzero(ca ^ cb)[0]]}
    # separation of the layouts' trajectories at each recorded step (rows still running in both layouts)
    separation = {}
    for a, b in combinations(lays, 2):
        ra, rb = lays[a]["rec"], lays[b]["rec"]
        m = min(len(ra), len(rb))
        out = {}
        for i in range(1, m):
            ok = ~np.isnan(ra[i, :, 0]) & ~np.isnan(rb[i, :, 0])
            if ok.sum() == 0:
                continue
            rel = np.linalg.norm(ra[i, ok] - rb[i, ok], axis=1) / np.linalg.norm(ra[i, ok], axis=1)
            out[str(i * every)] = {"rows": int(ok.sum()), "median": float(np.median(rel)), "max": float(rel.max()),
                                   "min": float(rel.min())}
        separation[f"{a}~{b}"] = out
    controls = []
    for c in args.control:
        cd, crec, cz, clays = load(c)
        for lab, L in clays.items():
            base = lab.split("-run")[0]
            M = lays[base]["rec"]
            R = L["rec"]
            k = min(len(R), len(M))
            rows = R.shape[1]
            eq = bool(np.array_equal(R[:k], M[:k, :rows], equal_nan=True))
            controls.append({"run": c, "layout": lab, "steps": crec["args"]["steps"], "rows": int(rows),
                             "starts_sha256_match": crec["starts_sha256"][:16] == rec["starts_sha256"][:16]
                             if crec["args"]["starts"] == rec["args"]["starts"] else None,
                             "recorded_states_bitwise_equal": eq})
    receipt = {"run": args.run, "args": rec["args"], "labels": rec["labels"], "starts_sha256": rec["starts_sha256"],
               "fate_npz_sha256": sha256(d / "fate.npz"), "fate_json_sha256": sha256(d / "fate.json"),
               "orbits": orbits, "per_layout": per_layout, "agreement": agreement,
               "separation_between_layouts": separation, "controls": controls,
               "seconds_per_layout": {lab: rec["summary"][lab]["seconds"] for lab in rec["labels"]}}
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("orbits", "per_layout", "agreement", "controls")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
