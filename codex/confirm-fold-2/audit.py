"""Audit of the Task 5b (PT2) passages, written and frozen with the design before any outcome.

For every row: the first strict exit u > U is reconstructed from the saved per-step trace (u after the last counted step
above U, every earlier one at most U), the count is compared with results.json, both frozen predictions are recomputed
from the design's coefficients, and the signed errors are recorded. Verdicts:
  primary   |N - T_arc| <= 1 on every row (unrounded continuous passage integral)
  secondary |N - N_quad| <= 1 on every row (discrete quadratic count)
  sign      N - N_quad <= 0 on every midpoint and full+transverse row (secondary hypothesis, not a criterion)
A not-run, failed or censored row passes nothing.

Usage: python codex/confirm-fold-2/audit.py --design codex/confirm-fold-2/design.json \
         --results C=/path/to/passage-C/run D=/path/to/passage-D/run --json OUT.json --markdown OUT.md
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quad_count(u, a, b, eps, U, cap=100000):
    for i in range(1, cap + 1):
        u = u + a * eps + b * u * u
        if u > U:
            return i
    return None


def arctan_count(u, a, b, eps, U):
    s = math.sqrt(b / (a * eps))
    return (math.atan(U * s) - math.atan(u * s)) / math.sqrt(a * b * eps)


def audit_target(t, run):
    out = {"label": t["label"], "status": t["status"], "rows": []}
    if t["status"] != "ready":
        out["rows"] = [dict(id=r["id"], status="not_run_setup_gate", primary_pass=False, secondary_pass=False) for r in t["rows"]]
        return out
    res = json.loads((Path(run) / "results.json").read_text())
    out["results_sha256"] = sha(Path(run) / "results.json")
    a, b, U = t["a_positive"], t["b_positive"], t["U"]
    by_id = {r["id"]: r for r in res["rows"]}
    for g in range(4):
        trace_file = Path(run) / f"group-{g}.npz"
        u_trace = np.load(trace_file)["u"] if trace_file.exists() else None
        for r in [r for r in t["rows"] if r["group"] == g]:
            rr = by_id[r["id"]]
            row = dict(id=r["id"], entrance=r["entrance"], q=r["q"], eps=r["eps"], status=rr["status"])
            n = rr.get("measured_G")
            qc, ac = quad_count(r["u_in"], a, b, r["eps"], U), arctan_count(r["u_in"], a, b, r["eps"], U)
            row.update(quadratic_G=qc, arctan_G=ac, frozen_quadratic_G=r["quadratic_G"], frozen_arctan_G=r["arctan_G"],
                       predictions_reproduced=bool(qc == r["quadratic_G"] and abs(ac - r["arctan_G"]) < 1e-9))
            if rr["status"] != "observed" or n is None or u_trace is None:
                row.update(measured_G=None, primary_pass=False, secondary_pass=False, crossing_verified=False)
            else:
                j = r["entrance_index"]
                col = u_trace[:n, j]
                verified = bool(col[-1] > U and np.all(col[:-1] <= U))
                row.update(measured_G=int(n), arctan_error=n - ac, quadratic_error=n - qc, crossing_verified=verified,
                           u_before_exit=float(col[-2]) if n > 1 else None, u_at_exit=float(col[-1]),
                           primary_pass=bool(verified and abs(n - ac) <= 1), secondary_pass=bool(verified and abs(n - qc) <= 1))
            out["rows"].append(row)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", required=True)
    p.add_argument("--results", nargs="+", required=True, help="LABEL=passage run folder (holding results.json)")
    p.add_argument("--json", required=True)
    p.add_argument("--markdown", required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.design).read_text())
    runs = dict(s.split("=", 1) for s in args.results)
    targets = [audit_target(t, runs.get(t["label"])) for t in plan["targets"]]
    rows = [r for t in targets for r in t["rows"]]
    sign_rows = [r for r in rows if r.get("entrance") in ("midpoint", "full+transverse") and r.get("measured_G") is not None]
    summary = dict(design_sha256=sha(args.design), rows=len(rows),
                   primary_all_pass=all(r["primary_pass"] for r in rows),
                   primary_passes=sum(r["primary_pass"] for r in rows),
                   secondary_all_pass=all(r["secondary_pass"] for r in rows),
                   secondary_passes=sum(r["secondary_pass"] for r in rows),
                   sign_hypothesis_holds=bool(sign_rows) and all(r["quadratic_error"] <= 0 for r in sign_rows),
                   sign_rows=len(sign_rows), sign_violations=[r["id"] for r in sign_rows if r["quadratic_error"] > 0],
                   all_crossings_verified=all(r.get("crossing_verified", False) for r in rows),
                   all_predictions_reproduced=all(r.get("predictions_reproduced", False) for r in rows),
                   smallest_frozen_arctan=min(r["frozen_arctan_G"] for r in rows if "frozen_arctan_G" in r),
                   max_abs_arctan_error=max((abs(r["arctan_error"]) for r in rows if "arctan_error" in r), default=None))
    Path(args.json).write_text(json.dumps(dict(summary=summary, targets=targets), indent=1) + "\n")
    lines = ["| row | q | eps | measured | T_arc (primary) | error | discrete (secondary) | error |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("measured_G") is None:
            lines.append(f"| {r['id']} | | | not observed | | | | |")
            continue
        lines.append(f"| {r['id']} | {r['q']} | {r['eps']:.3e} | {r['measured_G']} | {r['arctan_G']:.3f} | {r['arctan_error']:+.3f} | "
                     f"{r['quadratic_G']} | {r['quadratic_error']:+d} |")
    Path(args.markdown).write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
