"""Independent saved-evidence audit of a completed EXP-029 attempt.

Imports neither the batched autonomous engine nor the analysis labels. For a
seeded stratified sample of computed cells in every job, it re-integrates the
network with scipy's adaptive DOP853 and solver event detection, screens the
event states with separately written code, estimates the finite-time exponent
from a separately integrated variational system, recomputes the label and
compares it with the saved arrays; for sampled periodic cells it recomputes
Floquet multipliers from the saved event state and cycle time. Disagreements
are reported, never repaired.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matrix(alpha, gamma):
    return np.array([[1.5, 2.9, alpha], [-1.5, 1.18, 0.0], [gamma, -22.0, 0.47]])


def saved_label(saved, r, c, threshold):
    if bool(saved["diverged"][r, c]):
        return -1
    p = int(saved["period"][r, c])
    if p >= 1:
        return 1
    if bool(saved["equilibrium"][r, c]):
        return 0
    return 2 if saved["ftle"][r, c] > threshold else 3


def screen(events, max_period, repeats, atol, rtol):
    for p in range(1, max_period + 1):
        need = repeats * p + 1
        if len(events) < need:
            break
        seq = events[-need:]
        if np.all(np.abs(seq[p:] - seq[:-p]) <= atol + rtol * np.maximum(np.abs(seq[p:]), np.abs(seq[:-p]))):
            return p
    return 0


def audit_cell(w, x0, it, solver, threshold):
    def rhs(_t, x):
        return -x + w @ np.tanh(x)

    def peak(_t, x):
        return rhs(_t, x)[2]
    peak.direction = -1.0
    total = it["settle"] + it["measure"]
    settle = solve_ivp(rhs, (0.0, it["settle"]), x0, method="DOP853", rtol=solver["rtol"],
                       atol=solver["atol"])
    xs = settle.y[:, -1]
    run = solve_ivp(rhs, (it["settle"], total), xs, method="DOP853", rtol=solver["rtol"],
                    atol=solver["atol"], events=peak)
    events = run.y_events[0] if run.y_events and len(run.y_events[0]) else np.zeros((0, 3))
    period = screen(events, it["max_period"], it["repeats"], it["atol"], it["rtol"])
    x_end = run.y[:, -1]
    field = float(np.linalg.norm(rhs(0.0, x_end)))

    def variational(_t, z):
        x, v = z[:3], z[3:]
        s = 1.0 - np.tanh(x) ** 2
        return np.concatenate((-x + w @ np.tanh(x), -v + w @ (s * v)))

    v = np.random.default_rng(1).normal(size=3)
    v /= np.linalg.norm(v)
    z = np.concatenate((xs, v))
    growth, elapsed, t = 0.0, 0.0, it["settle"]
    segment = 10.0
    while t < total - 1e-9:
        h = min(segment, total - t)
        sol = solve_ivp(variational, (t, t + h), z, method="DOP853", rtol=solver["rtol"], atol=solver["atol"])
        z = sol.y[:, -1]
        norm = float(np.linalg.norm(z[3:]))
        if t >= it["settle"] + it["ftle_align"] - 1e-9:
            growth += math.log(norm)
            elapsed += h
        z[3:] /= norm
        t += h
    ftle = growth / elapsed if elapsed > 0 else float("nan")
    if period >= 1:
        label = 1
    elif len(events) == 0 and field < it["equilibrium_tolerance"]:
        label = 0
    elif ftle > threshold:
        label = 2
    else:
        label = 3
    return {"label": label, "period": period, "ftle": ftle, "events": int(len(events)), "field": field}


def audit_floquet(w, start, duration, solver):
    def system(_t, z):
        x, m = z[:3], z[3:].reshape(3, 3)
        s = 1.0 - np.tanh(x) ** 2
        return np.concatenate((-x + w @ np.tanh(x), (-m + w @ (s[:, None] * m)).ravel()))
    z0 = np.concatenate((start, np.eye(3).ravel()))
    sol = solve_ivp(system, (0.0, duration), z0, method="DOP853", rtol=solver["rtol"], atol=solver["atol"])
    mult = np.linalg.eigvals(sol.y[3:, -1].reshape(3, 3))
    neutral = int(np.argmin(np.abs(mult - 1.0)))
    others = np.delete(np.abs(mult), neutral)
    return float(np.max(others)), float(abs(mult[neutral] - 1.0))


def saved_transverse(saved, r, c):
    mult = saved["multiplier_real"][r, c] + 1j * saved["multiplier_imag"][r, c]
    neutral = int(np.argmin(np.abs(mult - 1.0)))
    return float(np.max(np.delete(np.abs(mult), neutral)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/track-b/EXP-029-plan.json"))
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    raw = Path(args.raw) if args.raw else (ROOT / plan["output"]["raw_root"] / plan["output"]["attempt"]).resolve()
    out = Path(args.out) if args.out else (ROOT / "../neural-shrimp-data/audits/EXP-029/validation-001").resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((raw / "receipt.json").read_text())
    mismatched = [f["path"] for f in receipt["files"] if sha256_file(raw / f["path"]) != f["sha256"]]
    it, solver = plan["integration"], plan["audit"]
    threshold = plan["labels"]["chaos_threshold_per_time"]
    report = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "receipt_mismatches": mismatched,
              "receipt_plan_hash_matches": receipt["plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest(),
              "jobs": {}}
    job_index = 0
    for spec in plan["maps"]:
        for start, state in spec["starts"].items():
            job = f"{spec['map_id']}-{start}"
            saved = np.load(raw / "maps" / f"{job}.npz")
            rows, cols = saved["period"].shape
            rng = np.random.default_rng(solver["audit_seed"] + job_index)
            job_index += 1
            computed = np.flatnonzero(saved["computed"].ravel())
            labels = np.array([saved_label(saved, *divmod(int(c), cols), threshold) for c in computed])
            sample = []
            for code in (-1, 0, 1, 2, 3):
                pool = computed[labels == code]
                take = min(solver["cells_per_label_per_job"], len(pool))
                if take:
                    sample += [int(c) for c in rng.choice(pool, size=take, replace=False)]
            rows_out = []
            for cell in sample:
                r, c = divmod(cell, cols)
                w = matrix(saved["alpha_axis"][c], saved["gamma_axis"][r])
                result = audit_cell(w, np.asarray(state, dtype=float), it, solver, threshold)
                entry = {"cell": cell, "saved_label": saved_label(saved, r, c, threshold),
                         "saved_period": int(saved["period"][r, c]), **result}
                if entry["saved_label"] == 1:
                    trans, neutral_error = audit_floquet(w, saved["last_event_state"][r, c],
                                                         float(saved["cycle_time"][r, c]), solver)
                    entry.update({"audit_transverse": trans, "saved_transverse": saved_transverse(saved, r, c),
                                  "audit_neutral_error": neutral_error})
                    entry["transverse_ok"] = abs(trans - entry["saved_transverse"]) <= solver["multiplier_tolerance"]
                rows_out.append(entry)
            regular = [e for e in rows_out if e["saved_label"] in (0, 1)]
            report["jobs"][job] = {
                "sampled": len(rows_out),
                "label_agreement": sum(e["label"] == e["saved_label"] for e in rows_out),
                "regular_label_and_period_agreement": sum(
                    e["label"] == e["saved_label"] and e["period"] == e["saved_period"] for e in regular),
                "regular_count": len(regular),
                "transverse_ok": sum(bool(e.get("transverse_ok")) for e in rows_out if e["saved_label"] == 1),
                "periodic_count": sum(e["saved_label"] == 1 for e in rows_out),
                "cells": rows_out}
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cells"}
                      for k, v in report["jobs"].items()}, indent=2))


if __name__ == "__main__":
    main()
