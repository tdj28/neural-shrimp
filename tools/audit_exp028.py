"""Independent saved-evidence audit of a completed EXP-028 attempt.

This file deliberately imports neither the batched engine nor the selection
helpers. It rewrites the two-layer training map for one cell, the tail
screen, the zoom rule, the component selection (via scipy.ndimage) and the
orbit closure and multiplier checks (finite-difference Jacobians). It reads
saved evidence only; disagreements are reported, never repaired.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
STATUS = {"divergent": 0, "fixed": 1, "periodic": 2, "chaotic": 3, "unresolved": 4}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(array):
    array = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    h = hashlib.sha256()
    h.update(repr(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


class CellMap:
    """One-cell gradient descent written independently of the batched engine."""

    def __init__(self, network, x, y, lr_hidden, lr_output):
        self.d, self.width = network["d"], network["width"]
        self.bias, self.ridge = network["bias"], network["ridge"]
        self.a_in, self.a_out = network["in_scale"], network["out_scale"]
        self.k = 2.0 * network["loss_scale"] / len(x)
        self.loss_scale = network["loss_scale"]
        self.x, self.y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n_hidden = self.d * self.width + (self.width if self.bias else 0)
        n_total = n_hidden + self.width + (1 if self.bias else 0)
        self.rates = np.r_[np.full(n_hidden, lr_hidden), np.full(n_total - n_hidden, lr_output)]
        self.n = n_total

    def split(self, theta):
        dw = self.d * self.width
        w0 = theta[:dw].reshape(self.d, self.width)
        offset = dw
        b0 = theta[offset:offset + self.width] if self.bias else np.zeros(self.width)
        offset += self.width if self.bias else 0
        w1 = theta[offset:offset + self.width]
        b1 = theta[offset + self.width] if self.bias else 0.0
        return w0, b0, w1, b1

    def loss_and_gradient(self, theta):
        w0, b0, w1, b1 = self.split(theta)
        z = self.a_in * (self.x @ w0) + b0
        h = np.tanh(z)
        r = self.a_out * (h @ w1) + b1 - self.y
        loss = self.loss_scale * float(np.mean(r * r)) + 0.5 * self.ridge * float(theta @ theta)
        e = self.k * r
        g_w1 = self.a_out * (h.T @ e)
        back = np.outer(e, self.a_out * w1) * (1.0 - h * h)
        g_w0 = self.a_in * (self.x.T @ back)
        parts = [g_w0.ravel()]
        if self.bias:
            parts.append(back.sum(axis=0))
        parts.append(g_w1)
        if self.bias:
            parts.append([e.sum()])
        return loss, np.concatenate(parts) + self.ridge * theta

    def __call__(self, theta):
        return theta - self.rates * self.loss_and_gradient(theta)[1]

    def jvp(self, theta, v):
        h = 1e-7 * max(1.0, float(np.max(np.abs(theta))))
        return (self(theta + h * v) - self(theta - h * v)) / (2.0 * h)

    def jacobian(self, theta):
        return np.column_stack([self.jvp(theta, e) for e in np.eye(self.n)])


def screen(tail, max_period, repeats, atol, rtol):
    if not np.all(np.isfinite(tail)):
        return -1
    for p in range(1, max_period + 1):
        span = p * repeats
        if span > len(tail):
            break
        window = tail[-span:]
        last = window[-p:]
        ok = True
        for j in range(repeats - 1):
            block = window[j * p:(j + 1) * p]
            if np.any(np.abs(block - last) > atol + rtol * np.maximum(np.abs(block), np.abs(last))):
                ok = False
                break
        if ok:
            return p
    return 0


def resimulate(cell_map, theta0, it):
    theta = np.array(theta0, dtype=np.float64)
    tail = []
    rng = np.random.default_rng(12345)
    v = rng.normal(size=cell_map.n)
    v /= np.linalg.norm(v)
    growth = 0.0
    start = it["steps"] - it["ftle_steps"] - it["ftle_align"]
    for step in range(it["steps"]):
        loss, grad = cell_map.loss_and_gradient(theta)
        new = theta - cell_map.rates * grad
        if (not np.isfinite(loss) or not np.all(np.isfinite(new))
                or np.max(np.abs(new)) > it["divergence_threshold"]):
            return {"status": STATUS["divergent"], "period": -1, "diverged_at": step,
                    "final_state": None, "ftle": None}
        if step >= start:
            w = cell_map.jvp(theta, v)
            norm = float(np.linalg.norm(w))
            if step >= start + it["ftle_align"] and norm > 0 and math.isfinite(norm):
                growth += math.log(norm)
            if norm > 0 and math.isfinite(norm):
                v = w / norm
        theta = new
        if step >= it["steps"] - it["tail_steps"]:
            tail.append(theta.copy())
    period = screen(np.asarray(tail), it["max_period"], it["min_repeats"], it["atol"], it["rtol"])
    ftle = growth / it["ftle_steps"]
    if period == 1:
        status = STATUS["fixed"]
    elif period >= 2:
        status = STATUS["periodic"]
    elif ftle > it["chaos_threshold"]:
        status = STATUS["chaotic"]
    else:
        status = STATUS["unresolved"]
    return {"status": status, "period": period, "diverged_at": -1, "final_state": theta, "ftle": ftle}


def build_inputs(spec):
    """Regenerate inputs independently for S1 and via the frozen recipe for S2."""
    if spec["system"] == "S1":
        width = spec["network"]["width"]
        streams = np.random.SeedSequence(int(spec["seed"])).spawn(3)
        n = spec["n_parameters"]
        x = np.random.default_rng(streams[0]).normal(size=(n, width))
        y = np.random.default_rng(streams[1]).normal(size=n)
        theta0 = np.random.default_rng(streams[2]).normal(size=n)
    else:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from neural_shrimp.models import TanhRegressor, make_teacher_regression
        data = make_teacher_regression(seed=spec["seed"], d=2, teacher_width=3, n_train=24,
                                       n_validation=48, n_test=96)
        x, y = data.train_x, data.train_y
        theta0 = TanhRegressor(2, 3).initialize(seed=spec["seed"] + 100000, scale=0.2)
    hashes = {"x": digest(x), "y": digest(y), "theta0": digest(theta0)}
    return x, y, theta0, hashes == spec["data_sha256"]


def select_audit_cells(status, count, seed):
    rng = np.random.default_rng(seed)
    flat = status.ravel()
    chosen = []
    per_class = count // len(STATUS)
    for code in STATUS.values():
        cells = np.flatnonzero(flat == code)
        take = min(per_class, len(cells))
        if take:
            chosen += [int(c) for c in rng.choice(cells, size=take, replace=False)]
    rest = np.setdiff1d(np.arange(flat.size), chosen)
    fill = count - len(chosen)
    if fill > 0:
        chosen += [int(c) for c in rng.choice(rest, size=min(fill, len(rest)), replace=False)]
    return chosen


def zoom_rule(status, period, window, stride):
    interest = ((status == STATUS["periodic"]) & (period >= 2)) | (status == STATUS["chaotic"])
    padded = np.pad(np.cumsum(np.cumsum(interest.astype(np.int64), 0), 1), ((1, 0), (1, 0)))
    best = None
    for r0 in range(0, status.shape[0] - window + 1, stride):
        for c0 in range(0, status.shape[1] - window + 1, stride):
            r1, c1 = r0 + window, c0 + window
            count = int(padded[r1, c1] - padded[r0, c1] - padded[r1, c0] + padded[r0, c0])
            if best is None or count > best[2]:
                best = (r0, c0, count)
    return None if best is None or best[2] == 0 else best


def components(status, period, fixed, min_size, limit):
    mask = (status == STATUS["fixed"]) if fixed else ((status == STATUS["periodic"]) & (period >= 2))
    chaos_near = ndimage.binary_dilation(status == STATUS["chaotic"], structure=np.ones((3, 3)))
    records = []
    for value in np.unique(period[mask]):
        labels, count = ndimage.label(mask & (period == value), structure=np.ones((3, 3)))
        for label in range(1, count + 1):
            members = np.flatnonzero(labels.ravel() == label)
            if len(members) < min_size:
                continue
            rows, cols = np.divmod(members, status.shape[1])
            dist = (rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2
            best = members[np.lexsort((members, dist))[0]]
            records.append({"period": int(value), "size": len(members), "first": int(members.min()),
                            "representative": int(best),
                            "borders_chaos": bool(chaos_near.ravel()[members].any())})
    records.sort(key=lambda r: (-r["size"], r["first"]))
    return records[:limit]


def recheck_certificate(record, spec, x, y):
    cert = record["certificate"]
    if not cert.get("verified"):
        return {"checked": False, "reason": "not verified by producer"}
    cell_map = CellMap(spec["network"], x, y, record["lr_hidden"], record["lr_output"])
    orbit = [np.asarray(p, dtype=np.float64) for p in cert["orbit"]]
    p = len(orbit)
    state = orbit[0].copy()
    for _ in range(p):
        state = cell_map(state)
    closure = float(np.max(np.abs(state - orbit[0]) / (1e-9 + 1e-8 * np.maximum(np.abs(state), np.abs(orbit[0])))))
    monodromy = np.eye(cell_map.n)
    for point in orbit:
        monodromy = cell_map.jacobian(point) @ monodromy
    radius = float(np.max(np.abs(np.linalg.eigvals(monodromy))))
    proper = []
    for divisor in range(1, p):
        if p % divisor == 0:
            proper.append(float(np.max(np.abs(orbit[divisor] - orbit[0]))))
    return {"checked": True, "normalized_closure": closure, "closure_ok": closure <= 1.0,
            "fd_spectral_radius": radius, "producer_spectral_radius": cert["spectral_radius"],
            "radius_difference": abs(radius - cert["spectral_radius"]),
            "radius_ok": abs(radius - cert["spectral_radius"]) <= 1e-5 + 1e-4 * cert["spectral_radius"],
            "attracting_consistent": (radius < 1.0) == bool(cert["attracting"]),
            "min_proper_divisor_separation": min(proper) if proper else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(ROOT / "experiments/trainability/EXP-028-plan.json"))
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    raw = Path(args.raw) if args.raw else (ROOT / plan["output"]["raw_root"] / plan["output"]["attempt"]).resolve()
    out = Path(args.out) if args.out else (ROOT / "../neural-shrimp-data/audits/EXP-028/validation-001").resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "raw": str(raw), "checks": {}}
    receipt = json.loads((raw / "receipt.json").read_text())
    mismatched = [f["path"] for f in receipt["files"] if sha256_file(raw / f["path"]) != f["sha256"]]
    report["checks"]["receipt"] = {"files": len(receipt["files"]), "mismatched": mismatched,
                                   "plan_hash_matches": receipt["plan_sha256"] == report["plan_sha256"]}
    it = plan["iteration"]
    cert_rule = plan["certification"]
    certificates = json.loads((raw / "stage3" / "certificates.json").read_text())
    by_map = {}
    for record in certificates:
        by_map.setdefault(record["map_id"], []).append(record)
    report["checks"]["maps"] = {}
    for spec in plan["maps"]:
        x, y, theta0, inputs_ok = build_inputs(spec)
        for suffix, stage in (("", "stage1"), ("-zoom", "stage2")):
            map_id = spec["map_id"] + suffix
            path = raw / stage / f"{map_id}.npz"
            if not path.exists():
                report["checks"]["maps"][map_id] = {"present": False}
                continue
            saved = np.load(path)
            status, period = saved["status"], saved["period"]
            entry = {"present": True, "inputs_hash_ok": inputs_ok}
            if suffix == "":
                z = plan["zoom"]
                recomputed = zoom_rule(status, period, z["window"], z["stride"])
                saved_zoom = json.loads((raw / "stage1" / f"{map_id}-zoom.json").read_text())["window"]
                entry["zoom_rule_ok"] = (recomputed is None and saved_zoom is None) or (
                    recomputed is not None and saved_zoom is not None and
                    [saved_zoom["row0"], saved_zoom["col0"], saved_zoom["count"]] == list(recomputed))
                cells = select_audit_cells(status, plan["audit"]["resimulated_cells_per_coarse_map"],
                                           plan["audit"]["audit_seed"] + spec["map_index"])
                rows = []
                cols_n = status.shape[1]
                for cell in cells:
                    r, c = divmod(cell, cols_n)
                    lh, lo = 10.0 ** saved["log10_hidden"][r], 10.0 ** saved["log10_output"][c]
                    result = resimulate(CellMap(spec["network"], x, y, lh, lo), theta0, it)
                    saved_status, saved_period = int(status[r, c]), int(period[r, c])
                    state_diff = None
                    if result["final_state"] is not None and saved_status in (1, 2):
                        ref = saved["final_state"][r, c]
                        state_diff = float(np.max(np.abs(result["final_state"] - ref) /
                                                  (1e-12 + np.abs(ref))))
                    rows.append({"cell": cell, "saved_status": saved_status, "audit_status": result["status"],
                                 "saved_period": saved_period, "audit_period": result["period"],
                                 "saved_ftle": None if not np.isfinite(saved["ftle"][r, c]) else float(saved["ftle"][r, c]),
                                 "audit_ftle": result["ftle"], "relative_state_difference": state_diff})
                regular = [row for row in rows if row["saved_status"] in (0, 1, 2)]
                irregular = [row for row in rows if row["saved_status"] in (3, 4)]
                entry["resimulation"] = {
                    "cells": rows,
                    "regular_label_agreement": sum(r["saved_status"] == r["audit_status"] and
                                                   r["saved_period"] == r["audit_period"] for r in regular),
                    "regular_count": len(regular),
                    "irregular_bounded_agreement": sum(r["audit_status"] in (3, 4) for r in irregular),
                    "irregular_exact_agreement": sum(r["saved_status"] == r["audit_status"] for r in irregular),
                    "irregular_count": len(irregular)}
            expected = {"periodic": components(status, period, False, cert_rule["min_component_size"],
                                               cert_rule["max_components"]),
                        "fixed": components(status, period, True, cert_rule["min_component_size"],
                                            cert_rule["max_fixed_components"])}
            produced = by_map.get(map_id, [])
            selection_ok = True
            for kind in ("periodic", "fixed"):
                mine = [(r["period"], r["size"], r["representative"], r["borders_chaos"]) for r in expected[kind]]
                theirs = [(r["period"], r["size"], r["representative"], r["borders_chaos"])
                          for r in produced if r["kind"] == kind]
                selection_ok &= mine == theirs
            seeds_ok = all(r["seed_state_sha256"] == digest(
                saved["final_state"].reshape(-1, saved["final_state"].shape[-1])[r["representative"]])
                for r in produced)
            entry["selection_ok"] = bool(selection_ok)
            entry["certificate_seeds_ok"] = bool(seeds_ok)
            entry["certificate_rechecks"] = [{"period": r["period"], "kind": r["kind"],
                                              **recheck_certificate(r, spec, x, y)} for r in produced]
            report["checks"]["maps"][map_id] = entry
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"out": str(out / "audit.json"), "receipt_mismatches": len(mismatched)}, indent=2))


if __name__ == "__main__":
    main()
