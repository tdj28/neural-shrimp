"""Independent saved-evidence audit of a completed EXP-028 attempt.

This file imports neither the batched engine, the orbit certifier nor the
selection helpers. It rewrites the one-cell training map, the full-tail
screen, the zoom rule and the component and ring rules (via scipy.ndimage),
recomputes rates from the saved axes, rechecks each certificate's closure,
primitivity and spectral radius with finite-difference Jacobians, and
re-simulates every ring-confirmation cell of each producer-qualifying
component. It then recomputes the stage-A1 gate on its own evidence. Under
the frozen protocol the gate is positive only if this audit confirms it.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DIVERGENT, FIXED, PERIODIC, CHAOTIC, UNRESOLVED, DRIFT = range(6)
EIGHT = np.ones((3, 3), dtype=bool)


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
        self.loss_scale = network["loss_scale"]
        self.k = 2.0 * self.loss_scale / len(x)
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
        h = np.tanh(self.a_in * (self.x @ w0) + b0)
        r = self.a_out * (h @ w1) + b1 - self.y
        loss = self.loss_scale * float(np.mean(r * r)) + 0.5 * self.ridge * float(theta @ theta)
        e = self.k * r
        back = np.outer(e, self.a_out * w1) * (1.0 - h * h)
        parts = [(self.a_in * (self.x.T @ back)).ravel()]
        if self.bias:
            parts.append(back.sum(axis=0))
        parts.append(self.a_out * (h.T @ e))
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
        if p * repeats > len(tail):
            break
        a, b = tail[p:], tail[:-p]
        if np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))):
            return p
    return 0


def resimulate(cell_map, theta0, it, steps=None, ftle_steps=None, ftle_align=None, tail_steps=None):
    steps = steps or it["steps"]
    ftle_steps = ftle_steps or it["ftle_steps"]
    ftle_align = it["ftle_align"] if ftle_align is None else ftle_align
    tail_steps = tail_steps or it["tail_steps"]
    theta = np.array(theta0, dtype=np.float64)
    tail = []
    v = np.random.default_rng(12345).normal(size=cell_map.n)
    v /= np.linalg.norm(v)
    growth = 0.0
    start = steps - ftle_steps - ftle_align
    window_norm = None
    for step in range(steps):
        if step == steps - ftle_steps:
            window_norm = float(np.linalg.norm(theta))
        loss, grad = cell_map.loss_and_gradient(theta)
        new = theta - cell_map.rates * grad
        if (not np.isfinite(loss) or not np.all(np.isfinite(new))
                or np.max(np.abs(new)) > it["divergence_threshold"]):
            return {"status": DIVERGENT, "period": -1, "final_state": None, "ftle": None}
        if step >= start:
            w = cell_map.jvp(theta, v)
            norm = float(np.linalg.norm(w))
            if norm > 0 and math.isfinite(norm):
                if step >= start + ftle_align:
                    growth += math.log(norm)
                v = w / norm
        theta = new
        if step >= steps - tail_steps:
            tail.append(theta.copy())
    period = screen(np.asarray(tail), it["max_period"], it["min_repeats"], it["atol"], it["rtol"])
    ftle = growth / ftle_steps
    if period == 1:
        status = FIXED
    elif period >= 2:
        status = PERIODIC
    elif ftle > it["chaos_threshold"]:
        status = CHAOTIC
    elif float(np.linalg.norm(theta)) / window_norm - 1.0 > it["drift_threshold"]:
        status = DRIFT
    else:
        status = UNRESOLVED
    return {"status": status, "period": period, "final_state": theta, "ftle": ftle}


def build_inputs(spec):
    """Regenerate inputs independently for S1/S1R and via the frozen recipe for S2."""
    if spec["system"] in ("S1", "S1R"):
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
    per_class = count // 6
    for code in range(6):
        cells = np.flatnonzero(flat == code)
        take = min(per_class, len(cells))
        if take:
            chosen += [int(c) for c in rng.choice(cells, size=take, replace=False)]
    rest = np.setdiff1d(np.arange(flat.size), chosen)
    fill = count - len(chosen)
    if fill > 0:
        chosen += [int(c) for c in rng.choice(rest, size=min(fill, len(rest)), replace=False)]
    return chosen


def _has_true_neighbor(mask):
    """True where at least one of the 8 neighbors (not the cell) is True."""
    counts = ndimage.convolve(mask.astype(int), EIGHT.astype(int), mode="constant") - mask.astype(int)
    return counts > 0


def zoom_rule(status, period, window, stride):
    chaotic = status == CHAOTIC
    primary = (status == PERIODIC) & (period >= 3) & _has_true_neighbor(chaotic)
    secondary = chaotic & _has_true_neighbor(~chaotic & (status != DIVERGENT))
    best = None
    for r0 in range(0, status.shape[0] - window + 1, stride):
        for c0 in range(0, status.shape[1] - window + 1, stride):
            score = (int(primary[r0:r0 + window, c0:c0 + window].sum()),
                     int(secondary[r0:r0 + window, c0:c0 + window].sum()))
            if best is None or score > best[2:]:
                best = (r0, c0) + score
    return None if best is None or best[2:] == (0, 0) else best


def ring_of(members_mask):
    return _has_true_neighbor(members_mask) & ~members_mask


def recheck_certificate(record, spec, x, y, saved, cert_rule):
    cert = record["certificate"] or {}
    cols = saved["status"].shape[1]
    r, c = divmod(record["representative"], cols)
    lh, lo = 10.0 ** saved["log10_hidden"][r], 10.0 ** saved["log10_output"][c]
    rates_ok = (abs(lh - record["lr_hidden"]) <= 1e-15 * lh and
                abs(lo - record["lr_output"]) <= 1e-15 * lo)
    out = {"rates_from_axes_ok": bool(rates_ok)}
    if not cert.get("verified"):
        return {**out, "checked": False, "reason": "not verified by producer"}
    cell_map = CellMap(spec["network"], x, y, lh, lo)
    orbit = [np.asarray(p, dtype=np.float64) for p in cert["orbit"]]
    p = len(orbit)
    state = orbit[0].copy()
    for _ in range(p):
        state = cell_map(state)
    scale = cert_rule["atol"] + cert_rule["rtol"] * np.maximum(np.abs(state), np.abs(orbit[0]))
    closure = float(np.max(np.abs(state - orbit[0]) / scale))
    monodromy = np.eye(cell_map.n)
    for point in orbit:
        monodromy = cell_map.jacobian(point) @ monodromy
    radius = float(np.max(np.abs(np.linalg.eigvals(monodromy))))
    base = max(1.0, float(np.max(np.abs(orbit[0]))))
    separations = [float(np.max(np.abs(orbit[d] - orbit[0]))) / base
                   for d in range(1, p) if p % d == 0]
    margin = cert_rule["stability_margin"]
    return {**out, "checked": True, "normalized_closure": closure, "closure_ok": closure <= 1.0,
            "fd_spectral_radius": radius, "producer_spectral_radius": cert["spectral_radius"],
            "radius_difference": abs(radius - cert["spectral_radius"]),
            "radius_ok": abs(radius - cert["spectral_radius"]) <= 1e-6,
            "fd_attracting": radius < 1.0 - margin,
            "attracting_consistent": (radius < 1.0 - margin) == bool(cert["attracting"]),
            "min_divisor_separation": min(separations) if separations else None,
            "primitive_ok": (not separations) or min(separations) >= cert_rule["divisor_separation"]}


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
                                   "status": receipt["status"],
                                   "plan_hash_matches": receipt["plan_sha256"] == report["plan_sha256"]}
    it, cert_rule, gate_rule = plan["iteration"], plan["certification"], plan["gate"]
    ring_rule = plan["ring_confirmation"]
    certificates = json.loads((raw / "stage3" / "certificates.json").read_text())
    ring_results = json.loads((raw / "stage3" / "ring-confirmation.json").read_text())
    report["checks"]["maps"] = {}
    audit_qualifying = []
    for spec in plan["maps"]:
        x, y, theta0, inputs_ok = build_inputs(spec)
        suffixes = [("", "stage1")] + ([("-zoom", "stage2")] if spec["pipeline"] == "full" else [])
        for suffix, stage in suffixes:
            map_id = spec["map_id"] + suffix
            path = raw / stage / f"{map_id}.npz"
            if not path.exists():
                report["checks"]["maps"][map_id] = {"present": False}
                continue
            saved = np.load(path)
            status, period = saved["status"], saved["period"]
            entry = {"present": True, "inputs_hash_ok": inputs_ok}
            if suffix == "" and spec["pipeline"] == "full":
                z = plan["zoom"]
                recomputed = zoom_rule(status, period, z["window"], z["stride"])
                zoom_spec = json.loads((raw / "stage1" / f"{map_id}-zoom.json").read_text())
                window = zoom_spec["window"]
                entry["zoom_rule_ok"] = (recomputed is None and window is None) or (
                    recomputed is not None and window is not None and
                    [window["row0"], window["col0"], window["primary"], window["secondary"]] == list(recomputed))
                if window is not None:
                    lh, lo = saved["log10_hidden"], saved["log10_output"]
                    expect_h = np.linspace(lh[window["row0"]], lh[window["row0"] + z["window"] - 1], z["resolution"])
                    expect_o = np.linspace(lo[window["col0"]], lo[window["col0"] + z["window"] - 1], z["resolution"])
                    zoomed = np.load(raw / "stage2" / f"{map_id}-zoom.npz")
                    entry["zoom_axes_ok"] = bool(np.array_equal(zoomed["log10_hidden"], expect_h) and
                                                 np.array_equal(zoomed["log10_output"], expect_o))
            if suffix == "":
                cells = select_audit_cells(status, plan["audit"]["resimulated_cells_per_coarse_map"],
                                           plan["audit"]["audit_seed"] + spec["map_index"])
                rows = []
                cols = status.shape[1]
                for cell in cells:
                    r, c = divmod(cell, cols)
                    lh, lo = 10.0 ** saved["log10_hidden"][r], 10.0 ** saved["log10_output"][c]
                    result = resimulate(CellMap(spec["network"], x, y, lh, lo), theta0, it)
                    saved_status, saved_period = int(status[r, c]), int(period[r, c])
                    diff = None
                    if result["final_state"] is not None and saved_status in (FIXED, PERIODIC):
                        ref = saved["final_state"][r, c]
                        diff = float(np.max(np.abs(result["final_state"] - ref) / (1e-12 + np.abs(ref))))
                    rows.append({"cell": cell, "saved_status": saved_status, "audit_status": result["status"],
                                 "saved_period": saved_period, "audit_period": result["period"],
                                 "audit_ftle": result["ftle"], "relative_state_difference": diff})
                regular = [r for r in rows if r["saved_status"] in (DIVERGENT, FIXED, PERIODIC)]
                irregular = [r for r in rows if r["saved_status"] in (CHAOTIC, UNRESOLVED, DRIFT)]
                entry["resimulation"] = {
                    "cells": rows, "regular_count": len(regular), "irregular_count": len(irregular),
                    "regular_label_agreement": sum(r["saved_status"] == r["audit_status"] and
                                                   r["saved_period"] == r["audit_period"] for r in regular),
                    "irregular_exact_agreement": sum(r["saved_status"] == r["audit_status"] for r in irregular)}
            if spec["pipeline"] != "full":
                report["checks"]["maps"][map_id] = entry
                continue
            mine = [r for r in certificates if r["map_id"] == map_id]
            component_checks = []
            for record in mine:
                members = np.zeros(status.shape, dtype=bool)
                labels, _ = ndimage.label((period == record["period"]) &
                                          ((status == FIXED) if record["kind"] == "fixed" else
                                           ((status == PERIODIC) & (period >= 2))), structure=EIGHT)
                r0, c0 = divmod(record["first_cell"], status.shape[1])
                members = labels == labels[r0, c0]
                ring = ring_of(members)
                ring_cells = np.flatnonzero(ring.ravel()).tolist()
                ring_status, ring_period = status.ravel()[ring_cells], period.ravel()[ring_cells]
                divisor = [int(cell) for cell, s, q in zip(ring_cells, ring_status, ring_period)
                           if s == FIXED or (s == PERIODIC and q < record["period"] and record["period"] % q == 0)]
                chaotic = [int(cell) for cell, s in zip(ring_cells, ring_status) if s == CHAOTIC]
                check = {"map_id": map_id, "component": record["component"], "kind": record["kind"],
                         "size_ok": int(members.sum()) == record["size"],
                         "ring_ok": ring_cells == record["ring_cells"] and divisor == record["ring_divisor_cells"]
                         and chaotic == record["ring_chaotic_cells"],
                         "seed_ok": record["seed_state_sha256"] == digest(
                             saved["final_state"].reshape(-1, saved["final_state"].shape[-1])[record["representative"]]),
                         **recheck_certificate(record, spec, x, y, saved, cert_rule)}
                cert = record["certificate"] or {}
                producer_candidate = (record["kind"] == "embedded" and cert.get("verified")
                                      and cert.get("attracting")
                                      and (cert.get("primitive_period") or 0) >= gate_rule["min_period"])
                if producer_candidate:
                    confirmed = 0
                    reruns = []
                    for cell in sorted(chaotic)[:ring_rule["max_cells_per_component"]]:
                        rr, cc = divmod(cell, status.shape[1])
                        lh, lo = 10.0 ** saved["log10_hidden"][rr], 10.0 ** saved["log10_output"][cc]
                        result = resimulate(CellMap(spec["network"], x, y, lh, lo), theta0,
                                            {**it, "chaos_threshold": ring_rule["chaos_threshold"]},
                                            steps=ring_rule["steps"], ftle_steps=ring_rule["ftle_steps"],
                                            ftle_align=ring_rule["ftle_align"], tail_steps=ring_rule["tail_steps"])
                        reruns.append({"cell": cell, "audit_status": result["status"], "audit_ftle": result["ftle"]})
                        confirmed += int(result["status"] == CHAOTIC)
                    fraction = confirmed / len(ring_cells) if ring_cells else 0.0
                    check["audit_ring_reruns"] = reruns
                    check["audit_confirmed_ring_fraction"] = fraction
                    passes = (check["ring_ok"] and not divisor and check["seed_ok"] and check["closure_ok"]
                              and check["radius_ok"] and check["fd_attracting"] and check["primitive_ok"]
                              and check["rates_from_axes_ok"]
                              and fraction >= gate_rule["min_confirmed_ring_fraction"])
                    check["audit_qualifies"] = bool(passes)
                    if passes:
                        audit_qualifying.append({"map_id": map_id, "component": record["component"],
                                                 "period": record["period"], "fraction": fraction})
                component_checks.append(check)
            entry["components"] = component_checks
            report["checks"]["maps"][map_id] = entry
    report["audit_gate"] = {"confirmed_positive": bool(audit_qualifying), "qualifying": audit_qualifying}
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"out": str(out / "audit.json"), "receipt_mismatches": len(mismatched),
                      "audit_gate": report["audit_gate"]["confirmed_positive"]}, indent=2))


if __name__ == "__main__":
    main()
