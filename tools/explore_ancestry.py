"""Exploratory A3: resonance-anchored ancestry of a phase-locked window.

Subcommands (each writes JSON under DATA/explore/ancestry/<name>/):

  controls   positive control (resonant Neimark-Sacker normal form, 1/5 tongue
             rooted at mu = 0, theta = 1/5), specificity control (the same map
             with the root displaced by 0.01 along the tangent), negative control
             (period-5 window of a logistic map with a contracting coordinate,
             born by a saddle-node, tested against its own fold point)
  root       S2: refine the parent period-2 Neimark-Sacker point whose birth
             rotation is m/q, the local frame, and the onset checks
  levels     S2: fold-bounded cross-sections at decreasing distances beyond the
             curve at a refined root, linked by orbit continuation
  connect    S2: follow the tongue's centre line from the top level towards the
             target window and test whether the window's certified orbit lies on
             the tracked branch

Criteria and outcome categories are those of the run card in docs/research-log.md
and src/neural_shrimp/ancestry.py (pass, alternative_ancestry, exhausted_budget,
no_qualifying_window, invalid).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.ancestry import (classify_ancestry, closest_to_one, continue_segment,  # noqa: E402
                                    cycle_amplitude, existence_ray, same_cycle)
from neural_shrimp.continuation import continue_ray  # noqa: E402
from neural_shrimp.gdmaps import GridMap, run_grid  # noqa: E402
from neural_shrimp.orbits import certify_orbit  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
OUT = DATA / "explore" / "ancestry"


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


# ----------------------------------------------------------------------------- controls

EPS = 0.5


def normal_form(params):
    mu, theta = float(params[0]), float(params[1])
    rot = np.exp(2j * np.pi * theta)

    def map_fn(state):
        z = complex(state[0], state[1])
        w = rot * ((1.0 + mu) * z - abs(z) ** 2 * z) + EPS * np.conj(z) ** 4
        return np.array([w.real, w.imag])

    def jacobian(state):
        z = complex(state[0], state[1])
        a = rot * ((1.0 + mu) - 2.0 * abs(z) ** 2)
        b = rot * (-(z ** 2)) + 4.0 * EPS * np.conj(z) ** 3
        return np.array([[(a + b).real, -(a - b).imag], [(a + b).imag, (a - b).real]])

    return map_fn, jacobian


def normal_form_batch(mu, theta, steps, start=0.05):
    z = np.full(len(theta), start + 0.01j)
    rot = np.exp(2j * np.pi * np.asarray(theta))
    tail = []
    for k in range(steps):
        z = rot * ((1.0 + mu) * z - np.abs(z) ** 2 * z) + EPS * np.conj(z) ** 4
        if k >= steps - 20:
            tail.append(z.copy())
    tail = np.array(tail)
    locked = np.abs(tail[-1] - tail[-6]) < 1e-11
    for d in (1,):
        locked &= np.abs(tail[-1] - tail[-1 - d]) > 1e-6
    return locked, tail[-1]


def logistic(params):
    r, c = float(params[0]), float(params[1])

    def map_fn(state):
        x, y = state
        return np.array([r * x * (1.0 - x) + c * y, 0.1 * y + 0.05 * x * (1.0 - x)])

    def jacobian(state):
        x, _ = state
        return np.array([[r * (1.0 - 2.0 * x), c], [0.05 * (1.0 - 2.0 * x), 0.1]])

    return map_fn, jacobian


def scan_generic(family, period, params_list, start, steps):
    """Iterate each parameter point from ``start``; return those settling on a period-``period`` orbit."""
    found = []
    for p in params_list:
        map_fn, jacobian = family(p)
        x = np.array(start, dtype=np.float64)
        ok = True
        for _ in range(steps):
            x = map_fn(x)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
                ok = False
                break
        if not ok:
            continue
        cert = certify_orbit(map_fn, jacobian, x, period, candidate_tolerance=0.05)
        if cert["verified"]:
            found.append((np.asarray(p), np.asarray(cert["orbit"]), cert))
    return found


def run_levels(family, period, parent_period, parent_seed, root, normal, tangent, deltas, first_orbit,
               first_u, s_max, h_fraction, root_tolerance, log, cert_options=None):
    """Cross-sections at decreasing ``deltas`` beyond the parent curve, linked by continuation."""
    root, normal, tangent = (np.asarray(v, dtype=np.float64) for v in (root, normal, tangent))
    levels = []
    orbit, u_current, delta_current = np.asarray(first_orbit), float(first_u), float(deltas[0])
    width_guess = None
    for k, delta in enumerate(deltas):
        record = {"delta": float(delta), "continuity": True, "primitive": True}
        if k > 0:
            # predicted centre: straight centre line through the root
            u_target = levels[-1]["center"] * delta / levels[-1]["delta"] if "center" in levels[-1] else u_current
            start = root + delta_current * normal + u_current * tangent
            end = root + delta * normal + u_target * tangent
            seg = continue_segment(family, orbit[0], period, start, end, h0=0.05, cert_options=cert_options)
            if not seg["ok"]:
                # second try: keep the tangential coordinate
                end = root + delta * normal + u_current * tangent
                seg = continue_segment(family, orbit[0], period, start, end, h0=0.05, cert_options=cert_options)
                u_target = u_current
            record["continuity"] = bool(seg["ok"])
            record["segment_steps"] = seg["steps"]
            if not seg["ok"]:
                record["reason"] = f"continuation to level failed: {seg['reason']} at t={seg['t_reached']:.4f}"
                levels.append(record)
                log(record)
                break
            orbit, u_current, delta_current = np.asarray(seg["orbit"]), u_target, delta
        base = root + delta * normal + u_current * tangent
        h0 = (width_guess / 40.0) if width_guess else s_max * h_fraction
        sides = {}
        for name, sign in (("left", -1.0), ("right", 1.0)):
            ray = existence_ray(family, orbit[0], period, base, sign * tangent, s_max, h0,
                                bisection_tolerance=max(1e-12, (width_guess or s_max) * 1e-7),
                                cert_options=cert_options)
            u = None if ray["s_boundary"] is None else u_current + sign * ray["s_boundary"]
            sides[name] = {"type": ray["type"], "u": u, "distance_to_one": ray.get("distance_to_one"),
                           "steps": ray.get("steps")}
        record.update(sides)
        map_fn, jacobian = family(base)
        cert = certify_orbit(map_fn, jacobian, orbit[0], period, candidate_tolerance=0.05,
                             **(cert_options or {}))
        record["primitive"] = bool(cert["verified"])
        record["min_divisor_separation"] = cert.get("min_divisor_separation")
        record["stability"] = cert.get("stability")
        record["spectral_radius"] = cert.get("spectral_radius")
        parent = certify_orbit(map_fn, jacobian, parent_seed, parent_period, candidate_tolerance=0.5)
        if parent["verified"]:
            parent_seed = np.asarray(parent["orbit"][0])
            record["amplitude"] = cycle_amplitude(cert["orbit"] if cert["verified"] else orbit, parent["orbit"])
        else:
            record["amplitude"] = None
            record["primitive"] = False
            record["reason"] = f"parent orbit not verified: {parent['reason']}"
        if sides["left"]["type"] == "fold" and sides["right"]["type"] == "fold":
            record["center"] = 0.5 * (sides["left"]["u"] + sides["right"]["u"])
            record["width"] = sides["right"]["u"] - sides["left"]["u"]
            width_guess = record["width"] * (0.25 ** 1.5)
            # move the interior orbit to the measured centre for the next level
            seg = continue_segment(family, orbit[0], period, base, root + delta * normal + record["center"] * tangent,
                                   h0=0.1, cert_options=cert_options)
            if seg["ok"]:
                orbit, u_current = np.asarray(seg["orbit"]), record["center"]
        levels.append(record)
        log(record)
    outcome, details = classify_ancestry(levels, root_tolerance)
    return levels, outcome, details


def measure_section(family, period, orbit, base, tangent, s_max, h0, bisection, cert_options):
    """Both fold edges of the existence interval through ``base`` along +-tangent (u relative to base)."""
    sides = {}
    for name, sign in (("left", -1.0), ("right", 1.0)):
        ray = existence_ray(family, orbit[0], period, base, sign * tangent, s_max, h0,
                            bisection_tolerance=bisection, cert_options=cert_options)
        sides[name] = {"type": ray["type"], "s": ray["s_boundary"], "distance_to_one": ray.get("distance_to_one"),
                       "steps": ray.get("steps")}
    return sides


def run_levels_tracking(family, period, parent_period, parent_seed, root, normal, tangent, deltas, first_orbit,
                        first_u, s_max, root_tolerance, log, cert_options=None, substeps=2, max_refine=4,
                        budget_seconds=5400):
    """Attempt-2 strategy: track the locked interval's centre between measurement levels.

    Between measurement levels (factor 4 in delta) the orbit moves in ``substeps``
    geometric sub-steps; at every sub-step the fold edges are re-measured and the
    orbit re-centred, with the next centre predicted by linear extrapolation in
    delta from the last two measured centres. A failed sub-step is refined by
    halving the sub-step ratio (up to ``max_refine`` times). Only the frozen
    measurement levels enter the classification.
    """
    root, normal, tangent = (np.asarray(v, dtype=np.float64) for v in (root, normal, tangent))
    started = time.time()
    options = cert_options or {}
    orbit = np.asarray(first_orbit)
    track = []  # (delta, centre, width) of every measured section, measurement levels and sub-steps

    def section(delta, u, orbit):
        base = root + delta * normal + u * tangent
        width_hint = track[-1][2] if track else s_max * 0.2
        sides = measure_section(family, period, orbit, base, tangent, s_max, max(width_hint / 40.0, 1e-9),
                                max(1e-13, width_hint * 1e-7), options)
        out = {"sides": sides, "u_base": u}
        if sides["left"]["type"] == "fold" and sides["right"]["type"] == "fold":
            left_u, right_u = u - sides["left"]["s"], u + sides["right"]["s"]
            centre = 0.5 * (left_u + right_u)
            seg = continue_segment(family, orbit[0], period, base, root + delta * normal + centre * tangent,
                                   h0=0.1, cert_options=options)
            out.update({"left_u": left_u, "right_u": right_u, "center": centre, "width": right_u - left_u,
                        "recentred": bool(seg["ok"])})
            if seg["ok"]:
                out["orbit"] = np.asarray(seg["orbit"])
        return out

    levels = []
    delta_now, u_now = float(deltas[0]), float(first_u)
    for k, delta in enumerate(deltas):
        record = {"delta": float(delta), "continuity": True, "primitive": True, "substeps": []}
        if time.time() - started > budget_seconds:
            record.update({"continuity": False, "reason": "time budget exhausted"})
            levels.append(record)
            log(record)
            break
        if k > 0:
            ratio = (delta / delta_now) ** (1.0 / substeps)
            targets = [delta_now * ratio ** (j + 1) for j in range(substeps)]
            j = 0
            refine = 0
            while j < len(targets):
                d_next = targets[j]
                if len(track) >= 2:
                    (d1, c1, _), (d2, c2, _) = track[-2], track[-1]
                    u_pred = c2 + (c2 - c1) * (d_next - d2) / (d2 - d1) if d2 != d1 else c2
                else:
                    u_pred = u_now
                start = root + delta_now * normal + u_now * tangent
                seg = None
                for u_try in (u_pred, u_now):
                    seg = continue_segment(family, orbit[0], period, start, root + d_next * normal + u_try * tangent,
                                           h0=0.05, cert_options=options)
                    if seg["ok"]:
                        break
                if not seg["ok"]:
                    if refine >= max_refine:
                        record.update({"continuity": False,
                                       "reason": f"sub-step to delta {d_next:.3g} failed after {refine} refinements: "
                                                 f"{seg['reason']} at t={seg['t_reached']:.4f}"})
                        break
                    refine += 1
                    # insert an intermediate target halfway (geometric) and retry
                    targets.insert(j, (delta_now * d_next) ** 0.5)
                    continue
                orbit, delta_now, u_now = np.asarray(seg["orbit"]), d_next, u_try
                sec = section(delta_now, u_now, orbit)
                record["substeps"].append({"delta": delta_now, "center": sec.get("center"), "width": sec.get("width"),
                                           "left": sec["sides"]["left"]["type"], "right": sec["sides"]["right"]["type"]})
                if "center" in sec and sec.get("recentred"):
                    orbit, u_now = sec["orbit"], sec["center"]
                    track.append((delta_now, sec["center"], sec["width"]))
                j += 1
            if not record["continuity"]:
                levels.append(record)
                log(record)
                break
            sec_final = record["substeps"][-1] if record["substeps"] else {}
        else:
            sec = section(delta_now, u_now, orbit)
            if "center" in sec and sec.get("recentred"):
                orbit, u_now = sec["orbit"], sec["center"]
                track.append((delta_now, sec["center"], sec["width"]))
            sec_final = {"center": sec.get("center"), "width": sec.get("width"),
                         "left": sec["sides"]["left"]["type"], "right": sec["sides"]["right"]["type"]}
        # measurement record at this level
        if sec_final.get("center") is not None:
            record["left"] = {"type": sec_final["left"], "u": sec_final["center"] - 0.5 * sec_final["width"]}
            record["right"] = {"type": sec_final["right"], "u": sec_final["center"] + 0.5 * sec_final["width"]}
            record["center"], record["width"] = sec_final["center"], sec_final["width"]
        else:
            record["left"] = {"type": sec_final.get("left"), "u": None}
            record["right"] = {"type": sec_final.get("right"), "u": None}
        base = root + delta_now * normal + u_now * tangent
        map_fn, jacobian = family(base)
        cert = certify_orbit(map_fn, jacobian, orbit[0], period, candidate_tolerance=0.05, **options)
        record["primitive"] = bool(cert["verified"])
        record["min_divisor_separation"] = cert.get("min_divisor_separation")
        record["stability"] = cert.get("stability")
        record["spectral_radius"] = cert.get("spectral_radius")
        record["orbit"] = cert["orbit"] if cert["verified"] else orbit.tolist()
        record["params"] = base.tolist()
        parent = certify_orbit(map_fn, jacobian, parent_seed, parent_period, candidate_tolerance=0.5)
        if parent["verified"]:
            parent_seed = np.asarray(parent["orbit"][0])
            record["parent_orbit"] = parent["orbit"]
            record["amplitude"] = cycle_amplitude(record["orbit"], parent["orbit"])
        else:
            record["amplitude"] = None
            record["primitive"] = False
            record["reason"] = f"parent orbit not verified: {parent['reason']}"
        levels.append(record)
        log(record)
    outcome, details = classify_ancestry(levels, root_tolerance)
    return levels, outcome, details


def cmd_controls(args):
    out = OUT / args.name
    results = {}

    def log(r):
        print(json.dumps({k: r.get(k) for k in ("delta", "center", "width", "amplitude", "continuity")}
                         | {"left": r.get("left", {}).get("type"), "right": r.get("right", {}).get("type")},
                         default=float), flush=True)

    deltas = [0.04 / 4 ** k for k in range(6)]
    # positive control: root at (mu, theta) = (0, 1/5); normal +mu, tangent +theta
    thetas = np.linspace(0.18, 0.22, 801)
    locked, final = normal_form_batch(deltas[0], thetas, 40000)
    idx = np.flatnonzero(locked)
    print("positive control: locked scan cells", len(idx), flush=True)
    if len(idx) == 0:
        results["positive"] = {"outcome": "exhausted_budget", "reason": "no locked cell at the top level"}
    else:
        mid = idx[len(idx) // 2]
        z = final[mid]
        cert = certify_orbit(*normal_form([deltas[0], thetas[mid]]), np.array([z.real, z.imag]), 5)
        levels, outcome, details = run_levels(normal_form, 5, 1, np.zeros(2), [0.0, 0.2], [1.0, 0.0], [0.0, 1.0],
                                              deltas, cert["orbit"], thetas[mid] - 0.2, 0.05, 0.01, 1e-4, log)
        results["positive"] = {"outcome": outcome, "details": details, "levels": levels}
        # specificity control: same branch, root displaced by +0.01 along the tangent
        shifted = [{**lv, **({"left": {**lv["left"], "u": lv["left"]["u"] - 0.01},
                              "right": {**lv["right"], "u": lv["right"]["u"] - 0.01}}
                             if lv.get("left", {}).get("u") is not None and lv.get("right", {}).get("u") is not None
                             else {})} for lv in levels]
        outcome_s, details_s = classify_ancestry(shifted, 1e-4)
        results["specificity"] = {"outcome": outcome_s, "details": details_s}
    print("positive:", results["positive"]["outcome"], "| specificity:", results.get("specificity", {}).get("outcome"),
          flush=True)
    # negative control: logistic period-5 window (saddle-node) with a contracting coordinate
    rs = np.linspace(3.736, 3.744, 161)
    found = scan_generic(logistic, 5, [np.array([r, 0.0]) for r in rs], [0.3, 0.0], 4000)
    print("negative control: period-5 cells", len(found), flush=True)
    if not found:
        results["negative"] = {"outcome": "exhausted_budget", "reason": "window not found"}
    else:
        p0, orbit0, _ = found[0]
        # locate the window's own fold (its birth) by continuing towards smaller r
        ray = existence_ray(logistic, orbit0[0], 5, p0, [-1.0, 0.0], 0.02, 1e-4, bisection_tolerance=1e-13)
        r_sn = p0[0] - ray["s_boundary"]
        print("negative control fold at r =", r_sn, ray["type"], flush=True)
        neg_deltas = [0.004 / 4 ** k for k in range(6)]
        start_params = np.array([r_sn + neg_deltas[0], 0.0])
        seg = continue_segment(logistic, orbit0[0], 5, p0, start_params)
        x_parent = 1.0 - 1.0 / r_sn
        levels_n, outcome_n, details_n = run_levels(logistic, 5, 1, np.array([x_parent, 0.05 * x_parent * (1 - x_parent) / 0.9]),
                                                    [r_sn, 0.0], [1.0, 0.0], [0.0, 1.0], neg_deltas, seg["orbit"],
                                                    0.0, 0.05, 0.01, 1e-4, log)
        results["negative"] = {"outcome": outcome_n, "details": details_n, "levels": levels_n,
                               "fold_r": r_sn, "fold_type": ray["type"]}
    passed = (results.get("positive", {}).get("outcome") == "pass"
              and results.get("specificity", {}).get("outcome") != "pass"
              and results.get("negative", {}).get("outcome") != "pass")
    results["controls_passed"] = bool(passed)
    dump(out / "controls.json", results)
    print("controls passed:", passed, {k: v.get("outcome") for k, v in results.items() if isinstance(v, dict)})


# ----------------------------------------------------------------------------- network


def s2_family(system, seed):
    net, x, y, theta0 = SYSTEMS[system](seed)

    def family(p):
        return GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]]).cell_map(0)

    return net, x, y, theta0, family


def ns_point(family, net, x, y, theta0, lo, start_h, s_max=0.5):
    grid = GridMap(net, x, y, [10.0 ** start_h], [10.0 ** lo])
    res = run_grid(grid, theta0, 40000, tail_steps=384, ftle_steps=1000, ftle_align=200, sd_horizons=())
    if int(res["period"][0]) != 2:
        return None
    ray = continue_ray(family, res["final_state"][0], 2, (start_h, lo), (1.0, 0.0), s_max, 0.002,
                       bisection_tolerance=1e-11)
    if ray["type"] != "neimark_sacker":
        return {"type": ray["type"]}
    mu = complex(*ray["dominant"])
    return {"type": "neimark_sacker", "lh": start_h + ray["s_boundary"], "lo": lo,
            "theta": abs(np.angle(mu)) / (2 * np.pi), "state": ray["last_attracting_state"]}


def circle_checks(net, x, y, family, p_curve, normal, state0, theta_birth, deltas, transient, samples,
                  orders=(4, 8, 16)):
    """Invariance residual of Fourier-fitted circles and rotation versus detuning at one curve point."""
    rows = []
    for d in deltas:
        p = np.asarray(p_curve) + d * np.asarray(normal)
        grid = GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]])
        state = np.asarray(state0)[None, :] + 1e-4
        for _ in range(transient):
            state = grid.step(state)
        pts = []
        for _ in range(samples):
            state = grid.step(grid.step(state))
            pts.append(state[0].copy())
        pts = np.array(pts)
        lobe = pts  # one lobe: the two-step map maps it to itself
        centre = lobe.mean(0)
        u, s, vt = np.linalg.svd(lobe - centre, full_matrices=False)
        plane = vt[:2]
        z = (lobe - centre) @ plane.T
        phi = np.arctan2(z[:, 1], z[:, 0])
        steps = (np.diff(phi) + np.pi) % (2 * np.pi) - np.pi
        rho = float(abs(steps.sum()) / (2 * np.pi * (len(phi) - 1)))
        residuals = {}
        for order in orders:
            basis = [np.ones_like(phi)]
            for k in range(1, order + 1):
                basis += [np.cos(k * phi), np.sin(k * phi)]
            basis = np.array(basis).T
            coef, *_ = np.linalg.lstsq(basis, lobe, rcond=None)

            def curve(angle):
                b = [np.ones_like(angle)]
                for k in range(1, order + 1):
                    b += [np.cos(k * angle), np.sin(k * angle)]
                return np.array(b).T @ coef

            grid_phi = np.linspace(-np.pi, np.pi, 256, endpoint=False)
            on = curve(grid_phi)
            mapped = GridMap(net, x, y, np.full(len(on), 10.0 ** p[0]), np.full(len(on), 10.0 ** p[1]))
            image = mapped.step(mapped.step(on))
            zi = (image - centre) @ plane.T
            back = curve(np.arctan2(zi[:, 1], zi[:, 0]))
            residuals[str(order)] = float(np.max(np.linalg.norm(image - back, axis=1)))
        radius = float(np.mean(np.linalg.norm(z, axis=1)))
        rows.append({"delta": float(d), "rho": rho, "planar_share": float((s[:2] ** 2).sum() / (s ** 2).sum()),
                     "radius": radius, "invariance_residual": residuals,
                     "residual_over_radius": residuals[str(orders[-1])] / max(radius, 1e-300)})
    ds = np.array([r["delta"] for r in rows])
    rhos = np.array([r["rho"] for r in rows])
    slope, intercept = np.polyfit(ds, rhos, 1)
    numerical = 1.0 / samples
    return {"rows": rows, "theta_birth": float(theta_birth), "intercept": float(intercept),
            "finite_detuning_slope": float(slope), "numerical_error_bound": numerical,
            "intercept_minus_birth": float(intercept - theta_birth),
            "rotation_consistent": bool(abs(intercept - theta_birth) <= 3 * numerical + 3 * np.std(
                rhos - (slope * ds + intercept))),
            "residuals_converge": bool(all(r["invariance_residual"][str(orders[-1])] <=
                                           r["invariance_residual"][str(orders[0])] for r in rows)
                                       and all(r["residual_over_radius"] < 1e-2 for r in rows))}


def cmd_root(args):
    net, x, y, theta0, family = s2_family(args.system, args.seed)
    target = args.m / args.q
    lo_a, lo_b = args.bracket
    a = ns_point(family, net, x, y, theta0, lo_a, args.start)
    b = ns_point(family, net, x, y, theta0, lo_b, args.start)
    if not a or not b or a["type"] != "neimark_sacker" or b["type"] != "neimark_sacker" or \
            (a["theta"] - target) * (b["theta"] - target) > 0:
        raise SystemExit(f"bracket does not straddle {args.m}/{args.q}: {a} {b}")
    for _ in range(args.iterations):
        mid_lo = 0.5 * (lo_a + lo_b)
        m = ns_point(family, net, x, y, theta0, mid_lo, args.start)
        if (m["theta"] - target) * (a["theta"] - target) > 0:
            lo_a, a = mid_lo, m
        else:
            lo_b, b = mid_lo, m
        if abs(lo_b - lo_a) < args.tolerance:
            break
    root = m
    # local frame from neighbouring curve points
    plus = ns_point(family, net, x, y, theta0, root["lo"] + args.frame_step, args.start)
    minus = ns_point(family, net, x, y, theta0, root["lo"] - args.frame_step, args.start)
    tangent = np.array([plus["lh"] - minus["lh"], plus["lo"] - minus["lo"]])
    tangent /= np.linalg.norm(tangent)
    normal = np.array([tangent[1], -tangent[0]])
    if normal[0] < 0:
        normal = -normal  # beyond the curve = larger hidden rate
    # onset checks at the root: two-step multipliers, transversality, supercritical small circle
    p_root = np.array([root["lh"], root["lo"]])
    onset = {}
    for label, d in (("before", -args.onset_step), ("at", 0.0), ("after", args.onset_step)):
        map_fn, jacobian = family(p_root + d * normal)
        cert = certify_orbit(map_fn, jacobian, np.asarray(root["state"]), 2, candidate_tolerance=0.05)
        mods = [m["modulus"] for m in cert["multipliers"]]
        onset[label] = {"verified": cert["verified"], "moduli_top4": mods[:4],
                        "pair": cert["multipliers"][:2]}
    pair_complex = abs(onset["at"]["pair"][0]["imag"]) > 1e-6 and \
        abs(onset["at"]["pair"][0]["imag"] + onset["at"]["pair"][1]["imag"]) < 1e-9
    others_inside = onset["at"]["moduli_top4"][2] < 1.0 - args.inside_margin
    transverse = onset["before"]["moduli_top4"][0] < 1.0 < onset["after"]["moduli_top4"][0]
    # supercriticality: small attracting invariant circle with radius ~ sqrt(detuning)
    radii = {}
    for d in args.circle_deltas:
        p = p_root + d * normal
        grid = GridMap(net, x, y, [10.0 ** p[0]], [10.0 ** p[1]])
        state = np.asarray(root["state"])[None, :] + 1e-4
        for _ in range(args.circle_steps):
            state = grid.step(state)
        tail = []
        for _ in range(4000):
            state = grid.step(grid.step(state))
            tail.append(state[0].copy())
        tail = np.array(tail)
        map_fn, jacobian = family(p)
        parent = certify_orbit(map_fn, jacobian, np.asarray(root["state"]), 2, candidate_tolerance=0.5)
        centre = np.asarray(parent["orbit"])
        dist = np.min(np.stack([np.linalg.norm(tail - c, axis=1) for c in centre]), axis=0)
        radii[str(d)] = {"mean_radius": float(dist.mean()), "max_radius": float(dist.max()),
                         "min_radius": float(dist.min()), "finite": bool(np.all(np.isfinite(tail)))}
    ds = np.array(args.circle_deltas)
    rs = np.array([radii[str(d)]["mean_radius"] for d in args.circle_deltas])
    slope = float(np.polyfit(np.log(ds), np.log(rs), 1)[0]) if np.all(rs > 0) else None
    supercritical = slope is not None and 0.3 < slope < 0.7 and all(v["finite"] for v in radii.values()) and \
        rs.max() < 0.5
    # criteria (2) and (3) at a prespecified non-resonant comparison point on the curve
    comparison = ns_point(family, net, x, y, theta0, root["lo"] + args.comparison_offset, args.start)
    checks = circle_checks(net, x, y, family, [comparison["lh"], comparison["lo"]], normal, comparison["state"],
                           comparison["theta"], args.detunings, args.circle_steps, args.circle_samples)
    result = {"m": args.m, "q": args.q, "root": {k: root[k] for k in ("lh", "lo", "theta")},
              "root_state": root["state"], "bracket_width": abs(lo_b - lo_a), "tangent": tangent.tolist(),
              "normal": normal.tolist(), "onset": onset, "criteria": {
                  "complex_pair": bool(pair_complex), "others_inside": bool(others_inside),
                  "transverse": bool(transverse), "supercritical_small_circle": bool(supercritical),
                  "radius_scaling_exponent": slope,
                  "invariance_residual_converges": checks["residuals_converge"],
                  "rotation_matches_birth_angle_at_zero_detuning": checks["rotation_consistent"]},
              "circle_radii": radii, "comparison_point": {k: comparison[k] for k in ("lh", "lo", "theta")},
              "circle_checks": checks}
    dump(OUT / args.name / "root.json", result)
    print(json.dumps({k: result[k] for k in ("root", "bracket_width", "criteria")}, indent=1, default=float))


def scan_cross_section(net, x, y, parent_state, root, normal, tangent, delta, us, steps, period):
    lh = root[0] + delta * normal[0] + us * tangent[0]
    lo = root[1] + delta * normal[1] + us * tangent[1]
    grid = GridMap(net, x, y, 10.0 ** lh, 10.0 ** lo)
    state = np.repeat(np.asarray(parent_state)[None, :] + 1e-4, len(us), axis=0)
    alive = np.ones(len(us), dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            state = grid.step(state)
            alive &= np.all(np.isfinite(state), axis=1)
            state[~alive] = 0.0
        tail = [state.copy()]
        for _ in range(2 * period):
            state = grid.step(state)
            tail.append(state.copy())
    tail = np.array(tail)
    scale = np.maximum(1.0, np.max(np.abs(tail[0]), axis=1))
    closes = np.max(np.abs(tail[period] - tail[0]), axis=1) / scale < 1e-5
    for d in (1, 2, 5):
        if period % d == 0 and d < period:
            closes &= np.max(np.abs(tail[d] - tail[0]), axis=1) / scale > 1e-3
    return alive & closes, tail[0]


def cmd_levels(args):
    net, x, y, theta0, family = s2_family(args.system, args.seed)
    root_info = json.loads((OUT / args.root_name / "root.json").read_text())
    root = np.array([root_info["root"]["lh"], root_info["root"]["lo"]])
    normal, tangent = np.array(root_info["normal"]), np.array(root_info["tangent"])
    parent_state = np.array(root_info["root_state"])
    deltas = [args.delta0 / 4 ** k for k in range(args.count)]
    started = time.time()
    us = np.linspace(-args.scan_half_width, args.scan_half_width, args.scan_cells)
    locked, final = scan_cross_section(net, x, y, parent_state, root, normal, tangent, deltas[0], us,
                                       args.scan_steps, args.period)
    idx = np.flatnonzero(locked)
    print(f"top level delta {deltas[0]}: locked cells {len(idx)} of {len(us)}", flush=True)
    result = {"root": root.tolist(), "deltas": deltas, "scan_locked_cells": int(len(idx)),
              "scan_u": [float(us[i]) for i in idx]}
    if len(idx) == 0:
        result.update({"outcome": "no_qualifying_window", "reason": "no locked period-10 cell on the top cross-section"})
        dump(OUT / args.name / "levels.json", result)
        print(result["outcome"])
        return
    # choose the locked run nearest the root's foot
    i = idx[np.argmin(np.abs(us[idx]))]
    p = root + deltas[0] * normal + us[i] * tangent
    map_fn, jacobian = family(p)
    options = {"newton_tolerance": args.newton_tolerance}
    cert = certify_orbit(map_fn, jacobian, final[i], args.period, candidate_tolerance=0.05, **options)
    if not cert["verified"]:
        result.update({"outcome": "exhausted_budget", "reason": f"scan orbit not certified: {cert['reason']}"})
        dump(OUT / args.name / "levels.json", result)
        print(result["outcome"])
        return

    def log(r):
        print(json.dumps({k: r.get(k) for k in ("delta", "center", "width", "amplitude", "continuity",
                                                  "stability", "reason")}
                         | {"left": r.get("left", {}).get("type"), "right": r.get("right", {}).get("type")},
                         default=float), flush=True)

    if args.tracking:
        levels, outcome, details = run_levels_tracking(family, args.period, 2, parent_state, root, normal, tangent,
                                                       deltas, cert["orbit"], us[i], args.s_max, args.root_tolerance,
                                                       log, cert_options=options, substeps=args.substeps,
                                                       budget_seconds=args.budget_seconds)
    else:
        levels, outcome, details = run_levels(family, args.period, 2, parent_state, root, normal, tangent, deltas,
                                              cert["orbit"], us[i], args.s_max, 0.01, args.root_tolerance, log,
                                              cert_options=options)
    result.update({"levels": levels, "outcome": outcome, "details": details, "seconds": time.time() - started})
    dump(OUT / args.name / "levels.json", result)
    print("outcome:", outcome, json.dumps({k: details.get(k) for k in ("extrapolated_center", "center_mismatch",
                                                                         "amplitude_ratio", "widths_shrink",
                                                                         "reason")}, default=float))


def cmd_connect(args):
    net, x, y, theta0, family = s2_family(args.system, args.seed)
    levels = json.loads((OUT / args.levels_name / "levels.json").read_text())
    root_info = json.loads((OUT / args.root_name / "root.json").read_text())
    root = np.array([root_info["root"]["lh"], root_info["root"]["lo"]])
    options = {"newton_tolerance": args.newton_tolerance}
    normal, tangent = np.array(root_info["normal"]), np.array(root_info["tangent"])
    certs = json.loads((DATA / "raw/EXP-028/attempt-002/stage3/certificates.json").read_text())
    rec = next(r for r in certs if r["map_id"] == args.map_id and r["component"] == args.component)
    window = np.log10([rec["lr_hidden"], rec["lr_output"]])
    window_orbit = np.array(rec["certificate"]["orbit"])
    top = next(lv for lv in levels["levels"] if "center" in lv)
    # re-find the top-level orbit by a short scan around its centre
    us = np.linspace(top["center"] - 0.25 * top["width"], top["center"] + 0.25 * top["width"], 9)
    locked, final = scan_cross_section(net, x, y, np.array(root_info["root_state"]), root, normal, tangent,
                                       top["delta"], us, args.scan_steps, args.period)
    idx = np.flatnonzero(locked)
    if len(idx) == 0:
        raise SystemExit("could not re-find the top-level orbit")
    i = idx[len(idx) // 2]
    p = root + top["delta"] * normal + us[i] * tangent
    cert = certify_orbit(*family(p), final[i], args.period, candidate_tolerance=0.05, **options)
    orbit = np.asarray(cert["orbit"])
    direction = normal.copy()
    track = []
    started = time.time()
    outcome = "exhausted_budget"
    for step in range(args.max_cross_sections):
        if time.time() - started > args.budget_seconds:
            break
        target = p + args.step * direction
        seg = continue_segment(family, orbit[0], args.period, p, target, h0=0.1, cert_options=options)
        if not seg["ok"]:
            track.append({"step": step, "params": target.tolist(), "reason": f"segment failed: {seg['reason']}"})
            break
        orbit = np.asarray(seg["orbit"])
        perp = np.array([-direction[1], direction[0]])
        sides = {}
        for name, sign in (("left", -1.0), ("right", 1.0)):
            ray = existence_ray(family, orbit[0], args.period, target, sign * perp, args.s_max, args.step / 20,
                                bisection_tolerance=1e-8, cert_options=options)
            sides[name] = {"type": ray["type"], "s": ray["s_boundary"]}
        entry = {"step": step, "params": target.tolist(), "left": sides["left"], "right": sides["right"],
                 "stability": seg["certificate"]["stability"], "radius": seg["certificate"]["spectral_radius"]}
        if sides["left"]["type"] == "fold" and sides["right"]["type"] == "fold":
            shift = 0.5 * (sides["right"]["s"] - sides["left"]["s"])
            centre = target + shift * perp
            entry["width"] = sides["right"]["s"] + sides["left"]["s"]
            seg2 = continue_segment(family, orbit[0], args.period, target, centre, h0=0.1, cert_options=options)
            if seg2["ok"]:
                orbit = np.asarray(seg2["orbit"])
                new_dir = centre - p
                direction = new_dir / np.linalg.norm(new_dir)
                p = centre
            else:
                p = target
        else:
            p = target
        entry["center"] = p.tolist()
        # does the window lie inside this cross-section and on this branch?
        rel = window - p
        along = float(rel @ direction)
        entry["window_along"] = along
        track.append(entry)
        print(json.dumps({k: entry.get(k) for k in ("step", "center", "width", "stability", "window_along")}
                         | {"left": sides["left"]["type"], "right": sides["right"]["type"]}, default=float),
              flush=True)
        if abs(along) <= args.step:
            seg3 = continue_segment(family, orbit[0], args.period, p, window, h0=0.05, cert_options=options)
            entry["window_reached"] = bool(seg3["ok"])
            if seg3["ok"]:
                entry["same_orbit_as_window"] = bool(same_cycle(np.asarray(seg3["orbit"]), window_orbit, 1e-7))
                outcome = "pass_connection" if entry["same_orbit_as_window"] else "alternative_branch_at_window"
            else:
                outcome = "window_outside_tracked_region"
            break
    dump(OUT / args.name / "connect.json", {"track": track, "outcome": outcome, "window": window.tolist(),
                                            "seconds": time.time() - started})
    print("connection outcome:", outcome)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("controls")
    c.add_argument("--name", default="controls")
    r = sub.add_parser("root")
    r.add_argument("--name", required=True)
    r.add_argument("--system", default="S2")
    r.add_argument("--seed", type=int, default=28011)
    r.add_argument("--m", type=int, default=1)
    r.add_argument("--q", type=int, default=5)
    r.add_argument("--bracket", type=float, nargs=2, required=True, help="log10 output rates straddling the root")
    r.add_argument("--start", type=float, default=1.65)
    r.add_argument("--iterations", type=int, default=40)
    r.add_argument("--tolerance", type=float, default=1e-7)
    r.add_argument("--frame-step", type=float, default=2e-3)
    r.add_argument("--onset-step", type=float, default=1e-4)
    r.add_argument("--inside-margin", type=float, default=1e-3)
    r.add_argument("--circle-deltas", type=float, nargs="+", default=[2e-4, 8e-4, 3.2e-3])
    r.add_argument("--circle-steps", type=int, default=200000)
    r.add_argument("--circle-samples", type=int, default=20000)
    r.add_argument("--comparison-offset", type=float, default=0.01,
                   help="log10 output offset of the prespecified comparison point on the curve")
    r.add_argument("--detunings", type=float, nargs="+", default=[1e-3, 5e-4, 2.5e-4])
    lv = sub.add_parser("levels")
    lv.add_argument("--name", required=True)
    lv.add_argument("--root-name", required=True)
    lv.add_argument("--system", default="S2")
    lv.add_argument("--seed", type=int, default=28011)
    lv.add_argument("--period", type=int, default=10)
    lv.add_argument("--delta0", type=float, default=0.0128)
    lv.add_argument("--count", type=int, default=6)
    lv.add_argument("--scan-half-width", type=float, default=0.03)
    lv.add_argument("--scan-cells", type=int, default=601)
    lv.add_argument("--scan-steps", type=int, default=100000)
    lv.add_argument("--s-max", type=float, default=0.05)
    lv.add_argument("--root-tolerance", type=float, default=1e-4)
    lv.add_argument("--tracking", action="store_true", help="attempt-2 centre-line tracking between levels")
    lv.add_argument("--substeps", type=int, default=2)
    lv.add_argument("--budget-seconds", type=float, default=5400)
    lv.add_argument("--newton-tolerance", type=float, default=1e-8,
                    help="relative Newton-correction tolerance for period-10 certification near the root")
    cn = sub.add_parser("connect")
    cn.add_argument("--name", required=True)
    cn.add_argument("--levels-name", required=True)
    cn.add_argument("--root-name", required=True)
    cn.add_argument("--system", default="S2")
    cn.add_argument("--seed", type=int, default=28011)
    cn.add_argument("--map-id", default="S2-28011")
    cn.add_argument("--component", type=int, default=139)
    cn.add_argument("--period", type=int, default=10)
    cn.add_argument("--step", type=float, default=0.01)
    cn.add_argument("--s-max", type=float, default=0.2)
    cn.add_argument("--scan-steps", type=int, default=100000)
    cn.add_argument("--newton-tolerance", type=float, default=1e-8)
    cn.add_argument("--max-cross-sections", type=int, default=200)
    cn.add_argument("--budget-seconds", type=float, default=5400)
    args = parser.parse_args()
    {"controls": cmd_controls, "root": cmd_root, "levels": cmd_levels, "connect": cmd_connect}[args.command](args)


if __name__ == "__main__":
    main()
