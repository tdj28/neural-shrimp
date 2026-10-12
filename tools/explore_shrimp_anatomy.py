"""Exploratory: what the S2-28011 network is doing inside the period-18 shrimp.

The window's attracting period-18 cycle is 2/9 locking of the motion that
appears beyond the edge of stability: a period-2 bounce whose deviation from
the (unstable) 2-cycle rotates by about 2/9 of a turn per two steps. This tool
makes that statement concrete in network terms at chosen rates:

  cycle   Find the attracting period-18 orbit (descent from the shared
          initial parameters, then Newton on F^18), the unstable 2-cycle it
          circles (Newton on F^2 from the mean of its even and odd points),
          and that 2-cycle's critical complex eigenvector. Report the loss
          along the cycle, the layer composition of the rotation plane, the
          preconditioned sharpness lambda_max(P H) and hidden-unit
          saturation along the cycle, the rotation angle per two steps
          around the 2-cycle, and the same quantities on a quasi-periodic
          trajectory at a comparison rate (for example just below the
          window's lower fold). Writes cycle.json and a figure.
  rotation-geometry  Test a geometric account of the rotation: near the edge of
          stability each step's Jacobian is close to a reflection across the
          sharpest preconditioned-Hessian direction, so two steps compose to
          a rotation by twice the angle alpha between the sharpest directions
          at the two points of the 2-cycle (turns ~ alpha/pi). Compares alpha/pi,
          the rotation of the two exact step Jacobians projected on the plane
          of the two sharp directions, and the measured multiplier angle, on
          2-cycles along a saved Neimark-Sacker locus and at given rates.
  slips   Loss signature just outside the window's lower fold: training is
          nearly 18-periodic with occasional phase slips. At rates moved below
          the fold by eps decades, iterate and record |theta_{t+18}-theta_t|;
          laminar stretches are runs below a threshold. For a saddle-node on
          the circle the mean laminar length should scale like eps^(-1/2).
  window  Characterize any census window from its representative rates and
          period: the attracting cycle (descent, Newton on F^p), its
          multipliers, layer composition, loss along the cycle, and its
          winding around the (rate-independent) minimum and around a 2-cycle
          near the cycle's centre, in the cycle's own principal plane. Meant
          to tell rotation-born windows from period-doubling (Henon-type)
          windows, for example for odd periods.
  edge    Cross a saved boundary node (fold or flip of the period-18 cycle)
          along the curve's normal by small signed distances; at each, start
          next to the node's orbit, iterate, and classify the attractor
          (period by recurrence up to 72, doubling amplitude |theta_t -
          theta_{t+18}| for period 36, or aperiodic with return statistics).
          A soft flip shows period 36 with amplitude ~ sqrt(distance).
  lyapunov  Full Lyapunov spectrum (QR iteration with exact step Jacobians)
          and Kaplan-Yorke dimension of the attractor at given rates, plus the
          principal-plane share of its two-step samples: is post-edge training
          effectively a low-dimensional (near two-dimensional) map?
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.fastcont import OrbitCurve, TwoRateMap, divisor_separations, newton  # noqa: E402
from neural_shrimp.trainability import make_s2  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
SEED = 28011


def fixed_rate_orbit(f, theta, p, period, max_iter=40):
    """Newton on F^N(theta) = theta at fixed rates (rates enter as fixed constants)."""
    curve = OrbitCurve(f, period, fixed=p[0])
    row = np.zeros(f.n + 1)
    row[-1] = 1.0
    z, history = newton(curve, np.concatenate([theta, [p[1]]]), max_iter=max_iter,
                        extra=lambda w: (row, w[-1] - p[1]))
    return (None if z is None else z[:f.n]), history


def hidden_preactivations(net, x, theta):
    w0, b0, w1, b1 = net._unpack(theta[None, :])
    return np.tensordot(x, w0[0], axes=([1], [0])) + b0[0][None, :]


def layer_split(vec, n_hidden):
    hidden = float(np.linalg.norm(vec[:n_hidden]) ** 2)
    total = float(np.linalg.norm(vec) ** 2)
    return hidden / total if total > 0 else float("nan")


def describe(f, net, x, y, states, p, center_even=None, plane=None):
    """Loss, sharpness, saturation and (optionally) rotation angles for a sequence of states."""
    n_hidden = net.hidden_parameter_count
    rates = f.rates(p)
    rows = []
    for k, theta in enumerate(states):
        loss = float(net.loss(theta[None, :], x, y)[0])
        h = f.hess(theta)
        sharp = float(np.max(np.linalg.eigvals(rates[:, None] * h).real))
        pre = hidden_preactivations(net, x, theta)
        saturation = float(np.mean(np.abs(np.tanh(pre)) > 0.9))
        rows.append({"k": k, "loss": loss, "preconditioned_sharpness": sharp, "saturated_fraction": saturation})
    angles = None
    if center_even is not None and plane is not None:
        even = states[0::2]
        planar = (even - center_even) @ plane
        angles = np.unwrap(np.arctan2(planar[:, 1], planar[:, 0]))
    return rows, angles, n_hidden


def cycle(args):
    net, x, y, theta0 = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    n = f.n
    n_hidden = net.hidden_parameter_count
    p = [args.lx, args.ly]
    theta = f.iterate(theta0, p, args.transient)
    orbit_state, hist18 = fixed_rate_orbit(f, theta, p, 18)
    record = {"rates": {"log10_output": args.lx, "log10_hidden": args.ly}, "newton18": hist18}
    if orbit_state is None:
        record["outcome"] = "no period-18 orbit from the descent endpoint"
        print(json.dumps(record))
        return
    _, m18, _, points = f.period_map(orbit_state, p, 18)
    states = points[:18]
    mult18 = np.linalg.eigvals(m18)
    record["period18"] = {"divisor_separations": divisor_separations(points, 18),
                          "spectral_radius": float(np.max(np.abs(mult18))),
                          "leading_multipliers": [[float(v.real), float(v.imag)]
                                                  for v in sorted(mult18, key=lambda v: -abs(v))[:3]]}
    # the unstable 2-cycle at the centre of the two nine-point loops
    seed2 = states[0::2].mean(axis=0)
    two, hist2 = fixed_rate_orbit(f, seed2, p, 2)
    record["newton2"] = hist2
    if two is None:
        record["outcome"] = "no 2-cycle found at the loop centre"
        print(json.dumps({k: record[k] for k in ("period18", "newton2")}))
        return
    _, m2, _, pts2 = f.period_map(two, p, 2)
    mult2, vec2 = np.linalg.eig(m2)
    i = int(np.argmax(np.abs(mult2) * (np.abs(mult2.imag) > 1e-9)))
    crit, v = mult2[i], vec2[:, i]
    plane, _ = np.linalg.qr(np.stack([np.real(v), np.imag(v)], axis=1))
    # pick the centre that the even points circle (either point of the 2-cycle)
    centre = pts2[0] if np.linalg.norm(states[0::2].mean(0) - pts2[0]) < np.linalg.norm(
        states[0::2].mean(0) - pts2[1]) else pts2[1]
    rows, angles, _ = describe(f, net, x, y, states, p, centre, plane)
    closing = (angles[0] - angles[-1] + np.pi) % (2 * np.pi) - np.pi
    turns = np.r_[np.diff(angles), closing] / (2 * np.pi)
    record["two_cycle"] = {"multiplier": [float(crit.real), float(crit.imag)], "modulus": float(abs(crit)),
                           "turns_per_two_step": float(abs(np.angle(crit)) / (2 * np.pi)),
                           "plane_hidden_fraction_re": layer_split(np.real(v), n_hidden),
                           "plane_hidden_fraction_im": layer_split(np.imag(v), n_hidden),
                           "separation": float(np.linalg.norm(pts2[1] - pts2[0]))}
    record["cycle_rows"] = rows
    record["cycle_turns_per_two_step"] = turns.tolist()
    record["cycle_total_turns"] = float(np.sum(turns))
    # deviation amplitude from the centre and bounce size along the cycle
    record["deviation_from_two_cycle"] = [float(np.linalg.norm(s - centre)) for s in states[0::2]]
    record["bounce"] = [float(np.linalg.norm(states[(k + 1) % 18] - states[k])) for k in range(18)]
    # layer composition of the motion: variance of the cycle in hidden vs output parameters
    centred = states - states.mean(axis=0)
    record["cycle_variance_hidden_fraction"] = float((centred[:, :n_hidden] ** 2).sum() / (centred ** 2).sum())
    # comparison trajectory
    if args.compare is not None:
        pc = [args.compare[0], args.compare[1]]
        tc = f.iterate(theta0, pc, args.transient)
        traj = f.orbit(tc, pc, args.compare_steps)[:-1]
        crow, cang, _ = describe(f, net, x, y, traj[: min(len(traj), 2 * args.compare_rows)], pc, centre, plane)
        record["compare"] = {"rates": {"log10_output": pc[0], "log10_hidden": pc[1]}, "rows": crow,
                             "loss_series": [float(net.loss(s[None, :], x, y)[0]) for s in traj]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "period18-states.npy", states)
    np.save(out / "two-cycle.npy", pts2)
    (out / "cycle.json").write_text(json.dumps(record, indent=1))
    figure(record, states, centre, plane, out)
    summary = {"spectral_radius_18": record["period18"]["spectral_radius"],
               "two_cycle": record["two_cycle"],
               "cycle_total_turns": record["cycle_total_turns"],
               "turns_per_two_step_mean": float(np.mean(turns)),
               "cycle_variance_hidden_fraction": record["cycle_variance_hidden_fraction"],
               "loss_range": [min(r["loss"] for r in rows), max(r["loss"] for r in rows)],
               "sharpness_range": [min(r["preconditioned_sharpness"] for r in rows),
                                   max(r["preconditioned_sharpness"] for r in rows)],
               "saturated_fraction_range": [min(r["saturated_fraction"] for r in rows),
                                            max(r["saturated_fraction"] for r in rows)]}
    print(json.dumps(summary, indent=1))


def figure(record, states, centre, plane, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = record["cycle_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    loss = [r["loss"] for r in rows] * 2
    ax.plot(range(len(loss)), loss, "o-", color="tab:red", label="period-18 cycle (two periods)")
    if "compare" in record:
        series = record["compare"]["loss_series"][:len(loss)]
        ax.plot(range(len(series)), series, ".--", color="tab:gray", label="comparison (quasi-periodic)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_title("loss: bounce every step, modulated with period 9 two-steps")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    proj_even = (states[0::2] - centre) @ plane
    ax.plot(np.r_[proj_even[:, 0], proj_even[0, 0]], np.r_[proj_even[:, 1], proj_even[0, 1]], "o-", color="tab:blue")
    for k, (a, b) in enumerate(proj_even):
        ax.annotate(str(2 * k), (a, b), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.plot(0, 0, "k+", ms=12)
    ax.set_aspect("equal")
    ax.set_title("even states around the unstable 2-cycle (rotation plane)\n"
                 f"{record['two_cycle']['turns_per_two_step']:.4f} turns/two-step at the 2-cycle; locked 2/9")
    ax = axes[1, 0]
    sharp = [r["preconditioned_sharpness"] for r in rows]
    sat = [r["saturated_fraction"] for r in rows]
    ax.plot(range(18), sharp, "s-", color="tab:purple", label="lambda_max(P H)")
    ax.axhline(2.0, color="k", lw=0.8)
    ax.set_xlabel("step in cycle")
    ax.set_ylabel("preconditioned sharpness")
    ax2 = ax.twinx()
    ax2.plot(range(18), sat, "^-", color="tab:green", label="saturated hidden fraction")
    ax2.set_ylabel("fraction |tanh| > 0.9")
    ax.set_title("edge-of-stability sharpness and hidden saturation along the cycle")
    ax = axes[1, 1]
    ax.bar(["rotation plane Re", "rotation plane Im", "cycle variance"],
           [record["two_cycle"]["plane_hidden_fraction_re"], record["two_cycle"]["plane_hidden_fraction_im"],
            record["cycle_variance_hidden_fraction"]], color=["tab:blue", "tab:cyan", "tab:red"])
    ax.axhline(9 / 13, color="k", lw=0.8, ls=":")
    ax.set_ylim(0, 1)
    ax.set_title("hidden-layer share (dotted: share of hidden parameters, 9/13)")
    fig.tight_layout()
    fig.savefig(out / "shrimp-cycle.png", dpi=110)


def rotation_geometry(args):
    net, x, y, _ = make_s2(SEED)
    f = TwoRateMap(net, x, y)

    def sym(theta, p):
        s = np.sqrt(f.rates(p))
        return s[:, None] * f.hess(theta) * s[None, :]

    def analyse(theta_a, p):
        _, m, _, pts = f.period_map(theta_a, p, 2)
        sa_mat, sb_mat = sym(pts[0], p), sym(pts[1], p)
        wa, va = np.linalg.eigh(sa_mat)
        wb, vb = np.linalg.eigh(sb_mat)
        sa, sb = va[:, -1], vb[:, -1]
        q, _ = np.linalg.qr(np.stack([sa, sb], axis=1))
        eye = np.eye(len(sa))
        prod = (q.T @ (eye - sb_mat) @ q) @ (q.T @ (eye - sa_mat) @ q)
        ev = np.linalg.eigvals(prod)
        c2 = ev[np.argmax(ev.imag)]
        mult = np.linalg.eigvals(m)
        upper = mult[mult.imag > 1e-9]
        c = upper[np.argmin(np.abs(np.abs(upper) - 1.0))] if len(upper) else None
        alpha = float(np.arccos(min(1.0, abs(sa @ sb))))
        return {"eta_lambda_a": float(wa[-1]), "eta_lambda_b": float(wb[-1]),
                "second_a": float(wa[-2]), "second_b": float(wb[-2]),
                "alpha_degrees": float(np.degrees(alpha)), "alpha_over_pi": alpha / np.pi,
                "two_plane_turns": float(abs(np.angle(c2)) / (2 * np.pi)) if abs(c2.imag) > 1e-12 else 0.0,
                "two_plane_modulus": float(abs(c2)),
                "measured_turns": None if c is None else float(abs(np.angle(c)) / (2 * np.pi)),
                "measured_modulus": None if c is None else float(abs(c))}

    rows = []
    if args.locus:
        data = json.loads(Path(args.locus).read_text())
        for r in [r for r in data["rows"] if "theta" in r][:: args.every]:
            rows.append({"lx": r["lx"], "ly": r["ly"], "source": "locus"} | analyse(np.array(r["theta"]), [r["lx"], r["ly"]]))
    for path, lx, ly in args.cycle or []:
        two = np.load(path)
        rows.append({"lx": float(lx), "ly": float(ly), "source": str(path)} | analyse(two[0], [float(lx), float(ly)]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rotation-geometry.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "source"}))


def slips(args):
    net, x, y, theta0 = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    node = json.loads(Path(args.node).read_text())
    point = [float(v) for v in node["point"]]
    start = np.array(node["orbit"][0], dtype=np.float64)
    lx, ly0 = math.log10(point[14]), math.log10(point[13])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for eps in args.eps:
        p = [lx, ly0 - eps]
        theta = start + 1e-6
        for _ in range(args.transient):
            theta = f.step(theta, p)
        window = [theta.copy()]
        for _ in range(17):
            window.append(f.step(window[-1], p))
        buf = list(window)
        dist = np.empty(args.steps)
        loss = np.empty(args.steps)
        t = window[-1]
        for k in range(args.steps):
            t = f.step(t, p)
            dist[k] = np.linalg.norm(t - buf[k % 18])
            buf[k % 18] = t
            loss[k] = net.loss(t[None, :], x, y)[0]
        laminar = dist < args.threshold
        lengths, run = [], 0
        for flag in laminar:
            if flag:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        lengths = lengths[1:] if len(lengths) > 2 else lengths      # drop a cut-off first run
        long_runs = [r for r in lengths if r >= args.min_run]       # flicker during a slip is not a phase
        row = {"eps": eps, "log10_hidden": ly0 - eps, "laminar_fraction": float(laminar.mean()),
               "laminar_runs": len(lengths), "mean_laminar_steps": float(np.mean(lengths)) if lengths else None,
               "median_laminar_steps": float(np.median(lengths)) if lengths else None,
               "long_runs": len(long_runs),
               "mean_long_run_steps": float(np.mean(long_runs)) if long_runs else None,
               "steps_per_slip": float(args.steps / len(long_runs)) if long_runs else None,
               "max_dist": float(dist.max()), "loss_mean": float(loss.mean())}
        rows.append(row)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}))
        np.savez_compressed(out / f"series-eps{eps:g}.npz", dist=dist[:40000], loss=loss[:40000])
    good = [r for r in rows if r["steps_per_slip"]]
    if len(good) >= 2:
        le = np.log([r["eps"] for r in good])
        slope = float(np.polyfit(le, np.log([r["steps_per_slip"] for r in good]), 1)[0])
        slope_mean = float(np.polyfit(le, np.log([r["mean_long_run_steps"] for r in good]), 1)[0])
    else:
        slope, slope_mean = None, None
    (out / "slips.json").write_text(json.dumps({"node": str(args.node), "rows": rows,
                                                "steps_per_slip_exponent": slope,
                                                "mean_long_run_exponent": slope_mean}, indent=1))
    print(json.dumps({"steps_per_slip_exponent": slope, "mean_long_run_exponent": slope_mean, "expected": -0.5}))


def winding(points, centre, plane):
    planar = (points - centre) @ plane
    ang = np.arctan2(planar[:, 1], planar[:, 0])
    steps = np.diff(np.r_[ang, ang[0]])
    steps = (steps + np.pi) % (2 * np.pi) - np.pi
    return float(steps.sum() / (2 * np.pi))


def window(args):
    net, x, y, theta0 = make_s2(args.seed)
    f = TwoRateMap(net, x, y)
    n, nh = f.n, net.hidden_parameter_count
    p = [args.lx, args.ly]
    theta = f.iterate(theta0, p, args.transient)
    orbit_state, hist = fixed_rate_orbit(f, theta, p, args.period)
    record = {"seed": args.seed, "rates": p, "period": args.period, "newton": hist}
    if orbit_state is None:
        print(json.dumps({"outcome": "no orbit", "newton": hist[-3:]}))
        return
    _, mono, _, pts = f.period_map(orbit_state, p, args.period)
    states = pts[:args.period]
    mult = np.linalg.eigvals(mono)
    order = np.argsort(-np.abs(mult))
    centred = states - states.mean(axis=0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    plane = vt[:2].T
    # the minimum (fixed point for every rate): descend at small rates along the same ratio, then Newton
    small = [args.lx - args.descend_shift, args.ly - args.descend_shift]
    tm = f.iterate(theta0, small, 200000)
    for _ in range(30):
        g = f.grad(tm)
        if np.linalg.norm(g) < 1e-13:
            break
        tm = tm - np.linalg.solve(f.hess(tm), g)
    minimum_ok = bool(np.linalg.norm(f.grad(tm)) < 1e-10)
    two, h2 = fixed_rate_orbit(f, states.mean(axis=0), p, 2)
    rec2 = None
    if two is not None:
        _, m2, _, p2 = f.period_map(two, p, 2)
        m2v = np.linalg.eigvals(m2)
        rec2 = {"separation": float(np.linalg.norm(p2[1] - p2[0])),
                "multipliers": [[float(v.real), float(v.imag)] for v in sorted(m2v, key=lambda v: -abs(v))[:3]],
                "winding_cycle_around_point0": winding(states, p2[0], plane),
                "winding_cycle_around_midpoint": winding(states, 0.5 * (p2[0] + p2[1]), plane)}
    record.update({
        "divisor_separations": divisor_separations(pts, args.period),
        "multipliers": [[float(mult[i].real), float(mult[i].imag)] for i in order[:5]],
        "spectral_radius": float(np.abs(mult[order[0]])),
        "plane_variance_share": float((sv[:2] ** 2).sum() / (sv ** 2).sum()),
        "hidden_variance_share": float((centred[:, :nh] ** 2).sum() / (centred ** 2).sum()),
        "loss": [float(v) for v in net.loss(states, x, y)],
        "minimum_found": minimum_ok, "minimum_loss": float(net.loss(tm[None, :], x, y)[0]),
        "winding_around_minimum": winding(states, tm, plane) if minimum_ok else None,
        "winding_around_cycle_mean": winding(states, states.mean(axis=0), plane),
        "two_cycle_near_centre": rec2})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "cycle-states.npy", states)
    (out / "window.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k not in ("loss", "newton")}, indent=1))


def lyapunov(args):
    net, x, y, theta0 = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    rows = []
    for lx, ly in args.points:
        p = [lx, ly]
        theta = f.iterate(theta0, p, args.transient)
        q = np.eye(f.n)
        sums = np.zeros(f.n)
        samples = []
        for k in range(args.steps):
            _, jac, _ = f.step_jac(theta, p)
            theta = f.step(theta, p)
            q, r = np.linalg.qr(jac @ q)
            sums += np.log(np.abs(np.diag(r)))
            if k % 2 == 0 and len(samples) < 4000:
                samples.append(theta.copy())
        lam = np.sort(sums / args.steps)[::-1]
        cum = np.cumsum(lam)
        j = int(np.max(np.where(cum >= 0)[0])) if np.any(cum >= 0) else -1
        ky = float(j + 1 + cum[j] / abs(lam[j + 1])) if 0 <= j < len(lam) - 1 else (0.0 if j < 0 else float(len(lam)))
        smp = np.array(samples)
        c = smp - smp.mean(axis=0)
        sv = np.linalg.svd(c, compute_uv=False)
        row = {"lx": lx, "ly": ly, "lyapunov": lam.tolist(), "kaplan_yorke": ky,
               "plane_share_two_step": float((sv[:2] ** 2).sum() / (sv ** 2).sum()),
               "three_share_two_step": float((sv[:3] ** 2).sum() / (sv ** 2).sum())}
        rows.append(row)
        print(json.dumps({"rates": [lx, ly], "top3": [round(v, 5) for v in lam[:3]], "kaplan_yorke": round(ky, 3),
                          "plane_share": round(row["plane_share_two_step"], 4),
                          "three_share": round(row["three_share_two_step"], 4)}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lyapunov.json").write_text(json.dumps(rows, indent=1))


def edge(args):
    net, x, y, theta0 = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    node = json.loads(Path(args.node).read_text())
    point = [float(v) for v in node["point"]]
    tangent = [float(v) for v in node["tangent"]]
    eta_h, eta_o = point[13], point[14]
    lx, ly = math.log10(eta_o), math.log10(eta_h)
    t_lx = tangent[14] / (eta_o * math.log(10.0))
    t_ly = tangent[13] / (eta_h * math.log(10.0))
    norm = math.hypot(t_lx, t_ly)
    normal = (-t_ly / norm, t_lx / norm)          # unit normal in (lx, ly)
    start = np.array(node["orbit"][0], dtype=np.float64)
    rows = []
    for off in args.offsets:
        p = [lx + off * normal[0], ly + off * normal[1]]
        theta = start + args.kick
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(args.transient):
                theta = f.step(theta, p)
            traj = [theta]
            for _ in range(160):
                traj.append(f.step(traj[-1], p))
        traj = np.array(traj)
        if not np.all(np.isfinite(traj)):
            rows.append({"offset": off, "rates": p, "attractor": "diverged"})
            print(json.dumps(rows[-1]))
            continue
        scale = max(1.0, float(np.linalg.norm(traj[0])))
        period = next((q for q in range(1, 73) if np.linalg.norm(traj[q] - traj[0]) < args.tol * scale), None)
        row = {"offset": off, "rates": p, "period": period,
               "return18": float(np.linalg.norm(traj[18] - traj[0])),
               "return36": float(np.linalg.norm(traj[36] - traj[0])),
               "loss_mean": float(np.mean(net.loss(traj[:144], x, y)))}
        row["attractor"] = (f"period {period}" if period else "aperiodic or slow")
        if period == 36:
            row["doubling_amplitude"] = float(np.max([np.linalg.norm(traj[k + 18] - traj[k]) for k in range(18)]))
        rows.append(row)
        print(json.dumps({k: (round(v, 7) if isinstance(v, float) else v) for k, v in row.items() if k != "rates"}))
    dbl = [(r["offset"], r["doubling_amplitude"]) for r in rows if r.get("doubling_amplitude")]
    exponent = None
    if len(dbl) >= 2:
        offs = np.array([abs(o) for o, _ in dbl])
        amps = np.array([a for _, a in dbl])
        exponent = float(np.polyfit(np.log(offs), np.log(amps), 1)[0])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge.json").write_text(json.dumps({"node": str(args.node), "normal": normal, "rows": rows,
                                               "doubling_amplitude_exponent": exponent}, indent=1))
    print(json.dumps({"normal_lx_ly": [round(normal[0], 4), round(normal[1], 4)],
                      "doubling_amplitude_exponent": exponent, "soft_flip_expectation": 0.5}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cy = sub.add_parser("cycle")
    cy.add_argument("--lx", type=float, required=True)
    cy.add_argument("--ly", type=float, required=True)
    cy.add_argument("--transient", type=int, default=60000)
    cy.add_argument("--compare", type=float, nargs=2, default=None, help="lx ly of a comparison trajectory")
    cy.add_argument("--compare-steps", type=int, default=400)
    cy.add_argument("--compare-rows", type=int, default=18)
    cy.add_argument("--out", required=True)
    rg = sub.add_parser("rotation-geometry")
    rg.add_argument("--locus", default=None, help="ns2-locus.json with 2-cycle states")
    rg.add_argument("--every", type=int, default=20)
    rg.add_argument("--cycle", nargs=3, action="append", metavar=("TWO_CYCLE_NPY", "LX", "LY"))
    rg.add_argument("--out", required=True)
    sl = sub.add_parser("slips")
    sl.add_argument("--node", default=str(DATA / "explore/s2-period18-boundary-curves/attempt-006-ray32-fold/run/node-+1-010.json"))
    sl.add_argument("--eps", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    sl.add_argument("--transient", type=int, default=50000)
    sl.add_argument("--steps", type=int, default=600000)
    sl.add_argument("--threshold", type=float, default=0.05)
    sl.add_argument("--min-run", type=int, default=36, help="shortest laminar stretch counted as a phase")
    sl.add_argument("--out", required=True)
    wi = sub.add_parser("window")
    wi.add_argument("--seed", type=int, default=SEED)
    wi.add_argument("--lx", type=float, required=True)
    wi.add_argument("--ly", type=float, required=True)
    wi.add_argument("--period", type=int, required=True)
    wi.add_argument("--transient", type=int, default=60000)
    wi.add_argument("--descend-shift", type=float, default=1.0, help="decades below the window for the minimum")
    wi.add_argument("--out", required=True)
    ly_ = sub.add_parser("lyapunov")
    ly_.add_argument("--points", type=float, nargs=2, action="append", required=True, metavar=("LX", "LY"))
    ly_.add_argument("--transient", type=int, default=40000)
    ly_.add_argument("--steps", type=int, default=40000)
    ly_.add_argument("--out", required=True)
    ed = sub.add_parser("edge")
    ed.add_argument("--node", required=True)
    ed.add_argument("--offsets", type=float, nargs="+",
                    default=[-1e-3, -3e-4, -1e-4, 1e-4, 3e-4, 1e-3, 3e-3])
    ed.add_argument("--kick", type=float, default=1e-6)
    ed.add_argument("--transient", type=int, default=200000)
    ed.add_argument("--tol", type=float, default=1e-7)
    ed.add_argument("--out", required=True)
    args = parser.parse_args()
    {"cycle": cycle, "rotation-geometry": rotation_geometry, "slips": slips, "window": window,
     "lyapunov": lyapunov, "edge": edge}[args.command](args)


if __name__ == "__main__":
    main()
