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
    args = parser.parse_args()
    {"cycle": cycle}[args.command](args)


if __name__ == "__main__":
    main()
