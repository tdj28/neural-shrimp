"""Exploratory certification targets for the hidden-parent organization of the period-18 window.

Subcommands:
  chain   The new 1:1 point on the window's upper fold (where the inherited
          saddle multiplier crosses +1, numerically near log10 rates
          (-0.0957, 1.9247)). Solve the generalized-chain system
              R = F^18(x) - x = 0,  A q = 0,  A r = q,  (q.q - 1)/2 = 0,  q.r = 0,
          A = D_x R, for (x, eta_h, eta_o, q, r) in float64 from a saved
          continuation node, report the spectrum, then attempt Codex's
          outward-rounded Krawczyk enclosure (rigorous_generalized_chain.
          chain_root_proof) at 256 bits. A passing root proves a locally
          unique primitive period-18 orbit with a nonzero length-two Jordan
          chain at +1; it does not by itself prove a generic 1:1 resonance
          (multiplicity, complementary spectrum and nondegeneracy are separate).
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.fastcont import TwoRateMap, divisor_separations  # noqa: E402
from neural_shrimp.trainability import make_s2  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
SEED = 28011
PERIOD = 18


class ChainSystem:
    """Generalized-chain residual and Jacobian in actual rates (float64)."""

    def __init__(self, fmap, h=1e-6):
        self.f, self.h = fmap, h

    def _logp(self, eh, eo):
        return [math.log10(eo), math.log10(eh)]

    def residual_jacobian(self, z):
        f, n = self.f, self.f.n
        x, eh, eo = z[:n], z[n], z[n + 1]
        q, r = z[n + 2:2 * n + 2], z[2 * n + 2:]
        p = self._logp(eh, eo)
        t, m, dp, points = f.period_map(x, p, PERIOD)
        a = m - np.eye(n)
        res = np.concatenate([t - x, a @ q, a @ r - q, [0.5 * (q @ q - 1.0)], [q @ r]])
        jac = np.zeros((3 * n + 2, 3 * n + 2))
        jac[:n, :n] = a
        jac[:n, n] = dp[:, 1] / (eh * math.log(10.0))
        jac[:n, n + 1] = dp[:, 0] / (eo * math.log(10.0))
        for vec, rows in ((q, slice(n, 2 * n)), (r, slice(2 * n, 3 * n))):
            for j in range(n + 2):
                xp, xm = x.copy(), x.copy()
                ehp = ehm = eh
                eop = eom = eo
                if j < n:
                    step = self.h * max(1.0, abs(x[j]))
                    xp[j] += step
                    xm[j] -= step
                elif j == n:
                    step = self.h * eh
                    ehp, ehm = eh + step, eh - step
                else:
                    step = self.h * eo
                    eop, eom = eo + step, eo - step
                _, vp = f.monodromy_vector(xp, self._logp(ehp, eop), PERIOD, vec)
                _, vm = f.monodromy_vector(xm, self._logp(ehm, eom), PERIOD, vec)
                jac[rows, j] = (vp - vm) / (2.0 * step)
        jac[n:2 * n, n + 2:2 * n + 2] = a
        jac[2 * n:3 * n, 2 * n + 2:] = a
        jac[2 * n:3 * n, n + 2:2 * n + 2] = -np.eye(n)
        jac[3 * n, n + 2:2 * n + 2] = q
        jac[3 * n + 1, n + 2:2 * n + 2] = r
        jac[3 * n + 1, 2 * n + 2:] = q
        return res, jac, m, points


def chain(args):
    net, x, y, _ = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    n = f.n
    nodes = np.load(args.nodes)["z"]
    w = nodes[args.index]
    xs, lx, ly = w[:n], w[26], w[27]
    eh, eo = 10.0 ** ly, 10.0 ** lx
    _, m, _, _ = f.period_map(xs, [lx, ly], PERIOD)
    a = m - np.eye(n)
    u, sv, vt = np.linalg.svd(a)
    q = vt[-1] / np.linalg.norm(vt[-1])
    r = np.linalg.lstsq(a, q, rcond=None)[0]
    r -= (r @ q) * q
    z = np.concatenate([xs, [eh, eo], q, r])
    system = ChainSystem(f)
    history = []
    for _ in range(args.max_newton):
        res, jac, m, points = system.residual_jacobian(z)
        history.append(float(np.linalg.norm(res)))
        if history[-1] < args.tol:
            break
        z = z + np.linalg.lstsq(jac, -res, rcond=None)[0]
    res, jac, m, points = system.residual_jacobian(z)
    history.append(float(np.linalg.norm(res)))
    mult = np.linalg.eigvals(m)
    order = np.argsort(np.abs(mult - 1.0))
    record = {"node_index": args.index, "newton": history,
              "eta_h": float(z[n]), "eta_o": float(z[n + 1]),
              "log10_hidden": math.log10(z[n]), "log10_output": math.log10(z[n + 1]),
              "multipliers_nearest_one": [[float(mult[i].real), float(mult[i].imag)] for i in order[:4]],
              "complement_radius": float(np.max(np.abs(mult[order[2:]]))),
              "min_divisor_separation": min(divisor_separations(points, PERIOD).values()),
              "jacobian_condition": float(np.linalg.cond(jac)),
              "z": z.tolist()}
    print(json.dumps({k: v for k, v in record.items() if k != "z"}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "chain-seed.json").write_text(json.dumps(record, indent=1))
    if not args.prove or history[-1] > 1e-9:
        return
    from flint import arb, ctx
    from neural_shrimp.rigorous_cusps import TwoRateCuspMap
    from neural_shrimp.rigorous_generalized_chain import chain_root_proof

    ctx.prec = args.precision
    from fractions import Fraction

    def q(v):
        return str(Fraction(float(v)))                # exact value of the binary64 constant

    inputs = {"schema": "neural-shrimp-c1-input-v1",
              "network": {"activation": "tanh", "bias": True, "d": 2, "width": 3, "in_scale": q(1.0),
                          "out_scale": q(1.0), "loss_scale": q(0.5), "ridge": q(net.ridge)},
              "x": [[q(v) for v in row] for row in np.asarray(x)], "y": [q(v) for v in np.asarray(y)],
              "lr_hidden": q(z[n]), "lr_output": q(z[n + 1])}
    model = TwoRateCuspMap(inputs)
    initial = [arb(float(v)) for v in z]
    tic = time.perf_counter()
    proof = chain_root_proof(model, initial, PERIOD, radius_exponent=args.radius_exponent,
                             newton_steps=args.proof_newton)
    seconds = time.perf_counter() - tic
    summary = {"passed": proof["passed"], "gates": proof["gates"], "contraction": proof["contraction"],
               "seconds": round(seconds, 1), "precision": args.precision,
               "radius_exponent": args.radius_exponent}
    print(json.dumps(summary))
    (out / "chain-proof.json").write_text(json.dumps(proof, indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ch = sub.add_parser("chain")
    ch.add_argument("--nodes", default=str(DATA / "explore/window-organization/upperfold-continuation-002/nodes-direction+1.npz"))
    ch.add_argument("--index", type=int, default=72)
    ch.add_argument("--max-newton", type=int, default=25)
    ch.add_argument("--tol", type=float, default=1e-12)
    ch.add_argument("--prove", action="store_true")
    ch.add_argument("--precision", type=int, default=256)
    ch.add_argument("--radius-exponent", type=int, default=-70)
    ch.add_argument("--proof-newton", type=int, default=8)
    ch.add_argument("--out", required=True)
    args = parser.parse_args()
    {"chain": chain}[args.command](args)


if __name__ == "__main__":
    main()
