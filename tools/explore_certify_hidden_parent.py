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
  resonance  The hidden tongue's root R3: a period-2 orbit at rates where the
          two-step Jacobian has the eigenvalue exp(i 2 pi p/q) exactly (2/9 by
          default). Unknowns (x, eta_h, eta_o, v_re, v_im); equations
          F^2(x) - x = 0, (M - e^{i w}) v = 0 (real and imaginary parts),
          (|v|^2 - 1)/2 = 0, v_re.v_im = 0. Float64 Newton from a saved
          period-2 state, then an outward-rounded Krawczyk enclosure modeled
          on Codex's chain proof (same mixed-jet derivatives), with cos and
          sin of the angle enclosed in Arb.
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
    if not proof["passed"]:
        return
    # conditional complementary spectrum over the proved box
    from neural_shrimp.rigorous_chain_spectrum import chain_complement
    from neural_shrimp.rigorous_generalized_chain import augmented_chain

    def ball(entry):
        m, e = entry["mid"]
        return arb(m) * arb(2) ** e

    radius = arb(0, arb(2) ** args.radius_exponent)
    box = [ball(v) + radius for v in proof["center"]]
    tic = time.perf_counter()
    _, _, data = augmented_chain(model, box, PERIOD)
    spectrum = chain_complement(data["monodromy"], data["q"], data["r"])
    discs = spectrum["spectral_exclusion"].get("discs", [])
    print(json.dumps({"complement_passed": spectrum["passed"],
                      "complement_attracting": spectrum["complement_attracting"],
                      "discs": len(discs), "seconds": round(time.perf_counter() - tic, 1)}))
    (out / "chain-complement.json").write_text(json.dumps(spectrum, indent=1, default=str))


class EigenSystem:
    """Float64 residual/Jacobian of the fixed-angle eigenpair system for a period-p orbit."""

    def __init__(self, fmap, period, omega, h=1e-6):
        self.f, self.p, self.c, self.s, self.h = fmap, period, math.cos(omega), math.sin(omega), h

    def residual_jacobian(self, z):
        f, n, c, s = self.f, self.f.n, self.c, self.s
        x, eh, eo = z[:n], z[n], z[n + 1]
        vr, vi = z[n + 2:2 * n + 2], z[2 * n + 2:]
        lp = [math.log10(eo), math.log10(eh)]
        t, m, dp, points = f.period_map(x, lp, self.p)
        a = m - np.eye(n)
        res = np.concatenate([t - x, a @ vr + (1 - c) * vr + s * vi, a @ vi + (1 - c) * vi - s * vr,
                              [0.5 * (vr @ vr + vi @ vi - 1.0)], [vr @ vi]])
        jac = np.zeros((3 * n + 2, 3 * n + 2))
        jac[:n, :n] = a
        jac[:n, n] = dp[:, 1] / (eh * math.log(10.0))
        jac[:n, n + 1] = dp[:, 0] / (eo * math.log(10.0))
        for vec, rows in ((vr, slice(n, 2 * n)), (vi, slice(2 * n, 3 * n))):
            for j in range(n + 2):
                xp, xm = x.copy(), x.copy()
                ehp = ehm = eh
                eop = eom = eo
                if j < n:
                    st = self.h * max(1.0, abs(x[j]))
                    xp[j] += st
                    xm[j] -= st
                elif j == n:
                    st = self.h * eh
                    ehp, ehm = eh + st, eh - st
                else:
                    st = self.h * eo
                    eop, eom = eo + st, eo - st
                _, mp_ = f.monodromy_vector(xp, [math.log10(eop), math.log10(ehp)], self.p, vec)
                _, mm_ = f.monodromy_vector(xm, [math.log10(eom), math.log10(ehm)], self.p, vec)
                jac[rows, j] = (mp_ - mm_) / (2.0 * st)
        eye = np.eye(n)
        jac[n:2 * n, n + 2:2 * n + 2] = a + (1 - c) * eye
        jac[n:2 * n, 2 * n + 2:] = s * eye
        jac[2 * n:3 * n, n + 2:2 * n + 2] = -s * eye
        jac[2 * n:3 * n, 2 * n + 2:] = a + (1 - c) * eye
        jac[3 * n, n + 2:2 * n + 2] = vr
        jac[3 * n, 2 * n + 2:] = vi
        jac[3 * n + 1, n + 2:2 * n + 2] = vi
        jac[3 * n + 1, 2 * n + 2:] = vr
        return res, jac, m, points


def augmented_eigenpair(model, z, period, cos_w, sin_w):
    """Arb residual and Jacobian of the fixed-angle eigenpair system (mixed jets as in augmented_chain)."""
    from flint import arb, arb_mat
    from neural_shrimp.rigorous_folds import MixedJet
    from neural_shrimp.rigorous_orbits import eye

    n = model.n
    base = n + 2
    x = z[:n]
    vr = z[base:base + n]
    vi = z[base + n:]

    def propagate(direction):
        state = [MixedJet.variable(x[i], base, i, direction[i]) for i in range(n)]
        a = MixedJet.variable(z[n], base, n)
        b = MixedJet.variable(z[n + 1], base, n + 1)
        initial = state
        orbit = [[v.f for v in state]]
        for _ in range(period):
            state = model.cusp_step(state, a, b)
            orbit.append([v.f for v in state])
        return [u - v for u, v in zip(state, initial)], orbit

    first, orbit = propagate(vr)
    second, _ = propagate(vi)
    amat = arb_mat([[v.g[k] for k in range(n)] for v in first])
    one_c = 1 - cos_w
    value = [v.f for v in first]
    value += [first[i].d + one_c * vr[i] + sin_w * vi[i] for i in range(n)]
    value += [second[i].d + one_c * vi[i] - sin_w * vr[i] for i in range(n)]
    value += [(sum((v * v for v in vr), arb(0)) + sum((v * v for v in vi), arb(0)) - 1) / 2,
              sum((vr[i] * vi[i] for i in range(n)), arb(0))]
    jac = [list(v.g) + [arb(0)] * (2 * n) for v in first]
    jac += [list(first[i].m) + [amat[i, k] + (one_c if i == k else 0) for k in range(n)]
            + [sin_w if i == k else arb(0) for k in range(n)] for i in range(n)]
    jac += [list(second[i].m) + [(-sin_w) if i == k else arb(0) for k in range(n)]
            + [amat[i, k] + (one_c if i == k else 0) for k in range(n)] for i in range(n)]
    jac += [[arb(0)] * base + list(vr) + list(vi), [arb(0)] * base + list(vi) + list(vr)]
    return arb_mat([[v] for v in value]), arb_mat(jac), {"orbit": orbit, "monodromy": amat + eye(n)}


def eigenpair_root_proof(model, initial, period, cos_w, sin_w, radius_exponent=-70, newton_steps=8):
    from flint import arb, arb_mat
    from neural_shrimp.rigorous_orbits import eye, midpoint_matrix, norm_inf_upper, ball_json, vector_json

    n = model.n
    dim = 3 * n + 2
    center = [v.mid() for v in initial]
    for _ in range(newton_steps):
        v, j, _ = augmented_eigenpair(model, center, period, cos_w, sin_w)
        step = j.inv() * v
        center = [(center[i] - step[i, 0]).mid() for i in range(dim)]
    value, j, _ = augmented_eigenpair(model, center, period, cos_w, sin_w)
    pre = midpoint_matrix(j.inv())
    radius = arb((1, radius_exponent))
    box = [arb(v, radius) for v in center]
    _, whole, data = augmented_eigenpair(model, box, period, cos_w, sin_w)
    defect = eye(dim) - pre * whole
    bound = norm_inf_upper(defect)
    image = -pre * value + defect * arb_mat([[arb(0, radius)] for _ in center])
    inclusions = [bool(box[i].contains_interior(center[i] + image[i, 0])) for i in range(dim)]
    root = bool(all(inclusions) and bound.is_finite() and bound < 1)
    divisors = []
    for d in range(1, period):
        if period % d:
            continue
        diff = [data["orbit"][d][i] - box[i] for i in range(n)]
        divisors.append({"divisor": d, "excluded": any(v.is_finite() and not v.contains(0) for v in diff)})
    gates = {"augmented_root_unique": root, "positive_actual_rates": bool(box[n] > 0 and box[n + 1] > 0),
             "primitive_period": all(v["excluded"] for v in divisors) if divisors else True}
    return {"passed": all(gates.values()), "gates": gates, "contraction": ball_json(bound),
            "center": vector_json(center), "box_radius_exponent": radius_exponent,
            "rates_box": [ball_json(box[n]), ball_json(box[n + 1])], "monodromy_box": data["monodromy"]}


def real_eigenpair_proof(mono, guess, radius_exponent=-60):
    """Krawczyk test for a real eigenpair (u, lam) valid for every matrix in the interval enclosure mono."""
    from flint import arb, arb_mat
    from neural_shrimp.rigorous_orbits import eye, midpoint_matrix, norm_inf_upper, ball_json

    n = mono.nrows()
    mid = np.array([[float(mono[i, j].mid()) for j in range(n)] for i in range(n)])
    vals, vecs = np.linalg.eig(mid)
    k = int(np.argmin(np.abs(vals - guess)))
    lam0 = float(vals[k].real)
    u0 = np.real(vecs[:, k])
    u0 /= np.linalg.norm(u0)
    centre = [arb(float(v)) for v in u0] + [arb(lam0)]

    def system(z, matrix):
        u, lam = z[:n], z[n]
        value = [sum((matrix[i, j] * u[j] for j in range(n)), arb(0)) - lam * u[i] for i in range(n)]
        value.append((sum((v * v for v in u), arb(0)) - 1) / 2)
        jac = [[matrix[i, j] - (lam if i == j else 0) for j in range(n)] + [-u[i]] for i in range(n)]
        jac.append(list(u) + [arb(0)])
        return arb_mat([[v] for v in value]), arb_mat(jac)

    value, jac = system(centre, mono)
    pre = midpoint_matrix(jac.inv())
    radius = arb((1, radius_exponent))
    box = [arb(c.mid(), radius) for c in centre]
    _, whole = system(box, mono)
    defect = eye(n + 1) - pre * whole
    bound = norm_inf_upper(defect)
    image = -pre * value + defect * arb_mat([[arb(0, radius)] for _ in box])
    inclusions = [bool(box[i].contains_interior(centre[i] + image[i, 0])) for i in range(n + 1)]
    passed = bool(all(inclusions) and bound.is_finite() and bound < 1)
    lam_box = centre[n] + image[n, 0]
    return {"passed": passed, "lambda": lam_box.str(20), "lambda_above_one": bool(passed and lam_box > 1),
            "contraction": bound.str(10), "statement": "For every matrix in the monodromy enclosure there is a real "
            "eigenvalue in the stated ball (Krawczyk on the eigenpair system); conditional on the enclosure."}


def resonance(args):
    net, x, y, _ = make_s2(SEED)
    f = TwoRateMap(net, x, y)
    n = f.n
    seed = json.loads(Path(args.seed_json).read_text())
    xs = np.array(seed["theta"])
    eh, eo = 10.0 ** seed["ly"], 10.0 ** seed["lx"]
    omega = 2 * math.pi * args.p / args.q
    _, m, _, _ = f.period_map(xs, [seed["lx"], seed["ly"]], 2)
    mult, vecs = np.linalg.eig(m)
    i = int(np.argmin(np.abs(mult - complex(math.cos(omega), math.sin(omega)))))
    v = vecs[:, i]
    v = v / np.linalg.norm(v)
    # rotate the complex phase so that v_re . v_im = 0
    a_, b_ = np.real(v), np.imag(v)
    phi = 0.5 * math.atan2(-2 * (a_ @ b_), (a_ @ a_ - b_ @ b_))
    v = v * complex(math.cos(phi), math.sin(phi))
    z = np.concatenate([xs, [eh, eo], np.real(v), np.imag(v)])
    system = EigenSystem(f, 2, omega)
    history = []
    for _ in range(args.max_newton):
        res, jac, m, pts = system.residual_jacobian(z)
        history.append(float(np.linalg.norm(res)))
        if history[-1] < 1e-13:
            break
        z = z + np.linalg.lstsq(jac, -res, rcond=None)[0]
    res, jac, m, pts = system.residual_jacobian(z)
    history.append(float(np.linalg.norm(res)))
    mult = np.sort_complex(np.linalg.eigvals(m))[::-1]
    mods = sorted(np.abs(np.linalg.eigvals(m)), reverse=True)
    record = {"p_over_q": f"{args.p}/{args.q}", "newton": history, "eta_h": float(z[n]), "eta_o": float(z[n + 1]),
              "log10_hidden": math.log10(z[n]), "log10_output": math.log10(z[n + 1]),
              "multiplier_moduli_top4": [float(v) for v in mods[:4]],
              "separation_F": float(np.linalg.norm(pts[1] - pts[0])), "jacobian_condition": float(np.linalg.cond(jac)),
              "z": z.tolist()}
    print(json.dumps({k: v for k, v in record.items() if k != "z"}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resonance-seed.json").write_text(json.dumps(record, indent=1))
    if not args.prove or history[-1] > 1e-9:
        return
    from fractions import Fraction
    from flint import arb, ctx
    from neural_shrimp.rigorous_cusps import TwoRateCuspMap
    from neural_shrimp.rigorous_cusps import spectral_exclusion

    ctx.prec = args.precision

    def qf(val):
        return str(Fraction(float(val)))

    inputs = {"schema": "neural-shrimp-c1-input-v1",
              "network": {"activation": "tanh", "bias": True, "d": 2, "width": 3, "in_scale": qf(1.0),
                          "out_scale": qf(1.0), "loss_scale": qf(0.5), "ridge": qf(net.ridge)},
              "x": [[qf(v) for v in row] for row in np.asarray(x)], "y": [qf(v) for v in np.asarray(y)],
              "lr_hidden": qf(z[n]), "lr_output": qf(z[n + 1])}
    model = TwoRateCuspMap(inputs)
    w = arb.pi() * 2 * args.p / args.q
    tic = time.perf_counter()
    proof = eigenpair_root_proof(model, [arb(float(v)) for v in z], 2, w.cos(), w.sin(),
                                 radius_exponent=args.radius_exponent)
    summary = {"passed": proof["passed"], "gates": proof["gates"], "contraction": proof["contraction"]["display"],
               "rates_box": [r["display"] for r in proof["rates_box"]], "seconds": round(time.perf_counter() - tic, 1)}
    print(json.dumps(summary))
    mono = proof.pop("monodromy_box")
    (out / "resonance-proof.json").write_text(json.dumps(proof, indent=1, default=str))
    if proof["passed"] and args.real_eigen is not None:
        res = real_eigenpair_proof(mono, args.real_eigen)
        print(json.dumps({"real_eigenvalue_proof": res["passed"], "lambda": res["lambda"],
                          "lambda_above_one": res["lambda_above_one"], "contraction": res["contraction"]}))
        (out / "real-eigenpair-proof.json").write_text(json.dumps(res, indent=1, default=str))


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
    rs = sub.add_parser("resonance")
    rs.add_argument("--seed-json", default=str(DATA / "explore/window-organization/collapse-period2-001/period2.json"))
    rs.add_argument("--p", type=int, default=2)
    rs.add_argument("--q", type=int, default=9)
    rs.add_argument("--max-newton", type=int, default=25)
    rs.add_argument("--prove", action="store_true")
    rs.add_argument("--precision", type=int, default=256)
    rs.add_argument("--radius-exponent", type=int, default=-70)
    rs.add_argument("--real-eigen", type=float, default=None, help="also prove a real eigenvalue near this value")
    rs.add_argument("--out", required=True)
    args = parser.parse_args()
    {"chain": chain, "resonance": resonance}[args.command](args)


if __name__ == "__main__":
    main()
