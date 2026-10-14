"""Does the alignment of successive step Jacobians make the period-8 orbit stable? (consult 7, recommendation 3)

Along the orbit, M = J_7 ... J_0 with every step expanding some direction (largest singular values 1.2-21) while
rho(M) = 0.011. Inserting orthogonal matrices R_j between the steps, M' = R_7 J_7 ... R_0 J_0, keeps every step's
singular values and changes only how the steps are aligned. If random alignments typically give rho(M') > 1, the
stability of the actual cycle rests on its particular alignment. Negative control: coordinate changes
J'_j = Q_(j+1) J_j Q_j^T (Q_8 = Q_0) leave the multipliers unchanged. Positive control: 2 x 2 steps that expand along
alternating axes (stable product) or the same axis (unstable). Endpoint: log rho / 8, nats per iterate.

Usage: python tools/alignment_intervention.py JACOBIANS.npz OUT.json [--draws 16 --seed 0]
"""
import argparse
import json

import numpy as np


def haar(d, rng):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    return q * np.sign(np.diag(r))


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def product(J, R=None):
    M = np.eye(J.shape[1])
    for j in range(len(J)):
        M = (J[j] if R is None else R[j] @ J[j]) @ M
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jacobians")
    ap.add_argument("out")
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    J = np.load(args.jacobians)["jacobians"]
    q, d = J.shape[0], J.shape[1]
    rng = np.random.default_rng(args.seed)
    base = rho(product(J))
    draws = []
    for _ in range(args.draws):
        R = [haar(d, rng) for _ in range(q)]
        draws.append(np.log(rho(product(J, R))) / q)
    Q = [haar(d, rng) for _ in range(q)]
    Jc = np.array([Q[(j + 1) % q] @ J[j] @ Q[j].T for j in range(q)])
    control = rho(product(Jc))
    A = np.diag([4.0, 0.1])
    P = np.array([[0.0, 1.0], [1.0, 0.0]])                      # swaps the axes
    alternating = rho(A @ P @ A @ P)
    aligned = rho(A @ A)
    out = {"steps": q, "dimension": d, "rho_actual": base, "log_rho_per_iterate_actual": float(np.log(base) / q),
           "log_rho_per_iterate_rotated": [float(x) for x in draws],
           "fraction_rotated_unstable": float(np.mean(np.array(draws) > 0)),
           "negative_control_rho": control, "negative_control_relative_error": abs(control - base) / base,
           "positive_control_rho_alternating": alternating, "positive_control_rho_aligned": aligned}
    open(args.out, "w").write(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                      if k != "log_rho_per_iterate_rotated"}))
    print("rotated log rho per iterate:", np.round(np.sort(draws), 3).tolist())


if __name__ == "__main__":
    main()
