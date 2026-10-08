"""Fail-closed Arb enclosures for exact gradient-descent maps (C1 pilot).

No floating-point eigenvalue or Newton convergence test can pass a proof gate.
Input constants are exact binary64 dyadics exported as numerator/denominator.
The checked map is real arithmetic with those constants, not float64 training.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from flint import arb, arb_mat, ctx


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def exact(value):
    """Only explicit rational strings are valid mathematical inputs."""
    if not isinstance(value, str):
        raise ValueError("expected an exact rational string")
    q = Fraction(value)
    return arb(q.numerator) / arb(q.denominator)


def dyadic(value):
    """Serialize a finite binary64 value without decimal reinterpretation."""
    n, d = float(value).as_integer_ratio()
    return f"{n}/{d}"


def ball_json(value):
    if not value.is_finite():
        return {"finite": False}
    return {"mid": [int(v) for v in value.mid().man_exp()],
            "rad": [int(v) for v in value.rad().man_exp()],
            "display": value.str(20)}


def vector_json(values):
    return [ball_json(x) for x in values]


def matrix_json(matrix):
    return [vector_json([matrix[i, j] for j in range(matrix.ncols())])
            for i in range(matrix.nrows())]


def eye(n):
    return arb_mat([[int(i == j) for j in range(n)] for i in range(n)])


def midpoint_matrix(matrix):
    return arb_mat([[matrix[i, j].mid() for j in range(matrix.ncols())]
                    for i in range(matrix.nrows())])


def norm_inf_upper(matrix):
    sums = [sum((abs(matrix[i, j]).upper() for j in range(matrix.ncols())), arb(0)).upper()
            for i in range(matrix.nrows())]
    if not all(x.is_finite() for x in sums):
        return arb("nan")
    return max(sums)


class BallTanhMap:
    """Analytic gradient and Hessian, all operations enclosed by Arb."""
    def __init__(self, inputs):
        if inputs.get("schema") != "neural-shrimp-c1-input-v1":
            raise ValueError("unknown input schema")
        net = inputs["network"]
        if net["activation"] != "tanh" or type(net["bias"]) is not bool:
            raise ValueError("unsupported model")
        self.d, self.w, self.bias = net["d"], net["width"], net["bias"]
        if type(self.d) is not int or type(self.w) is not int or not (1 <= self.d <= 8 and 1 <= self.w <= 8):
            raise ValueError("model outside bounded pilot scope")
        self.a = self.d * self.w
        self.hidden = self.a + self.w * self.bias
        self.n = self.hidden + self.w + self.bias
        self.ins, self.out, self.loss, self.ridge = [exact(net[k]) for k in
            ("in_scale", "out_scale", "loss_scale", "ridge")]
        if not all(v > 0 for v in (self.ins, self.out, self.loss)) or not self.ridge >= 0:
            raise ValueError("invalid network scales")
        self.x = [[exact(v) for v in row] for row in inputs["x"]]
        self.y = [exact(v) for v in inputs["y"]]
        if not self.x or len(self.x) != len(self.y) or any(len(row) != self.d for row in self.x):
            raise ValueError("invalid dataset dimensions")
        rh, ro = exact(inputs["lr_hidden"]), exact(inputs["lr_output"])
        if not rh > 0 or not ro > 0:
            raise ValueError("rates must be strictly positive")
        self.rates = [rh] * self.hidden + [ro] * (self.n - self.hidden)
        self.k = 2 * self.loss / len(self.x)

    def step(self, theta):
        if len(theta) != self.n:
            raise ValueError("state dimension mismatch")
        grad = [self.ridge * v for v in theta]
        hessian = [[self.ridge * int(i == j) for j in range(self.n)] for i in range(self.n)]
        for x, y in zip(self.x, self.y):
            acts, slopes, curvs, features = [], [], [], []
            for h in range(self.w):
                feature = [(d * self.w + h, self.ins * x[d]) for d in range(self.d)]
                if self.bias:
                    feature.append((self.a + h, arb(1)))
                z = sum((theta[i] * v for i, v in feature), arb(0))
                t = z.tanh()
                s = 1 - t * t
                acts.append(t)
                slopes.append(s)
                curvs.append(-2 * t * s)
                features.append(feature)
            prediction = self.out * sum((theta[self.hidden + h] * acts[h] for h in range(self.w)), arb(0))
            if self.bias:
                prediction += theta[-1]
            residual = prediction - y
            first = [arb(0) for _ in range(self.n)]
            second = [[arb(0) for _ in range(self.n)] for _ in range(self.n)]
            for h in range(self.w):
                oi = self.hidden + h
                ow = self.out * theta[oi]
                first[oi] = self.out * acts[h]
                for i, vi in features[h]:
                    first[i] = ow * slopes[h] * vi
                    second[i][oi] = second[oi][i] = self.out * slopes[h] * vi
                    for j, vj in features[h]:
                        second[i][j] = ow * curvs[h] * vi * vj
            if self.bias:
                first[-1] = arb(1)
            for i in range(self.n):
                grad[i] += self.k * residual * first[i]
                for j in range(self.n):
                    hessian[i][j] += self.k * (first[i] * first[j] + residual * second[i][j])
        result = [theta[i] - self.rates[i] * grad[i] for i in range(self.n)]
        jac = arb_mat([[int(i == j) - self.rates[i] * hessian[i][j]
                        for j in range(self.n)] for i in range(self.n)])
        return result, jac

    def iterate(self, theta, period):
        monodromy, orbit, jacobians = eye(self.n), [list(theta)], []
        for _ in range(period):
            theta, jac = self.step(theta)
            monodromy = jac * monodromy
            orbit.append(theta)
            jacobians.append(jac)
        return theta, monodromy, orbit, jacobians


def refine_center(model, center, period, iterations):
    """Produce dyadic proposals; this function confers no proof status."""
    for _ in range(iterations):
        end, monodromy, _, _ = model.iterate(center, period)
        residual = arb_mat([[end[i] - center[i]] for i in range(model.n)])
        correction = (monodromy - eye(model.n)).solve(residual)
        center = [(center[i] - correction[i, 0]).mid() for i in range(model.n)]
    return center


def check_orbit(inputs, *, precision=256, radius_exponent=-80, newton_steps=5, max_power=512):
    """Krawczyk inclusion + contraction, divisor exclusion and matrix-power norm.

    For H=F^p-I, R is an exact midpoint approximate inverse. We prove
    K=x-R H(x)+(I-R H'(X))(X-x) lies strictly inside X and
    ||I-R H'(X)||inf < 1. Banach gives existence/uniqueness in X;
    R is nonsingular because ||I-R H'(X)||inf<1. Proper divisors are
    excluded by disjoint coordinate intervals. A monodromy enclosure M
    with ||M^k||inf<1 proves strict attraction for the enclosed orbit.
    """
    if not 128 <= precision <= 2048 or not -1024 <= radius_exponent <= -1:
        raise ValueError("invalid proof precision or radius")
    period = inputs["period"]
    if type(period) is not int or not 1 <= period <= 64 or not 0 <= newton_steps <= 12 or not 1 <= max_power <= 4096:
        raise ValueError("invalid bounded proof settings")
    with ctx.workprec(precision):
        model = BallTanhMap(inputs)
        center = [exact(v).mid() for v in inputs["approximation"]]
        if len(center) != model.n:
            raise ValueError("approximation dimension mismatch")
        center = refine_center(model, center, period, newton_steps)
        end, monodromy0, _, _ = model.iterate(center, period)
        h0 = arb_mat([[end[i] - center[i]] for i in range(model.n)])
        inverse = midpoint_matrix((monodromy0 - eye(model.n)).inv())
        radius = arb((1, radius_exponent))
        box = [arb(c, radius) for c in center]
        _, monodromy, orbit, jacobians = model.iterate(box, period)
        defect = eye(model.n) - inverse * (monodromy - eye(model.n))
        contraction = norm_inf_upper(defect)
        delta = arb_mat([[arb(0, radius)] for _ in center])
        image_delta = -inverse * h0 + defect * delta
        krawczyk = [center[i] + image_delta[i, 0] for i in range(model.n)]
        inclusions = [bool(box[i].contains_interior(krawczyk[i])) for i in range(model.n)]
        existence = bool(all(inclusions) and contraction.is_finite() and contraction < 1)
        divisor_checks = []
        for d in range(1, period):
            if period % d:
                continue
            differences = [orbit[d][i] - box[i] for i in range(model.n)]
            excludes = [i for i, diff in enumerate(differences)
                        if diff.is_finite() and not diff.contains(0)]
            divisor_checks.append({"divisor": d, "excluded": bool(excludes),
                                   "separating_coordinates": excludes,
                                   "differences": vector_json(differences)})
        powers, power = [], eye(model.n)
        attracting, witness = False, None
        for k in range(1, max_power + 1):
            power = power * monodromy
            bound = norm_inf_upper(power)
            powers.append({"power": k, "norm_inf_upper": ball_json(bound)})
            if bound.is_finite() and bound < 1:
                attracting, witness = True, k
                break
            if not bound.is_finite():
                break
        primitive = all(d["excluded"] for d in divisor_checks)
        passed = bool(existence and primitive and attracting)
        return {"schema": "neural-shrimp-c1-result-v1", "passed": passed,
                "evidence_rung": 3 if passed else 0,
                "research_status": "exploratory certificate pilot; independent review pending",
                "exact_statement": {"primitive_period": period, "dimension": model.n,
                    "existence_and_local_uniqueness": existence, "primitive_period_excluded_divisors": primitive,
                    "attraction": attracting, "model": "exact-real GD with supplied exact dyadic constants"},
                "settings": {"precision": precision, "radius_exponent": radius_exponent,
                    "newton_steps": newton_steps, "max_power": max_power},
                "center": vector_json(center), "box": vector_json(box),
                "residual": matrix_json(h0), "preconditioner": matrix_json(inverse),
                "krawczyk_image": vector_json(krawczyk), "coordinate_inclusions": inclusions,
                "contraction_norm_upper": ball_json(contraction), "defect": matrix_json(defect),
                "divisors": divisor_checks, "monodromy": matrix_json(monodromy),
                "orbit_enclosures": [vector_json(o) for o in orbit],
                "jacobians": [matrix_json(j) for j in jacobians],
                "power_bounds": powers, "attraction_witness_power": witness,
                "attraction_witness": matrix_json(power) if attracting else None,
                "not_established": ["parameter-neighborhood/window geometry", "chaos", "invariant circle",
                    "training benefit", "floating-point implementation dynamics", "formal proof assistant verification"]}
