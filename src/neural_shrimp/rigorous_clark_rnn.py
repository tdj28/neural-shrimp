"""Native exact-real finite random RNN orbit certificate.

C1 arithmetic is transparently copied from rigorous_orbits.check_orbit, with
only the map factory/statement and an optional odd-companion disjointness check
changed. Historical GD kernels stay unchanged. No floating-point gate proves
existence, primitive period or attraction.
"""
from flint import arb,arb_mat,ctx
from .rigorous_orbits import (exact,eye,refine_center,midpoint_matrix,norm_inf_upper,
                             ball_json,vector_json,matrix_json)

class BallClarkRNN:
    def __init__(self,inputs):
        if inputs.get('schema')!='clark-rnn-c1-input-v1':raise ValueError('input schema')
        self.n=len(inputs['j0'])
        if not 1<=self.n<=16 or any(len(row)!=self.n for row in inputs['j0']):raise ValueError('matrix dimension')
        self.j=[[exact(v) for v in row] for row in inputs['j0']]
        self.g=exact(inputs['gain']);self.delta=exact(inputs['delta'])
        if not self.g>0 or not 0<self.delta<=1:raise ValueError('gain/leak range')
    def step(self,x):
        if len(x)!=self.n:raise ValueError('state dimension')
        t=[v.tanh() for v in x];s=[1-v*v for v in t]
        out=[(1-self.delta)*x[i]+self.delta*self.g*sum((self.j[i][k]*t[k] for k in range(self.n)),arb(0)) for i in range(self.n)]
        jac=arb_mat([[(1-self.delta)*int(i==k)+self.delta*self.g*self.j[i][k]*s[k] for k in range(self.n)] for i in range(self.n)])
        return out,jac
    def iterate(self,x,period):
        m=eye(self.n);orbit=[list(x)];js=[]
        for _ in range(period):
            x,a=self.step(x);m=a*m;orbit.append(x);js.append(a)
        return x,m,orbit,js

def check_rnn_orbit(inputs, *, precision=256, radius_exponent=-80, newton_steps=5, max_power=512):
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
        model = BallClarkRNN(inputs)
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
        # Squaring reuses small enclosures of M^(2^j), avoiding the
        # catastrophic wrapping from 512 sequential multiplications by M.
        powers, power = [], monodromy
        attracting, witness, k = False, None, 1
        while k <= max_power:
            bound = norm_inf_upper(power)
            powers.append({"power": k, "norm_inf_upper": ball_json(bound)})
            if bound.is_finite() and bound < 1:
                attracting, witness = True, k
                break
            if not bound.is_finite():
                break
            power = power * power
            k *= 2
        primitive = all(d["excluded"] for d in divisor_checks)
        passed = bool(existence and primitive and attracting)
        odd_checks = []
        for phase in range(period):
            differences = [orbit[phase][i] + box[i] for i in range(model.n)]
            excludes = [i for i, diff in enumerate(differences)
                        if diff.is_finite() and not diff.contains(0)]
            odd_checks.append({"phase": phase, "excluded": bool(excludes),
                               "separating_coordinates": excludes,
                               "differences": vector_json(differences)})
        return {"odd_companion_distinct": bool(passed and all(c["excluded"] for c in odd_checks)),
                "odd_companion_checks": odd_checks, "schema": "clark-rnn-c1-result-v1", "passed": passed,
                "evidence_rung": 3 if passed else 0,
                "research_status": "exploratory certificate pilot; independent review pending",
                "exact_statement": {"primitive_period": period, "dimension": model.n,
                    "existence_and_local_uniqueness": existence, "primitive_period_excluded_divisors": primitive,
                    "attraction": attracting, "model": "exact-real fixed random leaky RNN with supplied dyadic J0/g/delta"},
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
