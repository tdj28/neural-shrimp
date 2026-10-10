"""Known-root localization and physical stability on quadratic gauge slices.

Existence/primitive period come from the authenticated parent chart. The
intersection identity only tightens its already known root, for each fixed s.
"""
from fractions import Fraction
from flint import arb, arb_mat, ctx
from .rigorous_orbits import exact, eye, ball_json, vector_json, matrix_json
from .rigorous_connection import require, restore_ball
from .rigorous_connection_frames import restore_matrix
from .rigorous_arclength_connection import bordered
from .rigorous_periodic_lyapunov import periodic_witness


def localize_slice(model, chart, left, right, *, iterations=3, precision=256,
                   synthetic_profile=False, tick=lambda: None,
                   observe=lambda k, r: None):
    require(chart['passed'] and all(chart['gates'].values()) or synthetic_profile,
            'qualified parent required')
    require(type(iterations) is int and 1 <= iterations <= 3, 'intersection cap')
    require(type(precision) is int and 128 <= precision <= 512, 'precision cap')
    lo, hi = Fraction(left), Fraction(right)
    a, b = map(Fraction, chart['exact_parameter_endpoints'])
    require(a <= lo <= hi <= b, 'closed subinterval of parent')
    p, n = chart['statement']['period'], model.n
    size = p * n + 1
    require(1 <= p <= 18 and 1 <= n <= 13 and chart['statement']['phase_dimension'] == n,
            'native dimension identity')
    with ctx.workprec(precision):
        base = [restore_ball(v) for v in chart['center']]
        normal = [exact(v) for v in chart['normal']]
        origin = [exact(v) for v in chart['origin']]
        slope = restore_matrix(chart['predictor_slope'])
        curve = restore_matrix(chart['predictor_curvature'])
        pre = restore_matrix(chart['preconditioner'])
        known = restore_matrix(chart['image_correction'])
        require(len(base) == len(normal) == len(origin) == size, 'full native boxes')
        require(all(m.nrows() == size and m.ncols() == 1 for m in (slope, curve, known))
                and pre.nrows() == pre.ncols() == size, 'full predictor dimensions')
        fixed = base + normal + origin + [m[i, j] for m in (slope, curve, pre)
                                         for i in range(m.nrows()) for j in range(m.ncols())]
        require(all(v.is_finite() and v.is_exact() for v in fixed), 'exact fixed predictor and preconditioner')
        require(all(known[i, 0].is_finite() for i in range(size)), 'finite inherited K image')
        sc, half = (lo + hi) / 2, (hi - lo) / 2
        shift = exact(str(sc - Fraction(chart['parameter_center'])))
        d = arb(0, exact(str(half)))
        shifted = [base[i] + slope[i, 0] * shift + curve[i, 0] * shift * shift / 2 for i in range(size)]
        velocity = slope + curve * shift
        center = [(shifted[i] + known[i, 0].mid()).mid() for i in range(size)]
        error = arb_mat([[shifted[i] + known[i, 0] - center[i]] for i in range(size)])
        require(center[-1] > 0, 'positive center rate')
        center_residual, center_jac, _ = bordered(model, center, p, normal, origin, exact(str(sc)), tick)
        rhs = arb_mat([[int(i == size - 1)] for i in range(size)])
        start = {'exact_endpoints': [str(lo), str(hi)], 'center_parameter': str(sc),
                 'center': vector_json(center), 'shifted_parent_center': vector_json(shifted),
                 'slope': matrix_json(velocity), 'curvature': matrix_json(curve),
                 'normal': chart['normal'], 'origin': chart['origin'],
                 'preconditioner': matrix_json(pre), 'initial_error': matrix_json(error),
                 'inherited_image': chart['image_correction'],
                 'center_residual': matrix_json(center_residual), 'center_jacobian': matrix_json(center_jac)}
        observe('slice_start', start)
        rows = []
        predictor = [center[i] + velocity[i, 0] * d + curve[i, 0] * d * d / 2 for i in range(size)]
        path_velocity = velocity + curve * d
        for iteration in range(iterations):
            domain = [predictor[i] + arb(0).union(error[i, 0]) for i in range(size)]
            require(domain[-1].is_finite() and domain[-1] > 0, 'positive entire physical-rate domain')
            _, jac, _ = bordered(model, domain, p, normal, origin, arb(exact(str(sc)), exact(str(half))), tick)
            path = jac * path_velocity - rhs
            residual = center_residual + path * d
            defect = eye(size) - pre * jac
            image = -pre * residual + defect * error
            require(all(image[i, 0].is_finite() for i in range(size)), 'finite containment image')
            updated = arb_mat([[error[i, 0].intersection(image[i, 0])] for i in range(size)])
            row = {'iteration': iteration, 'prior_error': matrix_json(error),
                   'predictor': vector_json(predictor), 'path_velocity': matrix_json(path_velocity),
                   'derivative_domain': vector_json(domain), 'jacobian': matrix_json(jac),
                   'path_derivative': matrix_json(path), 'centered_residual': matrix_json(residual),
                   'defect': matrix_json(defect), 'image': matrix_json(image),
                   'retained_error': matrix_json(updated)}
            observe('slice_intersection', row)
            rows.append(row)
            error = updated
        final = [predictor[i] + error[i, 0] for i in range(size)]
        return {'schema': 'known-quadratic-gauge-slice-v1', 'passed': True,
                'scientific_claim': not synthetic_profile, 'floor_only': lo == hi,
                'exact_endpoints': [str(lo), str(hi)], 'precision': precision,
                'iterations': iterations, 'start': start, 'intersections': rows,
                'final_error': matrix_json(error), 'final_state_box': vector_json(final),
                'scope': 'same inherited primitive root for each gauge in this closed slice; no new existence or join',
                'theorem': 'Shift the exact quadratic predictor and include all recentering displacement in E. The full domain q(D)+conv(0,E) contains predictor paths and root-to-predictor segments, including actual hidden rate. G(q(d),sc+d) lies in G(center,sc)+(J_domain*(S_shift+T*D)-e_last)*D. For every fixed gauge, the inherited error e satisfies e=-C*G(q(d),sc+d)+(I-C*A)e. Intersect E with this image; no contraction is needed for conditional containment.'}


def slice_stability(model, chart, left, right, *, iterations=3, precision=256,
                    synthetic_profile=False, tick=lambda: None,
                    observe=lambda k, r: None):
    with ctx.workprec(precision):
        local = localize_slice(model, chart, left, right, iterations=iterations,
                               precision=precision, synthetic_profile=synthetic_profile,
                               tick=tick, observe=observe)
        box = [restore_ball(v) for v in local['final_state_box']]
        p, n = chart['statement']['period'], model.n
        require(box[-1].is_finite() and box[-1] > 0, 'positive final physical-rate interval')
        blocks = []
        for j in range(p):
            tick()
            _, jac, _ = model.step(box[j * n:(j + 1) * n], box[-1])
            blocks.append(jac)
        observe('slice_physical_jacobians', {'state_box': local['final_state_box'],
                                            'phase_jacobians': [matrix_json(m) for m in blocks]})
        witness = periodic_witness(blocks, precision=precision, tick=tick, observe=observe)
        return {'localization': local, 'stability': witness, 'passed': witness['passed'],
                'scope': 'attraction only for this inherited closed gauge slice if all periodic metric gates pass'}
