"""Uniform parameter-dependent multiple-shooting tubes, enclosed with Arb.

No float decides a gate. Passing means a continuous locally unique primitive
attracting branch on the declared exact-real interval, anchored only when the
separate common-parameter inclusion gate passes. No boundary connection follows.
"""
from fractions import Fraction
from flint import arb, arb_mat, ctx
from .rigorous_orbits import (BallTanhMap, exact, eye, midpoint_matrix,
    norm_inf_upper, ball_json, vector_json, matrix_json)


def require(value, message):
    if not value:
        raise ValueError(message)


def restore_ball(value):
    require(set(value) == {'mid', 'rad', 'display'}, 'finite serialized ball required')
    require(all(type(x) is int for k in ('mid', 'rad') for x in value[k])
            and len(value['mid']) == len(value['rad']) == 2, 'dyadic ball format')
    require(value['rad'][0] >= 0, 'nonnegative radius')
    return arb(arb(tuple(value['mid'])), arb(tuple(value['rad'])))


class HiddenRateMap:
    """Reuse the reviewed C1 objective derivatives; only actual hidden rate varies."""
    def __init__(self, inputs):
        self.unit = BallTanhMap(inputs)
        self.n, self.hidden = self.unit.n, self.unit.hidden
        self.output = exact(inputs['lr_output'])
        self.unit.rates = [arb(1)] * self.n

    def step(self, theta, hidden_rate):
        require(hidden_rate.is_finite() and hidden_rate > 0, 'positive rate interval')
        unit_image, unit_jac = self.unit.step(theta)
        grad = [theta[i] - unit_image[i] for i in range(self.n)]
        rates = [hidden_rate] * self.hidden + [self.output] * (self.n-self.hidden)
        image = [theta[i] - rates[i]*grad[i] for i in range(self.n)]
        jac = arb_mat([[int(i == j) - rates[i]*(int(i == j)-unit_jac[i,j])
                        for j in range(self.n)] for i in range(self.n)])
        parameter = [(-grad[i] if i < self.hidden else arb(0)) for i in range(self.n)]
        return image, jac, parameter


def shooting(model, states, rate, tick=lambda: None):
    period, n = len(states), model.n
    dim = period*n
    residual, parameter, blocks = [], [], []
    jac = arb_mat(dim, dim)
    for j, state in enumerate(states):
        tick()
        image, block, partial = model.step(state, rate)
        residual.extend(image[i]-states[(j+1)%period][i] for i in range(n))
        parameter.extend(partial)
        blocks.append(block)
        for i in range(n):
            for k in range(n):
                jac[j*n+i,j*n+k] += block[i,k]
            jac[j*n+i,((j+1)%period)*n+i] -= 1
    return arb_mat([[v] for v in residual]), jac, arb_mat([[v] for v in parameter]), blocks


def balanced_product(blocks):
    """Native J_(p-1)...J_0, balanced parenthesization limits interval wrapping."""
    require(bool(blocks), 'nonempty product')
    level = list(blocks)
    while len(level) > 1:
        level = [level[i+1]*level[i] if i+1 < len(level) else level[i]
                 for i in range(0,len(level),2)]
    return level[0]


def power_gate(monodromy, maximum):
    rows, matrix, exponent = [], monodromy, 1
    while exponent <= maximum:
        upper = norm_inf_upper(matrix)
        rows.append({'power': exponent, 'norm_inf_upper': ball_json(upper)})
        if upper.is_finite() and upper < 1:
            return True, exponent, rows
        if not upper.is_finite():
            break
        matrix = matrix*matrix
        exponent *= 2
    return False, None, rows


def tube(model, initial, rate, *, halfwidth_exponent=-20, radius_exponent=-28,
         precision=256, newton_steps=2, max_power=4096, anchor=None,
         tick=lambda:None, observe=lambda kind,record:None):
    """Prove H(p(h)+e,h)=0 for every h, e in a common radius box.

    p(h)=U+s(h-h0), D=[-delta,delta], E=[-r,r]^N, X=p(D)+E.
    H(p(h),h) in H(U,h0)+(H_U(X,L)s+H_h(X,L))*D by the mean value theorem.
    With exact dyadic C, K_h(E) is enclosed uniformly by
    -C*[H0+(A*s+b)D]+(I-C*A)E. Strict inclusion and norm(I-C*A)<1
    give existence, local uniqueness, nonsingular derivative and continuous
    dependence on h. An existing anchor root at h0 is identified only if its
    entire phase enclosure lies in the shooting uniqueness box U+E.
    """
    require(type(precision) is int and 128 <= precision <= 512, 'precision cap')
    require(type(newton_steps) is int and 0 <= newton_steps <= 4, 'proposal cap')
    require(type(max_power) is int and 1 <= max_power <= 4096, 'power cap')
    require(all(type(v) is int and -100 <= v <= -1 for v in
                (halfwidth_exponent, radius_exponent)), 'width bounds')
    period, n = len(initial), model.n
    require(1 <= period <= 18 and 1 <= n <= 13 and all(len(row)==n for row in initial), 'dimension cap')
    require(isinstance(rate,str), 'exact rational rate')
    with ctx.workprec(precision):
        h0 = exact(rate); delta = arb((1,halfwidth_exponent)); radius = arb((1,radius_exponent))
        interval = arb(h0,delta)
        require(h0.is_exact() and interval > 0, 'exact center and positive interval')
        states = [[exact(v).mid() for v in row] for row in initial]
        for iteration in range(newton_steps):
            residual, jac, _, _ = shooting(model,states,h0,tick)
            tick(); correction = jac.solve(residual)
            states = [[(states[j][i]-correction[j*n+i,0]).mid() for i in range(n)] for j in range(period)]
            observe('proposal',{'iteration':iteration,'states':[vector_json(row) for row in states],
                                'residual_before_update':ball_json(norm_inf_upper(residual))})
        hbase, abase, bbase, _ = shooting(model,states,h0,tick)
        tick(); inverse = midpoint_matrix(abase.inv())
        slope = midpoint_matrix(-inverse*bbase)
        d = arb(0,delta)
        flat = [v for row in states for v in row]
        boxes = [flat[i]+slope[i,0]*d+arb(0,radius) for i in range(period*n)]
        phase_boxes = [boxes[j*n:(j+1)*n] for j in range(period)]
        _, abox, bbox, blocks = shooting(model,phase_boxes,interval,tick)
        path_derivative = abox*slope+bbox
        residual_tube = hbase+path_derivative*d
        defect = eye(period*n)-inverse*abox
        contraction = norm_inf_upper(defect)
        error_box = arb_mat([[arb(0,radius)] for _ in flat])
        image = -inverse*residual_tube+defect*error_box
        inclusions = [bool(abs(image[i,0]).upper() < radius) for i in range(period*n)]
        root = bool(all(inclusions) and contraction.is_finite() and contraction < 1)
        observe('root_gate',{'root':root,'contraction':ball_json(contraction),
                            'max_error_image':ball_json(norm_inf_upper(image))})
        divisors = []
        for k in range(1,period):
            if period%k:continue
            differences = [phase_boxes[k][i]-phase_boxes[0][i] for i in range(n)]
            separating = [i for i,v in enumerate(differences) if v.is_finite() and not v.contains(0)]
            divisors.append({'divisor':k,'excluded':bool(separating),
                             'separating_coordinates':separating,'differences':vector_json(differences)})
        monodromy = balanced_product(blocks)
        tick(); attracting, power, powers = power_gate(monodromy,max_power)
        anchor_checks = []
        if anchor is not None:
            require(anchor['rate']==rate and len(anchor['phase_boxes'])==period, 'exact common anchor parameter')
            for j,row in enumerate(anchor['phase_boxes']):
                require(len(row)==n,'anchor dimensions')
                anchor_checks.extend(bool(abs(restore_ball(v)-states[j][i]).upper() <= radius) for i,v in enumerate(row))
        anchored = bool(anchor is not None and all(anchor_checks))
        gates = {'uniform_root':root,'primitive_period':all(row['excluded'] for row in divisors),
                 'uniform_attraction':attracting,'anchor_identified':anchored}
        # Endpoint image enclosures are narrow witnesses, not a later join claim.
        endpoint_boxes = []
        for sign in (-1,1):
            dd = sign*delta
            endpoint_image = -inverse*(hbase+path_derivative*dd)+defect*error_box
            endpoint_boxes.append(vector_json([flat[i]+slope[i,0]*dd+endpoint_image[i,0] for i in range(period*n)]))
        return {'schema':'neural-shrimp-shooting-tube-v1','passed':all(gates.values()),
            'evidence_rung':3 if all(gates.values()) else 0,'research_status':'exploratory; independent outcome review pending',
            'gates':gates,'statement':{'period':period,'phase_dimension':n,'shooting_dimension':period*n,
                'parameter':'actual hidden learning rate; fixed exact output rate',
                'scope':'continuous primitive attracting exact-real branch on one closed interval; anchored only if all gates pass'},
            'settings':dict(precision=precision,halfwidth_exponent=halfwidth_exponent,radius_exponent=radius_exponent,newton_steps=newton_steps,max_power=max_power),
            'parameter_interval':ball_json(interval),'parameter_center':rate,
            'exact_parameter_endpoints':[str(Fraction(rate)-Fraction(2)**halfwidth_exponent),
                                         str(Fraction(rate)+Fraction(2)**halfwidth_exponent)],
            'center':[vector_json(row) for row in states],'predictor_slope':matrix_json(slope),
            'phase_boxes':[vector_json(row) for row in phase_boxes],'correction_radius':ball_json(radius),
            'residual_center':matrix_json(hbase),'path_derivative':matrix_json(path_derivative),
            'residual_tube':matrix_json(residual_tube),'preconditioner':matrix_json(inverse),
            'jacobian_tube':matrix_json(abox),'parameter_partial_tube':matrix_json(bbox),
            'defect':matrix_json(defect),'contraction':ball_json(contraction),
            'image_correction':matrix_json(image),'coordinate_inclusions':inclusions,
            'divisors':divisors,'phase_jacobians':[matrix_json(j) for j in blocks],
            'monodromy':matrix_json(monodromy),'power_bounds':powers,'attraction_power':power,
            'anchor_inclusions':anchor_checks,'endpoint_root_enclosures':endpoint_boxes,
            'not_claimed':['connection to fold/flip','whole window','parameter interval beyond this tube',
                           'finite precision trajectory theorem','utility','formal proof assistant verification']}
