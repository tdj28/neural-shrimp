"""Prototype: uniform native shooting tubes on the normalized ±1 eigenpair locus.

Only MixedJet (second F) is required. Affine predictor residuals use a mean-value
bound, not an unimplemented quadratic/fourth-F Taylor remainder. A condensed
midpoint solve proposes an exact dyadic preconditioner; the complete 470D Arb
defect decides inclusion. No genericity, complement stability or target anchor
provenance is supplied by this low-level module.
"""
from fractions import Fraction
from flint import arb, arb_mat, ctx
from .rigorous_folds import MixedJet
from .rigorous_orbits import exact, eye, midpoint_matrix, ball_json, vector_json, matrix_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(matrix):
    return all(matrix[i,j].is_finite() for i in range(matrix.nrows()) for j in range(matrix.ncols()))


def column(values):
    return arb_mat([[v] for v in values])


def joined(left, right):
    require(left.nrows() == right.nrows(), 'block rows')
    return arb_mat([[left[i,j] for j in range(left.ncols())] +
                    [right[i,j] for j in range(right.ncols())] for i in range(left.nrows())])


def phase(model, state, mode, rates):
    """Local H derivative: base variables x and two rates; q held constant.

    Model.cusp_step accepts MixedJet arithmetic. The name is inherited from
    the reviewed two-actual-rate adapter; this function makes no cusp claim.
    """
    n = model.n
    require(len(state) == len(mode) == n and len(rates) == 2, 'phase dimensions')
    require(all(v.is_finite() for v in state+mode+rates), 'finite phase inputs')
    require(all(v > 0 for v in rates), 'positive actual rates')
    x = [MixedJet.variable(state[i], n+2, i, mode[i]) for i in range(n)]
    h, o = [MixedJet.variable(rates[i], n+2, n+i) for i in range(2)]
    result = model.cusp_step(x, h, o)
    require(len(result) == n and all(isinstance(v, MixedJet) and len(v.g) == n+2 for v in result), 'local jet result')
    block = arb_mat(2*n, 2*n)
    rate_block = arb_mat(2*n, 2)
    for i, v in enumerate(result):
        for j in range(n):
            block[i,j] = v.g[j]
            block[n+i,j] = v.m[j]
            block[n+i,n+j] = v.g[j]
        for j in range(2):
            rate_block[i,j] = v.g[n+j]
            rate_block[n+i,j] = v.m[n+j]
    return column([v.f for v in result]+[v.d for v in result]), block, rate_block


def assemble(model, flat, period, sigma, normal, origin, parameter,
             tick=lambda:None, observe=lambda kind,record:None):
    """Order per phase (x_i,q_i), then (eta_h,eta_o). No phase reordering."""
    n = model.n; m = 2*n; size = m*period+2
    require(type(period) is int and 1 <= period <= 18 and 1 <= n <= 13, 'native dimensions')
    require(type(sigma) is int and sigma in (-1, 1), 'critical sign')
    require(len(flat) == len(normal) == len(origin) == size, 'shooting dimensions')
    residual = arb_mat(size, 1); jac = arb_mat(size, size); local = []
    for k in range(period):
        offset = k*m; next_offset = ((k+1) % period)*m
        state, mode = flat[offset:offset+n], flat[offset+n:offset+m]
        tick()
        value, block, rate_block = phase(model, state, mode, flat[-2:])
        # Return before finite/deadline checks; a durable observer can preserve failure.
        observe('phase_return', {'phase':k, 'input':vector_json(state+mode+flat[-2:]),
                'value':matrix_json(value), 'block':matrix_json(block), 'rates':matrix_json(rate_block)})
        require(finite(value) and finite(block) and finite(rate_block), 'finite local return')
        for i in range(m):
            sign = sigma if k == period-1 and i >= n else 1
            residual[offset+i,0] = value[i,0] - sign*flat[next_offset+i]
            for j in range(m):
                jac[offset+i,offset+j] += block[i,j]
            jac[offset+i,next_offset+i] -= sign
            for j in range(2):
                jac[offset+i,size-2+j] = rate_block[i,j]
        local.append((block, rate_block))
    q0 = flat[n:m]
    residual[size-2,0] = (sum((v*v for v in q0), arb(0))-1)/2
    for i in range(n):
        jac[size-2,n+i] = q0[i]
    residual[size-1,0] = sum((normal[i]*(flat[i]-origin[i]) for i in range(size)), arb(0))-parameter
    for i in range(size):
        jac[size-1,i] = normal[i]
    require(finite(residual) and finite(jac), 'finite assembly')
    return residual, jac, local


def condensed_preconditioner(local, bottom_rows, sigma):
    """Propose C by block elimination to (2n+2) dimensions, no J_i inverse.

    For J u=b, u_(i+1)=T_i u_i+E_i rates-b_i. Terminal closure
    u_p=diag(I,sigma I)u_0. Both auxiliary rows are condensed by that same
    recursion. Midpoint rounding is allowed only here: C is subsequently
    checked against the complete interval Jacobian by the proof gate.
    """
    require(bool(local), 'nonempty local blocks')
    p = len(local); m = local[0][0].nrows(); size = p*m+2
    require(m % 2 == 0 and bottom_rows.nrows() == 2 and bottom_rows.ncols() == size, 'condensed dimensions')
    require(type(sigma) is int and sigma in (-1,1), 'critical sign')
    ps, qs, ns = [eye(m)], [arb_mat(m,2)], [arb_mat(m,size)]
    for k, (tt, ee) in enumerate(local):
        require(tt.nrows() == tt.ncols() == m and ee.nrows() == m and ee.ncols() == 2, 'local blocks')
        t, e = midpoint_matrix(tt), midpoint_matrix(ee)
        pp = midpoint_matrix(t*ps[-1]); qq = midpoint_matrix(t*qs[-1]+e)
        nn = t*ns[-1]
        for i in range(m): nn[i,k*m+i] -= 1
        ps.append(pp); qs.append(qq); ns.append(midpoint_matrix(nn))
    terminal = ps[-1]
    for i in range(m): terminal[i,i] -= sigma if i >= m//2 else 1
    small = arb_mat(m+2,m+2); rhs = arb_mat(m+2,size)
    for i in range(m):
        for j in range(m): small[i,j] = terminal[i,j]
        for j in range(2): small[i,m+j] = qs[-1][i,j]
        for j in range(size): rhs[i,j] = -ns[-1][i,j]
    for r in range(2):
        for k in range(p):
            weights = arb_mat([[bottom_rows[r,k*m+j].mid() for j in range(m)]])
            a, b, c = weights*ps[k], weights*qs[k], weights*ns[k]
            for j in range(m): small[m+r,j] += a[0,j]
            for j in range(2): small[m+r,m+j] += b[0,j]
            for j in range(size): rhs[m+r,j] -= c[0,j]
        for j in range(2): small[m+r,m+j] += bottom_rows[r,size-2+j].mid()
        rhs[m+r,size-2+r] += 1
    require(finite(small) and finite(rhs), 'finite condensed proposal')
    solved = midpoint_matrix(small.solve(rhs))
    u0 = arb_mat([[solved[i,j] for j in range(size)] for i in range(m)])
    rates = arb_mat([[solved[m+i,j] for j in range(size)] for i in range(2)])
    result = arb_mat(size,size)
    for k in range(p):
        part = midpoint_matrix(ps[k]*u0+qs[k]*rates+ns[k])
        for i in range(m):
            for j in range(size): result[k*m+i,j] = part[i,j]
    for i in range(2):
        for j in range(size): result[size-2+i,j] = rates[i,j]
    require(finite(result), 'finite proposed preconditioner')
    return result, midpoint_matrix(small)


def proposal(jac, local, sigma):
    size = jac.nrows()
    return condensed_preconditioner(local, arb_mat([[jac[size-2+i,j] for j in range(size)] for i in range(2)]), sigma)


def anchor_test(anchor, center, slope, normal, origin, sc, delta, radii, period, sigma, n):
    """Conditional on externally established prior root, never mere overlap."""
    if anchor is None: return {'passed':False, 'reason':'prior root not supplied'}
    require(anchor.get('period') == period and anchor.get('sigma') == sigma and anchor.get('dimension') == n, 'anchor convention')
    box = anchor['box']; size = len(center)
    require(len(box) == size and all(v.is_finite() for v in box), 'anchor box')
    projection = sum((normal[i]*(box[i]-origin[i]) for i in range(size)),arb(0))
    parameter = exact(anchor['exact_parameter']) if 'exact_parameter' in anchor else projection
    if 'exact_parameter' in anchor:
        require(parameter.is_exact() and parameter.is_finite(), 'exact prior gauge')
    d = parameter-sc
    checks = [bool(abs(box[i]-center[i]-slope[i,0]*d).upper() < radii[i]) for i in range(size)]
    inside = bool(abs(d).upper() <= delta)
    return {'passed':inside and all(checks), 'parameter_inside':inside,
            'parameter':ball_json(parameter), 'projection':ball_json(projection),
            'coordinate_inclusions':checks,
            'scope':'conditional on supplied root provenance and identical native order, model, sign, gauge'}


def locus_tube(model, initial, *, period, sigma, normal, origin, parameter,
               halfwidth_exponent=-12, radius_exponent=-16, weights=None,
               newton_steps=2, precision=256, anchor=None,
               tick=lambda:None, observe=lambda kind,record:None):
    """Uniform affine mean-value Krawczyk chart; no criticality/stability gate.

    H(z,s)=(native equations, normalization, normal.(z-origin)-s).
    r(P(s),s) in r(U,sc)+(J(P(I))*S-e_last)*(s-sc). The convex predictor
    interval is the derivative domain. Complete-box J encloses every correction
    segment. Strict weighted contraction/inclusion gives one root for each s.
    """
    require(type(precision) is int and 128 <= precision <= 512, 'precision cap')
    require(type(newton_steps) is int and 0 <= newton_steps <= 2, 'proposal cap')
    require(all(type(v) is int and -100 <= v <= -1 for v in (halfwidth_exponent,radius_exponent)), 'width caps')
    size = 2*model.n*period+2
    require(len(initial) == len(normal) == len(origin) == size, 'chart dimensions')
    weights = ['1']*size if weights is None else weights
    require(len(weights) == size, 'weight dimensions')
    with ctx.workprec(precision):
        center = [exact(v) for v in initial]; nn = [exact(v) for v in normal]; oo = [exact(v) for v in origin]
        w = [exact(v) for v in weights]; sc = exact(parameter)
        require(all(v.is_finite() and v.is_exact() for v in center+nn+oo+w+[sc]), 'exact finite chart data')
        require(any(v != 0 for v in nn) and all(v > 0 and v <= 1 for v in w), 'nonzero gauge, positive weights')
        delta = arb((1,halfwidth_exponent)); radius = arb((1,radius_exponent)); d = arb(0,delta)
        for iteration in range(newton_steps):
            g,j,local = assemble(model,center,period,sigma,nn,oo,sc,tick,observe)
            pre,small = proposal(j,local,sigma)
            correction = pre*g
            center = [(center[i]-correction[i,0]).mid() for i in range(size)]
            observe('proposal',{'iteration':iteration,'center':vector_json(center),'residual':matrix_json(g),
                               'correction':matrix_json(correction),'condensed_border':matrix_json(small)})
        g0,j0,local = assemble(model,center,period,sigma,nn,oo,sc,tick,observe)
        tick();pre,small = proposal(j0,local,sigma)
        observe('preconditioner_return',{'matrix':matrix_json(pre),'condensed_border':matrix_json(small)})
        slope = midpoint_matrix(pre*column([arb(0)]*(size-1)+[arb(1)]))
        predictor = [center[i]+slope[i,0]*d for i in range(size)]
        _,jp,_ = assemble(model,predictor,period,sigma,nn,oo,sc+d,tick,observe)
        path_derivative = jp*slope-column([arb(0)]*(size-1)+[arb(1)])
        rr = g0+path_derivative*d
        radii = [radius*v for v in w]; error = column([arb(0,r) for r in radii])
        box = [predictor[i]+error[i,0] for i in range(size)]
        require(all(v.is_finite() and v > 0 for v in box[-2:]), 'whole rate domain positive')
        _,jb,_ = assemble(model,box,period,sigma,nn,oo,sc+d,tick,observe)
        defect = eye(size)-pre*jb
        norm = max((sum((abs(defect[i,j]).upper()*w[j] for j in range(size)),arb(0))/w[i]).upper() for i in range(size))
        image = -pre*rr+defect*error
        inclusion = [bool(abs(image[i,0]).upper() < radii[i]) for i in range(size)]
        root = bool(all(inclusion) and norm.is_finite() and norm < 1)
        divisors = []
        for k in range(1,period):
            if period % k: continue
            diff = [box[k*2*model.n+i]-box[i] for i in range(model.n)]
            divisors.append({'divisor':k,'differences':vector_json(diff),
                             'excluded':any(v.is_finite() and not v.contains(0) for v in diff)})
        ac = anchor_test(anchor,center,slope,nn,oo,sc,delta,radii,period,sigma,model.n)
        gates = {'uniform_normalized_eigenpair_locus':root,
                 'primitive_period':all(v['excluded'] for v in divisors),'anchor_identified':ac['passed']}
        endpoint_boxes = []
        for sign in (-1,1):
            dd = sign*delta
            ee = -pre*(g0+path_derivative*dd)+defect*error
            endpoint_boxes.append([center[i]+slope[i,0]*dd+ee[i,0] for i in range(size)])
        result = {'schema':'neural-shrimp-eigenpair-locus-prototype-v1','passed':all(gates.values()),'gates':gates,
            'period':period,'dimension':model.n,'sigma':sigma,'shooting_dimension':size,
            'parameter_center':parameter,'parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],
            'normal':normal,'origin':origin,'center':vector_json(center),'slope':matrix_json(slope),
            'preconditioner':matrix_json(pre),'condensed_border':matrix_json(small),'center_residual':matrix_json(g0),
            'predictor_jacobian':matrix_json(jp),'path_derivative':matrix_json(path_derivative),'predictor_residual':matrix_json(rr),
            'whole_box':vector_json(box),'whole_jacobian':matrix_json(jb),'defect':matrix_json(defect),
            'weights':weights,'coordinate_radii':vector_json(radii),'image_correction':matrix_json(image),
            'contraction':ball_json(norm),'coordinate_inclusions':inclusion,'divisors':divisors,'anchor_check':ac,
            'endpoint_root_boxes':[vector_json(v) for v in endpoint_boxes],
            'settings':{'precision':precision,'newton_steps':newton_steps,'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent},
            'genericity':'not evaluated; other-mode crossings and vanishing fold coefficient permitted',
            'not_claimed':['simple critical multiplier','generic fold or flip','attraction','classical shrimp','distant boundary connection','target provenance']}
        observe('chart_return', result)
        # Native Arb values are for an immediate next chart; serialized proof is result.
        return result, {'center':center,'slope':slope,'endpoints':endpoint_boxes}


def exact_mid(value):
    integer, exponent = value.mid().man_exp()
    return str(Fraction(int(integer))*Fraction(2)**int(exponent))


def two_charts(model, initial, **settings):
    """At most two fixed-gauge charts; failure never advances the anchor."""
    first, data = locus_tube(model,initial,**settings)
    results = [first]
    if first['passed']:
        with ctx.workprec(settings.get('precision',256)):
            delta = Fraction(2)**settings.get('halfwidth_exponent',-12)
            dd = exact(str(2*delta))
            next_initial = [exact_mid(v+data['slope'][i,0]*dd) for i,v in enumerate(data['center'])]
            next_settings = dict(settings, parameter=str(Fraction(settings['parameter'])+2*delta),
                anchor={'box':data['endpoints'][1],'exact_parameter':str(Fraction(settings['parameter'])+delta),
                        'period':settings['period'],'sigma':settings['sigma'],'dimension':model.n})
            second,_ = locus_tube(model,next_initial,**next_settings)
        results.append(second)
    return {'schema':'neural-shrimp-two-eigenpair-charts-prototype-v1','passed':len(results)==2 and all(v['passed'] for v in results),'charts':results,
            'scope':'conditional native primitive normalized eigenpair-locus connection; genericity/stability separate'}
