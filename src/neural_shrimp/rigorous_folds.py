"""Validated folds of exact-real tanh GD: mixed forward jets + Arb Krawczyk.

No floating derivative, eigenvalue, or convergence decision enters a proof gate.
The variable parameter is the actual hidden learning rate, not its logarithm.
Existing C1 machinery supplies only exact constants and serialization helpers.
"""
from __future__ import annotations
from flint import arb, arb_mat, ctx
from .rigorous_orbits import (BallTanhMap, exact, eye, midpoint_matrix,
                             norm_inf_upper, ball_json, vector_json, matrix_json)


class MixedJet:
    """Value, base gradient, directional derivative, and its base gradient.

    If base x=(theta, eta), d=D_theta f v, then m_i=partial_x_i d with
    v held fixed. v itself is an interval constant. All entries are Arb balls.
    """
    __slots__ = ('f', 'g', 'd', 'm')

    def __init__(self, f, g, d, m):
        self.f, self.g, self.d, self.m = f, tuple(g), d, tuple(m)

    @classmethod
    def variable(cls, value, size, index, direction=0):
        return cls(arb(value), [arb(int(i == index)) for i in range(size)],
                   arb(direction), [arb(0) for _ in range(size)])

    def __add__(self, other):
        if not isinstance(other, MixedJet):
            return MixedJet(self.f + other, self.g, self.d, self.m)
        if len(self.g) != len(other.g):
            raise ValueError('jet dimensions differ')
        return MixedJet(self.f + other.f, [a+b for a,b in zip(self.g, other.g)],
                        self.d + other.d, [a+b for a,b in zip(self.m, other.m)])

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if not isinstance(other, MixedJet):
            return MixedJet(self.f*other, [x*other for x in self.g],
                            self.d*other, [x*other for x in self.m])
        if len(self.g) != len(other.g):
            raise ValueError('jet dimensions differ')
        return MixedJet(self.f*other.f,
            [a*other.f+self.f*b for a,b in zip(self.g, other.g)],
            self.d*other.f+self.f*other.d,
            [a*other.f+b*other.d+self.d*c+self.f*d
             for a,b,c,d in zip(self.m,self.g,other.g,other.m)])

    __rmul__ = __mul__

    def tanh(self):
        t = self.f.tanh()
        s = 1-t*t
        c = -2*t*s
        return MixedJet(t, [s*x for x in self.g], s*self.d,
                        [s*m+c*g*self.d for m,g in zip(self.m,self.g)])


class FoldTanhMap(BallTanhMap):
    """Differentiate the analytic loss-gradient formula using mixed jets.

    This is second-order AD of the gradient, hence includes the required
    third derivatives of the loss without allocating the full third tensor.
    Biases are outside in_scale/out_scale and both receive ridge penalties.
    """
    def jet_step(self, theta, hidden_rate):
        grad = [t*self.ridge for t in theta]
        for x,y in zip(self.x,self.y):
            acts, features = [], []
            for h in range(self.w):
                feature = [(d*self.w+h,self.ins*x[d]) for d in range(self.d)]
                if self.bias:
                    feature.append((self.a+h,arb(1)))
                acts.append(sum((theta[i]*v for i,v in feature),0).tanh())
                features.append(feature)
            prediction = sum((theta[self.hidden+h]*acts[h] for h in range(self.w)),0)*self.out
            if self.bias:
                prediction += theta[-1]
            residual = prediction-y
            for h in range(self.w):
                oi = self.hidden+h
                grad[oi] += residual*acts[h]*(self.k*self.out)
                slope = 1-acts[h]*acts[h]
                hidden_term = residual*theta[oi]*slope*(self.k*self.out)
                for i,feature in features[h]:
                    grad[i] += hidden_term*feature
            if self.bias:
                grad[-1] += residual*self.k
        return [t-grad[i]*(hidden_rate if i < self.hidden else self.rates[i])
                for i,t in enumerate(theta)]


def augmented(model, z, period, tick=lambda: None):
    """Return G, DG, and all jet/phase values for z=(theta,eta,v).

    G=(R,A v,(v^T v-1)/2).  DG blocks are [A,b,0],
    [D_theta(A v),D_eta(A v),A], [0,0,v^T].
    """
    n = model.n
    if len(z) != 2*n+1:
        raise ValueError('augmented state dimension mismatch')
    theta, eta, v = z[:n], z[n], z[n+1:]
    current = [MixedJet.variable(theta[i],n+1,i,v[i]) for i in range(n)]
    rate = MixedJet.variable(eta,n+1,n)
    initial = current
    orbit = [[q.f for q in current]]
    for _ in range(period):
        tick()
        current = model.jet_step(current,rate)
        orbit.append([q.f for q in current])
    residual = [q-p for q,p in zip(current,initial)]
    g = [q.f for q in residual]+[q.d for q in residual]+[(sum((a*a for a in v),arb(0))-1)/2]
    a = arb_mat([[q.g[j] for j in range(n)] for q in residual])
    j = arb_mat([list(q.g)+[arb(0)]*n for q in residual]
        +[list(q.m)+[a[i,k] for k in range(n)] for i,q in enumerate(residual)]
        +[[arb(0)]*(n+1)+list(v)])
    return g,j,{'residual_jets':residual,'a':a,'b':[q.g[n] for q in residual],
                'monodromy':a+eye(n),'orbit':orbit,'v':v}


def _proof(model, approximation, period, radius_exponent, newton_steps, max_power,
           tick=lambda:None, observe=lambda kind,record:None):
    """Internal generic engine; public input validation is in check_fold."""
    center = [x.mid() for x in approximation]
    for iteration in range(newton_steps):
        g,j,_ = augmented(model,center,period,tick)
        observe('newton_evaluation', {'iteration':iteration,'center':vector_json(center),
                                    'g':vector_json(g),'jacobian':matrix_json(j)})
        correction = j.solve(arb_mat([[x] for x in g]))
        center = [(x-correction[i,0]).mid() for i,x in enumerate(center)]
    g0,j0,_ = augmented(model,center,period,tick)
    inverse = midpoint_matrix(j0.inv())
    radius = arb((1,radius_exponent))
    box = [arb(x,radius) for x in center]
    gb,jb,data = augmented(model,box,period,tick)
    observe('whole_box_evaluation',{'box':vector_json(box),'g':vector_json(gb),
                                  'jacobian':matrix_json(jb)})
    defect = eye(len(center))-inverse*jb
    contraction = norm_inf_upper(defect)
    delta = arb_mat([[box[i]-center[i]] for i in range(len(center))])
    image_delta = -inverse*arb_mat([[q] for q in g0])+defect*delta
    image = [center[i]+image_delta[i,0] for i in range(len(center))]
    inclusions = [bool(x.contains_interior(y)) for x,y in zip(box,image)]
    root = bool(all(inclusions) and contraction.is_finite() and contraction < 1)
    n = model.n
    divisors = []
    for d in range(1,period):
        if period % d == 0:
            diff = [data['orbit'][d][i]-box[i] for i in range(n)]
            separated = [i for i,t in enumerate(diff) if t.is_finite() and not t.contains(0)]
            divisors.append({'divisor':d,'excluded':bool(separated),
                             'separating_coordinates':separated,'differences':vector_json(diff)})
    primitive = all(r['excluded'] for r in divisors)
    result = {'schema':'neural-shrimp-fold-result-v1','passed':False,
        'status':'augmented_root_only' if root else 'root_unproved',
        'settings':{'radius_exponent':radius_exponent,'newton_steps':newton_steps,'max_power':max_power},
        'gates':{'augmented_root_unique':root,'positive_hidden_rate':bool(box[n]>0),
                 'primitive_period':primitive,'simple_unit_multiplier':False,
                 'transversality':False,'quadratic_nondegeneracy':False,'other_modes_attracting':False},
        'center':vector_json(center),'box':vector_json(box),'g_center':vector_json(g0),
        'g_box':vector_json(gb),'jacobian_box':matrix_json(jb),'preconditioner':matrix_json(inverse),
        'krawczyk_image':vector_json(image),'coordinate_inclusions':inclusions,
        'defect':matrix_json(defect),'contraction_norm_upper':ball_json(contraction),
        'divisors':divisors,'orbit_enclosures':[vector_json(x) for x in data['orbit']],
        'monodromy':matrix_json(data['monodromy']),'parameter_derivative':vector_json(data['b']),
        'right_vector':vector_json(data['v']),'failures':[]}
    observe('root_gate',result)
    # Keep a partial augmented-root result if any subsequent validated solve fails.
    try:
        tick()
        a,v = data['a'],data['v']
        border = arb_mat([[a[j,i] for j in range(n)]+[v[i]] for i in range(n)]
                         +[list(v)+[arb(0)]])
        r = midpoint_matrix(border.inv())
        bd = eye(n+1)-r*border
        bc = norm_inf_upper(bd)
        invertible = bool(bc.is_finite() and bc < 1)
        left = border.solve(arb_mat([[arb(0)]]*n+[[arb(1)]]))
        w = [left[i,0] for i in range(n)]
        curvature = [sum((q.m[j]*v[j] for j in range(n)),arb(0))
                     for q in data['residual_jets']]
        aa = sum((w[i]*data['b'][i] for i in range(n)),arb(0))
        cc = sum((w[i]*curvature[i] for i in range(n)),arb(0))
        transverse = bool(aa.is_finite() and not aa.contains(0))
        quadratic = bool(cc.is_finite() and not cc.contains(0))
        deflated = data['monodromy']-arb_mat([[v[i]*w[j] for j in range(n)] for i in range(n)])
        power,k,powers,attracting = deflated,1,[],False
        while k <= max_power:
            tick()
            bound = norm_inf_upper(power)
            powers.append({'power':k,'norm_inf_upper':ball_json(bound)})
            if bound.is_finite() and bound < 1:
                attracting = True
                break
            if not bound.is_finite():
                break
            power = power*power
            k *= 2
        result.update({'left_border':matrix_json(border),'left_preconditioner':matrix_json(r),
            'left_defect':matrix_json(bd),'left_contraction_norm_upper':ball_json(bc),
            'left_vector':vector_json(w),'left_border_multiplier':ball_json(left[n,0]),
            'second_directional_residual':vector_json(curvature),'a':ball_json(aa),'c':ball_json(cc),
            'quadratic_normal_form_coefficient_c_over_2':ball_json(cc/2),
            'hidden_rate_second_derivative':ball_json(-cc/aa) if transverse else None,
            'deflated_monodromy':matrix_json(deflated),'power_bounds':powers,
            'complement_witness_power':k if attracting else None,
            'complement_witness':matrix_json(power) if attracting else None})
        result['gates'].update({'simple_unit_multiplier':bool(root and invertible),
            'transversality':bool(root and invertible and transverse),
            'quadratic_nondegeneracy':bool(root and invertible and quadratic),
            'other_modes_attracting':bool(root and invertible and attracting)})
    except (ValueError,ZeroDivisionError) as exc:
        result['failures'].append({'stage':'fold_gates','type':type(exc).__name__,'message':str(exc)})
    result['passed'] = all(result['gates'].values())
    result['evidence_rung'] = 3 if result['passed'] else 0
    if result['passed']:
        result['status'] = 'primitive_nondegenerate_fold_with_attracting_complement'
    result['exact_statement'] = {
        'period':period,'dimension':n,'augmented_dimension':2*n+1,
        'parameter':'actual hidden learning rate; fixed exact output rate',
        'scope':'exact-real GD with supplied rational constants; local fold only',
        'not_claimed':['attraction at the neutral fold','global branch identity','whole-window boundary proof'],
        'normalization':'v^T v=1 and w^T v=1 at the enclosed root; c=D_theta^2 R[v,v] paired with w'}
    return result


def check_fold(inputs, *, precision=256,radius_exponent=-80,newton_steps=5,max_power=4096,
               tick=lambda:None,observe=lambda kind,record:None):
    if inputs.get('schema') != 'neural-shrimp-fold-input-v1':
        raise ValueError('unknown fold input schema')
    for value,lo,hi in ((precision,128,2048),(radius_exponent,-1024,-1),
                        (newton_steps,0,12),(max_power,1,4096),(inputs['period'],1,64)):
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError('invalid bounded fold settings')
    with ctx.workprec(precision):
        model = FoldTanhMap(inputs['map_inputs'])
        approximation = [exact(x) for x in inputs['approximation']]
        if len(approximation) != 2*model.n+1 or not all(x.is_finite() for x in approximation):
            raise ValueError('invalid fold approximation')
        if not approximation[model.n] > 0:
            raise ValueError('nonpositive hidden-rate proposal')
        result = _proof(model,approximation,inputs['period'],radius_exponent,newton_steps,
                        max_power,tick,observe)
        result['settings']['precision'] = precision
        return result
