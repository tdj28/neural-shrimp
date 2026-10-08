"""Ball-polynomial proof gates for a selected planar period-2 NS point.

Polynomial Taylor jets contain normalized coefficients. All theorem gates use
Arb/Acb enclosures; Newton midpoints merely propose an exact dyadic box center.
"""
from fractions import Fraction
from itertools import product
from math import factorial

from flint import arb, acb, arb_mat, ctx
from .rigorous_orbits import (exact, ball_json, vector_json, matrix_json, eye,
                              midpoint_matrix, norm_inf_upper)

INDICES = tuple(a for a in product(range(4), repeat=3) if sum(a) <= 3)
ZERO = (0, 0, 0)


class Jet:
    """Truncated polynomial in three formal increments, total degree <=3."""
    def __init__(self, value=0, coefficients=None):
        self.c = {a: arb(0) for a in INDICES}
        self.c[ZERO] = arb(value)
        if coefficients:
            self.c.update(coefficients)

    @classmethod
    def variable(cls, value, i):
        out = cls(value)
        a = [0, 0, 0]; a[i] = 1
        out.c[tuple(a)] = arb(1)
        return out

    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        return Jet(coefficients={a: self.c[a]+other.c[a] for a in INDICES})

    __radd__ = __add__

    def __neg__(self):
        return Jet(coefficients={a: -self.c[a] for a in INDICES})

    def __sub__(self, other):
        return self + (-other if isinstance(other, Jet) else -arb(other))

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet(other)
        out = Jet()
        for a in INDICES:
            for b in INDICES:
                c = tuple(a[i]+b[i] for i in range(3))
                if sum(c) <= 3:
                    out.c[c] += self.c[a]*other.c[b]
        return out

    __rmul__ = __mul__

    def derivative(self, *axes):
        counts = tuple(axes.count(i) for i in range(3))
        multiplier = 1
        for count in counts:
            multiplier *= factorial(count)
        return self.c[counts]*multiplier


def scalar_step(x, eta, target, ridge, ratio):
    u, v = x
    error = u*v-target
    return [u-eta*(error*v+ridge*u), v-ratio*eta*(error*u+ridge*v)]


def scalar_jets(z, target, ridge, ratio):
    u, v, eta = [Jet.variable(z[i], i) for i in range(3)]
    y = scalar_step([u, v], eta, target, ridge, ratio)
    return scalar_step(y, eta, target, ridge, ratio)


def derivative_data(jets):
    A = [[jets[i].derivative(j) for j in range(2)] for i in range(2)]
    B = [[[jets[i].derivative(j,k) for k in range(2)] for j in range(2)] for i in range(2)]
    C = [[[[jets[i].derivative(j,k,l) for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)]
    D = A[0][0]*A[1][1]-A[0][1]*A[1][0]
    Dgrad = []
    for k in range(3):
        Ak = [[jets[i].derivative(j,k) for j in range(2)] for i in range(2)]
        Dgrad.append(Ak[0][0]*A[1][1]+A[0][0]*Ak[1][1]
                     -Ak[0][1]*A[1][0]-A[0][1]*Ak[1][0])
    return A, B, C, D, Dgrad


def augmented(z, evaluator):
    jets = evaluator(z)
    A, _, _, D, Dgrad = derivative_data(jets)
    H = [jets[i].derivative()-z[i] for i in range(2)] + [D-1]
    J = arb_mat([[jets[i].derivative(k)-int(i==k) for k in range(3)] for i in range(2)] + [Dgrad])
    return H, J


def mid_rational(x):
    m, e = x.mid().man_exp()
    return f'{int(m) << int(e)}/1' if e >= 0 else f'{int(m)}/{1 << -int(e)}'


def refine_proposal(center, evaluator, count):
    history = []
    for i in range(count):
        H, J = augmented(center, evaluator)
        correction = J.solve(arb_mat([[v] for v in H]))
        new_center = [(center[j]-correction[j,0]).mid() for j in range(3)]
        history.append(dict(iteration=i,center_rational=[mid_rational(v) for v in center],
                            residual=vector_json(H),correction=matrix_json(correction),
                            next_center_rational=[mid_rational(v) for v in new_center]))
        center = new_center
    return center, history


def inclusion(center, radius, evaluator):
    H, J = augmented(center, evaluator)
    R = midpoint_matrix(J.inv())
    box = [arb(v,radius) for v in center]
    _, Jbox = augmented(box,evaluator)
    defect = eye(3)-R*Jbox
    q = norm_inf_upper(defect)
    shift = -R*arb_mat([[v] for v in H])+defect*arb_mat([[arb(0,radius)] for _ in range(3)])
    image = [center[i]+shift[i,0] for i in range(3)]
    inside = [bool(box[i].contains_interior(image[i])) for i in range(3)]
    passed = bool(all(inside) and q.is_finite() and q<1)
    return box, dict(passed=passed,box=vector_json(box),center_rational=[mid_rational(v) for v in center],
                     residual=vector_json(H),preconditioner=matrix_json(R),
                     jacobian_enclosure=matrix_json(Jbox),defect=matrix_json(defect),
                     contraction_upper=ball_json(q),image=vector_json(image),coordinate_inclusions=inside)


def determinant2(A):
    return A[0][0]*A[1][1]-A[0][1]*A[1][0]


def inverse2(A):
    det = determinant2(A)
    if not det.is_finite() or det.contains(0):
        raise ValueError('unresolved matrix inverse')
    return [[A[1][1]/det,-A[0][1]/det],[-A[1][0]/det,A[0][0]/det]], det


def matvec(A,x):
    return [sum((A[i][j]*x[j] for j in range(2)),acb(0)) for i in range(2)]


def bilinear(B,x,y):
    return [sum((B[i][j][k]*x[j]*y[k] for j in range(2) for k in range(2)),acb(0)) for i in range(2)]


def trilinear(C,x,y,z):
    return [sum((C[i][j][k][l]*x[j]*y[k]*z[l] for j in range(2) for k in range(2) for l in range(2)),acb(0)) for i in range(2)]


def complex_json(z):
    return dict(real=ball_json(z.real),imag=ball_json(z.imag))


def coefficient(A,B,C):
    """Resolvent formula and an independent planar scalar-coefficient formula.

    Equivalence holds at a simple unit pair. Both expressions are enclosed;
    the polynomial root/linear gates establish the condition separately.
    """
    T, D = A[0][0]+A[1][1],determinant2(A)
    discriminant = 4*D-T*T
    if not discriminant>0 or A[0][1].contains(0):
        raise ValueError('unresolved complex eigenframe')
    mu = acb(T,discriminant.sqrt())/2
    q0 = [acb(A[0][1]),mu-A[0][0]]
    n2 = sum(((v.conjugate()*v).real for v in q0),arb(0))
    if not n2>0:
        raise ValueError('unresolved eigenvector norm')
    q = [v/n2.sqrt() for v in q0]; qb = [v.conjugate() for v in q]
    frame_inverse, frame_det = inverse2([[q[0],qb[0]],[q[1],qb[1]]])
    left = frame_inverse[0]
    pairing = lambda x:sum((left[i]*x[i] for i in range(2)),acb(0))
    inv1,det1 = inverse2([[int(i==j)-A[i][j] for j in range(2)] for i in range(2)])
    inv2,det2 = inverse2([[mu*mu*int(i==j)-A[i][j] for j in range(2)] for i in range(2)])
    bqq,bqqb,bqbqb = bilinear(B,q,q),bilinear(B,q,qb),bilinear(B,qb,qb)
    cqqb = trilinear(C,q,q,qb)
    w1,w2 = matvec(inv1,bqqb),matvec(inv2,bqq)
    term1,term2 = bilinear(B,q,w1),bilinear(B,qb,w2)
    c1 = pairing([cqqb[i]+2*term1[i]+term2[i] for i in range(2)])/2
    ell = (mu.conjugate()*c1).real
    # Kuznetsov planar formula: a separate algebraic evaluation with no resolvent solves.
    g20,g11,g02,g21 = pairing(bqq),pairing(bqqb),pairing(bqbqb),pairing(cqqb)
    planar = ((mu.conjugate()*g21)/2).real - (((1-2*mu)*(mu.conjugate()**2)/(2*(1-mu)))*g20*g11).real \
             -(g11*g11.conjugate()).real/2-(g02*g02.conjugate()).real/4
    return ell,planar,dict(mu=complex_json(mu),right=[complex_json(v) for v in q],
           left_row=[complex_json(v) for v in left],frame_determinant=complex_json(frame_det),
           resolvent_determinants=[complex_json(acb(det1)),complex_json(det2)],
           c1=complex_json(c1),ell=ball_json(ell),planar_ell=ball_json(planar),
           scalar_coefficients={name:complex_json(v) for name,v in zip(('g20','g11','g02','g21'),(g20,g11,g02,g21))},
           formula_difference=ball_json(ell-planar),
           both_strictly_negative=bool(ell<0 and planar<0),formula_enclosures_overlap=bool((ell-planar).contains(0)))


def local_conditions(jets):
    A,B,C,D,Dgrad = derivative_data(jets)
    T=A[0][0]+A[1][1]
    radicand=4*D-T*T
    resonance=[T-value for value in (-2,-1,0,2)]
    complete_complex=bool(radicand>0 and T>-2 and T<2)
    nonresonance=all(v.is_finite() and not v.contains(0) for v in resonance)
    minus_identity=[[A[i][j]-int(i==j) for j in range(2)] for i in range(2)]
    inv,det= inverse2(minus_identity)
    eta_partial=[jets[i].derivative(2) for i in range(2)]
    branch_derivative=[-sum((inv[i][j]*eta_partial[j] for j in range(2)),arb(0)) for i in range(2)]
    slope=Dgrad[2]+sum((Dgrad[i]*branch_derivative[i] for i in range(2)),arb(0))
    result=dict(A=matrix_json(arb_mat(A)),
        B=[[vector_json(row) for row in component] for component in B],
        C=[[[vector_json(row) for row in matrix] for matrix in component] for component in C],
        determinant_gradient=vector_json(Dgrad),eta_partial=vector_json(eta_partial),
        trace=ball_json(T),determinant=ball_json(D),
        complex_radicand=ball_json(radicand),trace_resonance_differences=vector_json(resonance),
        unit_complex_pair_if_augmented_root=complete_complex,nonresonance_orders_1_to_4=nonresonance,
        branch_denominator=ball_json(det),branch_derivative=vector_json(branch_derivative),
        determinant_slope=ball_json(slope),modulus_slope_at_root=ball_json(slope/2),
        positive_transversality=bool(slope>0),coefficient_status='not_attempted')
    if complete_complex and nonresonance:
        ell,planar,details=coefficient(A,B,C)
        result.update(coefficient_status='enclosed',coefficient=details,
                      negative_criticality=bool(ell<0 and planar<0 and (ell-planar).contains(0)))
    else:
        result['negative_criticality']=False
    return result


def check_scalar_ns(inputs,precision=256,radius_exponent=-40,newton_steps=5):
    if inputs.get('schema')!='scalar-ns-input-v1':raise ValueError('unknown input schema')
    if type(precision) is not int or precision!=256 or radius_exponent!=-40 or newton_steps!=5:
        raise ValueError('outside the single declared proof recipe')
    for value in [inputs[k] for k in ('target','ridge','ratio')]+inputs['approximation']:
        if not isinstance(value,str):raise ValueError('exact rational strings required')
        d=Fraction(value).denominator
        if d & (d-1):raise ValueError('target inputs must be dyadic')
    if len(inputs['approximation'])!=3:raise ValueError('three approximation coordinates required')
    with ctx.workprec(precision):
        target,ridge,ratio=[exact(inputs[k]) for k in ('target','ridge','ratio')]
        if not target>ridge or not ridge>0 or not ratio>0:raise ValueError('invalid model constants')
        evaluator=lambda z:scalar_jets(z,target,ridge,ratio)
        center=[exact(v).mid() for v in inputs['approximation']]
        center,history=refine_proposal(center,evaluator,newton_steps)
        box,inc=inclusion(center,arb((1,radius_exponent)),evaluator)
        differences=[v-box[i] for i,v in enumerate(scalar_step(box[:2],box[2],target,ridge,ratio))]
        separating=[i for i,v in enumerate(differences) if v.is_finite() and not v.contains(0)]
        local=local_conditions(evaluator(box))
        gates=dict(augmented_root=inc['passed'],positive_rate=bool(box[2]>0),
                   primitive_period_two=bool(separating),unit_complex_pair=local['unit_complex_pair_if_augmented_root'],
                   nonresonance=local['nonresonance_orders_1_to_4'],transversality=local['positive_transversality'],
                   negative_criticality=local['negative_criticality'])
        passed=all(gates.values())
        return dict(schema='scalar-ns-result-v1',passed=passed,evidence_rung=3 if passed else 0,
          research_status='exploratory certificate; independent review pending',gates=gates,
          settings=dict(precision=precision,radius_exponent=radius_exponent,newton_steps=newton_steps),
          proposal_history=history,inclusion=inc,primitive=dict(separating_coordinates=separating,differences=vector_json(differences)),
          local=local,statement=dict(model='exact-real scalar factorization GD with exact dyadic fixed coefficients',
              period=2,local_unique_unit_pair_point=bool(inc['passed'] and gates['primitive_period_two'] and gates['unit_complex_pair']),
              generic_local_supercritical_ns=passed,eta_is_unknown_exact_real=True,
              global_branch_connection=False,finite_parameter_tube=False,finite_detuning_invariant_curve_enclosure=False,
              floating_point_dynamics=False,formalized=False),
          theorem_dependence='Krawczyk/Banach, implicit function theorem, planar generic NS theorem; Arb/FLINT trusted')
