"""Independent384-bit common-variable native cocycle and metric reduction.

Only serialized derivative enclosures and fixed exact proposals are read.
No producer polynomial/norm/metric routines, model calls or floating solver.
"""
from fractions import Fraction
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json
from review_localization_math import original_ball,original_matrix,exact
from review_endpoint_attraction import positive
import review_joined_period18_math as M
need=M.need


def normf(A):
    total=arb(0)
    for row in A.tolist():
        for x in row:total+=abs(x).upper()**2
    return total.sqrt().upper()


def norminf(A):return max(sum((abs(x).upper() for x in row),arb(0)).upper() for row in A.tolist())

def multiply(left,right):
    """Exact noncommutative coefficient convolution, all degrees retained."""
    n=left[0].nrows();out=[arb_mat(n,n) for _ in range(len(left)+len(right)-1)]
    for degree in range(len(out)):
        for i in range(len(left)):
            j=degree-i
            if 0<=j<len(right):out[degree]+=left[i]*right[j]
    return out


def interval_polynomial(coeff):
    # Exact range enclosures for each common power t^k, -1<=t<=1.
    out=coeff[0]+arb_mat(coeff[0].nrows(),coeff[0].ncols())
    for k in range(1,len(coeff)):out+=coeff[k]*(arb(0,1) if k%2 else arb('0.5','0.5'))
    return out


def summary(w):
    return {'passed':w['passed'],'minimum_row_margin':min(w['row_margins'],key=lambda v:M.interval(v)[0]),'inverse_defect':w['inverse_defect']}


def audit(phases,frames,norm_phases,metric,halfwidth):
    p=len(phases);need(1<=p<=18 and len(norm_phases)==len(frames['witnesses'])==p and frames['cyclic_last_index']==0,'complete cyclic native inventory')
    need([r['phase'] for r in phases]==[r['phase'] for r in norm_phases]==[r['phase'] for r in frames['witnesses']]==list(range(p)),'native phase ordering')
    halfwidth=Fraction(halfwidth);need(halfwidth>0,'positive exact halfwidth')
    with ctx.workprec(384):
        bases=[original_matrix(v['B']) for v in frames['witnesses']];n=bases[0].nrows();need(1<=n<=13,'native dimension')
        inverses=[]
        for B in bases:
            need(B.nrows()==B.ncols()==n and all(x.is_finite() and x.is_exact() for row in B.tolist() for x in row),'exact fixed cyclic frames')
            inverse=B.inv();mid=arb_mat([[x.mid() for x in row] for row in inverse.tolist()]);need(norminf(M.ident(n)-mid*B)<1,'independent nonsingular frame')
            inverses.append(inverse)
        poly=[M.ident(n),arb_mat(n,n),arb_mat(n,n)];err=arb(0);centerbound=arb(1);checks=[]
        for j in range(p):
            J=original_matrix(phases[j]['center_J']);Jprime=original_matrix(phases[j]['whole_total_Jprime']);need(J.nrows()==J.ncols()==Jprime.nrows()==Jprime.ncols()==n,'native physical derivatives')
            scaled=Jprime*exact(halfwidth)
            A=arb_mat([[J[a,b].mid() for b in range(n)] for a in range(n)])
            B=arb_mat([[scaled[a,b].mid() for b in range(n)] for a in range(n)])
            remainder=(J-A)+(scaled-B)*arb(0,1)
            left,right=bases[(j+1)%p],inverses[j]
            phase=[left*A*right,left*B*right,arb_mat(n,n)];epsilon=normf(left*remainder*right)
            op=norm_phases[j]['operator_bound'];gamma=original_ball(op['gamma']);need(gamma.is_exact() and gamma>0,'positive exact operator proposal')
            gc=positive(M.ident(n)*(gamma*gamma)-phase[0].transpose()*phase[0],original_matrix(op['positive']['basis']));need(gc['passed'],'independent strict phase operator bound')
            a_bound=gamma+normf(phase[1])+normf(phase[2]);b_bound=centerbound+normf(poly[1])+normf(poly[2])
            full=multiply(phase,poly)
            err=(a_bound*err+epsilon*b_bound+epsilon*err+normf(full[3])+normf(full[4])).upper()
            centerbound=(gamma*centerbound).upper();poly=full[:3]
            checks.append({'phase':j,'operator':summary(gc),'remainder_norm_upper':ball_json(err)})
        P=original_matrix(metric['P']);need(P.nrows()==P.ncols()==n and all(P[i,j].is_exact() and P[i,j]==P[j,i] for i in range(n) for j in range(n)),'exact symmetric Lyapunov metric')
        pc=positive(P,original_matrix(metric['P_positive']['basis']));need(pc['passed'],'independent strict metric positivity')
        # Work degree-by-degree, keeping every term of Q(t)^T P Q(t).
        squares=multiply([v.transpose()*P for v in poly],poly)
        decrease=[P-squares[0]]+[-v for v in squares[1:]]
        qbound=centerbound+normf(poly[1])+normf(poly[2]);penalty=(norminf(P)*(2*qbound*err+err*err)).upper()
        D=interval_polynomial(decrease)-M.ident(n)*penalty
        dc=positive(D,original_matrix(metric['complete_D_positive']['basis']))
        return {'passed':bool(dc['passed']),'precision':384,'phases':p,'dimension':n,'phase_operator_bounds':checks,'P_positive':summary(pc),'complete_D_positive':summary(dc),'remainder_norm_upper':ball_json(err),'remainder_penalty':ball_json(penalty),'scope':'Independent common-scalar coefficient convolution, cyclic similarity, complete operator remainder and strict congruence/Gershgorin proof. Conditional on independently authenticated full root and total physical derivative domains.'}
