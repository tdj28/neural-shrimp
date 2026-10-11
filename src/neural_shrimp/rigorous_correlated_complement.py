"""Conditional common-parameter complementary cocycle on a native root tube.

Critical q/r columns vary with the exact root. Their total derivatives are
retained in the inverse-frame derivative; fixed complement/metric proposals
are exact dyadic constants and are validated before any spectral conclusion.
"""
from flint import arb,arb_mat
from .rigorous_folds import MixedJet
from .rigorous_connection import require
from .rigorous_orbits import eye,matrix_json,ball_json,vector_json
from .rigorous_correlated_cocycle import phase_model
from .rigorous_norm_cocycle import operator_upper,frobenius_upper,advance,scalar_remainder_metric


def finite(A):return all(v.is_finite()for row in A.tolist()for v in row)


def total_two_rate_jacobian(model,state,rates,state_tangent,rate_tangent):
    """DF with physical rates held fixed, plus its total root-path derivative."""
    n=model.n
    require(1<=n<=13 and len(state)==len(state_tangent)==n and len(rates)==len(rate_tangent)==2,'full physical state and both rate tangents')
    require(all(v.is_finite()for v in state+rates+state_tangent+rate_tangent)and all(v>0 for v in rates),'finite positive actual native domains')
    x=[MixedJet.variable(state[i],n,i,state_tangent[i])for i in range(n)]
    rates_ad=[MixedJet(rates[j],[arb(0)]*n,rate_tangent[j],[arb(0)]*n)for j in range(2)]
    result=model.cusp_step(x,*rates_ad);require(len(result)==n,'all native returns')
    A=arb_mat([list(v.g)for v in result]);Ap=arb_mat([list(v.m)for v in result])
    require(A.nrows()==A.ncols()==Ap.nrows()==Ap.ncols()==n and finite(A)and finite(Ap),'finite native first/total second derivatives')
    return A,Ap


def frames(flat,tangent,complements,n,p):
    require(len(flat)==len(tangent)==3*n*p+2 and len(complements)==p,'full state, mode, rate and tangent inventory')
    require(all(v.is_finite()for v in flat+tangent),'finite root and tangent domains')
    F=[];Fp=[];inverses=[]
    for j,E in enumerate(complements):
        require(E.nrows()==n and E.ncols()==n-2 and all(v.is_finite()and v.is_exact()for row in E.tolist()for v in row),'fixed exact complementary columns')
        k=3*n*j
        frame=arb_mat([[flat[k+n+i],flat[k+2*n+i]]+[E[i,h]for h in range(n-2)]for i in range(n)])
        derivative=arb_mat([[tangent[k+n+i],tangent[k+2*n+i]]+[arb(0)]*(n-2)for i in range(n)])
        inverse=frame.inv();require(finite(inverse),'validated entire variable-frame inverse')
        F.append(frame);Fp.append(derivative);inverses.append(inverse)
    return F,Fp,inverses


def quotient_derivative(A,Ap,F,Fp,after_inverse,after_derivative):
    """T=F_next^-1 A F and T'=F_next^-1(A'F+AF'-F_next'T)."""
    n=A.nrows();require(all(v.nrows()==v.ncols()==n and finite(v)for v in (A,Ap,F,Fp,after_inverse,after_derivative)),'complete finite quotient derivative data')
    T=after_inverse*A*F
    Tp=after_inverse*(Ap*F+A*Fp-after_derivative*T)
    require(finite(T)and finite(Tp),'finite total transformed derivative')
    return T,Tp


def correlated_complement(rows,center,whole,tangent,complements,metrics,halfwidth,terminal,*,tick=lambda:None,observe=lambda k,v:None):
    p=len(rows);require(1<=p<=18,'bounded complete native phase count')
    n=rows[0]['center_A'].nrows();m=n-2
    require(4<=n<=13 and len(metrics)==p and terminal.nrows()==terminal.ncols()==2 and finite(terminal),'full native and critical dimensions')
    require(halfwidth.is_finite()and halfwidth.is_exact()and halfwidth>0,'exact positive parameter halfwidth')
    require(all(v>0 for v in center[-2:]+whole[-2:]),'both actual rates positive')
    zeros=[arb(0)]*len(center)
    Fc,_,Ic=frames(center,zeros,complements,n,p);Fw,Fp,Iw=frames(whole,tangent,complements,n,p)
    invB=[]
    for B in metrics:
        require(B.nrows()==B.ncols()==m and all(v.is_finite()and v.is_exact()for row in B.tolist()for v in row),'fixed exact quotient metric')
        inverse=B.inv();require(finite(inverse),'validated quotient metric inverse');invB.append(inverse)
    observe('frames',{'center':[matrix_json(v)for v in Fc],'whole':[matrix_json(v)for v in Fw],'whole_derivative':[matrix_json(v)for v in Fp],
      'center_inverse':[matrix_json(v)for v in Ic],'whole_inverse':[matrix_json(v)for v in Iw],'metric':[matrix_json(v)for v in metrics],'metric_inverse':[matrix_json(v)for v in invB],'cyclic_final_index':0})
    acc=([eye(m),arb_mat(m,m),arb_mat(m,m)],arb(0),arb(1))
    for j,row in enumerate(rows):
        tick();require(row['phase']==j,'literal native phase order');k=(j+1)%p
        A0,Aw,Ap=(row[name]for name in ('center_A','whole_A','whole_Aprime'))
        require(all(v.nrows()==v.ncols()==n and finite(v)for v in (A0,Aw,Ap)),'full finite actual phase derivatives')
        T0=Ic[k]*A0*Fc[j];Tw,Tp=quotient_derivative(Aw,Ap,Fw[j],Fp[j],Iw[k],Fp[k])
        expected=terminal if j==p-1 else eye(2)
        require(all(Tw[i,h].contains(0)for i in range(2,n)for h in range(2)),'conditional invariant critical plane retained')
        require(all(Tw[i,h].overlaps(expected[i,h])for i in range(2)for h in range(2)),'conditional exact terminal critical block retained')
        cut=lambda T:arb_mat([[T[i,h]for h in range(2,n)]for i in range(2,n)])
        Q0=metrics[k]*cut(T0)*invB[j];Qp=metrics[k]*cut(Tp)*invB[j]
        coeff,error=phase_model(Q0,Qp,halfwidth);op,gamma=operator_upper(coeff[0],tick);require(op['passed'],'validated phase center operator bound')
        epsilon=frobenius_upper(error);acc,detail=advance(acc,(coeff,epsilon,gamma))
        observe('phase',{'phase':j,'center_transformed':matrix_json(T0),'whole_transformed':matrix_json(Tw),'whole_transformed_derivative':matrix_json(Tp),
          'center_quotient':matrix_json(Q0),'whole_quotient_derivative':matrix_json(Qp),'coefficients':[matrix_json(v)for v in coeff],'matrix_remainder':matrix_json(error),
          'remainder_norm_upper':ball_json(epsilon),'operator_bound':op,'product_bound':detail})
    metric=scalar_remainder_metric(*acc,tick=tick);observe('metric',metric)
    return {'passed':bool(metric['passed']),'dimension':n,'period':p,'complement_dimension':m,'halfwidth':ball_json(halfwidth),
      'coefficients':[matrix_json(v)for v in acc[0]],'remainder_norm_upper':ball_json(acc[1]),'metric':metric,
      'scope':'Conditional uniform attraction of all n−2 complementary modes on the authenticated primitive root curve. Exact root q/r-plane identities, actual total state/q/r/both-rate tangents, and native mixed derivatives are caller prerequisites. Same parameter retained across all phases; no signed NS coefficient, stability side or completed shrimp window.'}
