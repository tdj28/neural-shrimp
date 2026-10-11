"""Conditional 1:1 quadratic coefficients and two-actual-rate transversality.

Requires an already proved same-box primitive root with exact length-two Jordan
chain and no other +1 eigenvalue. Input Jacobian is that of augmented_chain.
No midpoint spectral decision or numerical derivative enters these enclosures.
"""
from flint import arb,arb_mat
from .rigorous_eigenpair_locus import require,finite
from .rigorous_orbits import matrix_json,vector_json,ball_json


def dot(a,b):return sum((u*v for u,v in zip(a,b)),arb(0))
def mul(a,b):return [sum((a[i,j]*b[j]for j in range(a.ncols())),arb(0))for i in range(a.nrows())]

def chain_coefficients(box,jac):
    n=(len(box)-2)//3;d=3*n+2
    require(n>=2 and len(box)==d and jac.nrows()==jac.ncols()==d,'same augmented dimensions')
    require(all(v.is_finite()for v in box)and finite(jac),'finite same-root data')
    q=box[n+2:2*n+2];r=box[2*n+2:];a=arb_mat([[jac[i,j]for j in range(n)]for i in range(n)])
    # At the exact root, Aq=0, Ar=q. This bordered adjoint has a unique inverse.
    # q^T times its top row equations forces the auxiliary variable to zero.
    border=arb_mat(n+1,n+1)
    for i in range(n):
        for j in range(n):border[i,j]=a[j,i]
        border[i,n]=q[i];border[n,i]=r[i]
    inverse=border.inv();require(finite(inverse),'validated adjoint border inverse')
    sol0=mul(inverse,[arb(0)]*n+[arb(1)]);p0=sol0[:n]
    sol1=mul(inverse,p0+[arb(0)]);p1=sol1[:n]
    require(sol0[-1].contains(0)and sol1[-1].contains(0),'conditional adjoint auxiliary zeros')
    require(dot(p0,q).contains(0)and dot(p0,r).contains(1)and dot(p1,q).contains(1)and dot(p1,r).contains(0),'conditional dual normalization')
    bq=arb_mat([[jac[n+i,j]for j in range(n)]for i in range(n)])
    br=arb_mat([[jac[2*n+i,j]for j in range(n)]for i in range(n)])
    bqq=mul(bq,q);bqr=mul(bq,r)
    quadratic=dot(p0,bqq);mixed=dot(p0,bqr)+dot(p1,bqq)
    result={'schema':'neural-shrimp-conditional-chain-unfolding-v1','dimension':n,'adjoint_border':matrix_json(border),'adjoint_inverse':matrix_json(inverse),'p0':vector_json(p0),'p1':vector_json(p1),'Bqq':vector_json(bqq),'Bqr':vector_json(bqr),'a':ball_json(quadratic),'b':ball_json(mixed),'b_minus_a':ball_json(mixed-quadratic),'s':ball_json((mixed-quadratic)*quadratic),'quadratic_nondegenerate':not quadratic.contains(0),'resonant_nondegenerate':not(mixed-quadratic).contains(0),'unfolding_transverse':False,'scope':'Conditional on proved exact same-root chain and complementary spectrum. Convention: second coordinate w2+a*w1^2/2+b*w1*w2. No branch connection or global boundary topology is proved by these coefficients.'}
    if quadratic.contains(0):return result
    forward=arb_mat(n+1,n+1)
    for i in range(n):
        for j in range(n):forward[i,j]=a[i,j]
        forward[i,n]=r[i];forward[n,i]=q[i]
    fi=forward.inv();require(finite(fi),'validated state border inverse')
    c=[];tau=[];rows=[]
    for k in range(2):
        ra=[jac[i,n+k]for i in range(n)];aq=[jac[n+i,n+k]for i in range(n)];ar=[jac[2*n+i,n+k]for i in range(n)]
        solution=mul(fi,[-v for v in ra]+[arb(0)]);h=solution[:n];cc=dot(p0,ra)
        require((solution[-1]+cc).contains(0),'conditional range solvability')
        dd=dot(p0,[u+v for u,v in zip(mul(bq,h),aq)])
        hh=[h[i]-q[i]*dd/quadratic for i in range(n)]
        tt=dot(p1,[u+v for u,v in zip(mul(bq,hh),aq)])+dot(p0,[u+v for u,v in zip(mul(br,hh),ar)])
        c.append(cc);tau.append(tt);rows.append({'rate_index':k,'range_state':vector_json(h),'range_obstruction':ball_json(cc),'fold_derivative':ball_json(dd),'fold_state_extension':vector_json(hh),'center_trace_extension':ball_json(tt)})
    det=c[0]*tau[1]-c[1]*tau[0]
    result.update(state_border=matrix_json(forward),state_border_inverse=matrix_json(fi),rate_reductions=rows,unfolding_matrix=matrix_json(arb_mat([c,tau])),unfolding_determinant=ball_json(det),unfolding_transverse=not det.contains(0))
    return result
