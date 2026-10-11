"""Independent saved-ball Lyapunov--Schmidt and Krawczyk reductions.

Uses saved derivative enclosures, not the producer coefficient/border helper.
The original byte replay binds those derivative enclosures to the analytic map.
"""
from fractions import Fraction
from flint import arb,arb_mat,acb,acb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,matrix_json,vector_json
from review_clark_rnn_c1_math import need,restore,exact,eye,norm,endpoints,strict_nested

def matrix(v):return arb_mat([[restore(x)for x in r]for r in v])
def vector(v):return arb_mat([[restore(x)]for x in v])
def same(a,b):return matrix_json(a)==matrix_json(b)
def left_border_inverse(A,q):
    n=A.nrows()
    primal=arb_mat([[A[i,j]for j in range(n)]+[q[i,0]]for i in range(n)]+[[q[i,0]for i in range(n)]+[arb(0)]])
    return primal,primal.inv().transpose()


def arithmetic(proof,precision=384,observe=lambda kind,row:None):
    need(proof['passed']and all(v is True for v in proof['gates'].values()),'producer gates')
    n=len(proof['monodromy']);dim=3*n+2;need(len(proof['box'])==dim,'full augmented dimension')
    with ctx.workprec(precision):
        box=[restore(x)for x in proof['box']];center=[restore(x)for x in proof['center']]
        need(all(x.is_exact()for x in center)and all(x==b.mid()for x,b in zip(center,box)),'exact center')
        C=matrix(proof['preconditioner']);need(C.nrows()==dim and C.ncols()==dim and all(C[i,j].is_exact()for i in range(dim)for j in range(dim)),'exact full preconditioner')
        J=matrix(proof['whole_jacobian']);R=matrix(proof['residual']);D=eye(dim)-C*J
        E=arb_mat([[restore({'mid':[0,0],'rad':b['rad']})]for b in proof['box']]);K=vector(proof['center'])-C*R+D*E;contraction=norm(D)
        inc=[bool(box[i].contains_interior(K[i,0]))for i in range(dim)];need(all(inc)and contraction<1,'independent strict Krawczyk gate')
        need(box[n]>0 and box[n+1]>0,'positive actual rates')
        M=matrix(proof['monodromy']);A=M-eye(n);q=arb_mat([[v]for v in box[n+2:2*n+2]]);h=arb_mat([[v]for v in box[2*n+2:]])
        grad=matrix(proof['base_gradients']);g1=matrix(proof['direction_gradients']);g2=matrix(proof['acceleration_gradients'])
        need(all(x.nrows()==n and x.ncols()==dim for x in (grad,g1,g2)),'full derivative arrays')
        need(proof['monodromy']==[row[:n]for row in proof['base_gradients']],'native monodromy/base join')
        # Derivative block identities use original precision/order for exact balls.
        with ctx.workprec(256):
            jr=arb_mat([[grad[i,j]-int(i==j)for j in range(dim)]for i in range(n)]+[[g1[i,j]-int(n+2+i==j)for j in range(dim)]for i in range(n)]+[[g2[i,j]-int(2*n+2+i==j)for j in range(dim)]for i in range(n)]+[[arb(0)]*(n+2)+[q[i,0]for i in range(n)]+[arb(0)]*n, [arb(0)]*(n+2)+[h[i,0]for i in range(n)]+[q[i,0]for i in range(n)]])
            need(matrix_json(jr)==proof['whole_jacobian'],'exact saved augmented derivative assembly')
        primal,inverse=left_border_inverse(A,q);B=primal.transpose()
        left=inverse*arb_mat([[arb(0)]]*n+[[arb(1)]]);w=arb_mat([[left[i,0]for i in range(n)]])
        need(all(inverse[i,j].is_finite()for i in range(n+1)for j in range(n+1)),'left border invertibility')
        need(left[n,0].contains(0)and (w*q)[0,0].contains(1)and (q.transpose()*q)[0,0].contains(1)and(q.transpose()*h)[0,0].contains(0),'normalized augmented relations')
        tangent_h=arb_mat([[sum((g1[i,k]*h[k,0]for k in range(n)),arb(0))]for i in range(n)])
        third=arb_mat([[sum((g2[i,k]*q[k,0]for k in range(n)),arb(0))]for i in range(n)])+2*tangent_h
        cubic=(w*third)[0,0]/6
        U=arb_mat(2,2);ys=[]
        for j in range(2):
            par=arb_mat([[grad[i,n+j]]for i in range(n)]);y=inverse.transpose()*arb_mat([[-par[i,0]]for i in range(n)]+[[arb(0)]])
            U[0,j]=(w*par)[0,0]
            U[1,j]=(w*arb_mat([[g1[i,n+j]+sum((g1[i,k]*y[k,0]for k in range(n)),arb(0))]for i in range(n)]))[0,0]
            need((y[n,0]+U[0,j]).contains(0),'parameter complement multiplier sign');ys.append(matrix_json(y))
        determinant=U.det();observe('coefficient_reduction',{'left_border':matrix_json(B),'inverse':matrix_json(inverse),'left_augmented':matrix_json(left),'third':matrix_json(third),'cubic':ball_json(cubic),'unfolding':matrix_json(U),'determinant':ball_json(determinant),'parameter_solutions':ys});need(cubic.is_finite()and not cubic.contains(0)and determinant.is_finite()and not determinant.contains(0),'independent cubic/versal gates')
        need(cubic.overlaps(restore(proof['cubic_coefficient']))and determinant.overlaps(restore(proof['unfolding_determinant'])),'quoted coefficient consistency')
        Q=M-q*w;basis=acb_mat([[acb(restore(z['real']),restore(z['imag']))for z in row]for row in proof['spectral_exclusion']['proposal_basis']])
        need(basis.nrows()==n and basis.ncols()==n and all(basis[i,j].real.is_exact()and basis[i,j].imag.is_exact()for i in range(n)for j in range(n)),'exact complex proposal basis')
        inv=basis.inv();T=inv*acb_mat([[acb(Q[i,j])for j in range(n)]for i in range(n)])*basis;discs=[]
        for i in range(n):
            c=acb(T[i,i].real.mid(),T[i,i].imag.mid());radius=(abs(T[i,i]-c)+sum((abs(T[i,j])for j in range(n)if i!=j),arb(0))).upper();mod=abs(c)
            inside=bool(mod+radius<1);outside=bool(mod-radius>1);need(inside or outside,'independent full noncritical unit-circle exclusion')
            discs.append({'real':ball_json(c.real),'imag':ball_json(c.imag),'radius':ball_json(radius),'inside':inside,'outside':outside})
        period=proof['statement']['primitive_period'];need(type(period)is int and period>=1,'period');div=[d for d in range(1,period)if period%d==0]
        need([x['divisor']for x in proof['divisors']]==div,'all proper divisors')
        orbit=[[restore(x)for x in r]for r in proof['orbit_enclosures']];need(len(orbit)==period+1 and all(len(r)==n for r in orbit),'native orbit shape')
        separators={str(d):[i for i in range(n)if not(orbit[d][i]-box[i]).contains(0)]for d in div};need(all(separators.values()),'independent proper-divisor separation')
        return {'passed':True,'precision':precision,'dimension':n,'augmented_dimension':dim,'period':period,'inclusions':inc,'contraction':ball_json(contraction),'krawczyk':matrix_json(K),'defect':matrix_json(D),'left_border':matrix_json(B),'left_inverse':matrix_json(inverse),'left_augmented':matrix_json(left),'cubic_numerator':matrix_json(third),'cubic_coefficient':ball_json(cubic),'unfolding':matrix_json(U),'unfolding_determinant':ball_json(determinant),'parameter_solutions':ys,'deflated':matrix_json(Q),'spectral_discs':discs,'proper_divisor_separators':separators,'scope':'Independent reduction of saved analytic enclosures; original byte replay supplies derivative acquisition binding. No new derivative calls in this reduction.'}
