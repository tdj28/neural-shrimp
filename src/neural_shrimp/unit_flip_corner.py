"""Physical stability-boundary gradients at a proved complex-unit/simple-flip root.

Conditional on the authenticated same-root critical equations and stable
complement. Uses no new native map or derivative acquisition. The left spectral
frame is recovered by a Sylvester solve, including nonnormal cross coupling.
"""
from flint import arb,arb_mat
from . import junction_audit as A

def block(M,rows,cols):return arb_mat([[M[i,j]for j in cols]for i in rows])
def finite(M):return all(M[i,j].is_finite()for i in range(M.nrows())for j in range(M.ncols()))
def zeros(M):return all(M[i,j].contains(0)for i in range(M.nrows())for j in range(M.ncols()))

def gradients(M,J,box,columns):
    n=M.nrows();k=3*n+2;dim=4*n+3
    A.need(4<=n<=13 and M.ncols()==n and J.nrows()==J.ncols()==dim and len(box)==dim,'complete native root dimensions')
    A.need(finite(M)and finite(J)and all(v.is_finite()for v in box),'finite saved derivative enclosure')
    A.need(len(columns)==n-3 and len(set(columns))==n-3 and all(type(i)is int and 0<=i<n for i in columns),'complete fixed frame proposal')
    q,r,w=box[n+2:2*n+2],box[2*n+2:k],box[k+1:];t=box[k]
    A.need(t>0 and t<arb(1)/4,'nonreal unit pair away from+1')
    angle=t.sqrt();c=angle.cos();e=angle.sin()/angle;d=t*e
    B=arb_mat([[c,e,0],[-d,c,0],[0,0,-1]])
    frame=arb_mat([[q[i],r[i],w[i]]+[arb(int(i==j))for j in columns]for i in range(n)])
    inv=A.validated_solve(frame,A.eye(n));K=inv*M*frame
    A.need(zeros(block(K,range(3,n),range(3)))and zeros(block(K,range(3),range(3))-B),'same-root exact critical identities enclosed')
    D=block(K,range(3),range(3,n));Q=block(K,range(3,n),range(3,n));m=n-3
    # BX-XQ=D, column-major vectorization. Then L=[I,X] frame^-1,
    # LR=I and LM=BL at the exact proved critical root.
    S=arb_mat([[B[i,j]*int(a==b)-int(i==j)*Q[b,a]for b in range(m)for j in range(3)]for a in range(m)for i in range(3)])
    rhs=arb_mat([[D[i,a]]for a in range(m)for i in range(3)])
    v=A.validated_solve(S,rhs);X=arb_mat([[v[3*a+i,0]for a in range(m)]for i in range(3)])
    L=arb_mat([[arb(int(i==j))for j in range(3)]+[X[i,a]for a in range(m)]for i in range(3)])*inv
    R=block(frame,range(n),range(3))
    A.need(zeros(L*R-A.eye(3))and zeros(L*M-B*L),'conditional exact dual critical invariance enclosed')
    # Native fixed point of F^p continues with both physical learning rates.
    # (I-M)x_rate=F_rate. All terms come from the same augmented whole box.
    Fp=block(J,range(n),range(n,n+2));T=A.validated_solve(A.eye(n)-M,Fp)
    physical=arb_mat([[T[i,j]if i<n else arb(int(i-n==j))for j in range(2)]for i in range(n+2)])
    rows=[];actions=[];B2=block(B,range(2),range(2));Binv=A.validated_solve(B2,A.eye(2))
    for p in range(2):
        a=arb_mat([[sum((J[offset+i,j]*physical[j,p]for j in range(n+2)),arb(0))for offset in(n,2*n,k)]for i in range(n)])
        H=L*a;C=Binv*block(H,range(2),range(2));sigma=(C[0,0]+C[1,1])/2;tau=H[2,2]
        rows.append((sigma,tau));actions.append(a)
    G=arb_mat([[rows[p][i]for p in range(2)]for i in range(2)]);det=G[0,0]*G[1,1]-G[0,1]*G[1,0]
    A.need(finite(G)and det.is_finite(),'finite full physical stability gradients')
    crossing=None
    if not G[0,1].contains(0):crossing=-det/G[0,1]
    return dict(passed=not det.contains(0),frame=frame,frame_inverse=inv,critical_block=B,transformed=K,sylvester=S,sylvester_rhs=rhs,sylvester_solution=X,left_critical=L,physical_state_tangents=T,full_physical_tangents=physical,critical_actions=actions,physical_gradient=G,determinant=det,flip_derivative_along_unit_hidden=crossing,
      scope='Conditional same-root full-state IFT and transverse physical stability boundaries. sigma=log|complex multiplier|, tau=real multiplier+1. Stable local sector sigma<0,tau>0; qualitative open neighborhood only, no numerical wedge size or nonlinear normal-form assertion.')
