"""Independent complex bordered derivatives, conditional on a proved same root.

No producer spectral reduction, Sylvester solve, model or derivative import.
Realification keeps all complex interval solves in Arb matrix arithmetic.
"""
from flint import arb, arb_mat

def need(v,s):
    if not v: raise ValueError(s)
def eye(n):return arb_mat([[int(i==j)for j in range(n)]for i in range(n)])
def finite(A):return all(v.is_finite()for row in A.tolist()for v in row)
def zero(A):return all(v.contains(0)for row in A.tolist()for v in row)
def norm(A):return max(sum((abs(v)for v in row),arb(0)).upper()for row in A.tolist())
def mid(A):return arb_mat([[v.mid()for v in row]for row in A.tolist()])
def inverse(A):
    n=A.nrows();need(A.ncols()==n and finite(A),'finite square inverse domain')
    C=mid(mid(A).inv());D=eye(n)-C*A
    need(finite(C)and norm(D)<1,'strict independent inverse defect')
    X=(C*A).inv()*C
    need(finite(X)and zero(A*X-eye(n))and zero(X*A-eye(n)),'validated two-sided inverse')
    return X

def complex_left(M,q,y,c,s):
    """v=q+i*y, lambda=c+i*s, bilinear l^T v=1 (not Hermitian)."""
    n=M.nrows();m=n+1
    R=arb_mat([[M[j,i]-c*int(i==j)if i<n and j<n else q[i]if i<n else q[j]if j<n else arb(0)for j in range(m)]for i in range(m)])
    I=arb_mat([[-s*int(i==j)if i<n and j<n else -y[i]if i<n else y[j]if j<n else arb(0)for j in range(m)]for i in range(m)])
    B=arb_mat([[R[i%m,j%m]if (i<m)==(j<m)else -I[i,j-m]if i<m else I[i-m,j]for j in range(2*m)]for i in range(2*m)])
    rhs=arb_mat([[int(i==n)]for i in range(2*m)]);left=inverse(B)*rhs
    lr=[left[i,0]for i in range(n)];li=[left[m+i,0]for i in range(n)]
    need(left[n,0].contains(0)and left[m+n,0].contains(0),'conditional complex border multiplier zero')
    need((sum((lr[i]*q[i]-li[i]*y[i]for i in range(n)),arb(0))-1).contains(0)and sum((lr[i]*y[i]+li[i]*q[i]for i in range(n)),arb(0)).contains(0),'complex bilinear normalization')
    return lr,li,B,left

def physical_gradients(M,J,box):
    n=M.nrows();k=3*n+2;dim=4*n+3
    need(4<=n<=13 and M.ncols()==n and J.nrows()==J.ncols()==dim and len(box)==dim,'complete dimensions')
    need(finite(M)and finite(J)and all(v.is_finite()for v in box),'finite actual domains')
    t=box[k];need(t>0 and t<arb(1)/4 and box[n]>0 and box[n+1]>0,'positive physical rates and nonreal unit pair')
    a=t.sqrt();c=a.cos();s=a.sin();q=box[n+2:2*n+2];r=box[2*n+2:k];w=box[k+1:];y=[a*v for v in r]
    for i in range(n):
        mq=sum((M[i,j]*q[j]for j in range(n)),arb(0));my=sum((M[i,j]*y[j]for j in range(n)),arb(0));mw=sum((M[i,j]*w[j]for j in range(n)),arb(0))
        need((mq-c*q[i]+s*y[i]).contains(0)and(my-s*q[i]-c*y[i]).contains(0)and(mw+w[i]).contains(0),'conditional actual critical eigenvector identities')
    lr,li,border,left=complex_left(M,q,y,c,s)
    realborder=arb_mat([[M[j,i]+int(i==j)if i<n and j<n else w[i]if i<n else w[j]if j<n else arb(0)for j in range(n+1)]for i in range(n+1)])
    realleft=inverse(realborder)*arb_mat([[int(i==n)]for i in range(n+1)])
    need(realleft[n,0].contains(0)and(sum((realleft[i,0]*w[i]for i in range(n)),arb(0))-1).contains(0),'conditional real left normalization')
    T=inverse(eye(n)-M)*arb_mat([[J[i,n+p]for p in range(2)]for i in range(n)])
    physical=arb_mat([[T[i,p]if i<n else arb(int(i-n==p))for p in range(2)]for i in range(n+2)])
    rates=[];actions=[];complex_derivatives=[]
    for p in range(2):
        A=arb_mat([[sum((J[offset+i,j]*physical[j,p]for j in range(n+2)),arb(0))for offset in(n,2*n,k)]for i in range(n)])
        re=sum((lr[i]*A[i,0]-li[i]*a*A[i,1]for i in range(n)),arb(0));im=sum((lr[i]*a*A[i,1]+li[i]*A[i,0]for i in range(n)),arb(0))
        sigma=(re*c+im*s)/(c*c+s*s);tau=sum((realleft[i,0]*A[i,2]for i in range(n)),arb(0))
        rates.append((sigma,tau));actions.append(A);complex_derivatives.append([re,im])
    G=arb_mat([[rates[p][i]for p in range(2)]for i in range(2)]);det=G[0,0]*G[1,1]-G[0,1]*G[1,0]
    crossing=None if G[0,1].contains(0)else -det/G[0,1]
    direction=[arb(-1),arb(11)/20000];directional=[sum((G[i,j]*direction[j]for j in range(2)),arb(0))for i in range(2)]
    need(finite(G)and det.is_finite(),'finite physical gradients')
    return dict(passed=not det.contains(0),physical_gradient=G,determinant=det,flip_derivative_along_unit_hidden=crossing,complex_border=border,complex_left_realification=left,real_border=realborder,real_left=realleft,physical_state_tangents=T,full_physical_tangents=physical,critical_actions=actions,complex_eigenvalue_derivatives=complex_derivatives,direction=direction,directional_derivatives=directional,attracting_direction=bool(directional[0]<0 and directional[1]>0),scope='Conditional on independently proved same-root identities and ten stable complementary modes. sigma=log|lambda_complex|; tau=lambda_real+1. Direction (-1,11/20000) chosen after producer outcome. Strict signs give a qualitative local side only, not a finite-distance claim.')
