"""Conditional full-domain polynomial inverse of degree two, no acquisitions.

Caller authenticates J0(E), J1(E), J2(I,E) on the SAME full cubic correction
 tube and r0..r3 plus whole quartic r4. Finite C0,C1,C2 are proposals only.
"""
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import eye,midpoint_matrix
from neural_shrimp.rigorous_eigenpair_locus import require,finite


def propose_quadratic_inverse(C0,J):
    n=C0.nrows();require(C0.ncols()==n and len(J)==3 and all(M.nrows()==M.ncols()==n and finite(M)for M in J+[C0]),'full finite proposal matrices')
    require(all(v.is_exact()for row in C0.tolist()for v in row),'finite exact C0')
    A1=midpoint_matrix(J[1]);A2=midpoint_matrix(J[2]);C1=midpoint_matrix(-C0*A1*C0)
    C2=midpoint_matrix(-(C0*A1*C1+C0*A2*C0))
    return[C0,C1,C2]


def quadratic_inverse_data(C,J,residual,weights,radius,halfwidth,*,tick=lambda:None):
    n=len(weights);require(1<=n<=705 and len(C)==3 and len(J)==3 and len(residual)==5,'full fixed degree2/2/4 shapes')
    require(all(M.nrows()==M.ncols()==n and finite(M)for M in C+J),'full finite square matrices')
    require(all(M.nrows()==n and M.ncols()==1 and finite(M)for M in residual),'all full residual coefficients')
    require(all(v.is_exact()for M in C for row in M.tolist()for v in row),'exact finite fixed C polynomial')
    require(all(v.is_exact()and v.is_finite()and v>0 for v in weights+[radius,halfwidth]),'finite exact positive scales')
    D=[arb_mat(n,n)for _ in range(5)];D[0]=eye(n)
    for a in range(3):
        for b in range(3):D[a+b]-=C[a]*J[b]
    tick()
    rhs=[arb_mat(n,1)for _ in range(7)]
    for a in range(3):
        for b in range(5):rhs[a+b]-=C[a]*residual[b]
    powers=[arb(1)];hk=arb(1)
    for k in range(1,7):
        hk*=halfwidth;powers.append(arb(0,hk)if k%2 else arb(hk/2,hk/2))
    defect=sum((v*powers[k]for k,v in enumerate(D)),arb_mat(n,n));forcing=sum((v*powers[k]for k,v in enumerate(rhs)),arb_mat(n,1))
    radii=[radius*w for w in weights];error=arb_mat([[arb(0,r.upper())]for r in radii]);image=forcing+defect*error;tick()
    require(finite(defect)and finite(forcing)and finite(image),'finite whole polynomial bounds')
    q=max((sum((abs(defect[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n));inside=[bool(abs(image[i,0]).upper()<radii[i])for i in range(n)]
    return dict(passed=bool(q<1 and all(inside)),dimension=n,contraction=q,coordinate_inclusions=inside,radii=radii,defect_coefficients=D,forcing_coefficients=rhs,whole_defect=defect,whole_image=image)
