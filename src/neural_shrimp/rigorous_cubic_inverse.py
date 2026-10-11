"""Affine inverse times quadratic Jacobian and quartic residual enclosures.

All products retain noncommuting order. Common-parameter parity encloses powers
through5. This is conditional algebra only; caller binds identical actual full
cubic correction tube and fresh J0(E),J1(E),J2(I,E)/quartic remainder domains.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye
from .rigorous_eigenpair_locus import require,finite


def cubic_inverse_data(C,J,residual,weights,radius,halfwidth,*,tick=lambda:None):
    n=len(weights)
    require(1<=n<=705 and len(C)==2 and len(J)==3 and len(residual)==5,'full bounded polynomial shapes')
    require(all(v.nrows()==v.ncols()==n and finite(v)for v in C+J),'full finite matrix polynomials')
    require(all(v.nrows()==n and v.ncols()==1 and finite(v)for v in residual),'full finite residual coefficients')
    require(all(v.is_exact()for m in C for row in m.tolist()for v in row),'fixed finite polynomial inverse proposal')
    require(all(v.is_finite()and v.is_exact()and v>0 for v in weights+[radius,halfwidth]),'positive finite exact scales')
    # Never commute C and J: these are left-preconditioned full native rows.
    dc=[eye(n)-C[0]*J[0],-C[0]*J[1]-C[1]*J[0],-C[0]*J[2]-C[1]*J[1],-C[1]*J[2]];tick()
    rc=[arb_mat(n,1)for _ in range(6)]
    for a in range(2):
        for b in range(5):rc[a+b]-=C[a]*residual[b]
    h2=halfwidth*halfwidth;h3=h2*halfwidth;h4=h2*h2
    # Arb's generic real power is undefined on negative-base intervals.
    # These exact parity enclosures cover each real d in [-h,h].
    powers=[arb(1),arb(0,halfwidth),arb(h2/2,h2/2),arb(0,h3),arb(h4/2,h4/2),arb(0,h4*halfwidth)]
    defect=sum((dc[k]*powers[k]for k in range(4)),arb_mat(n,n))
    forcing=sum((rc[k]*powers[k]for k in range(6)),arb_mat(n,1))
    radii=[radius*w for w in weights];error=arb_mat([[arb(0,r.upper())]for r in radii])
    image=forcing+defect*error;tick()
    require(finite(defect)and finite(forcing)and finite(image),'finite full correlated images')
    bound=max((sum((abs(defect[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n))
    included=[bool(abs(image[i,0]).upper()<radii[i])for i in range(n)]
    return {'passed':bool(bound<1 and all(included)),'dimension':n,'contraction':bound,'coordinate_inclusions':included,'radii':radii,'defect_coefficients':dc,'forcing_coefficients':rc,'whole_defect':defect,'whole_image':image}


