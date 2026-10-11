"""Conditional correlated Krawczyk bounds with a parameter-dependent inverse.

Caller must authenticate, on the *same* actual curved correction tube, every
coefficient enclosure J(d,e)=J0(e)+J1(e)d+J2([−h,h],e)d² and the residual
H(p(d),s0+d)=r0+r1d+r2d²+r3([−h,h])d³. This helper acquires no derivatives.
C0+C1d is a finite exact polynomial proposal. Full native/gauge/primitive and
known-root connection obligations are separate and cannot be inferred here.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,matrix_json,ball_json,vector_json
from .rigorous_eigenpair_locus import require,finite


def correlated_inverse_data(C,J,residual,weights,radius,halfwidth,*,tick=lambda:None):
    n=len(weights)
    require(1<=n<=705 and len(C)==2 and len(J)==3 and len(residual)==4,'full bounded polynomial shapes')
    require(all(v.nrows()==v.ncols()==n and finite(v)for v in C+J),'full finite matrix polynomials')
    require(all(v.nrows()==n and v.ncols()==1 and finite(v)for v in residual),'full finite residual coefficients')
    require(all(v.is_exact()for m in C for row in m.tolist()for v in row),'fixed finite polynomial inverse proposal')
    require(all(v.is_finite()and v.is_exact()and v>0 for v in weights+[radius,halfwidth]),'positive finite exact scales')
    # Never commute C and J: these are left-preconditioned full native rows.
    dc=[eye(n)-C[0]*J[0],-C[0]*J[1]-C[1]*J[0],-C[0]*J[2]-C[1]*J[1],-C[1]*J[2]];tick()
    rc=[arb_mat(n,1)for _ in range(5)]
    for a in range(2):
        for b in range(4):rc[a+b]-=C[a]*residual[b]
    h2=halfwidth*halfwidth;h3=h2*halfwidth;h4=h2*h2
    # Arb's generic real power is undefined on negative-base intervals.
    # These exact parity enclosures cover each real d in [-h,h].
    powers=[arb(1),arb(0,halfwidth),arb(h2/2,h2/2),arb(0,h3),arb(h4/2,h4/2)]
    defect=sum((dc[k]*powers[k]for k in range(4)),arb_mat(n,n))
    forcing=sum((rc[k]*powers[k]for k in range(5)),arb_mat(n,1))
    radii=[radius*w for w in weights];error=arb_mat([[arb(0,r.upper())]for r in radii])
    image=forcing+defect*error;tick()
    require(finite(defect)and finite(forcing)and finite(image),'finite full correlated images')
    bound=max((sum((abs(defect[i,j]).upper()*weights[j]for j in range(n)),arb(0))/weights[i]).upper()for i in range(n))
    included=[bool(abs(image[i,0]).upper()<radii[i])for i in range(n)]
    return {'passed':bool(bound<1 and all(included)),'dimension':n,'contraction':bound,'coordinate_inclusions':included,'radii':radii,'defect_coefficients':dc,'forcing_coefficients':rc,'whole_defect':defect,'whole_image':image}


def serialize_correlated_inverse(data,*,dense=True):
    """Serialization only; dense=False retains gates while derived arrays replay.

    Native input/returns and the finite exact C polynomial are mandatory caller
    evidence. They uniquely determine every omitted derived matrix at the
    recorded arithmetic settings; this flag changes no mathematical bound.
    """
    require(type(dense)is bool,'explicit derived-array retention mode')
    result={'passed':data['passed'],'dimension':data['dimension'],'contraction':ball_json(data['contraction']),'coordinate_inclusions':data['coordinate_inclusions'],'radii':vector_json(data['radii']),'scope':'Conditional same-domain full-dimensional Krawczyk sufficient inequalities only. Requires independently authenticated Jacobian/residual Taylor enclosures and exact inverse proposals on the actual curved correction tube; no primitive orbit, connection, spectrum or new derivative claim.'}
    if dense:
        result.update(defect_coefficients=[matrix_json(v)for v in data['defect_coefficients']],forcing_coefficients=[matrix_json(v)for v in data['forcing_coefficients']],whole_defect=matrix_json(data['whole_defect']),whole_image=matrix_json(data['whole_image']))
    else:
        result.update(derived_matrices_retained=False,whole_image=matrix_json(data['whole_image']),image_radius_ratio_upper=vector_json([(abs(data['whole_image'][i,0]).upper()/r).upper()for i,r in enumerate(data['radii'])]),reconstruction='Full noncommuting C(d)*J(d,e) and C(d)*residual convolution from retained native analytic outputs and finite exact inverse proposal; all mathematical operations and inequalities unchanged.')
    return result


def correlated_inverse_bound(C,J,residual,weights,radius,halfwidth,*,tick=lambda:None):
    """Historical full derived-array serialization, unchanged mathematical API."""
    return serialize_correlated_inverse(correlated_inverse_data(C,J,residual,weights,radius,halfwidth,tick=tick))
