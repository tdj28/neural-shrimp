"""Fresh full-native PAL acquisition with a finite affine inverse proposal.

No Newton or source/known-root provenance is established here. Caller supplies
exact predictor/preconditioner proposals and must bind any conditional anchor.
"""
from fractions import Fraction
from flint import arb,arb_mat
from .rigorous_cubic_gradient_path import cubic_jacobian_coefficients as jacobian_path_coefficients
from .rigorous_cubic_inverse import cubic_inverse_data as correlated_inverse_data
from .rigorous_polynomial_inverse import serialize_correlated_inverse
from .rigorous_cubic_unit_pair import cubic_residual_coefficients as path_coefficients
from .rigorous_eigenpair_locus import require,finite,exact_mid
from .rigorous_eigenpair_march import decode
from .rigorous_orbits import exact,vector_json,matrix_json,midpoint_matrix


def cubic_correlated_tube(model,center,slope,curvature,cubic,C0,*,period,normal,origin,parameter,
                    halfwidth_exponent,radius_exponent,weights,anchor=None,
                    tick=lambda:None,observe=lambda k,v:None,compact=True):
    require(compact is True,'cubic curves require complete native/compact-derived evidence')
    n=model.n;size=3*n*period+3
    require(all(len(v)==size for v in [center,slope,curvature,cubic,normal,origin,weights]),'full native705 proposals')
    require(all(v.is_finite()and v.is_exact()for v in center+slope+curvature+cubic+normal+origin+weights),'exact finite proposals')
    require(C0.nrows()==C0.ncols()==size and finite(C0)and all(v.is_exact()for row in C0.tolist()for v in row),'exact full inverse proposal')
    require(all(type(k)is int and -100<=k<=-1 for k in [halfwidth_exponent,radius_exponent]),'bounded exact dyadic scales')
    require(all(v>0 for v in weights),'positive weights')
    sc=exact(parameter);h=arb((1,halfwidth_exponent));rho=arb((1,radius_exponent));d=arb(0,h)
    require(sc.is_exact()and sc.is_finite(),'exact finite gauge center')
    radii=[rho*w for w in weights];error=[arb(0,v.upper())for v in radii]
    predictor=[center[i]+slope[i]*d+curvature[i]*d*d/2+cubic[i]*d*d*d for i in range(size)]
    box=[predictor[i]+error[i]for i in range(size)]
    require(box[-3]>0 and box[-2]>0 and abs(box[-1]).upper()<=exact('1/4'),'whole actual-rate and angle domain')
    observe('cubic_correlated_domain',{'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'cubic':vector_json(cubic),'error':vector_json(error),'whole_box':vector_json(box),'parameter':parameter,'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent})
    # Both acquisitions use precisely the same *constructed* full error box.
    jzero=jacobian_path_coefficients(model,center,slope,curvature,cubic,error,arb(0),period,normal,tick=tick,observe=observe,retain_assembly=not compact)
    jwhole=jacobian_path_coefficients(model,center,slope,curvature,cubic,error,d,period,normal,tick=tick,observe=observe,retain_assembly=not compact)
    J=[jzero[0],jzero[1],jwhole[2]]
    C1=midpoint_matrix(-C0*midpoint_matrix(J[1])*C0)
    coeff=path_coefficients(model,center,slope,curvature,cubic,arb(0),period,normal,origin,sc,tick=tick,observe=observe)
    whole=path_coefficients(model,center,slope,curvature,cubic,d,period,normal,origin,sc,tick=tick,observe=observe)
    residual=[arb_mat([[coeff[i,k]if k<4 else whole[i,4]]for i in range(size)])for k in range(5)]
    raw_bound=correlated_inverse_data([C0,C1],J,residual,weights,rho,h,tick=tick)
    bound=serialize_correlated_inverse(raw_bound,dense=not compact)
    divisors=[]
    for k in range(1,period):
        if period%k:continue
        diff=[box[3*n*k+i]-box[i]for i in range(n)]
        divisors.append({'divisor':k,'differences':vector_json(diff),'excluded':any(not v.contains(0)for v in diff)})
    ac=cubic_anchor_test(anchor,center,slope,curvature,cubic,normal,origin,sc,h,radii,n,period)
    gates={'uniform_full_native_unit_pair_locus':bound['passed'],'primitive_period':all(v['excluded']for v in divisors),'realified_frame_independent':any(not v.contains(0)for v in box[2*n:3*n]),'whole_positive_angle':bool(box[-1]>0),'anchor_identified':ac['passed']}
    defect=raw_bound['whole_defect']if compact else arb_mat([[decode(v)for v in row]for row in bound['whole_defect']]);E=arb_mat([[v]for v in error]);endpoints=[]
    for dd in [-h,h]:
        rr=sum((residual[k]*(dd*dd*dd*dd if k==4 else dd*dd*dd if k==3 else dd*dd if k==2 else dd if k==1 else arb(1))for k in range(5)),arb_mat(size,1))
        ee=-(C0+C1*dd)*rr+defect*E
        endpoints.append([center[i]+slope[i]*dd+curvature[i]*dd*dd/2+cubic[i]*dd*dd*dd+ee[i,0]for i in range(size)])
    result={'schema':'neural-shrimp-cubic-correlated-native-pal-v1','passed':all(gates.values()),'gates':gates,'dimension':n,'period':period,'shooting_dimension':size,'parameter_center':parameter,'parameter_endpoints':[str(Fraction(parameter)-Fraction(2)**halfwidth_exponent),str(Fraction(parameter)+Fraction(2)**halfwidth_exponent)],'normal':[exact_mid(v)for v in normal],'origin':[exact_mid(v)for v in origin],'center':vector_json(center),'slope':vector_json(slope),'curvature':vector_json(curvature),'cubic':vector_json(cubic),'residual_coefficients':[matrix_json(v)for v in residual],'whole_box':vector_json(box),'weights':[exact_mid(v)for v in weights],'settings':{'halfwidth_exponent':halfwidth_exponent,'radius_exponent':radius_exponent},'bound':bound,'divisors':divisors,'anchor_check':ac,'endpoint_root_boxes':[vector_json(v)for v in endpoints],'scope':'Conditional pointwise full-dimensional unit-pair curve. Exact model/source and known-root provenance are caller obligations; complementary spectrum, criticality and global topology are separate.'}
    if compact:
        result['inverse_polynomial_encoding']='exact-dyadic-pairs'
        result['inverse_polynomial_exact']=[[[[int(a)for a in v.mid().man_exp()]for v in row]for row in M.tolist()]for M in [C0,C1]]
        result['derived_matrices_retained']=False
    else:result['inverse_polynomial']=[matrix_json(C0),matrix_json(C1)]
    observe('cubic_correlated_chart_return',result)
    return result


def cubic_anchor_test(anchor,center,slope,curvature,cubic,normal,origin,sc,h,radii,n,period):
    if anchor is None:return {'passed':False,'reason':'known native root not supplied'}
    require(anchor['dimension']==n and anchor['period']==period and len(anchor['box'])==len(center),'same full cubic native convention')
    box=anchor['box'];require(all(v.is_finite()for v in box),'finite full known-root enclosure')
    projection=sum((normal[i]*(box[i]-origin[i])for i in range(len(center))),arb(0))
    parameter=exact(anchor['exact_parameter'])if 'exact_parameter'in anchor else projection
    if 'exact_parameter'in anchor:require(parameter.is_finite()and parameter.is_exact(),'authenticated exact same-gauge prior parameter')
    d=parameter-sc;inside=bool(abs(d).upper()<=h)
    checks=[bool(abs(box[i]-center[i]-slope[i]*d-curvature[i]*d*d/2-cubic[i]*d*d*d).upper()<radii[i])for i in range(len(center))]
    return {'passed':inside and all(checks),'parameter_inside':inside,'projection':vector_json([projection])[0],'parameter':vector_json([parameter])[0],'coordinate_inclusions':checks,'scope':'Conditional full native prior-root membership. Exact parameter requires matching source-authenticated gauge; otherwise whole projected parameter is mandatory.'}
