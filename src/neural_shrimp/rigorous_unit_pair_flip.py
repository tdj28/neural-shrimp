"""Outward-rounded full native NS/flip augmented root and conditional spectrum."""
import itertools
import numpy as np
from flint import arb,arb_mat
from .rigorous_folds import MixedJet
from .rigorous_unit_pair import entire_pair
from .rigorous_eigenpair_locus import require,finite
from .rigorous_cusps import spectral_exclusion
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,vector_json,matrix_json,ball_json


def augmented(model,y,period,tick=lambda:None):
    n=model.n;base=n+2;k=3*n+2;dim=4*n+3
    require(type(n)is int and 3<=n<=13 and len(y)==dim and 1<=period<=18,'native junction dimensions')
    x=y[:n];rates=y[n:n+2];q=y[base:base+n];r=y[base+n:k];t=y[k];w=y[k+1:]
    require(all(v.is_finite()for v in y),'finite full native domain')
    returns=[]
    for direction in(q,r,w):
        state=[MixedJet.variable(x[i],base,i,direction[i])for i in range(n)]
        h=MixedJet.variable(rates[0],base,n);o=MixedJet.variable(rates[1],base,n+1)
        orbit=[[v.f for v in state]];modes=[direction]
        for _ in range(period):
            tick();state=model.cusp_step(state,h,o);orbit.append([v.f for v in state]);modes.append([v.d for v in state])
        returns.append((state,orbit,modes))
    first,orbit,qp=returns[0];second,_,rp=returns[1];third,_,wp=returns[2]
    M=arb_mat([[v.g[j]for j in range(n)]for v in first]);c,e,d=entire_pair(t);cp,ep,dp=entire_pair(t,1)
    value=[v.f-x[i]for i,v in enumerate(first)]+[v.d-c*q[i]+d*r[i]for i,v in enumerate(first)]+[v.d-e*q[i]-c*r[i]for i,v in enumerate(second)]
    value +=[(sum((v*v for v in q),arb(0))-1)/2,sum((q[i]*r[i]for i in range(n)),arb(0))]
    value +=[v.d+w[i]for i,v in enumerate(third)]+[(sum((v*v for v in w),arb(0))-1)/2]
    J=arb_mat(dim,dim)
    for i in range(n):
        for j in range(base):J[i,j]=first[i].g[j];J[n+i,j]=first[i].m[j];J[2*n+i,j]=second[i].m[j];J[k+i,j]=third[i].m[j]
        J[i,i]-=1
        for j in range(n):
            J[n+i,base+j]=M[i,j]-c*int(i==j);J[n+i,base+n+j]=d*int(i==j)
            J[2*n+i,base+j]=-e*int(i==j);J[2*n+i,base+n+j]=M[i,j]-c*int(i==j)
            J[k+i,k+1+j]=M[i,j]+int(i==j)
        J[n+i,k]=-cp*q[i]+dp*r[i];J[2*n+i,k]=-ep*q[i]-cp*r[i]
        J[k-2,base+i]=q[i];J[k-1,base+i]=r[i];J[k-1,base+n+i]=q[i];J[dim-1,k+1+i]=w[i]
    V=arb_mat([[v]for v in value]);require(finite(V)and finite(J),'finite native augmented arithmetic')
    return V,J,{'orbit':orbit,'monodromy':M,'q_phases':qp,'r_phases':rp,'w_phases':wp}


def actual_box_delta(center,box):
    """Cover the constructed Arb box, including its outward-rounded radii."""
    require(len(center)==len(box)and all(box[i].mid()==center[i]for i in range(len(box))),'exact constructed-box centers')
    return arb_mat([[arb(0,v.rad())]for v in box])


def root_proof(model,initial,period,*,radius_exponent=-60,newton_steps=4,tick=lambda:None,observe=lambda k,v:None):
    n=model.n;dim=4*n+3;k=3*n+2
    require(len(initial)==dim and type(newton_steps)is int and 0<=newton_steps<=6 and type(radius_exponent)is int and -256<=radius_exponent<=-1,'fixed proof settings')
    center=[v.mid()for v in initial]
    for iteration in range(newton_steps):
        value,J,_=augmented(model,center,period,tick);observe('newton_evaluation',{'iteration':iteration,'center':vector_json(center),'value':matrix_json(value),'jacobian':matrix_json(J)})
        step=J.inv()*value;require(finite(step),'finite Newton proposal');center=[(center[i]-step[i,0]).mid()for i in range(dim)];observe('newton_proposal',{'iteration':iteration,'center':vector_json(center)})
    value,J,_=augmented(model,center,period,tick);observe('center_evaluation',{'center':vector_json(center),'value':matrix_json(value),'jacobian':matrix_json(J)})
    pre=midpoint_matrix(J.inv());radius=arb((1,radius_exponent));box=[arb(v,radius)for v in center]
    _,whole,data=augmented(model,box,period,tick);observe('box_evaluation',{'box':vector_json(box),'jacobian':matrix_json(whole),'monodromy':matrix_json(data['monodromy'])})
    defect=eye(dim)-pre*whole;contraction=norm_inf_upper(defect);delta=actual_box_delta(center,box);image=-pre*value+defect*delta
    inclusion=[bool(box[i].contains_interior(center[i]+image[i,0]))for i in range(dim)]
    divisors=[]
    for d in range(1,period):
        if period%d:continue
        diff=[data['orbit'][d][i]-box[i]for i in range(n)];divisors.append({'divisor':d,'excluded':any(not v.contains(0)for v in diff),'differences':vector_json(diff)})
    gates={'strict_native_augmented_root':bool(all(inclusion)and contraction<1),'primitive_period':all(v['excluded']for v in divisors),'positive_actual_rates':bool(box[n]>0 and box[n+1]>0),'small_positive_angle':bool(box[k]>0 and box[k]<arb(1)/4)}
    result={'passed':all(gates.values()),'gates':gates,'dimension':n,'period':period,'augmented_dimension':dim,'center':vector_json(center),'box':vector_json(box),'value':matrix_json(value),'preconditioner':matrix_json(pre),'whole_jacobian':matrix_json(whole),'defect':matrix_json(defect),'actual_box_delta':matrix_json(delta),'image_correction':matrix_json(image),'coordinate_inclusions':inclusion,'contraction':ball_json(contraction),'orbit_enclosures':[vector_json(v)for v in data['orbit']],'monodromy':matrix_json(data['monodromy']),'divisors':divisors,'settings':{'newton_steps':newton_steps,'radius_exponent':radius_exponent},'scope':'Locally unique normalized native NS/flip augmented root only. Exact critical multiplicities, complementary attraction, crossing and connection require their same-root qualifications.'}
    observe('root_gate',result);return result,data


def complement(M,q,r,w,t):
    n=len(q);require(n>=4 and len(r)==len(w)==n and M.nrows()==M.ncols()==n,'full critical frame')
    proposal=np.array([[float(q[i].mid()),float(r[i].mid()),float(w[i].mid())]for i in range(n)])
    require(np.isfinite(proposal).all(),'finite minor proposal');pivots=max(itertools.combinations(range(n),3),key=lambda rows:abs(np.linalg.det(proposal[list(rows)])))
    columns=[i for i in range(n)if i not in pivots];frame=arb_mat([[q[i],r[i],w[i]]+[arb(int(i==j))for j in columns]for i in range(n)])
    inverse=frame.inv();require(finite(inverse),'validated three-mode frame inverse');transformed=inverse*M*frame
    c,e,d=entire_pair(t);expected=arb_mat([[c,e,0],[-d,c,0],[0,0,-1]])
    require(all(transformed[i,j].contains(0)for i in range(3,n)for j in range(3)),'conditional exact critical invariance enclosed')
    require(all(transformed[i,j].overlaps(expected[i,j])for i in range(3)for j in range(3)),'conditional exact critical block enclosed')
    quotient=arb_mat([[transformed[i,j]for j in range(3,n)]for i in range(3,n)]);spectral=spectral_exclusion(quotient)
    return {'passed':bool(spectral['passed']and all(v['strict_inside']for v in spectral['discs'])),'coordinate_pivots':list(pivots),'coordinate_columns':columns,'frame':matrix_json(frame),'validated_inverse':matrix_json(inverse),'transformed':matrix_json(transformed),'quotient':matrix_json(quotient),'spectral_exclusion':spectral,'scope':'Conditional on a same-root exact proved native unit-pair and flip identities. Validated frame and strictly attracting quotient imply one simple conjugate unit pair, one simple minus1 and allother n−3 strictly attracting; no parameter unfolding or branch connection.'}


def transverse_crossing(whole_J,M,w,n):
    """Conditional same-root flip derivative along the normalized unit-pair locus."""
    k=3*n+2;require(whole_J.nrows()==whole_J.ncols()==4*n+3 and len(w)==n,'whole-root differential dimensions')
    section=arb_mat([[whole_J[i,j]for j in range(k)]for i in range(k)])
    for i in range(k):section[i,n]=whole_J[i,k]
    inverse=section.inv();require(finite(inverse),'validated native section inverse')
    tangent=-inverse*arb_mat([[whole_J[i,n]]for i in range(k)])
    physical=arb_mat([[arb(1)if i==n else tangent[i,0]]for i in range(n+2)])
    border=arb_mat(n+1,n+1)
    for i in range(n):
        for j in range(n):border[i,j]=M[j,i]+int(i==j)
        border[i,n]=w[i];border[n,i]=w[i]
    border_inverse=border.inv();require(finite(border_inverse),'validated dual border inverse')
    # At the proved root (M+I)w=0 and ||w||=1, the last solution component
    # vanishes exactly; the first n give left(M+I)=0 and left.w=1.
    left=[border_inverse[i,n]for i in range(n)]
    mixed=arb_mat([[whole_J[k+i,j]for j in range(n+2)]for i in range(n)])
    derivative=(arb_mat([left])*mixed*physical)[0,0]
    return {'passed':bool(derivative.is_finite()and not derivative.contains(0)),'section':matrix_json(section),'validated_section_inverse':matrix_json(inverse),'section_tangent':matrix_json(tangent),'physical_state_rate_tangent':matrix_json(physical),'dual_border':matrix_json(border),'validated_dual_border_inverse':matrix_json(border_inverse),'critical_left':vector_json(left),'multiplier_hidden_rate_derivative':ball_json(derivative),'scope':'Conditional on same-root proved native critical equations and same-domain Jacobian. Validated fixed-hidden-rate section and nonzero total flip derivative establish a transverse flip crossing along that unit-pair locus; no normal-form criticality or global connection.'}
