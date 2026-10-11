"""Pure supplied-array curvature diagnostic. No I/O, model or acquisition code.

Floating spectral diagnostics only; supplied matrices do not establish an orbit.
Decompositions are explicit and retained: <=2 eigh, <=1 eig, <=1 full basis SVD.
"""
import math
import numpy as np
from scipy.linalg import eig

TOL=dict(symmetry=1e-12,top_gap=1e-6,aligned=1e-10,discriminant=1e-10,
    pair=1e-10,residual=1e-10,product_atol=1e-10,product_rtol=1e-10,
    attribution_radius=.02)


def require(ok,message):
    if not ok:raise ValueError(message)


def real(a,shape=None):
    a=np.asarray(a)
    require(a.dtype==np.float64 and (shape is None or a.shape==shape) and np.isfinite(a).all(),'finite float64 input shape')
    return a


def fro(a):return float(np.sqrt(np.sum(np.abs(a)**2)))


class Decompositions:
    def __init__(self,observer=None):
        self.counts=dict(eigh=0,eig=0,svd=0);self.records=[];self.observer=observer or (lambda *_:None)
    def call(self,kind,matrix):
        require(self.counts[kind]<(2 if kind=='eigh' else 1),'decomposition cap')
        ident=len(self.records);self.counts[kind]+=1;row=dict(id=ident,kind=kind,input=matrix.copy(),status='started');self.records.append(row)
        self.observer('decomposition_start',row)
        try:
            if kind=='eigh':w,V=np.linalg.eigh(matrix);value=dict(values=w,vectors=V)
            elif kind=='eig':w,left,right=eig(matrix,left=True,right=True,check_finite=True);value=dict(values=w,left=left,right=right)
            else:U,s,Vh=np.linalg.svd(matrix,full_matrices=True);value=dict(U=U,s=s,Vh=Vh)
        except BaseException as error:
            row.update(status='raised',error=type(error).__name__+': '+str(error));self.observer('decomposition_exception',row);raise
        row.update(status='returned',output=value);self.observer('decomposition_return',row)
        return value


def leading_pair(values):
    values=np.asarray(values,dtype=np.complex128);require(values.ndim==1 and len(values)>=2 and np.isfinite(values).all(),'complete finite spectrum')
    order=sorted(range(len(values)),key=lambda i:(-float(abs(values[i])),-float(values[i].real),-float(values[i].imag),i))
    pair=values[order[:2]];scale=max(1.,float(np.max(np.abs(values))));eps=TOL['pair']*scale
    real_modes=np.abs(pair.imag)==0
    if bool(np.all(real_modes)):kind='real_pair';qualified=True
    elif bool(np.any(real_modes)):kind='mixed_pair';qualified=False
    elif float(np.min(np.abs(pair.imag)))<=eps:kind='near_real_unresolved';qualified=False
    elif abs(pair[0]-np.conj(pair[1]))<=eps and pair[0].imag*pair[1].imag<0:kind='complex_conjugate_pair';qualified=True
    else:kind='nonconjugate_pair';qualified=False
    tie=len(values)>2 and abs(abs(values[order[1]])-abs(values[order[2]]))<=eps
    return dict(indices=np.array(order[:2]),values=pair,kind=kind,qualified=qualified,
        cutoff_modulus_tie=bool(tie),order=np.array(order),spectral_radius=float(np.max(np.abs(values))))


def full_model_spectrum(n,a,b,alpha,beta,c):
    require(type(n) is int and n>=2,'ambient dimension >=2');require(all(math.isfinite(v) for v in (a,b,alpha,beta,c)) and abs(c)<=1+1e-12,'finite fit strengths/overlap')
    c=float(np.clip(c,-1,1));T=2*a*b-b*alpha-a*beta+alpha*beta*c*c;D=a*b*(a-alpha)*(b-beta)
    delta=T*T-4*D;band=TOL['discriminant']*max(1.,T*T,4*abs(D));aligned=1-c*c<=TOL['aligned']
    if aligned:
        values=np.array([(a-alpha)*(b-beta)]+[a*b]*(n-1),dtype=complex);kind='aligned_one_dimensional';components=['selected']+['bulk']*(n-1)
    else:
        root=np.sqrt(complex(delta));values=np.array([(T+root)/2,(T-root)/2]+[a*b]*(n-2),dtype=complex)
        kind='complex' if delta < -band else 'real' if delta>band else 'discriminant_unresolved';components=['plane','plane']+['bulk']*(n-2)
    return dict(values=values,components=components,trace=float(T),determinant=float(D),discriminant=float(delta),discriminant_band=float(band),span_case=kind,leading=leading_pair(values),complex_plane_modulus=math.sqrt(D) if kind=='complex' and D>0 else None)


def pair_error(predicted,target,radius):
    a=np.asarray(predicted,dtype=complex);b=np.asarray(target,dtype=complex)
    errors=[float(np.max(np.abs(a-b))),float(np.max(np.abs(a-b[::-1])))];choice=int(errors[1]<errors[0]);matched=b[::-1] if choice else b
    return dict(max_absolute=errors[choice],normalized=errors[choice]/max(1.,radius),assignment=np.array([1,0] if choice else [0,1]),modulus_error=float(np.max(np.abs(np.abs(a)-np.abs(matched)))))


def phase_fit(H,rates,decomp):
    n=len(rates);asym=fro(H-H.T)/max(1.,fro(H));require(asym<=TOL['symmetry'],'nonsymmetric_hessian')
    Hsym=(H+H.T)/2;root=np.sqrt(rates);K=root[:,None]*Hsym*root[None,:]
    e=decomp.call('eigh',K);w,V=e['values'],e['vectors'];require(np.isfinite(w).all() and np.isfinite(V).all(),'nonfinite_phase_eigensystem')
    residual=fro(K@V-V*w)/max(1.,fro(K));orthogonality=fro(V.T@V-np.eye(n));require(max(residual,orthogonality)<=TOL['residual'],'phase_eigensystem_residual')
    gap=float(w[-1]-w[-2]);normK=float(np.max(np.abs(w)));require(gap>TOL['top_gap']*max(1.,normK),'ambiguous_top_curvature')
    u=V[:,-1]/fro(V[:,-1]);bulk=float((np.trace(K)-w[-1])/(n-1));a=1-bulk;alpha=float(w[-1]-bulk)
    require(alpha!=0,'zero_selected_strength');S=np.eye(n)-K;S0=a*np.eye(n)-alpha*np.outer(u,u);E=S-S0
    return dict(H_unsymmetrized=H,H_symmetric=Hsym,K=K,S=S,S0=S0,E=E,asymmetry=asym,
        eigenvalues=w,eigenvectors=V,eigen_residual=residual,orthogonality=orthogonality,top_gap=gap,K_norm2=normK,
        u=u,bulk_curvature=bulk,bulk_factor=a,strength=alpha,model_norm2=max(abs(a),abs(a-alpha)),error_frobenius=fro(E))


def numerical_radius(M,Mmodel,delta_op,decomp,e):
    w,left,V=e['values'],e['left'],e['right'];n=len(M)
    require(w.shape==(n,) and left.shape==V.shape==(n,n) and all(np.isfinite(v).all() for v in (w,left,V)),'complete_finite_general_eigensystem')
    sv=decomp.call('svd',V);U,s,Vh=sv['U'],sv['s'],sv['Vh']
    require(U.shape==Vh.shape==(n,n) and s.shape==(n,) and all(np.isfinite(v).all() for v in (U,s,Vh)),'complete_finite_basis_svd')
    scale=max(1.,fro(V));svd_residual=max(fro(V@Vh.conj().T-U*s)/scale,fro(V.conj().T@U-Vh.conj().T*s)/scale,fro(V-(U*s)@Vh)/scale,fro(U.conj().T@U-np.eye(n)),fro(Vh@Vh.conj().T-np.eye(n)))
    require(np.all(s>=0) and np.all(s[:-1]>=s[1:]),'ordered_basis_singular_values')
    minimum=float(s[-1]);maximum=float(s[0]);basis_ok=minimum>0 and math.isfinite(maximum/minimum) and svd_residual<=TOL['residual']
    R=M@V-V*w;LR=M.conj().T@left-left*np.conj(w)
    rn=np.sqrt(np.sum(np.abs(V)**2,axis=0));ln=np.sqrt(np.sum(np.abs(left)**2,axis=0));require(np.all(rn>0) and np.all(ln>0),'nonzero_eigenvectors')
    # Frobenius residual normalization is explicit, with no hidden operator-norm SVD.
    denominator=np.maximum(1.,fro(M)+np.abs(w));rr=np.sqrt(np.sum(np.abs(R)**2,axis=0))/(denominator*rn);lr=np.sqrt(np.sum(np.abs(LR)**2,axis=0))/(denominator*ln)
    overlaps=np.abs(np.sum(np.conj(left)*V,axis=0))/(ln*rn);conditions=[float(1/v) if v>0 and np.isfinite(1/v) else None for v in overlaps]
    kappa=maximum/minimum if basis_ok else None;diag=fro(R)/minimum if basis_ok else None;rho=diag+kappa*delta_op if basis_ok else None
    if rho is not None and not math.isfinite(rho):rho=None;basis_ok=False
    return dict(R_V=R,left_residual_matrix=LR,right_relative_residual=rr,left_relative_residual=lr,eigenvalue_conditions=conditions,
        sigma_min=minimum,sigma_max=maximum,basis_svd_residual=svd_residual,kappa2=kappa,delta_diag=diag,rho_num=rho,
        basis_qualified=bool(basis_ok),eigen_residuals_qualified=bool(np.isfinite(rr).all() and np.isfinite(lr).all() and max(float(rr.max()),float(lr.max()))<=TOL['residual']),
        product_difference=M-Mmodel,scope='Floating spectral diagnostic only; no enclosure, eigenvalue counts or validated invertibility.')


def analyze_curvature(gradients,hessians,rates,actual_monodromies=None,observer=None):
    """Analyze one supplied reference; all orbit/input provenance is caller-owned.

    actual_monodromies are original-coordinate native matrices. The first is
    the target, whitened by P^-1/2 M P^1/2. Its mismatch from S_b S_a enters
    the operator diagnostic. Without them the target is exactly the supplied
    Hessian pair product, with no claim that it represents an actual orbit.
    """
    d=Decompositions(observer);out=dict(status='unresolved',reasons=[],decomposition_counts=d.counts,decompositions=d.records,
        scope='Supplied-array numerical analysis. Closed-orbit qualification, actual model/control provenance and empirical panel gate are external and pending.')
    try:
        p=real(rates);require(p.ndim==1 and 2<=len(p)<=72 and np.all(p>0),'positive fixed rates and dimension2..72');n=len(p)
        H=real(hessians,(2,n,n));g=real(gradients,(2,n));out['inputs']=dict(gradients=g.copy(),hessians=H.copy(),rates=p.copy(),actual_monodromies=None if actual_monodromies is None else np.asarray(actual_monodromies).copy())
        a=phase_fit(H[0],p,d);b=phase_fit(H[1],p,d);out['phases']=[a,b]
        product=b['S']@a['S'];reverse=a['S']@b['S'];model=b['S0']@a['S0'];c=float(a['u']@b['u']);root=np.sqrt(p)
        if actual_monodromies is None:target=product;formation=0.;other_formation=0.;target_kind='Hessian_pair_product_only';agreement=True
        else:
            actual=real(actual_monodromies,(2,n,n));white=actual*root[None,None,:]/root[None,:,None]
            target=white[0];formation=fro(target-product);other_formation=fro(white[1]-reverse);target_kind='supplied_external_native0'
            agreement=bool(np.all(np.abs(target-product)<=TOL['product_atol']+TOL['product_rtol']*np.abs(product)) and np.all(np.abs(white[1]-reverse)<=TOL['product_atol']+TOL['product_rtol']*np.abs(reverse)))
        delta_fit=b['model_norm2']*a['error_frobenius']+b['error_frobenius']*a['model_norm2']+b['error_frobenius']*a['error_frobenius'];delta_total=delta_fit+formation
        out.update(pair_product=product,reverse_pair_product=reverse,model_product=model,target=target,target_kind=target_kind,
            formation_error_frobenius=formation,opposite_phase_formation_error_frobenius=other_formation,actual_pair_product_agreement=agreement,
            orientation_cosine=c,delta_fit=delta_fit,delta_op_total=delta_total,relative_product_error=fro(target-model)/max(1.,fro(target)),whitening_condition=float(np.max(root)/np.min(root)))
        strengths=(n,a['bulk_factor'],b['bulk_factor'],a['strength'],b['strength'])
        spectra={name:full_model_spectrum(*strengths,overlap) for name,overlap in [('rotating',c),('aligned',1.),('orthogonal',0.)]};out['models']=spectra
        e=d.call('eig',target);out['target_eigensystem']=e;target_pair=leading_pair(e['values']);out['target_pair']=target_pair
        diag=numerical_radius(target,model,delta_total,d,e);out['spectral_diagnostic']=diag
        comparisons={}
        for name,spec in spectra.items():
            chosen=spec['leading'];comparison=pair_error(chosen['values'],target_pair['values'],target_pair['spectral_radius'])
            comparison.update(model_pair_kind=chosen['kind'],target_pair_kind=target_pair['kind'],pair_qualified=bool(chosen['qualified'] and target_pair['qualified']),angle_error=None)
            if chosen['kind']==target_pair['kind']=='complex_conjugate_pair':
                x=chosen['values'][np.argmax(chosen['values'].imag)];y=target_pair['values'][np.argmax(target_pair['values'].imag)]
                comparison['angle_error']=abs(float(np.angle(x)-np.angle(y)))
            comparisons[name]=comparison
        out['comparisons']=comparisons;out['oracle_commuting_error']=min(comparisons['aligned']['normalized'],comparisons['orthogonal']['normalized'])
        out['oracle_label']='Better of two fixed commuting controls; not deployable predictor.'
        modeled=spectra['rotating'];selected=modeled['leading'];rho=diag['rho_num'];separations=[abs(e['values'][i]-e['values'][j]) for i in target_pair['indices'] for j in range(n) if j not in target_pair['indices']]
        sep_other=float(min(separations)) if separations else None;sep_axis=float(np.min(np.abs(target_pair['values'].imag)))
        assigned=target_pair['values'][comparisons['rotating']['assignment']];disk_errors=np.abs(selected['values']-assigned)
        attribution=bool(target_pair['kind']==selected['kind']=='complex_conjugate_pair' and modeled['span_case']=='complex' and not target_pair['cutoff_modulus_tie'] and not selected['cutoff_modulus_tie'] and rho is not None and rho<=TOL['attribution_radius'] and rho<.5*sep_axis and (sep_other is None or rho<.5*sep_other) and np.all(disk_errors<=rho) and diag['basis_qualified'] and diag['eigen_residuals_qualified'] and agreement)
        out['attribution']=dict(qualified=attribution,rho_num=rho,separation_other=sep_other,separation_real_axis=sep_axis,assigned_root_distances=disk_errors,scope='Floating pair attribution only, never a certified disk count or enclosure.')
        out['strength_only']=dict(complex_plane_modulus=modeled['complex_plane_modulus'],orientation_credit_for_modulus=False,conditional_on=modeled['span_case'])
        reasons=[]
        if not agreement:reasons.append('actual_pair_product_mismatch')
        if not target_pair['qualified'] or not selected['qualified']:reasons.append('leading_pair_unqualified')
        if target_pair['cutoff_modulus_tie'] or selected['cutoff_modulus_tie']:reasons.append('leading_cutoff_tie')
        if modeled['span_case']=='discriminant_unresolved':reasons.append('model_discriminant_unresolved')
        if not diag['basis_qualified'] or not diag['eigen_residuals_qualified']:reasons.append('eigenbasis_diagnostic_failure')
        out['reasons']=reasons;out['status']='analysis_complete' if not reasons else 'unresolved'
    except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError) as error:out['reasons'].append(type(error).__name__+': '+str(error))
    return out
