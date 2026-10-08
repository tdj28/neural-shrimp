"""Fixed-state lower-gap diagnostics; no corrective solve or branch acceptance."""
import time
import numpy as np
LIMITS = dict(fixed_states=6, maps=12, jacobians=12, svds=24, internal_seconds=15.,
    external_seconds=20, kill_grace_seconds=5, store_bytes=7*1024**2, launcher_bytes=1024**2)
TOL = dict(agreement_absolute=2e-11, closure_absolute=1e-9, closure_relative=1e-8,
    primitive_separation=1e-6, rank_low=1e-12, rank_high=1e-10,
    svd_residual=1e-10, tangent_residual=1e-10, soft_isolation_ratio=.1,
    orientation_cosine_min=1e-8, nearzero_residual=1e-12,
    material_forcing=1e-6, backward_error=1e-12, h_scale=.000125)
STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def require(value, message):
    if not value: raise ValueError(message)


def finite(value, shape=None):
    a = np.asarray(value)
    require(a.dtype == np.float64 and (shape is None or a.shape == shape) and np.isfinite(a).all(), 'finite float64 shape')
    return a


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


class Budget:
    def __init__(self, emit, seconds=15.):
        self.emit, self.deadline = emit, time.monotonic()+seconds
        self.attempted = dict(maps=0, jacobians=0, svds=0)
        self.returned = dict(maps=0, jacobians=0, svds=0)
    def check(self):
        if STOP: raise InterruptedError('cooperative SIGTERM')
        if time.monotonic() >= self.deadline: raise TimeoutError('fixed-state internal deadline')
    def begin(self, kinds):
        self.check()
        require(all(self.attempted[k] < LIMITS[k] for k in kinds), 'operation cap')
        for k in kinds: self.attempted[k] += 1
        self.emit('operation_started', dict(kinds=list(kinds), attempted=dict(self.attempted)))
    def returned_values(self, kinds, label, payload):
        # Save every returned matrix/decomposition before later checks.
        for k in kinds: self.returned[k] += 1
        self.emit(label, payload)


def phase(theta, rates, hidden, derivative, budget, label):
    theta, rates = finite(theta), finite(rates)
    require(theta.ndim == 1 and theta.shape == rates.shape and np.all(rates > 0), 'state/rates')
    require(type(hidden) is int and 0 < hidden < len(theta), 'hidden block')
    budget.begin(('maps', 'jacobians'))
    loss, gradient, hessian = derivative(theta)
    gradient, hessian = np.asarray(gradient), np.asarray(hessian)
    budget.emit(label+'_analytic_return',dict(loss=float(loss) if np.isfinite(loss) else None,
        gradient=gradient,hessian=hessian))
    finite(gradient,theta.shape);finite(hessian,(len(theta),len(theta)))
    one = theta-rates*gradient
    jacobian = np.eye(len(theta))-rates[:, None]*hessian
    rate_derivative = np.r_[np.log(10.)*rates[:hidden], np.zeros(len(theta)-hidden)]
    parameter = -rate_derivative*gradient
    result = dict(theta=theta, loss=float(loss) if np.isfinite(loss) else None, gradient=gradient, hessian=hessian,
                  image=one, jacobian=jacobian, parameter=parameter)
    budget.returned_values(('maps', 'jacobians'), label,
        {k:v for k,v in result.items() if k not in ('gradient','hessian')})
    for key in ('gradient', 'image', 'parameter'): finite(result[key], theta.shape)
    for key in ('hessian', 'jacobian'): finite(result[key], (len(theta), len(theta)))
    require(np.isfinite(loss), 'finite loss')
    return result


def evaluate_state(theta, rates, hidden, derivative, budget, role):
    first = phase(theta, rates, hidden, derivative, budget, role+'_first_phase')
    second = phase(first['image'], rates, hidden, derivative, budget, role+'_second_phase')
    result = dict(state=theta, map1=first['image'], map2=second['image'],
        J0=first['jacobian'], J1=second['jacobian'], partial_h_F=first['parameter'],
        R=second['image']-theta, M=second['jacobian']@first['jacobian'],
        b=second['jacobian']@first['parameter']+second['parameter'])
    result['A'] = result['M']-np.eye(len(theta))
    budget.emit(role+'_assembly', {k:v for k,v in result.items() if k not in ('J0','J1')})
    for key in ('R','b'): finite(result[key],theta.shape)
    for key in ('A','M'): finite(result[key],(len(theta),len(theta)))
    return result


def rank_summary(s):
    scale = max(1., float(s[0]))
    low, high = TOL['rank_low']*scale, TOL['rank_high']*scale
    ambiguous = bool(np.any((s > low) & (s <= high)))
    status = 'ambiguous_rank' if ambiguous else 'full_row_rank_resolved' if s[-1] > high else 'numerically_rank_deficient'
    return dict(status=status, low=low, high=high, small_count=int(np.sum(s <= low)),
        resolved_count=int(np.sum(s > high)), sigma_max=float(s[0]), sigma_min=float(s[-1]),
        condition=float(s[0]/s[-1]) if s[-1] > 0 and np.isfinite(s[0]/s[-1]) else None,
        isolated_soft_direction=bool(len(s)>1 and s[-2]>high and s[-1]<=TOL['soft_isolation_ratio']*s[-2]))


def decompose(matrix, budget, label):
    matrix = finite(matrix)
    budget.begin(('svds',))
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    result = dict(matrix=matrix, U=u, singular_values=s, Vh=vh)
    budget.returned_values(('svds',), label+'_svd', result)
    m, n = matrix.shape
    require(u.shape == (m,m) and s.shape == (min(m,n),) and vh.shape == (n,n), 'complete SVD shapes')
    for a in (u,s,vh): finite(a)
    require(np.all(s >= 0) and np.all(s[:-1] >= s[1:]), 'ordered singular values')
    v=vh[:m].T; norm=max(1., float(np.linalg.norm(matrix)))
    right=np.linalg.norm(matrix@v-u*s, axis=0)/norm
    left=np.linalg.norm(matrix.T@u-v*s, axis=0)/norm
    reconstruction=float(np.linalg.norm(matrix-(u*s)@vh[:m]))/norm
    orthogonal=max(float(np.linalg.norm(u.T@u-np.eye(m))), float(np.linalg.norm(vh@vh.T-np.eye(n))))
    diagnostics=dict(right_residual=right, left_residual=left, reconstruction=reconstruction,
                     orthogonality=orthogonal, rank=rank_summary(s))
    budget.emit(label+'_svd_checks', diagnostics)
    require(max(float(right.max()),float(left.max()),reconstruction,orthogonal)<=TOL['svd_residual'], 'SVD residual/orthogonality')
    return result, diagnostics


def oriented_tangent(decomposition, diagnostics, columns, secant, raw_matrix):
    status=diagnostics['rank']['status']
    if status != 'full_row_rank_resolved':
        return dict(status='abstain_'+status, tangent=None)
    v=decomposition['Vh'][-1].copy(); columns=finite(columns, v.shape)
    require(np.all(columns>0), 'positive column scales')
    raw=columns*v
    raw/=np.linalg.norm(raw)
    matrix=decomposition['matrix']
    scaled_residual=float(np.linalg.norm(matrix@v))/max(1.,float(np.linalg.norm(matrix)))
    raw_residual=float(np.linalg.norm(raw_matrix@raw))/max(1.,float(np.linalg.norm(raw_matrix)))
    if max(raw_residual,scaled_residual)>TOL['tangent_residual']:
        return dict(status='abstain_null_residual', tangent=None, raw_residual=raw_residual, scaled_residual=scaled_residual)
    orientation='unoriented_no_secant'; cosine=None
    if secant is not None:
        secant=finite(secant, v.shape)
        norm=float(np.linalg.norm(secant/columns))
        if norm==0: return dict(status='abstain_zero_secant', tangent=None)
        cosine=float(v@(secant/columns)/norm)
        if abs(cosine)<=TOL['orientation_cosine_min']:
            return dict(status='abstain_ambiguous_orientation', tangent=None, orientation_cosine=cosine)
        if cosine<0: v=-v; raw=-raw; cosine=-cosine
        orientation='toward_saved_predecessor_to_last_secant'
    return dict(status='local_null_tangent', tangent=raw, scaled_unit_tangent=v,
        orientation=orientation, secant_cosine=cosine, raw_residual=raw_residual,
        scaled_residual=scaled_residual, h_component=float(raw[-1]),
        scope='kernel of local residual derivative; branch tangent only at a qualified root')


def geometry(assembly, state_scales, secant, budget, role):
    A,b=assembly['A'],assembly['b']; n=len(b)
    d=finite(state_scales,(n,)); require(np.all(d>=1), 'fixed state scales')
    raw=np.column_stack((A,b)); columns=np.r_[d,TOL['h_scale']]
    scaled_A=(A*d[None,:])/d[:,None]
    scaled_B=(raw*columns[None,:])/d[:,None]
    budget.emit(role+'_scales',dict(row_scale=1/d,state_column_scale=d,augmented_column_scale=columns,
        rule='D^-1 A D and D^-1 [A,b] diag(D,h_scale); original tangent=column_scale*scaled tangent',
        scaled_residual=assembly['R']/d if 'R' in assembly else None))
    result={}
    for name, matrix,aug,col in [('raw_A',A,False,np.ones(n)),('raw_B',raw,True,np.ones(n+1)),
                               ('scaled_A',scaled_A,False,d),('scaled_B',scaled_B,True,columns)]:
        decomp,checks=decompose(matrix,budget,role+'_'+name)
        row=dict(rank=checks['rank'])
        if aug: row['null_tangent']=oriented_tangent(decomp,checks,col,secant,raw)
        else:
            bcoord=b if name=='raw_A' else b*TOL['h_scale']/d
            coupling=float(decomp['U'][:,-1]@bcoord)
            row.update(smallest_left_parameter_coupling=coupling,
                normalized_left_parameter_coupling=abs(coupling)/float(np.linalg.norm(bcoord)) if np.linalg.norm(bcoord)>0 else None,
                coupling_qualified=checks['rank']['isolated_soft_direction'],
                caveat='No fold claim; singular directions are coordinate dependent and may be non-isolated.')
        result[name]=row
        budget.emit(role+'_'+name+'_summary',row)
    return result


def saved_linear_check(A,R,theta,step,saved_R):
    for a in (A,R,theta,step,saved_R): finite(a)
    error=A@step+R; old_error=A@step+saved_R
    rnorm=float(np.linalg.norm(R)); threshold=TOL['nearzero_residual']*max(1.,float(np.linalg.norm(theta)))
    absolute=float(np.linalg.norm(error)); denominator=float(np.linalg.norm(A))*float(np.linalg.norm(step))+rnorm
    agreement=float(np.linalg.norm(R-saved_R))
    ratio=absolute/rnorm if rnorm>threshold else None
    backward=absolute/denominator if denominator>0 else None
    verdict='abstain_nearzero_residual' if ratio is None else 'arithmetic_resolution_limited' if agreement>TOL['material_forcing']*rnorm else 'linear_equation_resolved' if ratio<=TOL['material_forcing'] and backward<=TOL['backward_error'] else 'material_linear_equation_mismatch'
    return dict(error=error,error_using_saved_R=old_error,absolute_linear_residual=absolute,
        true_relative_forcing=ratio,nearzero_threshold=threshold,residual_norm=rnorm,
        residual_reconstruction_difference=agreement,normwise_backward_error_frobenius=backward,
        verdict=verdict,scope='No corrective step applied; inconsistency does not identify its cause.')


def carried_secants(parent, child, parent_h, child_h, phase_exchange):
    require(phase_exchange in (0,1), 'discrete phase exchange')
    require(parent.shape==child.shape and parent.ndim==2 and parent.shape[0]==2, 'full pair secant')
    output={}
    for p in (0,1):
        q=p^phase_exchange
        delta=np.r_[child[q]-parent[p], child_h-parent_h]
        output[('parent',p)]=delta;output[('child',q)]=delta
    return output


def saved_trials(iteration, trials):
    require(len(trials)==13,'exact thirteen saved trials')
    theta,R,p=iteration['state'],iteration['residual'],iteration['step']
    for a in (theta,R,p):finite(a)
    norm=float(np.max(np.abs(R))); result=[]
    for half,row in enumerate(trials):
        require(row['call']==iteration['call'] and row['iteration']==iteration['iteration'] and row['halving']==half,'saved trial sequence')
        require(exact(row['state'],theta+(2.**-half)*p),'saved trial step bytes')
        residual=finite(row['residual'],R.shape); value=float(np.max(np.abs(residual)))
        require(row['accepted']==bool(value<norm),'saved infinity-merit decision')
        result.append(dict(halving=half,alpha=2.**-half,residual_inf=value,
            ratio=value/norm if norm>0 else None,accepted=row['accepted']))
    return result
