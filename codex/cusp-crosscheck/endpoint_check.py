"""Task14 numerical endpoint reductions. No input loading or model construction.

No Newton, retuning, selected FD plateau, endpoint interpolation or cusp claim.
The fixed target wrapper must bind every callback and its original full state.
"""
import math
import time
from collections import Counter
import numpy as np
import torch
import numerics as N

# Source-prepared thresholds. The exact target card must repeat these unchanged.
TOLERANCES=dict(closure=1e-10,primitive=1e-6,phase_composition=1e-12,
    critical=1e-8,imaginary=1e-10,unit_circle_gap=1e-3,isolation=1e-3,
    eigen=1e-8,dense_jvp=1e-10,finite_difference=1e-3,
    finite_difference_floor=1e-5,finite_difference_reduction=.3,
    homological=1e-8,near_mu_coefficient=1e-3,nonzero_a=1e-10)
FD_STEPS=(1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5)
PARAMETER_STEPS=(7e-6,2e-6,7e-7)


def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    if isinstance(value,(int,float,complex,np.number)):return bool(np.isfinite(value))
    return True


def array(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().copy()
    return np.asarray(value,dtype=np.float64)


def relative(value,reference,*,floor=1e-30):
    value,reference=array(value),array(reference)
    answer=float(np.linalg.norm(value-reference)/max(floor,float(np.linalg.norm(reference))))
    if not math.isfinite(answer):raise ValueError('nonfinite relative diagnostic')
    return answer


def scalar_error(value,reference):
    """Absolute/relative projected diagnostic; left projection may amplify errors."""
    answer=abs(float(value)-float(reference))/max(abs(float(reference)),1.)
    if not math.isfinite(answer):raise ValueError('nonfinite projected diagnostic')
    return answer


def sign_resolved(reference,estimates):
    """Numerical sign agreement on the fixed finest two rows, not an enclosure."""
    values=[float(reference),*[float(x) for x in estimates[-2:]]]
    return len(estimates)>=2 and all(math.isfinite(v) and abs(v)>TOLERANCES['nonzero_a'] for v in values) and (
        all(v>0 for v in values) or all(v<0 for v in values))


class GateFailure(RuntimeError):
    def __init__(self,stage,checks):
        self.stage,self.checks=stage,checks
        super().__init__(stage+' failed: '+','.join(k for k,v in checks.items() if not v))


def gates(stage,checks,emit):
    checks={k:bool(v) for k,v in checks.items()}
    emit(stage+'-gates',checks)
    if not all(checks.values()):raise GateFailure(stage,checks)


class Operations:
    def __init__(self,model,alpha,log_temperature,period,emit,check,batch=16):
        self.model,self.alpha,self.log_temperature,self.period=model,float(alpha),float(log_temperature),int(period)
        self.emit,self.check,self.batch=emit,check,batch
        self.counts=Counter();self.timings=[];self.cache=None
    def common(self):return dict(alpha=self.alpha,log10_temperature=self.log_temperature,period=self.period)
    def call(self,name,payload,function):
        self.check();self.counts[name]+=1
        tag=f'{name}-{self.counts[name]:03d}'
        self.emit(tag+'-input',dict(payload,**self.common()));begin=time.monotonic()
        value=function()
        # Complete raw return before finiteness, time or other validation.
        self.emit(tag+'-return',dict(value=value,seconds=time.monotonic()-begin))
        self.timings.append(dict(operation=name,seconds=time.monotonic()-begin))
        if not finite(value):raise ValueError('nonfinite completed '+name)
        self.check();return value
    def prefix(self):
        self.check();self.emit('prefix-input',self.common());begin=time.monotonic()
        self.cache=self.model.prefix_cache(self.alpha)
        self.emit('prefix-return',dict(alpha=self.cache.alpha,layers=self.cache.layers,seconds=time.monotonic()-begin))
        if not finite(self.cache.layers):raise ValueError('nonfinite completed prefix')
        self.counts['prefix']+=1;self.check()
    def function(self,x):return self.model.period_map(x,self.alpha,self.log_temperature,self.period,cache=self.cache)
    def composed(self,x):return self.call('composed',dict(state=x),lambda:self.function(x))
    def step(self,x):return self.call('step',dict(state=x),lambda:self.model.forward(x,self.alpha,self.log_temperature,cache=self.cache))
    def jvp(self,x,v):return self.call('composed-jvp',dict(state=x,direction=v),lambda:self.model.jvp(x,v,self.alpha,self.log_temperature,period=self.period,cache=self.cache))
    def derivative(self,x,directions):
        return self.call('state-ad-'+str(len(directions)),dict(state=x,directions=directions),lambda:N.directional(self.function,x,directions))
    def parameter(self,x):
        scalar=torch.tensor(self.alpha,dtype=torch.float64)
        # External cache deliberately omitted: physical alpha changes the prefix.
        return self.call('alpha-ad',dict(state=x),lambda:torch.func.jvp(
            lambda a:self.model.period_map(x,a,self.log_temperature,self.period),(scalar,),(torch.ones_like(scalar),))[1])
    def parameter_map(self,x,offset):
        actual=self.alpha+offset
        return self.call('alpha-offset-map',dict(state=x,actual_alpha=actual,offset=offset),
            lambda:self.model.period_map(x,actual,self.log_temperature,self.period))
    def dense(self,x):
        n=len(x);eye=torch.eye(n,dtype=torch.float64);rows=[]
        for start in range(0,n,self.batch):
            directions=eye[start:start+self.batch];states=x[None,:].expand(len(directions),-1).clone()
            output=self.call('dense-batch',dict(state=states,directions=directions,column_start=start),
                lambda:self.model.jvp(states,directions,self.alpha,self.log_temperature,period=self.period,cache=self.cache))
            if output.shape!=directions.shape:raise ValueError('full dense JVP shape mismatch')
            rows.append(output)
        matrix=torch.cat(rows,dim=0).T.contiguous()
        self.emit('assembled-monodromy',dict(matrix=matrix,columns=n));return array(matrix)


def homological_ok(row):
    return (finite(row) and max(row['bordered_relative_residual'],row['unbordered_relative_residual'],
        abs(row['orthogonality'])/max(1.,row['h2_norm']))<=TOLERANCES['homological'])


def ladder_gate(errors):
    """All rows retained; fixed finest two and endpoint convergence, no pair selection."""
    errors=np.asarray(errors,dtype=float)
    if len(errors)<3 or not np.isfinite(errors).all():return False
    finest=bool(max(errors[-2:])<=TOLERANCES['finite_difference'])
    convergence=bool(max(errors[-2:])<=TOLERANCES['finite_difference_floor'] or
                     max(errors[-2:])<=TOLERANCES['finite_difference_reduction']*max(errors[0],1e-30))
    return finest and convergence


def check_endpoint(ops,x,*,saved_matrix=None,steps=FD_STEPS,parameter_steps=PARAMETER_STEPS):
    """Full numeric row; failure raises only after the available inputs/returns persist."""
    if x.ndim!=1 or x.dtype!=torch.float64 or x.device.type!='cpu' or not finite(x):raise ValueError('full finite float64 CPU state required')
    ops.prefix();scale=max(1.,float(torch.linalg.vector_norm(x)))
    current=x;phases=[x]
    for _ in range(ops.period):current=ops.step(current);phases.append(current)
    composed=ops.composed(x)
    closure=float(torch.linalg.vector_norm(composed-x))/scale
    phase_error=relative(current,composed)
    divisors=[d for d in range(1,ops.period) if ops.period%d==0]
    separations={str(d):float(torch.linalg.vector_norm(phases[d]-x))/scale for d in divisors}
    ops.emit('orbit-diagnostics',dict(closure=closure,composition_relative=phase_error,proper_divisor_separations=separations))
    gates('orbit',dict(closure=closure<=TOLERANCES['closure'],phase_composition=phase_error<=TOLERANCES['phase_composition'],
        primitive=all(v>TOLERANCES['primitive'] for v in separations.values())),ops.emit)
    matrix=ops.dense(x) if saved_matrix is None else array(saved_matrix)
    if matrix.shape!=(len(x),len(x)) or not finite(matrix):raise ValueError('finite full-state matrix required')
    ops.emit('monodromy-source',dict(matrix=matrix,reused=saved_matrix is not None))
    # Retain the complete numerical spectrum even when the classifier refuses it.
    eigenvalues=np.linalg.eigvals(matrix);ops.emit('raw-full-spectrum',dict(eigenvalues=eigenvalues))
    try:
        basis=N.fold_basis(matrix,critical_tolerance=TOLERANCES['critical'],imaginary_tolerance=TOLERANCES['imaginary'],
            unit_circle_gap=TOLERANCES['unit_circle_gap'],isolation=TOLERANCES['isolation'],eigen_tolerance=TOLERANCES['eigen'])
    except ValueError as exc:
        ops.emit('classification-refusal',dict(error=str(exc)))
        raise GateFailure('fold-classification',dict(real_isolated_hyperbolic_plus_one=False)) from exc
    ops.emit('classified-fold-basis',basis)
    r=torch.tensor(basis['right'],dtype=torch.float64);l=basis['left'];mu=basis['multiplier']
    probe=torch.arange(1,len(x)+1,dtype=torch.float64);probe/=torch.linalg.vector_norm(probe)
    directional={}
    for name,v in [('critical',r),('probe',probe)]:
        action=ops.jvp(x,v);error=relative(matrix@array(v),action)
        directional[name]=dict(vector=v,action=action,dense_relative_error=error)
    ops.emit('dense-directional',directional)
    gates('dense-composed', {k:v['dense_relative_error']<=TOLERANCES['dense_jvp'] for k,v in directional.items()},ops.emit)
    # Coefficients are computed only after orbit, multiplier and dense gates.
    B=ops.derivative(x,(r,r));C=ops.derivative(x,(r,r,r));A=ops.parameter(x)
    fold=N.homological(matrix,array(r),l,array(B))
    near=N.homological(matrix,array(r),l,array(B),multiplier=mu)
    ops.emit('homological-solves',dict(fold=fold,near_mu=near))
    gates('homological',dict(fold=homological_ok(fold),near_mu=homological_ok(near)),ops.emit)
    H=torch.tensor(fold['h2'],dtype=torch.float64);Hmu=torch.tensor(near['h2'],dtype=torch.float64)
    BH=ops.derivative(x,(r,H));BHmu=ops.derivative(x,(r,Hmu))
    coefficient=N.cubic(l,array(C),array(BH));near_coefficient=N.cubic(l,array(C),array(BHmu))
    a=float(l@array(A));b=fold['b']
    near_relative=abs(coefficient['c']-near_coefficient['c'])/max(abs(coefficient['c']),1.)
    raw=dict(a_physical_alpha=a,b=b,ab=a*b,b_sign_a=(b*math.copysign(1.,a) if abs(a)>TOLERANCES['nonzero_a'] else None),
        coefficient=coefficient,near_mu_coefficient=near_coefficient,near_mu_relative=near_relative,
        h2_relative_change=relative(Hmu,H),multiplier_minus_one=mu-1,
        normalization=dict(norm_right=float(torch.linalg.vector_norm(r)),left_right=float(l@array(r))),
        definition='x(u)=x*+r*u+h2*u^2/2+..., ||r||2=1,l^Tr=1; clock one full period',
        orientation=basis['orientation'])
    ops.emit('raw-ad-coefficients',raw)
    # Every requested step and every projected component is retained; AD is primary.
    fd=[];j0=directional['critical']['action'];absolute_scale=float(torch.linalg.vector_norm(x)) or 1.
    for h in steps:
        s=float(h)*absolute_scale;xp,xm=x+s*r,x-s*r
        jp,jm=ops.jvp(xp,r),ops.jvp(xm,r)
        bfd=(jp-jm)/(2*s);cfd=(jp-2*j0+jm)/(s*s)
        vectors=torch.stack((H,Hmu));hp=ops.jvp(xp[None,:].expand(2,-1).clone(),vectors);hm=ops.jvp(xm[None,:].expand(2,-1).clone(),vectors)
        mixed=(hp-hm)/(2*s)
        cparts=N.cubic(l,array(cfd),array(mixed[0]));near_parts=N.cubic(l,array(cfd),array(mixed[1]))
        map_p,map_m=ops.composed(xp),ops.composed(xm)
        record=dict(h_relative=float(h),h_absolute=s,b=.5*float(l@array(bfd)),coefficient=cparts,near_mu_coefficient=near_parts,
            errors=dict(b_projected=scalar_error(.5*float(l@array(bfd)),b),Brr=relative(bfd,B,floor=1.),Crrr=relative(cfd,C,floor=1.),Brh2=relative(mixed[0],BH,floor=1.),Brh2_near=relative(mixed[1],BHmu,floor=1.),
                critical_map_jvp=relative((map_p-map_m)/(2*s),j0,floor=1.),
                c_direct=abs(cparts['direct']-coefficient['direct'])/max(abs(coefficient['direct']),1.),
                c_center=abs(cparts['center_manifold']-coefficient['center_manifold'])/max(abs(coefficient['center_manifold']),1.),
                c_total=abs(cparts['c']-coefficient['c'])/max(abs(coefficient['c']),1.)))
        ops.emit('finite-difference-row',record);fd.append(record)
    # Independent full-state probe's first derivative has its own complete ladder.
    probe_rows=[]
    for h in steps:
        s=float(h)*absolute_scale
        plus,minus=ops.composed(x+s*probe),ops.composed(x-s*probe)
        row=dict(h_relative=float(h),h_absolute=s,error=relative((plus-minus)/(2*s),directional['probe']['action'],floor=1.))
        ops.emit('probe-finite-difference-row',row);probe_rows.append(row)
    alpha_rows=[]
    for h in parameter_steps:
        plus,minus=ops.parameter_map(x,float(h)),ops.parameter_map(x,-float(h));estimate=(plus-minus)/(2*float(h))
        row=dict(h_absolute=float(h),a_physical_alpha=float(l@array(estimate)),error=relative(estimate,A,floor=1.),
            projected_error=scalar_error(float(l@array(estimate)),a))
        ops.emit('alpha-finite-difference-row',row);alpha_rows.append(row)
    checks={k:ladder_gate([row['errors'][k] for row in fd]) for k in fd[0]['errors']}
    checks.update(physical_alpha_projected=ladder_gate([v['projected_error'] for v in alpha_rows]),probe_jvp=ladder_gate([v['error'] for v in probe_rows]),physical_alpha=ladder_gate([v['error'] for v in alpha_rows]),
        near_mu=math.isfinite(near_relative) and near_relative<=TOLERANCES['near_mu_coefficient'])
    gates('coefficient-precision',checks,ops.emit)
    signs=dict(a=sign_resolved(a,[v['a_physical_alpha'] for v in alpha_rows]),
        b=sign_resolved(b,[v['b'] for v in fd]),c=sign_resolved(coefficient['c'],[v['coefficient']['c'] for v in fd]))
    ops.emit('numerical-sign-resolution',dict(signs=signs,definition='AD and fixed finest two FD rows; no rigorous enclosure'))
    return dict(status='numerical_endpoint_qualified',numerical_qualified=True,certificate_passed=False,cusp_qualified=False,
        closure=closure,proper_divisor_separations=separations,critical_multiplier=mu,
        next_modulus=basis['next_modulus'],other_unit_circle_distance=basis['nearest_other_unit_circle'],
        complement_attracting=basis['complement_attracting'],orientation_bracket_ready=signs['a'] and signs['b'],cubic_nonzero_supported=signs['c'],
        numerical_sign_resolution=signs,coefficients=raw,fd_rows=fd,probe_fd_rows=probe_rows,
        alpha_fd_rows=alpha_rows,precision_gates=checks,operation_counts=dict(ops.counts),timings=ops.timings)
