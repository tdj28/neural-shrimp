"""Fixed-reference numerical acquisition; pure adapters have no target I/O."""
import time
import hashlib,json
import numpy as np
from scipy.optimize import linear_sum_assignment
from .curvature_mechanism import analyze_curvature, require
from .period2_branch import PilotStop

LIMITS=dict(analytic=24,eigh=12,eig=6,svd=6,seconds=60.,store_bytes=16*1024**2)
STOP=False

def request_stop(*_):
    global STOP
    STOP=True

def finite(a,shape=None):
    a=np.asarray(a,dtype=np.float64)
    require((shape is None or a.shape==shape) and np.isfinite(a).all(),'finite array shape')
    return a

def loss_gradient_hessian(theta,x,y,model):
    """Analytic tanh prediction Jacobian plus residual Hessian; both bias groups."""
    d,w=model['d'],model['width'];bias=model['bias'];hidden=d*w+bias*w;n=hidden+w+bias
    require(model['activation']=='tanh' and type(bias) is bool,'tanh model')
    theta=finite(theta,(n,));x=finite(x);y=finite(y,(len(x),));require(x.shape[1]==d,'input dimension')
    ins,out=model['in_scale'],model['out_scale'];ridge=model['ridge'];scale=2*model['loss_scale']/len(x)
    W=theta[:d*w].reshape(d,w);v=theta[hidden:hidden+w]
    features=np.column_stack((ins*x,np.ones(len(x)))) if bias else ins*x
    pre=ins*x@W+(theta[d*w:hidden] if bias else 0)
    h=np.tanh(pre);s=1-h*h;s2=-2*h*s
    residual=out*h@v+(theta[-1] if bias else 0)-y
    first=np.zeros((len(x),n))
    first[:,:d*w]=(ins*x[:,:,None]*(out*s*v)[:,None,:]).reshape(len(x),d*w)
    if bias:first[:,d*w:hidden]=out*s*v;first[:,-1]=1.
    first[:,hidden:hidden+w]=out*h
    gradient=scale*(first.T@residual)+ridge*theta
    hessian=scale*(first.T@first)+ridge*np.eye(n)
    for unit in range(w):
        ids=np.r_[np.arange(d)*w+unit,d*w+unit] if bias else np.arange(d)*w+unit
        oi=hidden+unit
        hessian[np.ix_(ids,ids)]+=scale*out*v[unit]*(features.T@((residual*s2[:,unit])[:,None]*features))
        cross=scale*out*(features.T@(residual*s[:,unit]));hessian[ids,oi]+=cross;hessian[oi,ids]+=cross
    loss=model['loss_scale']*np.mean(residual**2)+ridge*(theta@theta)/2
    return float(loss),gradient,hessian

class Budget:
    def __init__(self,emit,seconds=60.,clock=time.monotonic):
        self.emit=emit;self.clock=clock;self.start=clock();self.seconds=seconds
        self.attempted=dict(analytic=0,eigh=0,eig=0,svd=0);self.returned=self.attempted.copy()
    def check(self):
        if STOP or self.clock()-self.start>=self.seconds:raise PilotStop('signal_or_deadline')
    def begin(self,kind,payload):
        self.check();require(self.attempted[kind]<LIMITS[kind],kind+' operation cap')
        self.attempted[kind]+=1;self.emit('operation_start',dict(kind=kind,number=self.attempted[kind],**payload))
    def returned_event(self,kind,payload):
        self.returned[kind]+=1;self.emit('operation_return',dict(kind=kind,number=self.returned[kind],**payload));self.check()

class Acquisition:
    def __init__(self,budget,slot,x,y,model,rates,derivative=loss_gradient_hessian):
        self.budget=budget;self.slot=slot;self.x=x;self.y=y;self.model=model;self.rates=finite(rates)
        require(np.all(self.rates>0),'positive rates');self.derivative=derivative;self.cache={};self.serial=0;self.context=self.context_key()
    def context_key(self):
        def array_key(value):
            if value is None:return None
            a=np.ascontiguousarray(value);return (a.dtype.str,a.shape,hashlib.sha256(a.tobytes()).hexdigest())
        return (json.dumps(self.model,sort_keys=True,separators=(',',':')),array_key(self.x),array_key(self.y),array_key(self.rates))
    def phase(self,theta):
        require(self.context_key()==self.context,'immutable model/data/rate cache context')
        theta=finite(theta,(len(self.rates),));key=(self.context,theta.dtype.str,theta.shape,theta.tobytes())
        self.budget.check()
        if key in self.cache:
            value=self.cache[key];self.budget.emit('cache',dict(slot=self.slot,id=value['id'],theta=theta));return value
        ident=self.serial;self.serial+=1
        self.budget.begin('analytic',dict(slot=self.slot,id=ident,theta=theta))
        started=time.monotonic();loss,g,H=self.derivative(theta,self.x,self.y,self.model)
        # Retain every returned derivative before finite/shape validation or map construction.
        raw=dict(slot=self.slot,id=ident,theta=theta.copy(),loss=loss,gradient=np.asarray(g),hessian=np.asarray(H),elapsed_seconds=time.monotonic()-started)
        self.budget.returned_event('analytic',raw)
        require(np.isfinite(loss),'finite loss');g=finite(g,theta.shape);H=finite(H,(len(theta),len(theta)))
        image=theta-self.rates*g;J=np.eye(len(theta))-self.rates[:,None]*H
        value={**raw,'image':image,'jacobian':J}
        self.budget.emit('phase',value);self.budget.check();finite(image,theta.shape);finite(J,H.shape)
        self.cache[key]=value;return value
    def observer(self,kind,row):
        payload=dict(slot=self.slot,record=row)
        if kind=='decomposition_start':self.budget.begin(row['kind'],payload)
        elif kind=='decomposition_return':self.budget.returned_event(row['kind'],payload)
        else:self.budget.emit(kind,payload)


def normalized(a,b):return float(np.max(np.abs(a-b)/(1e-9+1e-8*np.maximum(np.abs(a),np.abs(b)))))
def agrees(a,b):return bool(np.all(np.abs(a-b)<=1e-10+1e-10*np.abs(b)))
def match_values(new,old):
    new=np.asarray(new,dtype=complex);old=np.asarray(old,dtype=complex)
    if new.shape!=old.shape:return dict(passed=False,reason='spectrum count mismatch')
    rows,cols=linear_sum_assignment(np.abs(new[:,None]-old[None,:]));errors=np.abs(new[rows]-old[cols])
    return dict(passed=bool(np.all(errors<=1e-10+1e-10*np.abs(old[cols]))),rows=rows,cols=cols,errors=errors)


def evaluate_slot(slot,x,y,model,budget,derivative=loss_gradient_hessian,reuse=None):
    rates=slot['rate_vector'];acq=Acquisition(budget,slot['slot'],x,y,model,rates,derivative)
    if reuse is not None:
        require(reuse['theta'].tobytes()==slot['pair'][0].tobytes(),'profile exact theta')
        key=(acq.context,reuse['theta'].dtype.str,reuse['theta'].shape,reuse['theta'].tobytes());acq.cache[key]=reuse
        acq.serial=1;budget.attempted['analytic']+=1;budget.returned['analytic']+=1
        budget.emit('profile_reuse',dict(slot=slot['slot'],phase=reuse))
    p0=acq.phase(slot['pair'][0]);theta1=p0['image'] if len(slot['pair'])==1 else slot['pair'][1]
    p1=acq.phase(theta1);q0=acq.phase(p0['image']);q1=acq.phase(p1['image'])
    pair=np.stack((p0['theta'],p1['theta']));M=np.stack((q0['jacobian']@p0['jacobian'],q1['jacobian']@p1['jacobian']))
    closure=[normalized(q0['image'],pair[0]),normalized(q1['image'],pair[1])]
    transitions=[normalized(p0['image'],pair[1]),normalized(p1['image'],pair[0])]
    separation=float(np.max(np.abs(pair[0]-pair[1])))/max(1.,float(np.max(np.abs(pair))))
    qualification=dict(closure=closure,transitions=transitions,separation=separation,
        physical_passed=bool(max(closure+transitions)<=1 and separation>=1e-6),saved_matrix_agreement=None)
    if slot.get('saved_matrices') is not None:qualification['saved_matrix_agreement']=agrees(M,slot['saved_matrices'])
    budget.emit('native_assembly',dict(slot=slot['slot'],pair=pair,monodromies=M,phase_ids=[p0['id'],p1['id'],q0['id'],q1['id']],qualification=qualification))
    analysis=analyze_curvature(np.stack((p0['gradient'],p1['gradient'])),np.stack((p0['hessian'],p1['hessian'])),rates,M,acq.observer)
    budget.emit('analysis',dict(slot=slot['slot'],analysis=analysis));budget.check()
    reasons=[]
    if not qualification['physical_passed']:reasons.append('physical_period_two_gate')
    if qualification['saved_matrix_agreement'] is False:reasons.append('historical_stored_phase_product_disagreement')
    if analysis['status']!='analysis_complete':reasons.extend(analysis['reasons'])
    target=analysis.get('target_pair');spectrum=analysis.get('target_eigensystem',{}).get('values')
    history=None
    if spectrum is not None and slot.get('saved_values') is not None:
        history=match_values(spectrum,slot['saved_values'])
        if not history['passed']:reasons.append('historical_spectrum_disagreement')
    if spectrum is not None and slot.get('saved_mu') is not None:
        z=complex(*slot['saved_mu']);error=float(np.min(np.abs(spectrum-z)))
        history=dict(mu_error=error,passed=error<=1e-10+1e-10*abs(z))
        if not history['passed']:reasons.append('historical_critical_mode_disagreement')
    if target is None:reasons.append('no_target_spectrum')
    else:
        expected='real_pair' if slot['stratum']=='stable-real' else 'complex_conjugate_pair'
        if target['kind']!=expected:reasons.append('historical_class_or_complex_slot_disagreement')
        if slot['stratum'].startswith('stable') and not target['spectral_radius']<1-1e-7:reasons.append('historical_attraction_margin')
    comparisons=analysis.get('comparisons',{});rot=comparisons.get('rotating',{})
    return dict(slot=slot['slot'],qualified=not reasons,reasons=reasons,qualification=qualification,historical_spectrum=history,
        target_kind=None if target is None else target['kind'],modeled_kind=rot.get('model_pair_kind'),
        pair_error=rot.get('normalized'),angle_error=rot.get('angle_error'),modulus_error=rot.get('modulus_error'),
        commuting_error=analysis.get('oracle_commuting_error'),attribution=analysis.get('attribution',{}).get('qualified',False),
        stratum=slot['stratum'],scope='Exploratory fixed reference; native0 spectrum analyzed, native1 matrix/closure retained; phases not replicates.')


def aggregate(rows):
    all_slots=len(rows)==6 and all(r.get('qualified',False) for r in rows)
    real=[r for r in rows if r.get('stratum')=='stable-real'];complex_rows=[r for r in rows if r.get('stratum')!='stable-real']
    character=len(real)==2 and len(complex_rows)==4 and all(r.get('modeled_kind')=='real_pair' for r in real) and all(r.get('modeled_kind')==r.get('target_kind')=='complex_conjugate_pair' for r in complex_rows)
    finite_errors=all(isinstance(r.get('pair_error'),(int,float)) and np.isfinite(r['pair_error']) and isinstance(r.get('commuting_error'),(int,float)) and np.isfinite(r['commuting_error']) for r in rows)
    accuracy=finite_errors and all(r['pair_error']<=.02 for r in rows)
    angles=all(isinstance(r.get('angle_error'),(int,float)) and r['angle_error']<=.05 for r in complex_rows)
    mean=float(np.mean([r['pair_error'] for r in rows])) if finite_errors and rows else None
    baseline=float(np.mean([r['commuting_error'] for r in rows])) if finite_errors and rows else None
    gain=mean is not None and mean<=.75*baseline and baseline-mean>=.001
    attribution=len(complex_rows)==4 and all(r.get('attribution',False) for r in complex_rows)
    gates=dict(all_six_slots_qualified=all_slots,character=character,pair_accuracy=accuracy,angles=angles,mean_gain=bool(gain),complex_attribution=attribution)
    categories=[]
    if not all_slots:categories.append('unresolved_reference_or_model_qualification')
    if not attribution:categories.append('unresolved_floating_spectral_attribution')
    if all_slots and not all((character,accuracy,angles,gain)):categories.append('empirical_engineering_gate_miss')
    return dict(passed=bool(all(gates.values())),failure_categories=categories,gates=gates,mean_pair_error=mean,mean_oracle_commuting_error=baseline,
        scope='Exploratory engineering screen for this fixed rank-one/bulk approximation; failure is not nonexistence of a curvature mechanism; no training or inference claim.')
