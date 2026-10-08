"""Algebraic four-fork dose feasibility; never advances a model state."""
import hashlib
import math
import numpy as np

TOTAL_BYTES=1024**2
LAUNCHER_BYTES=256*1024
STORE_BYTES=TOTAL_BYTES-LAUNCHER_BYTES
INPUT_MANIFEST='2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f'
INPUT_REVIEW='42d901c0957221180aec5966b2c1726ea999b11b'
STATE_HASHES=(
 '55c3effe673c620fd4d6be8b8b0276dee5d6c8d5564ee8e5ea50c0bf3c35e075',
 'bb49b69dd75af049fc3bad609027b42c6d29821b0912f46e949ea2caad5adae2',
 'cf9e59e755e2626c35a235c3fd7eac0a3cee05846df3d5ad9e52f40c4d668762',
 '1cfb60ebc30e10609137b35114a7276b96621055cfb40b6deead65a4980fad8d')

def require(condition,message):
    if not condition:raise ValueError(message)

def training_data():
    """Exact existing S2 training split only; no validation/test generation."""
    streams=np.random.SeedSequence(28011).spawn(4)
    teacher=np.random.default_rng(streams[0])
    w=teacher.normal(size=(2,3))/np.sqrt(2)
    b=teacher.normal(scale=.3,size=3)
    v=teacher.normal(size=3)/np.sqrt(3)
    c=float(teacher.normal(scale=.1))
    rng=np.random.default_rng(streams[1])
    x=rng.normal(size=(24,2)).astype(np.float64)
    y=np.tanh(x@w+b)@v+c
    y=y+0.*rng.normal(size=24)
    return x,y.astype(np.float64)

def plan_array_hash(array):
    a=np.ascontiguousarray(array,dtype=np.float64)
    return hashlib.sha256(repr(a.shape).encode()+a.tobytes()).hexdigest()

def dose_prediction(gradient,rate_pair,hidden_count):
    """Treatment-minus-sham dose and total step norms, without theta +/- step."""
    g=np.asarray(gradient,dtype=np.float64)
    rates=np.asarray(rate_pair,dtype=np.float64)
    require(g.ndim==1 and type(hidden_count) is int and 0<hidden_count<len(g),'disjoint nonempty block packing')
    require(rates.shape==(2,) and np.isfinite(rates).all() and np.all(rates>0),'finite positive rates')
    out=dict(status='infeasible',reason=None,hidden_slice=[0,hidden_count],output_slice=[hidden_count,len(g)],
             uniform_fraction=.05,maximum_group_fraction=.25,fractions=None,arm_predictions=None)
    if not np.isfinite(g).all():out['reason']='nonfinite_gradient';return out
    full_rates=np.r_[np.full(hidden_count,rates[0]),np.full(len(g)-hidden_count,rates[1])]
    with np.errstate(over='ignore',invalid='ignore'):
        scaled=g*full_rates
    out['scaled_gradient']=scaled
    if not np.isfinite(scaled).all():out['reason']='nonfinite_scaled_gradient';return out
    hn=math.hypot(*map(float,scaled[:hidden_count]));on=math.hypot(*map(float,scaled[hidden_count:]));total=math.hypot(hn,on)
    out.update(hidden_norm=hn if math.isfinite(hn) else None,output_norm=on if math.isfinite(on) else None,total_norm=total if math.isfinite(total) else None)
    if not all(math.isfinite(n) and n>0 for n in (hn,on,total)):
        out['reason']='zero_or_nonfinite_block_norm';return out
    radius=.05*total
    fractions=dict(uniform=.05,hidden=radius/hn,output=radius/on)
    out.update(radius=radius if math.isfinite(radius) else None,fractions={k:v if math.isfinite(v) else None for k,v in fractions.items()})
    if not math.isfinite(radius) or radius<=0 or not all(math.isfinite(f) and f>0 for f in fractions.values()):
        out['reason']='nonfinite_or_zero_dose';return out
    arms={}
    for name,hfrac,ofrac in [('sham',0.,0.),('uniform',.05,.05),('hidden',fractions['hidden'],0.),('output',0.,fractions['output'])]:
        reduction=np.r_[np.full(hidden_count,hfrac),np.full(len(g)-hidden_count,ofrac)]
        # Pure predicted increments; no model update and no new state is formed.
        delta=reduction*scaled
        step=-(1.-reduction)*scaled
        dose=math.hypot(*map(float,delta))
        expected=0. if name=='sham' else radius
        error=abs(dose-expected)
        arms[name]=dict(reduction_fractions=[hfrac,ofrac],treatment_minus_sham_increment=delta,
                        predicted_update_increment=step,predicted_total_step_length=math.hypot(*map(float,step)),
                        dose=dose,absolute_dose_error=error,dose_tolerance=max(1e-14,1e-12*expected),
                        dose_equality_pass=bool(math.isfinite(dose) and error<=max(1e-14,1e-12*expected)))
    out['arm_predictions']=arms
    feasible=fractions['hidden']<=.25 and fractions['output']<=.25 and all(v['dose_equality_pass'] for v in arms.values())
    out.update(status='feasible' if feasible else 'infeasible',reason='fraction_cap' if max(fractions.values())>.25 else 'dose_equality' if not feasible else None)
    return out
