#!/usr/bin/env python3
"""Independent EXP014 saved-evidence audit. No producer imports or ODE calls.

Calculation and tolerance choices were written before EXP014 outcomes.
This backed-up snapshot includes the recorded post-outcome JSON scalar
serialization fix; original failed output and source remain preserved.
Original scientific thresholds are unchanged.
Independent arithmetic comparison: rtol1e-10/atol1e-12 for scalar readouts;
rtol1e-10/atol1e-11 for independently interpolated arrays. Categories,
identities, counts, hashes and shape/dtype are exact. CubicHermiteSpline and
math.erf provide implementations independent of the producer's Hermite/CDF.
Raw evidence is read-only; the requested report must be a new external file.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

SETTINGS={'core':(1.064,10.260722222222222),'failed-near-edge':(1.0661333333333334,10.297166666666667)}
METHODS=('rk4-0.0025','rk4-0.00125','dop853')
DIRECTIONS=('phase_write','phase_null','neuron_one')
DOSES=(1e-4,1e-3,1e-2);DELAYS=(1,2,4,8);NOISES=(1e-5,1e-4,1e-3)
TIMES=np.arange(12801,dtype=np.float64)*.025;TIMES[-1]=320.


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def norm(v):return float(np.linalg.norm(v))


class Audit:
    def __init__(self):self.checks=0;self.failures=[];self.max_scalar_error=0.;self.max_array_error=0.
    def check(self,passed,label,details=None):
        self.checks+=1
        if not bool(passed):self.failures.append(dict(check=label,details=details))
        return bool(passed)
    def near(self,a,b,label,*,array=False):
        a,b=np.asarray(a),np.asarray(b)
        finite=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        error=float(np.max(np.abs(a-b))) if finite and a.size else 0. if finite else None
        if error is not None:
            if array:self.max_array_error=max(self.max_array_error,error)
            else:self.max_scalar_error=max(self.max_scalar_error,error)
        return self.check(finite and np.allclose(a,b,rtol=1e-10,atol=1e-11 if array else 1e-12),label,dict(maximum_absolute_difference=error))
    def exact(self,a,b,label):return self.check(type(a) is type(b) and a==b,label)


def decode_packet(path,audit):
    """Independent inverse of the declared numeric JSON/NPZ storage contract."""
    path=Path(path);metadata=read(path);entry=metadata['arrays'];raw=path.parent/entry['path']
    if Path(entry['path']).name!=entry['path'] or not raw.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Packet path escape')
    audit.check(metadata['schema']=='numeric-packet-v1','packet schema:'+str(path))
    audit.check(raw.stat().st_size==entry['bytes'] and sha(raw)==entry['sha256'],'packet bytes:'+str(path))
    with np.load(raw,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    used=set()
    def walk(v):
        if isinstance(v,list):return [walk(x) for x in v]
        if isinstance(v,dict):
            if '__array__' in v:
                key=v['__array__'];audit.check(set(v)=={'__array__','dtype','shape'} and key not in used,'unique array marker')
                used.add(key);array=arrays[key]
                audit.check(array.dtype.str==v['dtype'] and list(array.shape)==v['shape'] and not array.dtype.hasobject,'array dtype/shape')
                return array
            if '__nonfinite__' in v:
                audit.check(set(v)=={'__nonfinite__'} and v['__nonfinite__'] in ('nan','inf','-inf'),'nonfinite marker')
                return float(v['__nonfinite__'])
            return {k:walk(x) for k,x in v.items()}
        return v
    value=walk(metadata['tree']);audit.check(used==set(arrays),'NPZ complete member census')
    digest=hashlib.sha256(canonical(dict(tree=metadata['tree'],arrays={k:hashlib.sha256(v.tobytes(order='C')).hexdigest() for k,v in arrays.items()}))).hexdigest()
    audit.check(digest==metadata['semantic_sha256'],'packet semantic hash')
    return value


def tasks():
    result=[]
    for setting in SETTINGS:
        for phase in range(8):
            for dose,epsilon in enumerate(DOSES):
                for direction in DIRECTIONS:
                    for sign in (-1,1):
                        for method in METHODS:
                            result.append(dict(id=f'{setting}--p{phase}--d{dose}--{direction}--s{sign:+d}--{method}',setting=setting,
                                phase_index=phase,kind='pulse',dose_index=dose,epsilon=epsilon,direction=direction,sign=sign,method=method))
            for method in METHODS:result.append(dict(id=f'{setting}--p{phase}--sham--{method}',setting=setting,phase_index=phase,
                kind='sham',dose_index=None,epsilon=None,direction=None,sign=None,method=method))
    return result


def rhs(states,alpha,gamma):
    matrix=np.array([[1.5,2.9,alpha],[-1.5,1.18,0.],[gamma,-22.,.47]])
    return -states+np.tanh(states)@matrix.T


def jac(states,alpha,gamma):
    matrix=np.array([[1.5,2.9,alpha],[-1.5,1.18,0.],[gamma,-22.,.47]])
    return -np.eye(3)+matrix*(1.-np.tanh(states)**2)[...,None,:]


def spline(times,states,derivatives,*,period=None):
    if period is not None:
        base=CubicHermiteSpline(np.r_[times[:-1],period],np.concatenate((states[:-1],states[:1])),
            np.concatenate((derivatives[:-1],derivatives[:1])),axis=0,extrapolate=False)
        return lambda q,derivative=0:base(np.remainder(q,period),nu=derivative)
    base=CubicHermiteSpline(times,states,derivatives,axis=0,extrapolate=False)
    return lambda q,derivative=0:base(q,nu=derivative)


def clock(template,period,epoch,delay):
    times=period*(delay-1+(np.arange(32)+.5)/32)
    states=template(times+epoch);activations=np.tanh(states)
    gradient=(1.-activations**2)*template(times+epoch,1)
    energy=float(np.sum(gradient*gradient))
    if not np.isfinite(energy) or energy<=0:raise ValueError('Invalid clock energy')
    weights=gradient/energy
    return dict(times=times,reference=activations,weights=weights,gain=norm(weights),energy=energy)


def accuracy(qplus,qminus,gain,orientation,sigma):
    if gain<=0 or sigma<=0 or orientation not in (-1,1):raise ValueError('Invalid observation model')
    a=orientation*qplus/(sigma*gain);b=-orientation*qminus/(sigma*gain)
    return ((.5+.5*math.erf(a/math.sqrt(2)))+(.5+.5*math.erf(b/math.sqrt(2))))/2


def direction_table(z):
    write=z/norm(z);projector=np.eye(3)-np.outer(z,z)/(z@z)
    coordinate=int(np.argmax(np.sum(projector*projector,axis=0)))
    null=projector[:,coordinate]/norm(projector[:,coordinate])
    return dict(phase_write=write,phase_null=null,neuron_one=np.array([1.,0.,0.])),coordinate


def variational_audit(packet,alpha,gamma,period,initial,audit,prefix):
    times=packet['times'];states=packet['states'];phi=packet['phi']
    audit.check(times.dtype==np.float64 and np.array_equal(times,np.linspace(0.,period,4097)),prefix+' grid')
    audit.check(states.dtype==phi.dtype==np.float64 and states.shape==(4097,3) and phi.shape==(4097,3,3) and np.isfinite(states).all() and np.isfinite(phi).all(),prefix+' finite shapes')
    audit.check(np.array_equal(states[0],initial) and np.array_equal(phi[0],np.eye(3)) and np.array_equal(phi[-1],packet['monodromy']),prefix+' endpoint identity')
    flow=rhs(states,alpha,gamma);dphi=np.einsum('nij,njk->nik',jac(states,alpha,gamma),phi)
    audit.near(packet['derivatives'],flow,prefix+' saved RHS',array=True)
    audit.near(packet['phi_derivatives'],dphi,prefix+' saved tangent derivative',array=True)
    closure=norm(states[-1]-initial)/max(1.,norm(initial))
    neutral=norm(phi[-1]@flow[0]-flow[0])/norm(flow[0])
    transported=norm(phi[-1]@flow[0]-flow[-1])/norm(flow[0])
    for key,value in [('relative_closure',closure),('neutral_flow_relative_residual',neutral),('transported_flow_relative_residual',transported)]:audit.near(packet[key],value,prefix+' '+key)
    return dict(closure=closure,neutral_flow=neutral,transported_flow=transported)



def certificate_audit(certificate,alpha,gamma,audit):
    """Replay the original gate arithmetic from its retained numerical evidence.

    Root finding and event integration are not repeated; saved interior-return
    distances are the original certificate's evidence for the primitive gate.
    """
    state=np.asarray(certificate['section_state']);orbit=np.asarray(certificate['orbit_states'])
    matrix=np.asarray(certificate['monodromy']);flow=rhs(state[None,:],alpha,gamma)[0]
    eigenvalues,eigenvectors=np.linalg.eig(matrix);index=int(np.argmin(np.abs(eigenvalues-1.)))
    neutral_error=float(abs(eigenvalues[index]-1));flow_residual=norm(matrix@flow-flow)/norm(flow)
    alignment=float(abs(np.vdot(eigenvectors[:,index],flow))/(norm(eigenvectors[:,index])*norm(flow)))
    radius=float(np.max(np.abs(np.delete(eigenvalues,index))))
    closure=norm(orbit[-1]-state)/max(1.,norm(state));curvature=float((jac(state[None,:],alpha,gamma)[0]@flow)[2])
    distances=np.asarray(certificate['interior_maximum_return_distances']);peak_count=1+len(certificate['interior_maximum_times'])
    gates=dict(closure=closure<=1e-8,phase_condition=abs(flow[2])<=1e-8,section_is_maximum=curvature < -1e-8,
        nonstationary=norm(flow)>1e-7 and float(np.max(np.ptp(orbit,axis=0)))>1e-6,
        expected_peak_count=peak_count==9,primitive=bool(np.all(distances>1e-7)),
        neutral_mode=neutral_error<=1e-6 and flow_residual<=1e-6 and alignment>=.99,
        transverse_stability=radius<1.-1e-5)
    audit.check(certificate['gates']==gates,'original certificate gate arithmetic')
    for key,value in [('relative_closure',closure),('neutral_multiplier_error',neutral_error),('neutral_flow_residual',flow_residual),
        ('neutral_flow_alignment',alignment),('transverse_spectral_radius',radius),('section_curvature',curvature)]:
        audit.near(certificate[key],value,'original certificate '+key)
    audit.check(certificate['peak_count']==peak_count,'original certificate full-state peak count')
    audit.check(certificate['thresholds']==dict(relative_closure=1e-8,neutral_error=1e-6,neutral_flow_residual=1e-6,
        neutral_flow_alignment_min=.99,transverse_stability_margin=1e-5),'original unchanged certificate thresholds')
    return all(gates.values())


def reference_audit(reference,audit):
    setting=reference['setting'];alpha,gamma=SETTINGS[setting]
    audit.check((reference['alpha'],reference['gamma'])==(alpha,gamma),setting+' reference parameters')
    if reference.get('qualified') is not True:
        return dict(qualified=False,reason=reference.get('error'),failed_stage=reference.get('failed_stage'))
    certificate=reference['certificate'];period=reference['period'];orbit=reference['orbit'];adjoint=reference['adjoint']
    audit.check(certificate['verified'] is True and certificate['attracting'] is True and certificate['peak_count']==9 and certificate['primitive'] is True and all(certificate['gates'].values()),setting+' original certificate all gates')
    audit.check(0<period<=40 and period==certificate['period'],setting+' fixed horizon')
    original_pass=certificate_audit(certificate,alpha,gamma,audit)
    residual=variational_audit(orbit,alpha,gamma,period,np.asarray(certificate['section_state']),audit,setting+' fine')
    flow=rhs(orbit['states'],alpha,gamma);z=adjoint['values'];matrix=orbit['monodromy']
    _,singular,vh=np.linalg.svd(matrix.T-np.eye(3));unit=vh[-1];normal=unit/(unit@flow[0])
    audit.near(z[-1],normal,setting+' neutral adjoint terminal',array=True)
    normalization=np.einsum('ij,ij->i',z,flow)-1
    periodic=norm(z[0]-z[-1])/max(norm(z[0]),norm(z[-1]))
    dz=-np.einsum('nji,nj->ni',jac(orbit['states'],alpha,gamma),z)
    audit.near(adjoint['derivatives'],dz,setting+' adjoint derivative',array=True)
    audit.near(adjoint['normalization_residuals'],normalization,setting+' all normalization residuals',array=True)
    audit.near(adjoint['relative_periodic_residual'],periodic,setting+' adjoint periodicity')
    left=norm(matrix.T@normal-normal)/norm(normal);cosine=abs(unit@flow[0])/norm(flow[0])
    gates=dict(original_certificate=original_pass,closure=residual['closure']<=1e-8,neutral_flow=residual['neutral_flow']<=1e-7,neutral_left=left<=1e-7,
        singular_unique=singular[-2]>1e-12*max(1.,singular[0]),conditioning=cosine>=1e-12,
        initial_normalization=abs(normal@flow[0]-1)<=1e-12,adjoint_periodic=periodic<=1e-7,
        normalization=float(np.max(np.abs(normalization)))<=1e-7)
    audit.check(all(gates.values()),setting+' independent reference qualification',gates)
    phase_indices=[phase['phase_index'] for phase in reference['phases']]
    phase_identity=all(type(j) is int for j in phase_indices) and phase_indices==list(range(8))
    audit.check(phase_identity,setting+' exact ordered phase-index census')
    for phase in reference['phases']:
        j=phase['phase_index']
        if type(j) is not int or not 0<=j<8:
            continue
        index=512*j;directions,coordinate=direction_table(z[index])
        gradient_bound=np.array_equal(phase['phase_gradient'],z[index])
        flow_bound=np.allclose(phase['flow'],flow[index],rtol=1e-10,atol=1e-11)
        phase_identity=phase_identity and gradient_bound and flow_bound
        audit.check(gradient_bound,setting+f' phase{j} exact adjoint binding')
        audit.near(phase['flow'],flow[index],setting+f' phase{j} independent RHS binding',array=True)
        audit.check(phase['phase_epoch']==orbit['times'][index] and np.array_equal(phase['initial'],orbit['states'][index]),setting+f' phase{j} index')
        audit.check(phase['null_coordinate_index']==coordinate,setting+f' phase{j} axis choice')
        for name,d in directions.items():
            audit.near(phase['directions'][name],d,setting+f' phase{j} {name}',array=True)
            audit.check(abs(norm(d)-1)<=1e-12,setting+f' phase{j} unit {name}')
        audit.check(abs(z[index]@directions['phase_null'])/norm(z[index])<=1e-12,setting+f' phase{j} null residual')
    return dict(qualified=all(gates.values()) and phase_identity,gates=gates,period=period,closure=residual['closure'],adjoint_periodic=periodic,
        normalization_maximum=float(np.max(np.abs(normalization))),neutral_flow=residual['neutral_flow'],neutral_left=left)


def score(task,trace,reference,propagator,prepared,audit):
    name=task['id'];alpha,gamma=SETTINGS[task['setting']];period=reference['period'];j=task['phase_index'];phase=reference['phases'][j]
    audit.check(canonical(trace['task'])==canonical(task),name+' exact task')
    audit.check(trace['complete'] is True and trace['status']=='complete',name+' complete')
    audit.check(trace['states'].dtype==np.float64 and trace['states'].shape==(12801,3) and np.isfinite(trace['states']).all() and np.array_equal(trace['times'],TIMES),name+' grid/finite')
    audit.check((trace['alpha'],trace['gamma'])==(alpha,gamma),name+' parameters')
    z=phase['phase_gradient'];base=phase['initial'];directions,_=direction_table(z)
    requested=np.zeros(3) if task['kind']=='sham' else task['sign']*task['epsilon']*directions[task['direction']]
    initial=base+requested;realized=initial-base
    # Arithmetic-equivalent null projector can differ by a few ulps; the actual
    # sealed direction controls the exact addition identity, independently unit/null checked above.
    exact_requested=np.zeros(3) if task['kind']=='sham' else task['sign']*task['epsilon']*phase['directions'][task['direction']]
    audit.near(trace['requested_delta'],requested,name+' requested pulse',array=True)
    audit.check(np.array_equal(trace['initial'],base+exact_requested) and np.array_equal(trace['states'][0],trace['initial']),name+' exact initial addition')
    exact_realized=trace['initial']-base
    audit.check(np.array_equal(trace['realized_delta'],exact_realized) and np.array_equal(trace['addition_rounding_residual'],exact_realized-exact_requested),name+' exact realization')
    audit.near(trace['realized_norm'],norm(exact_realized),name+' pulse norm')
    audit.check(np.array_equal(trace['initial_activation_displacement'],np.tanh(trace['initial'])-np.tanh(base)),name+' activation displacement')
    requested,realized=exact_requested,exact_realized
    orientation=1 if task['kind']=='sham' or task['direction']=='phase_null' else (-1 if float(z@directions[task['direction']])<0 else 1)
    orbit=reference['orbit'];template=spline(orbit['times'],orbit['states'],orbit['derivatives'],period=period)
    trajectory=spline(TIMES,trace['states'],rhs(trace['states'],alpha,gamma))
    propagator_spline=spline(propagator['times'],propagator['phi'],propagator['phi_derivatives'])
    epoch=phase['phase_epoch'];phase_shift=float(z@requested);case=prepared['cases'][name.rsplit('--',1)[0]]
    result=dict(orientation=orientation,readouts={},realized_norm=norm(realized))
    for delay in DELAYS:
        c=clock(template,period,epoch,delay);t=c['times'];actual=np.tanh(trajectory(t));q=float(np.sum(c['weights']*(actual-c['reference'])))
        null=np.tanh(template(t+epoch+phase_shift));qnull=float(np.sum(c['weights']*(null-c['reference'])))
        cycles=np.floor(t/period).astype(int);remainder=t-cycles*period
        linear=np.asarray([propagator_spline(s)@np.linalg.matrix_power(propagator['monodromy'],int(k))@realized for k,s in zip(cycles,remainder,strict=True)])
        activation_linear=(1-c['reference']**2)*linear;qlinear=float(np.sum(c['weights']*activation_linear))
        context=prepared['contexts'][f'{task["setting"]}--p{j}']['readouts'][str(delay)]
        audit.near(context['weights'],c['weights'],name+f' weights {delay}',array=True)
        audit.near(context['derivative_energy'],c['energy'],name+f' energy {delay}')
        audit.near(case['readouts'][str(delay)]['phase_only_q'],qnull,name+f' sealed null q {delay}')
        audit.near(case['readouts'][str(delay)]['full_linear_q'],qlinear,name+f' sealed linear q {delay}')
        result['readouts'][delay]=dict(q=q,phase_only_q=qnull,full_linear_q=qlinear,gain=c['gain'])
    return result



def prepared_predictions(prepared,references,props,audit):
    """Rebuild every reference-only null independently, including unobserved cases."""
    predictions={};cache={}
    for task in tasks():
        if task['method']!='rk4-0.00125':continue
        caseid=task['id'].rsplit('--',1)[0];setting=task['setting'];j=task['phase_index'];context=f'{setting}--p{j}'
        saved=prepared['cases'][caseid]
        if saved['status']!='complete':continue
        ref=references[setting];phase=ref['phases'][j];period=ref['period'];epoch=phase['phase_epoch']
        if context not in cache:
            orbit=ref['orbit'];template=spline(orbit['times'],orbit['states'],orbit['derivatives'],period=period)
            prop=props[context];psi=spline(prop['times'],prop['phi'],prop['phi_derivatives'])
            cache[context]=(template,psi,prop['monodromy'])
        template,psi,matrix=cache[context]
        delta=np.zeros(3) if task['kind']=='sham' else task['sign']*task['epsilon']*phase['directions'][task['direction']]
        realized=(phase['initial']+delta)-phase['initial'];shift=float(phase['phase_gradient']@delta)
        orientation=1 if task['kind']=='sham' or task['direction']=='phase_null' else (-1 if phase['phase_gradient']@phase['directions'][task['direction']]<0 else 1)
        result=dict(orientation=orientation,readouts={})
        for delay in DELAYS:
            c=clock(template,period,epoch,delay);t=c['times'];null_states=template(t+epoch+shift);null=np.tanh(null_states)
            q=float(np.sum(c['weights']*(null-c['reference'])))
            cycles=np.floor(t/period).astype(int);remainder=t-cycles*period
            linear=np.asarray([psi(u)@np.linalg.matrix_power(matrix,int(k))@realized for k,u in zip(cycles,remainder,strict=True)])
            linear_activation=(1-c['reference']**2)*linear;qlinear=float(np.sum(c['weights']*linear_activation))
            row=saved['readouts'][str(delay)]
            for key,a,b in [('phase_only_states',row['phase_only_states'],null_states),('phase_only_activations',row['phase_only_activations'],null),
                            ('full_linear_state_delta',row['full_linear_state_delta'],linear),('full_linear_activation_delta',row['full_linear_activation_delta'],linear_activation)]:
                audit.near(a,b,caseid+' reference-only '+key,array=True)
            audit.near(row['phase_only_q'],q,caseid+' all-null q')
            audit.near(row['full_linear_q'],qlinear,caseid+' all-linear q')
            audit.near(row['noise_gain'],c['gain'],caseid+' all-null gain')
            result['readouts'][delay]=dict(q=q,linear=qlinear,gain=c['gain'])
        predictions[caseid]=result
    saved_cells={(r['setting'],r['phase_index'],r['dose_index'],r['direction'],r['delay_cycles'],r['sigma']):r for r in prepared['paired_nulls']}
    count=0
    for task in tasks():
        if task['kind']!='pulse' or task['sign']!=-1 or task['method']!='rk4-0.00125':continue
        minus=task['id'].rsplit('--',1)[0];plus=minus.replace('--s-1','--s+1')
        if minus not in predictions or plus not in predictions:continue
        for delay in DELAYS:
            a=predictions[plus]['readouts'][delay];b=predictions[minus]['readouts'][delay]
            for sigma in NOISES:
                key=(task['setting'],task['phase_index'],task['dose_index'],task['direction'],delay,sigma)
                cell=saved_cells[key];audit.check(cell['status']=='complete','prepared null paired status')
                for metric,column in [('q','phase_only_balanced_accuracy'),('linear','full_linear_balanced_accuracy')]:
                    audit.near(cell[column],accuracy(a[metric],b[metric],a['gain'],predictions[plus]['orientation'],sigma),'all1728 prepared null BA')
                count+=1
    return dict(independently_rebuilt_cases=len(predictions),independently_rebuilt_null_ba_cells=count)


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def artifact_audit(raw,repo,audit):
    receipt=read(raw/'receipt.json');manifest=read(raw/'manifest.json');plan=manifest['plan']
    inventory=[dict(path=str(p.relative_to(raw)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(raw.rglob('*')) if p.is_file() and p.name!='receipt.json']
    audit.check(inventory==receipt['artifacts'],'complete receipt inventory')
    audit.check(sum(x['bytes'] for x in inventory)==receipt['artifact_bytes'],'receipt byte count')
    audit.check(receipt['source_commit']==manifest['source_commit'] and receipt['plan_sha256']==manifest['plan_sha256'],'receipt/manifest binding')
    source=receipt['source_commit']
    for entry in plan['source_files']:
        data=subprocess.check_output(['git','-C',str(repo),'show',f'{source}:{entry["path"]}'])
        audit.check(len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'],'frozen Git source:'+entry['path'])
    plan_name='EXP-014-recovery-plan.json' if any(r['path']=='tools/run_phase_memory_recovery.py' for r in plan['source_files']) else 'EXP-014-plan.json'
    committed=subprocess.check_output(['git','-C',str(repo),'show',f'{source}:experiments/hopfield/{plan_name}'])
    audit.check(hashlib.sha256(committed).hexdigest()==receipt['plan_sha256'] and json.loads(committed)==plan,'committed plan')
    for entry in plan['historical_inputs']:
        p=repo/entry['path'];audit.check(p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],'historical input:'+entry['path'])
    audit.check(canonical(plan['tasks'])==canonical(tasks()),'independent ordered912census')
    return receipt,manifest,plan,dict(artifacts=len(inventory),bytes=sum(x['bytes'] for x in inventory),sources=len(plan['source_files']),historical_inputs=len(plan['historical_inputs']))


def run(raw,repo,audit):
    raw=Path(raw).resolve();repo=Path(repo).resolve();receipt,manifest,plan,inventory=artifact_audit(raw,repo,audit)
    terminal=read(raw/'all-terminal-records.json');summary=read(raw/'summary.json');analysis=decode_packet(raw/'analysis.json',audit)
    prepared=decode_packet(raw/'reference-seal.json',audit) if (raw/'reference-seal.json').exists() else None
    expected=dict(references=list(SETTINGS),propagators=[f'{s}--p{p}' for s in SETTINGS for p in range(8)],traces=[t['id'] for t in tasks()])
    packets={};cleanup=[];live=[];status_counts={}
    for stage,ids in expected.items():
        rows=terminal[stage];audit.check([r['id'] for r in rows]==ids,'full terminal census:'+stage)
        status_counts[stage]=dict(Counter(r['status'] for r in rows));audit.check(summary['status_counts'][stage]==status_counts[stage],'summary terminal counts:'+stage)
        packets[stage]={};byid={r['id']:r for r in rows}
        if (raw/stage).exists():
            cleanup_record=read(raw/stage/'cleanup.json');path=raw/stage/'processes.jsonl'
            processes=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            audit.check(cleanup_record['all_registered_workers_joined'] is True and cleanup_record['all_original_identities_absent'] is True,stage+' cleanup passed')
            audit.check(len(processes)==len(cleanup_record['workers']),stage+' cleanup census')
            for owner,closed in zip(processes,cleanup_record['workers'],strict=True):
                audit.check(all(closed[k]==v for k,v in owner.items()) and closed['original_identity_absent'] is True,stage+' historical worker absence')
                now=process_identity(owner['pid']);absent=now is None or now!=owner['identity']
                cleanup.append(dict(stage=stage,task_id=owner['task_id'],pid=owner['pid'],original_identity_absent_now=absent))
                if not absent:live.append(owner)
        for row in rows:
            if row['status']!='complete':continue
            record=read(raw/stage/'records'/f'{row["id"]}.json');audit.check(record==row,stage+' exact worker record:'+row['id'])
            p=raw/stage/row['packet_path'];audit.check(p.resolve().is_relative_to((raw/stage).resolve()) and sha(p)==row['packet_sha256'],stage+' packet binding')
            packet=decode_packet(p,audit);packets[stage][row['id']]=packet
    audit.check(sum(map(len,terminal.values()))==930,'all930terminal rows')
    references=packets['references'];props=packets['propagators'];traces=packets['traces'];reference_checks={};prop_checks={}
    for setting in SETTINGS:
        reference_checks[setting]=reference_audit(references[setting],audit) if setting in references else dict(qualified=False,reason='missing_or_failed_reference')
        for j in range(8):
            key=f'{setting}--p{j}'
            if key in props and reference_checks[setting]['qualified']:
                r=references[setting];p=props[key];initial=r['phases'][j]['initial']
                audit.check(p['setting']==setting and type(p['phase_index']) is int and p['phase_index']==j and p['phase_epoch']==r['phases'][j]['phase_epoch'],key+' identity')
                v=variational_audit(p,*SETTINGS[setting],r['period'],initial,audit,key)
                prop_checks[key]=dict(qualified=v['closure']<=1e-8,**v)
            else:prop_checks[key]=dict(qualified=False)
    if prepared is not None:
        audit.check(prepared['tasks']==tasks() and len(prepared['contexts'])==16 and len(prepared['cases'])==304 and len(prepared['paired_nulls'])==1728,'prepared complete census')
    prepared_counts=prepared_predictions(prepared,references,props,audit) if prepared is not None else dict(independently_rebuilt_cases=0,independently_rebuilt_null_ba_cells=0)
    scored={};saved_scores={r['task']['id']:r for r in analysis['scored_records']};sham_gates={};pulse_norms=[];null_differences=[]
    for task in tasks():
        name=task['id'];setting=task['setting'];key=f'{setting}--p{task["phase_index"]}'
        if name not in traces:continue
        try:
            r=score(task,traces[name],references[setting],props[key],prepared,audit);scored[name]=r
            saved=saved_scores[name];audit.check(saved['status']=='complete' and saved['orientation']==r['orientation'],name+' scored status/orientation')
            for delay,values in r['readouts'].items():
                for metric in ('q','phase_only_q','full_linear_q'):audit.near(saved['readouts'][str(delay)][metric],values[metric],name+f' {metric}@{delay}')
                audit.near(saved['readouts'][str(delay)]['noise_gain'],values['gain'],name+f' noise gain@{delay}')
                if task['kind']=='sham':sham_gates[(name,delay)]=abs(values['q']/(1e-4*values['gain']))<=.01
                else:null_differences.append(dict(id=name,delay=delay,standardized_actual_minus_phase_only=(values['q']-values['phase_only_q'])/(1e-4*values['gain'])))
            if task['kind']=='pulse':pulse_norms.append(dict(id=name,requested=task['epsilon'],realized=r['realized_norm'],absolute_error=abs(r['realized_norm']-task['epsilon'])))
        except (KeyError,ValueError,TypeError,IndexError) as error:audit.check(False,name+' independent scoring exception',repr(error))
    comparison_gates={};comparison_details=[];saved_comparisons={tuple(r['ids']):r for r in analysis['trace_comparisons']}
    for task in tasks():
        if task['method']=='dop853':continue
        peer=task['id'].rsplit('--',1)[0]+'--'+METHODS[METHODS.index(task['method'])+1];pair=(task['id'],peer)
        passed=False
        if all(i in scored for i in pair):
            state=float(np.max(np.abs(traces[pair[0]]['states']-traces[pair[1]]['states'])))
            activation=float(np.max(np.abs(np.tanh(traces[pair[0]]['states'])-np.tanh(traces[pair[1]]['states']))))
            standardized={k:abs(scored[pair[0]]['readouts'][k]['q']-scored[pair[1]]['readouts'][k]['q'])/(1e-4*scored[pair[0]]['readouts'][k]['gain']) for k in DELAYS}
            passed=state<=1e-6 and activation<=1e-7 and max(standardized.values())<=.01
            row=saved_comparisons[pair];audit.near(row['state_difference']['maximum_absolute'],state,'comparison state max')
            audit.near(row['activation_difference']['maximum_absolute'],activation,'comparison activation max')
            for k,v in standardized.items():audit.near(row['readouts'][str(k)]['standardized_mean_difference'],v,'comparison standardized mean')
            audit.check(row['qualified']==passed,'comparison qualification')
            comparison_details.append(dict(ids=list(pair),setting=task['setting'],state_max=state,activation_max=activation,standardized_max=max(standardized.values()),qualified=passed))
        comparison_gates[pair]=passed
    saved_cells={(r['setting'],r['phase_index'],r['dose_index'],r['direction'],r['method'],r['delay_cycles'],r['sigma']):r for r in analysis['summary']['paired_readouts']}
    cells={};ba_gates={};ba_differences=[]
    for task in tasks():
        if task['kind']!='pulse' or task['sign']!=-1:continue
        minus=task['id'];plus=minus.replace('--s-1--','--s+1--')
        if minus not in scored or plus not in scored:continue
        for delay in DELAYS:
            a=scored[plus]['readouts'][delay];b=scored[minus]['readouts'][delay]
            for sigma in NOISES:
                key=(task['setting'],task['phase_index'],task['dose_index'],task['direction'],task['method'],delay,sigma)
                values={metric:accuracy(a[metric],b[metric],a['gain'],scored[plus]['orientation'],sigma) for metric in ('q','phase_only_q','full_linear_q')}
                cells[key]=values;saved=saved_cells[key]
                audit.check(saved['status']=='complete','BA cell status')
                for metric,column in [('q','balanced_accuracy'),('phase_only_q','phase_only_balanced_accuracy'),('full_linear_q','full_linear_balanced_accuracy')]:audit.near(saved[column],values[metric],'independent Gaussian BA')
                audit.check(saved['zero_balanced_accuracy']==.5,'zero-response BA')
    saved_odd_even={(r['setting'],r['phase_index'],r['dose_index'],r['direction'],r['method'],r['delay_cycles']):r for r in analysis['summary']['signed_response_readouts']}
    for task in tasks():
        if task['kind']!='pulse' or task['sign']!=-1:continue
        minus=task['id'];plus=minus.replace('--s-1--','--s+1--')
        if minus not in scored or plus not in scored:continue
        for delay in DELAYS:
            key=(task['setting'],task['phase_index'],task['dose_index'],task['direction'],task['method'],delay)
            saved=saved_odd_even[key];a=scored[plus]['readouts'][delay];b=scored[minus]['readouts'][delay]
            audit.check(saved['status']=='complete','odd/even readout status')
            for metric in ('q','phase_only_q','full_linear_q'):
                audit.near(saved[metric+'_odd'],(a[metric]-b[metric])/2,'independent odd readout')
                audit.near(saved[metric+'_even'],(a[metric]+b[metric])/2,'independent even readout')
    saved_ba_gates={(r['setting'],r['phase_index'],r['dose_index'],r['direction'],tuple(r['methods']),r['delay_cycles'],r['sigma']):r for r in analysis['summary']['balanced_accuracy_comparisons']}
    for key,value in cells.items():
        if key[4]=='dop853':continue
        other=key[:4]+(METHODS[METHODS.index(key[4])+1],)+key[5:]
        difference=abs(value['q']-cells[other]['q']) if other in cells else None
        ba_gates[(key,other)]=difference is not None and difference<=.001
        lookup=key[:4]+((key[4],other[4]),)+key[5:];saved=saved_ba_gates[lookup]
        audit.check(saved['qualified']==ba_gates[(key,other)],'paired BA qualification gate')
        if difference is not None:
            audit.near(saved['absolute_ba_difference'],difference,'paired BA method difference')
            ba_differences.append(dict(setting=key[0],key=list(key),absolute_difference=difference))
    settings={}
    for setting in SETTINGS:
        needed=[t for t in tasks() if t['setting']==setting]
        complete=all(t['id'] in scored for t in needed)
        qualified=(complete and reference_checks[setting]['qualified'] and all(prop_checks[f'{setting}--p{j}']['qualified'] for j in range(8))
            and all(v for pair,v in comparison_gates.items() if pair[0].startswith(setting+'--'))
            and all(v for (ident,k),v in sham_gates.items() if ident.startswith(setting+'--'))
            and all(v for (key,other),v in ba_gates.items() if key[0]==setting))
        audit.check(analysis['summary']['settings'][setting]['complete']==complete and analysis['summary']['settings'][setting]['qualified']==qualified,setting+' independent qualification')
        primary=None
        if qualified:
            primary={}
            for direction in DIRECTIONS:
                selected=[v for k,v in cells.items() if (k[0],k[3],k[4],k[5],k[6])==(setting,direction,'rk4-0.00125',4,1e-4)]
                audit.check(len(selected)==24,setting+' full24primary denominator')
                primary[direction]={column:float(np.mean([x[name] for x in selected])) for name,column in [('q','balanced_accuracy'),('phase_only_q','phase_only_balanced_accuracy'),('full_linear_q','full_linear_balanced_accuracy')]}
                primary[direction]['actual_minus_phase_only']=primary[direction]['balanced_accuracy']-primary[direction]['phase_only_balanced_accuracy']
            primary['phase_write_minus_phase_null']=primary['phase_write']['balanced_accuracy']-primary['phase_null']['balanced_accuracy']
            saved=next(x for x in analysis['summary']['settings'][setting]['aggregates'] if x['primary'])
            audit.near(saved['phase_write_minus_phase_null'],primary['phase_write_minus_phase_null'],setting+' primary contrast')
            for direction in DIRECTIONS:audit.near(saved['means'][direction]['balanced_accuracy'],primary[direction]['balanced_accuracy'],setting+' primary '+direction)
        else:audit.check(analysis['summary']['settings'][setting]['aggregates'] is None,setting+' no qualified aggregate on failure')
        settings[setting]=dict(complete=complete,qualified=qualified,primary=primary,observed_traces=sum(t['id'] in scored for t in needed))
    counts=dict(planned_terminal=930,planned_traces=912,observed_traces=len(traces),scored_traces=len(scored),signed_readouts=len(scored)*4,
        raw_comparison_rows=len(analysis['summary']['trace_comparisons']),paired_ba_cells=len(analysis['summary']['paired_readouts']),
        ba_comparison_rows=len(analysis['summary']['balanced_accuracy_comparisons']),odd_even_pairs=len(analysis['paired_dynamics']),odd_even_readouts=len(analysis['summary']['signed_response_readouts']))
    for key,expected_count in [('raw_comparison_rows',608),('paired_ba_cells',5184),('ba_comparison_rows',3456),('odd_even_pairs',432),('odd_even_readouts',1728)]:audit.check(counts[key]==expected_count,'analysis full denominator:'+key)
    complete=all(r['status']=='complete' for rows in terminal.values() for r in rows)
    qualified=complete and all(s['qualified'] for s in settings.values()) and summary['error'] is None and summary['postflight_error'] is None
    audit.check(summary['complete']==complete and summary['qualified']==qualified,'independent study conclusion')
    events=[json.loads(s) for s in (raw/'events.jsonl').read_text().splitlines()]
    seals=[e for e in events if e['stage']=='reference_stage_sealed'];starts=[e for e in events if e['stage']=='traces_started']
    if traces:
        audit.check(len(seals)==len(starts)==1 and seals[0]['monotonic_ns']<starts[0]['monotonic_ns'],'reference-before-trace chronology')
        audit.check(seals[0]['sha256']==sha(raw/'reference-seal.json')==starts[0]['reference_seal_sha256'],'reference seal byte binding')
    producer_now=process_identity(manifest['producer_pid']);producer_absent=producer_now is None or producer_now!=manifest['producer_identity']
    audit.check(not live,'current namespace original owned worker identities absent')
    audit.check(producer_absent,'current namespace producer original identity absent')
    return dict(raw_directory=str(raw),source_commit=receipt['source_commit'],receipt_sha256=sha(raw/'receipt.json'),inventory=inventory,prepared_counts=prepared_counts,counts=counts,status_counts=status_counts,
        references=reference_checks,propagators=prop_checks,settings=settings,complete=complete,qualified=qualified,
        comparison_maxima={key:max((r[key] for r in comparison_details),default=None) for key in ('state_max','activation_max','standardized_max')},
        failing_comparisons=[r for r in comparison_details if not r['qualified']],
        maximum_ba_method_difference=max((r['absolute_difference'] for r in ba_differences),default=None),
        maximum_realized_norm_error=max((r['absolute_error'] for r in pulse_norms),default=None),
        phase_only_residuals={s:dict(maximum_abs_standardized_q_difference=max((abs(r['standardized_actual_minus_phase_only']) for r in null_differences if r['id'].startswith(s+'--')),default=None),
            scope='All available signed cases and delays, not a qualified primary aggregate') for s in SETTINGS},
        cleanup=dict(recorded_workers=len(cleanup),current_original_identities_remaining=live,producer_original_identity_absent=producer_absent,
            check_scope='Current process namespace; invocation on the host must be established separately',pid_namespace=os.readlink('/proc/self/ns/pid'),hostname=platform.node()),
        scope='Saved evidence only; no new integration, fitting, tolerance tuning, phase optimization or sample omission.')


def json_default(value):
    """Normalize NumPy report scalars without changing scientific values."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f"Unsupported report value {type(value).__name__}")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit=Audit();start=time.monotonic();result={}
    if args.output.resolve().is_relative_to(args.raw.resolve()) or args.output.resolve().is_relative_to(args.repo.resolve()):raise ValueError('Audit output must be external to raw/repository')
    if args.output.exists():raise FileExistsError('Existing audit output')
    try:result=run(args.raw,args.repo,audit)
    except Exception as error:audit.check(False,'fatal independent audit exception',repr(error))
    result.update(audit_passed=not audit.failures,checks=audit.checks,failures=audit.failures,maximum_scalar_replay_error=audit.max_scalar_error,
        maximum_array_replay_error=audit.max_array_error,elapsed_seconds=time.monotonic()-start,audit_source_sha256=sha(__file__),
        arithmetic_tolerances=dict(rtol=1e-10,scalar_atol=1e-12,array_atol=1e-11,scientific_gates='unchanged prospective thresholds'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False,default=json_default);f.write('\n')
    print(json.dumps(dict(audit_passed=result['audit_passed'],checks=audit.checks,failures=len(audit.failures),output=str(args.output))))
    raise SystemExit(0 if result['audit_passed'] else 1)


if __name__=='__main__':main()
