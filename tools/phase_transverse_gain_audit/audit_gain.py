#!/usr/bin/env python3
"""Independent EXP015 saved-only audit. No producer imports, ODEs or fitting.

The evidence lane checks hashes, censuses, recorded gate arithmetic and exact
categorical identities. The alternate-arithmetic lane uses SciPy Hermite
polynomials, explicit matrix-vector products and math.fsum contractions. Its
strict componentwise tolerance is the existing 1e-12 relative / 1e-14 absolute
replay tolerance; disagreements are retained separately, never used to repair
or reinterpret the original decisions. Raw evidence is read-only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from independent_codec import Audit, decode_packet, sha, read, canonical

ROOT=Path('/data2/neural-shrimp')
RAW=Path('/data2/neural-shrimp-data/raw/EXP-015/attempt-001')
OLD=Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002')
SOURCE='e62b1dac4291c8aca288b723c70aea85259800d1'
RECEIPT='502a2613b7bac2704a7f93030d0450d7ae7a501c8c9dd8c8a47114dfac30c986'
OLD_RECEIPT='2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8'
SETTINGS=('core','failed-near-edge');DIRECTIONS=('phase_write','phase_null','neuron_one')
DELAYS=(1,2,4,8);DOSES=(1e-4,1e-3,1e-2);NOISES=(1e-5,1e-4,1e-3)
RTOL=1e-12;ATOL=1e-14


class Arithmetic(Audit):
    def near(self,a,b,label,**ignored):
        a,b=np.asarray(a),np.asarray(b)
        finite=a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        err=float(np.max(np.abs(a-b))) if finite and a.size else None
        if err is not None:
            if a.ndim:self.max_array_error=max(self.max_array_error,err)
            else:self.max_scalar_error=max(self.max_scalar_error,err)
        return self.check(finite and np.all(np.abs(a-b)<=ATOL+RTOL*np.abs(b)),label,
                          dict(maximum_absolute_difference=err))


def exact(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b


def census():
    return [dict(id=f'{s}--p{p}--{d}--k{k}',setting=s,phase_index=p,direction=d,delay_cycles=k)
            for s,p,d,k in itertools.product(SETTINGS,range(8),DIRECTIONS,DELAYS)]


def pulse_ids(cell):
    return [f'{cell["setting"]}--p{cell["phase_index"]}--d{dose}--{cell["direction"]}--s{sign:+d}'
            for dose,sign in itertools.product(range(3),(-1,1))]


def stats(value):
    value=np.asarray(value);index=np.unravel_index(np.abs(value).argmax(),value.shape)
    peak=float(np.max(np.abs(value)));scaled=value/peak if peak else value
    return dict(maximum_absolute=peak,rms=float(peak*np.sqrt(np.mean(scaled*scaled))),
                maximum_location=list(map(int,index)),value_at_maximum=float(value[index]))


def close(a,b):
    a,b=np.asarray(a),np.asarray(b);delta=a-b;bound=ATOL+RTOL*np.abs(b)
    return dict(passed=bool(a.shape==b.shape and np.isfinite(delta).all() and np.all(np.abs(delta)<=bound)),
                residual=stats(delta),maximum_tolerance_ratio=float(np.max(np.abs(delta)/bound)))


def compare_tree(a,b,evidence,arithmetic,label):
    if isinstance(a,dict) and isinstance(b,dict):
        evidence.check(a.keys()==b.keys(),label+':keys')
        for key in a.keys()&b.keys():compare_tree(a[key],b[key],evidence,arithmetic,label+'/'+key)
    elif isinstance(a,list) and isinstance(b,list):
        evidence.check(len(a)==len(b),label+':length')
        for i,(left,right) in enumerate(zip(a,b)):compare_tree(left,right,evidence,arithmetic,f'{label}/{i}')
    elif isinstance(a,(float,np.floating,np.ndarray)) or isinstance(b,(float,np.floating,np.ndarray)):
        arithmetic.near(a,b,label)
    else:evidence.check(exact(a,b),label,dict(saved=a,expected=b))


def power(matrix,n):
    answer=np.eye(3,dtype=np.float64);base=matrix.copy()
    while n:
        if n&1:answer=answer@base
        n//=2
        if n:base=base@base
    return answer


def context_arrays(reference,propagator,context,clock):
    period=reference['period'];orbit=reference['orbit'];t=clock['times']
    # The periodic state closes on the initial sample; raw endpoint is retained
    # in the source packet and is not substituted into this clock template.
    values=np.vstack((orbit['states'][:-1],orbit['states'][:1]))
    deriv=np.vstack((orbit['derivatives'][:-1],orbit['derivatives'][:1]))
    template=CubicHermiteSpline(orbit['times'],values,deriv,axis=0)
    query=np.remainder(t+context['phase_epoch'],period)
    states=template(query);velocity=template(query,1)
    segment=CubicHermiteSpline(propagator['times'],propagator['phi'],propagator['phi_derivatives'],axis=0)
    cycles=np.floor(t/period).astype(int);remainder=t-cycles*period
    psi=np.stack([segment(float(s))@power(propagator['monodromy'],int(n)) for s,n in zip(remainder,cycles)])
    a,g=reference['alpha'],reference['gamma']
    matrix=np.array([[1.5,2.9,a],[-1.5,1.18,0.],[g,-22.,.47]],dtype=np.float64)
    physical=np.stack([-x+matrix@np.tanh(x) for x in states])
    return dict(psi=psi,states=states,velocity=velocity,physical=physical,
                f0=context['flow'],z=context['phase_gradient'],w=clock['weights'])


def derive(common,direction):
    psi=common['psi'];f0=common['f0'];z=common['z'];w=common['w']
    p=np.eye(3)-np.outer(f0,z);a=float(z@direction)
    transported=np.stack([m@f0 for m in psi]);full=np.stack([m@direction for m in psi])
    state=dict(total=full,phase_algebraic=transported*a,
        transverse=np.stack([m@(p@direction) for m in psi]),phase_physical=common['physical']*a,
        phase_clock=common['velocity']*a,transport_defect=(transported-common['physical'])*a,
        clock_flow_defect=(common['physical']-common['velocity'])*a)
    derivative=1-np.tanh(common['states'])**2
    activation={key:derivative*value for key,value in state.items()}
    q={key:math.fsum((w*value).ravel().tolist()) for key,value in activation.items()}
    q.update(initial_time_shift=a,clock_normalization_defect=q['phase_clock']-a,
        ideal_accounting_residual=q['total']-(a+q['transverse']+q['transport_defect']+q['clock_flow_defect']+(q['phase_clock']-a)))
    row=np.array([math.fsum((w*derivative*psi[:,:,j]).ravel().tolist()) for j in range(3)])
    norms=np.linalg.norm(common['physical'],axis=1)
    defects=dict(normalization=abs(float(z@f0)-1.),normalization_signed=float(z@f0)-1.,
        projector_idempotence=stats(p@p-p),phase_null_leakage=stats(z@p),
        flow_transport=stats(transported-common['physical']),clock_flow=stats(common['physical']-common['velocity']),
        maximum_relative_physical_flow_transport=None if np.any(norms==0) else float(np.max(np.linalg.norm(transported-common['physical'],axis=1)/norms)),
        zero_reference_flow_samples=int(np.sum(norms==0)),
        clock_weight_derivative_inner_product=math.fsum((w*derivative*common['velocity']).ravel().tolist()),
        clock_weight_derivative_normalization=abs(math.fsum((w*derivative*common['velocity']).ravel().tolist())-1.))
    return dict(state=state,activation=activation,observable=q,row_operator=row,
                projector=p,noise_gain=float(np.linalg.norm(w)),reference_defects=defects)


def recorded_checks(mathematics,direction):
    state=mathematics['state'];act=mathematics['activation'];q=mathematics['observable'];a=q['initial_time_shift']
    return dict(state_partition=close(state['phase_algebraic']+state['transverse'],state['total']),
        activation_partition=close(act['phase_algebraic']+act['transverse'],act['total']),
        readout_partition=close(q['phase_algebraic']+q['transverse'],q['total']),
        row_operator=close(float(mathematics['row_operator']@direction),q['total']),
        ideal_accounting=close(a+q['transverse']+q['transport_defect']+q['clock_flow_defect']+(q['phase_clock']-a),q['total']))


def approximation(mathematics):
    state=mathematics['state'];act=mathematics['activation'];q=mathematics['observable'];rows={}
    for key in ('transport_defect','clock_flow_defect','combined_ideal_clock_defect'):
        if key.startswith('combined'):
            x=state['phase_algebraic']-state['phase_clock'];u=act['phase_algebraic']-act['phase_clock'];score=q['phase_algebraic']-q['phase_clock']
        else:x=state[key];u=act[key];score=q[key]
        dx=float(np.max(np.abs(x)));du=float(np.max(np.abs(u)));dq=abs(score)/(1e-4*mathematics['noise_gain'])
        rows[key]=dict(state_max=dx,activation_max=du,primary_standardized_readout=dq,
                       passed=bool(dx<=1e-6 and du<=1e-7 and dq<=.01))
    return rows


def audit_math(saved,derived,direction,orientation,common,evidence,arithmetic,alternate_decisions,label):
    for key in derived:compare_tree(saved[key],derived[key],alternate_decisions,arithmetic,label+'/'+key)
    # Recorded gate decisions are replayed from the actual saved operands, while
    # alternate interpolation/reduction discrepancies remain in their own lane.
    expected=recorded_checks(saved,direction)
    for key,value in expected.items():compare_tree(saved['algebraic_checks'][key],value,evidence,arithmetic,label+'/checks/'+key)
    for field,arrays in [('state_gain',saved['state']),('activation_gain',saved['activation'])]:
        for key,value in arrays.items():compare_tree(saved[field][key],stats(value),evidence,arithmetic,label+'/'+field+'/'+key)
    for key,value in saved['activation'].items():
        arithmetic.near(saved['observable'][key],math.fsum((common['w']*value).ravel().tolist()),label+'/independent_contraction/'+key)
    d=saved['reference_defects'];p=np.eye(3)-np.outer(common['f0'],common['z'])
    expected_normalization=float(common['z']@common['f0'])-1.
    arithmetic.near(d['normalization_signed'],expected_normalization,label+'/signed_normalization')
    arithmetic.near(d['normalization'],abs(expected_normalization),label+'/normalization')
    compare_tree(d['projector_idempotence'],stats(p@p-p),evidence,arithmetic,label+'/idempotence')
    compare_tree(d['phase_null_leakage'],stats(common['z']@p),evidence,arithmetic,label+'/null_leakage')
    noise=saved.get('noise_rows')
    if noise is not None:audit_noise(noise,saved,orientation,evidence,arithmetic,label)
    return expected


def audit_noise(rows,mathematics,orientation,evidence,arithmetic,label):
    evidence.check([r['sigma'] for r in rows]==list(NOISES),label+'/noise_census')
    for row,sigma in zip(rows,NOISES):
        sd=sigma*mathematics['noise_gain'];arithmetic.near(row['noise_sd'],sd,label+'/noise_sd')
        expected={k:orientation*v/sd for k,v in mathematics['observable'].items()}
        compare_tree(row['signed_standardized'],expected,evidence,arithmetic,label+f'/sigma{sigma}')


def packet(path,evidence):
    evidence.check(not Path(path).is_symlink(),'packet not symlink:'+str(path))
    return decode_packet(path,evidence)


def file_binding(path,relative=None):
    path=Path(path);return dict(path=str(path.relative_to(relative)) if relative else str(path),bytes=path.stat().st_size,sha256=sha(path))


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=fields[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError,IndexError):return None


def audit(directory,report):
    evidence=Audit();arithmetic=Arithmetic();alternate_decisions=Audit()
    report.update(evidence=evidence.__dict__,alternate_arithmetic=arithmetic.__dict__,alternate_arithmetic_categories=alternate_decisions.__dict__)
    evidence.check(directory.resolve()==RAW,'exact target directory')
    evidence.check(sha(directory/'receipt.json')==RECEIPT,'exact original target receipt')
    receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    evidence.check(receipt['source_commit']==manifest['source_commit']==SOURCE,'exact frozen source')
    files=[file_binding(p,directory) for p in sorted(directory.rglob('*')) if p.is_file() and p!=directory/'receipt.json']
    evidence.check(files==receipt['artifacts'],'complete artifact inventory')
    evidence.check(all(not p.is_symlink() for p in directory.rglob('*')),'no raw aliases')
    evidence.check(sum(f['bytes'] for f in files)==receipt['artifact_bytes'],'artifact byte total')
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':experiments/hopfield/EXP-015-plan.json'])
    evidence.check(json.loads(committed)==plan and hashlib.sha256(committed).hexdigest()==receipt['plan_sha256']==manifest['plan_sha256'],'exact committed plan')
    for item in plan['source_files']+[manifest['review']]:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',SOURCE+':'+item['path']])
        evidence.check(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'source:'+item['path'])
    evidence.check(sha(OLD/'receipt.json')==OLD_RECEIPT,'old receipt pin')
    for item in plan['input_files']:evidence.check(file_binding(OLD/item['path'],OLD)==item,'old input:'+item['path'])
    bundle=packet(directory/'inputs-seal.json',evidence)
    evidence.check(len(bundle['references'])==2 and len(bundle['propagators'])==16 and len(bundle['contexts'])==16 and len(bundle['cases'])==288,'compact input census')
    # Independent full decode and explicit projection of original sealed arrays.
    original=packet(OLD/'reference-seal.json',evidence)
    for key,value in bundle['contexts'].items():
        expected={k:v for k,v in original['contexts'][key].items() if k not in {'diagnostic_times','reference_states','reference_velocity','reference_adjoint'}}
        evidence.check(exact(value,expected),'unchanged compact context:'+key)
    for key,value in bundle['cases'].items():
        expected={k:v for k,v in original['cases'][key].items() if k not in {'full_linear_state_delta','full_linear_activation_delta','phase_only_activations'}}
        evidence.check(exact(value,expected),'unchanged compact case:'+key)
    del original
    for setting,reference in bundle['references'].items():
        old=packet(OLD/f'references/payloads/{setting}/packet.json',evidence)
        evidence.check(exact(reference,old),'unchanged reference:'+setting)
        evidence.check(reference['qualified'] is True and all(reference['gates'].values()),'qualified inherited reference:'+setting)
    for key,prop in bundle['propagators'].items():
        old=packet(OLD/f'propagators/payloads/{key}/packet.json',evidence)
        evidence.check(exact(prop,old),'unchanged propagator:'+key)
        evidence.check(prop['qualified'] is True and all(prop['gates'].values()),'qualified inherited propagator:'+key)
    cells=census();rows=read(directory/'terminal-records.json');journal=[json.loads(line) for line in (directory/'cell-journal.jsonl').read_text().splitlines()]
    evidence.check(plan['cells_census']==cells==read(directory/'planned-cells.json')==[r['cell'] for r in rows],'exact ordered192 census')
    evidence.check(journal==rows,'full journal/terminal identity')
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    seals=[r for r in events if r['event']=='inputs_sealed'];starts=[r for r in events if r['event']=='cells_started']
    evidence.check(len(seals)==len(starts)==1 and seals[0]['monotonic_ns']<starts[0]['monotonic_ns'],'seal precedes calculation')
    evidence.check(seals[0]['sha256']==sha(directory/'inputs-seal.json') and seals[0]['semantic_sha256']==read(directory/'inputs-seal.json')['semantic_sha256'],'seal event hashes')
    qualifications=[];compatibility_count=unit_noise=pulse_noise=0;alternate_gate_differences=[];failed_cells=[]
    for cell,row in zip(cells,rows,strict=True):
        label=cell['id'];evidence.check(row['status']=='complete','complete cell:'+label)
        path=directory/row['packet_path'];evidence.check(row['packet_path']=='cells/'+label+'.json' and sha(path)==row['packet_sha256'],'cell file:'+label)
        result=packet(path,evidence);evidence.check(result['cell']==cell and result['complete'] is True and result['status']=='complete','cell identity:'+label)
        evidence.check(row['packet_semantic_sha256']==read(path)['semantic_sha256'],'cell semantic row:'+label)
        ids=pulse_ids(cell);evidence.check(row['compatibility_ids']==ids==[r['id'] for r in result['compatibility']],'six realized identities:'+label)
        evidence.check(row['compatibility_counts']==dict(required=6,complete=6,failed=0,unavailable=0),'six counts:'+label)
        context=bundle['contexts'][f'{cell["setting"]}--p{cell["phase_index"]}'];ref=bundle['references'][cell['setting']];prop=bundle['propagators'][context['id']]
        clock=context['readouts'][str(cell['delay_cycles'])];common=context_arrays(ref,prop,context,clock)
        digest=hashlib.sha256(canonical({key:clock[key] for key in ('derivative_energy','phase_epoch','period','delay_cycles')}))
        for key in ('times','reference_activations','weights'):digest.update(key.encode());digest.update(clock[key].tobytes(order='C'))
        evidence.check(result['clock_sha256']==digest.hexdigest(),'exact sealed clock digest:'+label)
        expected_times=ref['period']*(cell['delay_cycles']-1+(np.arange(32)+.5)/32)
        evidence.check(np.array_equal(clock['times'],expected_times),'exact clock grid:'+label)
        d=(1-np.tanh(common['states'])**2)*common['velocity'];energy=float(np.sum(d*d))
        arithmetic.near(clock['weights'],d/energy,label+'/independent_clock_weights')
        arithmetic.near(clock['reference_activations'],np.tanh(common['states']),label+'/independent_clock_reference')
        direction=context['directions'][cell['direction']];orientation=context['classifier_orientations'][cell['direction']]
        phase=ref['phases'][cell['phase_index']];sample=512*cell['phase_index']
        for name,expected in [('initial',ref['orbit']['states'][sample]),('flow',ref['orbit']['derivatives'][sample]),('phase_gradient',ref['adjoint']['values'][sample])]:
            evidence.check(np.array_equal(context[name],expected) and np.array_equal(phase[name],expected),label+'/phase_binding/'+name)
        evidence.check(abs(np.linalg.norm(direction)-1.)<=1e-12,label+'/unit_direction')
        unit=result['unit'];derived=derive(common,direction)
        evidence.check(unit['orientation']==orientation,'frozen orientation:'+label)
        checks=audit_math(unit,derived,direction,orientation,common,evidence,arithmetic,alternate_decisions,label+'/unit');unit_noise+=len(unit['noise_rows'])
        # Independently reconstructed Psi also audits the direct-propagation
        # comparisons. Their floating-point residual differences are diagnostic.
        directcheck=close(unit['state']['total'],derived['state']['total'])
        compare_tree(unit['algebraic_checks']['old_direct_unit_propagation'],directcheck,alternate_decisions,arithmetic,label+'/alternate_direct_unit')
        algebra=all(v['passed'] for v in unit['algebraic_checks'].values());compat_ok=True;ideal_ok=True
        independent_algebra=all(v['passed'] for v in checks.values()) and directcheck['passed']
        for saved,ident,(dose,sign) in zip(result['compatibility'],ids,itertools.product(range(3),(-1,1)),strict=True):
            compatibility_count+=1;old=bundle['cases'][ident];pred=old['readouts'][str(cell['delay_cycles'])]
            requested=sign*DOSES[dose]*direction;initial=context['initial']+requested;realized=initial-context['initial']
            for name,expected in [('requested_delta',requested),('realized_delta',realized),('initial',initial),('addition_rounding_residual',realized-requested)]:
                evidence.check(np.array_equal(old['pulse'][name],expected),label+'/'+ident+'/pulse/'+name)
            evidence.check(np.array_equal(saved['requested_delta'],requested) and np.array_equal(saved['realized_delta'],realized),label+'/'+ident+'/realization')
            value=saved['mathematics'];independent=derive(common,realized)
            current=audit_math(value,independent,realized,orientation,common,evidence,arithmetic,alternate_decisions,label+'/'+ident)
            expected_checks=dict(state=close(value['state']['total'],pred['full_linear_state_delta']),activation=close(value['activation']['total'],pred['full_linear_activation_delta']),q=close(value['observable']['total'],pred['full_linear_q']),noise_gain=close(value['noise_gain'],pred['noise_gain']))
            for key,expect in expected_checks.items():compare_tree(saved['checks'][key],expect,evidence,arithmetic,label+'/'+ident+'/compatibility/'+key)
            alternate_direct=close(value['state']['total'],independent['state']['total'])
            compare_tree(saved['checks']['direct_propagation'],alternate_direct,alternate_decisions,arithmetic,label+'/'+ident+'/alternate_direct')
            expect=approximation(value);compare_tree(saved['approximation'],expect,evidence,arithmetic,label+'/'+ident+'/ideal_gates')
            audit_noise(saved['noise_rows'],value,orientation,evidence,arithmetic,label+'/'+ident);pulse_noise+=len(saved['noise_rows'])
            evidence.check(saved['original_finite_phase_only_q']==pred['phase_only_q'],'requested finite comparator:'+label+'/'+ident)
            algebra &= all(v['passed'] for v in value['algebraic_checks'].values())
            compat_ok &= all(v['passed'] for v in saved['checks'].values())
            ideal_ok &= all(v['passed'] for v in expect.values())
            independent_algebra &= all(v['passed'] for v in current.values()) and alternate_direct['passed']
        gates=dict(input_identity=True,algebraic=bool(algebra),old_linear_compatibility=bool(compat_ok),normalization=bool(abs(float(common['z']@common['f0'])-1)<=1e-7),ideal_approximation=bool(ideal_ok))
        evidence.check(result['gates']==gates,'all recorded gate arithmetic:'+label,dict(saved=result['gates'],expected=gates))
        qa=bool(gates['input_identity'] and gates['algebraic'] and gates['old_linear_compatibility']);qi=bool(qa and gates['normalization'] and gates['ideal_approximation'])
        evidence.check(result['qualified_algebraic']==row['qualified_algebraic']==qa and result['qualified_ideal_neutral']==row['qualified_ideal_neutral']==result['qualified']==row['qualified']==qi,'qualification:'+label)
        qualifications.append(dict(id=label,algebraic=qa,ideal=qi))
        if not qi:failed_cells.append(dict(cell=cell,gates=gates,qualification_algebraic=qa,qualification_ideal=qi,
            failing_checks={ident:{key:v for key,v in saved['approximation'].items() if not v['passed']} for ident,saved in zip(ids,result['compatibility']) if any(not v['passed'] for v in saved['approximation'].values())},
            failed_algebraic_unit={k:v for k,v in unit['algebraic_checks'].items() if not v['passed']},
            failed_realized_checks={ident:dict(algebra={k:v for k,v in saved['mathematics']['algebraic_checks'].items() if not v['passed']},compatibility={k:v for k,v in saved['checks'].items() if not v['passed']}) for ident,saved in zip(ids,result['compatibility']) if any(not v['passed'] for v in list(saved['mathematics']['algebraic_checks'].values())+list(saved['checks'].values()))}))
        if independent_algebra != qa:alternate_gate_differences.append(dict(cell=label,original_algebraic=qa,independent_direct_arithmetic=bool(independent_algebra)))
    summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    totals=dict(required_cells=192,calculated_cells=192,algebraically_qualified_cells=sum(r['algebraic'] for r in qualifications),ideal_neutral_qualified_cells=sum(r['ideal'] for r in qualifications),all_gates_pass_cells=sum(r['ideal'] for r in qualifications),realized_pulse_rows=dict(required=1152,complete=1152,failed=0,unavailable=0))
    for key,value in totals.items():evidence.check(summary[key]==value,'summary:'+key)
    evidence.check(compatibility_count==1152 and unit_noise==576 and pulse_noise==3456,'complete arithmetic/noise census')
    evidence.check(summary['setting_means'] is None and summary['original_study_requalified'] is False,'no setting means/requalification')
    evidence.check(summary['complete'] is True and summary['source_valid'] is True and summary['error'] is None and summary['postflight_error'] is None and post['error'] is None and receipt['status']=='complete','operational completion')
    current=process_identity(manifest['producer_pid']);absent=current is None or current!=manifest['producer_identity'];evidence.check(absent,'original owned producer identity absent')
    report.update(census=dict(cells=192,realized_pulses=compatibility_count,unit_noise=unit_noise,realized_noise=pulse_noise,artifacts=len(files),artifact_bytes=sum(f['bytes'] for f in files),source_files=len(plan['source_files']),old_input_files=len(plan['input_files'])),totals=totals,failed_cells=failed_cells,alternate_gate_differences=alternate_gate_differences,qualification_rows=qualifications,cleanup=dict(checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid_namespace=os.readlink('/proc/self/ns/pid'),original=manifest['producer_identity'],current=current,original_identity_absent=absent),evidence=evidence.__dict__,alternate_arithmetic=arithmetic.__dict__)
    report['evidence_and_recorded_decisions_passed']=not evidence.failures
    report['alternate_strict_arithmetic_passed']=not arithmetic.failures


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=RAW);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT) or args.output.resolve().is_relative_to(args.directory.resolve()):raise ValueError('New external output required')
    began=time.monotonic();report=dict(audit_scope=__doc__,target_directory=str(args.directory),expected_source=SOURCE,expected_receipt=RECEIPT,python=platform.python_version(),numpy=np.__version__,strict_tolerance=dict(rtol=RTOL,atol=ATOL),new_odes=0,new_fits=0)
    try:audit(args.directory,report)
    except BaseException as exc:report['audit_exception']=repr(exc)
    report['elapsed_seconds']=time.monotonic()-began
    report['audit_sources']=[file_binding(Path(__file__)),file_binding(Path(__file__).with_name('independent_codec.py'))]
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),bytes=args.output.stat().st_size,evidence_passed=report.get('evidence_and_recorded_decisions_passed'),strict_arithmetic_passed=report.get('alternate_strict_arithmetic_passed'),audit_exception=report.get('audit_exception'),elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report.get('evidence_and_recorded_decisions_passed') and not report.get('audit_exception') else 1


if __name__=='__main__':raise SystemExit(main())
