"""Task37 independent manual-map R pair. Held packet; no producer numerical imports.

Execution requires an admitted card and external supervisor. Every proposal,
Newton operator call, dense column, full eigensystem and phase logit is retained.
This is finite numerical evidence, never interval certification.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
RAW = Path('/data2/neural-shrimp-data/scratch/audit-codex-1019')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024**2), b''): h.update(b)
    return h.hexdigest()

def verify(pin):
    p = Path(pin['path'])
    if (type(pin.get('bytes')) is not int or pin['bytes'] < 0
            or not _hex_digest(pin.get('sha256'), 64)):
        raise ValueError('Complete byte/hash pin required: '+str(p))
    if p.stat().st_size != pin['bytes'] or sha(p) != pin['sha256']:
        raise ValueError('Changed pinned file: '+str(p))

def dump(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')

def _hex_digest(value, length):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)

def _pin_map(pins):
    if not isinstance(pins, list):
        raise ValueError('Explicit pin inventory required')
    result = {}
    for pin in pins:
        if not isinstance(pin, dict) or not isinstance(pin.get('path'), str):
            raise ValueError('Invalid file pin')
        name = pin['path']
        if not Path(name).is_absolute() or str(Path(name)) != name or '..' in Path(name).parts or name in result:
            raise ValueError('Canonical unique absolute pin paths required')
        result[name] = pin
    return result

def _identity(pid):
    # Read only this worker or its known parent, never enumerate other jobs.
    fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid':pid, 'starttime':int(fields[19]), 'pgid':int(fields[2]), 'sid':int(fields[3])}

def _worker_argv():
    return [s.decode() for s in Path('/proc/self/cmdline').read_bytes().split(b'\0') if s]

def _utcnow():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)

def _capacity_window(window):
    from datetime import datetime, timezone
    if (not isinstance(window, dict) or window.get('confirmed') is not True
            or window.get('owner') != 'task37-r-pair-001'
            or not isinstance(window.get('coordination_reference'),str) or not window['coordination_reference'].strip()):
        raise ValueError('Explicit confirmed Task37 capacity window required')
    try:
        begin = datetime.fromisoformat(window['not_before_utc'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(window['not_after_utc'].replace('Z', '+00:00'))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Dated capacity window required') from error
    if (begin.tzinfo is None or end.tzinfo is None or begin.utcoffset() != timezone.utc.utcoffset(begin)
            or end.utcoffset() != timezone.utc.utcoffset(end) or not begin <= _utcnow() <= end
            or (end - _utcnow()).total_seconds() < 15300):
        raise ValueError('Capacity confirmation is not current in UTC')

def preflight(card_path, out_path, admission_path):
    folder = ROOT/'codex/presubmit-task37'
    entry = folder/'p6_kite.py'
    card_file, plan_file = folder/'r-pair-001-card.json', folder/'r-pair-001-plan.json'
    output, supervision = RAW/'r-pair-001', RAW/'r-pair-001-supervision'
    if Path(__file__).resolve() != entry.resolve():
        raise ValueError('Execution must use the committed canonical wrapper')
    if Path(card_path) != card_file or Path(out_path) != output or output.exists():
        raise ValueError('Exact card and fresh allocation required; no retry')
    if Path(admission_path) != supervision/'admission.json':
        raise ValueError('Wrong original supervisor admission')
    card = json.loads(card_file.read_text())
    if card.get('schema') != 'codex-task37-p6-r-pair-card-v1' or card.get('status') != 'reviewed_for_execution':
        raise ValueError('Card is held or wrong schema')
    if card.get('target') != {'label':'R','alpha':.072912,'log10_temperature':-.154104,'period':6}:
        raise ValueError('Only the requested R pair is admitted by this packet')
    _capacity_window(card.get('capacity_window'))
    admission = json.loads(Path(admission_path).read_text())
    source = admission.get('source_commit')
    if not _hex_digest(source, 40):
        raise ValueError('Exact source commit required')
    if admission.get('lane') != 'r_pair' or admission.get('plan', {}).get('path') != str(plan_file):
        raise ValueError('Wrong supervisor plan or lane')
    verify(admission['plan'])
    plan = json.loads(plan_file.read_text())
    if (plan.get('status') != 'reviewed_for_execution' or plan.get('repository') != str(ROOT)
            or plan.get('raw_root') != str(RAW) or plan.get('cpu_threads') != 1
            or set(plan.get('lanes', {})) != {'r_pair'}):
        raise ValueError('Wrong reviewed supervisor plan')
    lane = plan['lanes']['r_pair']
    limits = {'as_bytes':18*1024**3,'cpu_seconds':15000,'wall_seconds':15300,'output_bytes':512*1024**2}
    if (lane.get('output') != str(output) or lane.get('supervision') != str(supervision)
            or lane.get('entry') != str(entry) or lane.get('limits') != limits or admission.get('limits') != limits):
        raise ValueError('Supervisor allocation or limits mismatch')
    arguments = ['--card', str(card_file), '--out', str(output), '--admission', str(supervision/'admission.json')]
    if lane.get('arguments') != arguments or not isinstance(plan.get('python'), str):
        raise ValueError('Wrong worker arguments')
    expected_argv = [plan['python'],'-I','-B',str(entry),*arguments]
    if admission.get('argv') != expected_argv or _worker_argv() != expected_argv:
        raise ValueError('Executing argv does not match admitted lane')
    if Path(sys.executable).resolve() != Path(plan['python']).resolve():
        raise ValueError('Wrong worker interpreter')
    sources = _pin_map(card.get('source_files'))
    required_sources = {str(entry), str(folder/'periodic_audit.py'), str(ROOT/'codex/crosscheck/qwen_map.py'),
                        str(ROOT/'codex/certify-chaos/supervise.py')}
    if set(sources) != required_sources:
        raise ValueError('Exact wrapper/helper/manual-map/supervisor source inventory required')
    inventory_file = folder/'p6-seed-inventory.json'
    snapshot = Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775')
    p6 = Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/p6s')
    if card.get('seed_inventory') != str(inventory_file) or card.get('tokenizer') != str(snapshot/'tokenizer.json'):
        raise ValueError('Wrong seed inventory or tokenizer')
    prior = card.get('prior_closed_receipt')
    if (prior != '/data2/neural-shrimp-data/codex/certificate-ladder/phase-cascade-p6-001-supervision/receipt.json'
            or prior not in lane.get('prior_receipts', [])):
        raise ValueError('Explicit prior closed receipt required')
    required_inputs = {str(inventory_file), prior, str(snapshot/'tokenizer.json'), str(snapshot/'model.safetensors'),
                       str(snapshot/'config.json'), '/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz'}
    for name in ('p6s-ad-curve-plus','p6s-td-curve-minus'):
        required_inputs.update({str(p6/name/'states.npy'), str(p6/name/'curve.json')})
    inputs = _pin_map(card.get('input_files'))
    if len(required_inputs) != 10 or set(inputs) != required_inputs:
        raise ValueError('Exact ten original/input/receipt pins required')
    plan_sources = _pin_map(plan.get('source_files'))
    if set(plan_sources) != required_sources | {str(card_file)}:
        raise ValueError('Plan source inventory must include the exact committed card')
    for name,pin in sources.items():
        if any(plan_sources[name].get(k) != pin.get(k) for k in ('bytes','sha256')):
            raise ValueError('Card and plan source pins disagree')
    preflight_pins = _pin_map(lane.get('preflight_pins'))
    if not set(inputs) <= set(preflight_pins):
        raise ValueError('Supervisor did not pin every worker input')
    for name,pin in inputs.items():
        if any(preflight_pins[name].get(k) != pin.get(k) for k in ('bytes','sha256')):
            raise ValueError('Card and supervisor input pins disagree')
    blobs = admission.get('source_blobs')
    expected_blobs = {str(Path(name).relative_to(ROOT)):(p.get('bytes'),p.get('sha256')) for name,p in plan_sources.items()}
    if (not isinstance(blobs,list) or len(blobs) != len(expected_blobs)
            or {b.get('relative_path'):(b.get('bytes'),b.get('sha256')) for b in blobs} != expected_blobs):
        raise ValueError('Supervisor did not admit the exact source inventory')
    # Pin both metadata documents to the same actual source commit, not only
    # arbitrary source paths supplied inside an editable card.
    for name,pin in [*plan_sources.items(), (str(plan_file), admission['plan'])]:
        verify(pin)
        rel = str(Path(name).relative_to(ROOT))
        original = subprocess.check_output(['git','show',source+':'+rel],cwd=ROOT)
        if len(original) != pin['bytes'] or hashlib.sha256(original).hexdigest() != pin['sha256']:
            raise ValueError('Source commit does not contain reviewed bytes')
    for pin in inputs.values():verify(pin)
    for ref in ('HEAD','origin/main'):
        if subprocess.check_output(['git','rev-parse',ref],cwd=ROOT).decode().strip() != source:
            raise ValueError('Private source checkpoint equality no longer holds')
    prior_record = json.loads(Path(prior).read_text())
    if prior_record.get('worker_reaped') is not True or prior_record.get('worker_group_session_absent') is not True:
        raise ValueError('Prior worker ownership is not closed')
    supervisor = admission.get('supervisor', {})
    if supervisor.get('pid') != os.getppid():
        raise ValueError('Worker is not a child of the admitted supervisor')
    if any(_identity(os.getppid())[k] != supervisor.get(k) for k in ('pid','starttime','pgid','sid')):
        raise ValueError('Supervisor identity changed')
    startup = json.loads((supervision/'worker-startup.json').read_text())
    worker = _identity(os.getpid())
    if (any(worker[k] != startup.get('identity', {}).get(k) for k in worker)
            or worker['pgid'] != os.getpid() or worker['sid'] != os.getpid()
            or startup.get('planned_argv') != expected_argv):
        raise ValueError('Worker preexec identity/argv mismatch')
    if startup.get('affinity') != sorted(os.sched_getaffinity(0)) or len(startup['affinity']) != 1:
        raise ValueError('Exact one-CPU worker affinity required')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1' or admission.get('threads', {}).get(key) != '1':
            raise ValueError('One-thread environment required')
    if os.environ.get('HF_HUB_OFFLINE') != '1' or os.getpriority(os.PRIO_PROCESS,0) != 10:
        raise ValueError('Offline assets and nice10 required')
    for key,rlim,expected in [('address_space',resource.RLIMIT_AS,limits['as_bytes']),
                              ('cpu_seconds',resource.RLIMIT_CPU,limits['cpu_seconds']),
                              ('file_size',resource.RLIMIT_FSIZE,limits['output_bytes'])]:
        if resource.getrlimit(rlim) != (expected,)*2 or startup.get(key) != [expected]*2:
            raise ValueError('External '+key+' cap required')
    capacity = admission.get('capacity', {})
    startup_gib = plan.get('startup_ram_gib',32)
    if (not isinstance(startup_gib,(int,float)) or startup_gib < 32
            or capacity.get('available_ram_bytes',0) < startup_gib*1024**3
            or capacity.get('disk_free_bytes',0) < 16*1024**3):
        raise ValueError('Supervisor capacity admission below required floor')
    return card, admission

def run_science(a, card, admission):
    # Scientific imports occur only after admission and exact source checks.
    import numpy as np
    import torch
    import scipy
    import tokenizers
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).parent))
    from periodic_audit import AuditOperator, Limits, solve, dense_monodromy, spectrum, derivative_gate, primitive_and_alignment
    path=ROOT/'codex/crosscheck/qwen_map.py'
    spec=importlib.util.spec_from_file_location('task37_manual_map',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    out=Path(a.out);began=time.monotonic()
    inventory=json.loads(Path(card['seed_inventory']).read_text())
    rows=inventory['seeds']
    if [(r['name'],r['index']) for r in rows] != [('p6s-ad-curve-plus',6),('p6s-td-curve-minus',2)]:
        raise ValueError('Exact two seed rows required')
    seeds=[]
    for r in rows:
        verify(r['states']);verify(r['curve'])
        states=np.load(r['states']['path'],allow_pickle=False)
        points=json.loads(Path(r['curve']['path']).read_text())['points']
        if states.shape!=(len(points),896) or states.dtype!=np.float64 or not np.isfinite(states).all():
            raise ValueError('Invalid original seed array')
        if [p['index'] for p in points]!=list(range(len(points))):raise ValueError('Ambiguous saved index')
        point=points[r['index']]
        if any(point[k]!=r['seed_parameters'][k] for k in ('alpha','log10_temperature')):
            raise ValueError('Seed metadata changed')
        seed=states[r['index']].copy()
        if hashlib.sha256(seed.tobytes()).hexdigest()!=r['state_row_sha256_native_float64']:
            raise ValueError('Saved seed row changed')
        seeds.append(seed)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    dump(out/'model-load-start.json',{'model_load_started':True,'prefix_evaluations_started':0})
    model=module.ManualQwenSoftMap.from_pinned_files()
    if (model.d,model.layers,model.vocab,model.steering_layer)!=(896,24,151936,12):
        raise ValueError('Wrong map architecture')
    alpha,lt=card['target']['alpha'],card['target']['log10_temperature']
    dump(out/'prefix-start.json',{'model_loaded':True,'prefix_evaluations_started':1})
    cache=model.prefix_cache(alpha)
    tokenizer=Tokenizer.from_file(card['tokenizer'])
    dump(out/'model-ready.json',{'source_commit':admission['source_commit'],'target':card['target'],
         'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,
         'tokenizers':tokenizers.__version__,'python':sys.version,'prefix_evaluations':1,
         'temperature_binary64':float((10.**torch.tensor(lt,dtype=torch.float64)).item()),
         'map_source_sha256':sha(path),'num_threads':torch.get_num_threads(),'device':'cpu'})
    def t(x):return torch.from_numpy(np.array(x,dtype=np.float64,copy=True))
    def f(x):return model.period_map(t(x),alpha,lt,6,cache).detach().numpy()
    def j(x,v):return model.jvp(t(x),t(v),alpha,lt,6,cache).detach().numpy()
    def bj(x,v):
        return model.jvp(t(np.broadcast_to(x,v.shape).copy()),t(v),alpha,lt,6,cache).detach().numpy()
    orbits=[];results=[]
    try:
        for k,seed in enumerate(seeds):
            with AuditOperator(f,j,out/f'seed{k}',batch_jvp=bj,limits=Limits(**card['operator_limits']),
                               metadata={'original_seed':rows[k],'target':card['target']}) as op:
                x,status=solve(op,seed)
                result={'seed':k,'newton':status,'qualified':False,'certificate_passed':False}
                results.append(result)
                if not status['converged']:
                    dump(op.path/'result.json',result);continue
                phases=[];tokens=[];cur=x.copy()
                for phase in range(6):
                    phases.append(cur.copy())
                    op.emit('phase_logits_start',phase=phase)
                    logits=model.logits(t(cur),alpha,cache).detach()
                    probs=torch.softmax(logits/(10.**torch.tensor(lt,dtype=torch.float64)),dim=-1)
                    token=int(torch.argmax(probs).item())
                    np.save(op.path/f'phase-{phase}-logits.npy',logits.numpy(),allow_pickle=False)
                    next_state=(probs@model.embedding).numpy()
                    np.save(op.path/f'phase-{phase}-image.npy',next_state,allow_pickle=False)
                    tokens.append({'phase':phase,'token_id':token,'text':tokenizer.decode([token],skip_special_tokens=False),
                                   'probability':float(probs[token]),'logit_gap':float(torch.topk(logits,2).values.diff().abs()[0])})
                    op.emit('phase_logits_end',phase=phase)
                    cur=next_state
                orbit=np.stack(phases);np.save(op.path/'phases.npy',orbit,allow_pickle=False)
                closure=float(np.linalg.norm(cur-x)/np.linalg.norm(x))
                result.update(phase_closure_relative=closure,tokens=tokens,
                              primitive=primitive_and_alignment(orbit),one_step_phase_calls=6)
                if closure>card['operator_limits']['residual_tolerance']:
                    result['failure']='separate phase walk failed closure';dump(op.path/'result.json',result);continue
                all_phase_residuals=[]
                for phase,state in enumerate(orbit):
                    image=op.value(state)
                    all_phase_residuals.append(float(np.linalg.norm(image-state)/np.linalg.norm(state)))
                result['all_phase_return_residuals_relative']=all_phase_residuals
                if max(all_phase_residuals)>card['operator_limits']['residual_tolerance']:
                    result['failure']='at least one phase failed composed-return closure'
                    dump(op.path/'result.json',result);continue
                M=dense_monodromy(op,x,chunk_size=8)
                spec_result=spectrum(op,x,M)
                eigenvalues=np.array([complex(*z) for z in spec_result['all_multipliers']])
                near=int(np.argmin(abs(eigenvalues+1)))
                result['nearest_minus_one']={'sorted_index':near,'real':float(eigenvalues[near].real),
                    'imag':float(eigenvalues[near].imag),'distance':float(abs(eigenvalues[near]+1)),
                    'rounded_R_is_not_a_bisected_crossing':True}
                modes=np.load(op.path/'eigenvectors.npy',allow_pickle=False)
                fd=[];rng=np.random.default_rng(370001);random_checks=[]
                for index in range(3):
                    direction=rng.normal(size=896);direction/=np.linalg.norm(direction)
                    actual=op.jvp(x,direction);predicted=M@direction
                    random_checks.append(float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(actual),1e-14)))
                directions=[('random0',rng.normal(size=896)),('random1',rng.normal(size=896)),
                            ('critical_real',modes[:,0].real.copy()),('critical_imag',modes[:,0].imag.copy())]
                for name,d in directions:
                    if np.linalg.norm(d)>1e-12:
                        fd.append({'component':name,**derivative_gate(op,x,d)})
                random_quadratic=any(v['quadratic_interval_seen'] for v in fd if v['component'].startswith('random'))
                result.update(spectrum=spec_result,composed_fd=fd,random_dense_scalar_jvp_errors=random_checks,
                              nonlinear_random_quadratic_interval=random_quadratic,
                              qualified=bool(result['primitive']['primitive'] and spec_result['eigen_gate']
                                             and max(random_checks)<=1e-10 and random_quadratic
                                             and fd and all(v['passed'] for v in fd)),counts=op.counts.copy())
                dump(op.path/'result.json',result)
                orbits.append((k,orbit,result['qualified']))
        both_closed=len(orbits)==2 and all(z[2] for z in orbits)
        comparison=primitive_and_alignment(orbits[0][1],orbits[1][1]) if both_closed else None
        result={'status':'completed','target':card['target'],'orbits':results,
                'both_orbits_numerically_qualified':both_closed,'alignment':comparison,
                'body_points_eligible':bool(both_closed and comparison['comparison_if_both_closed']=='distinct_numerically'),
                'body_points_run':False,'elapsed_seconds':time.monotonic()-began,'certificate_passed':False}
        for pin in card['source_files']+card['input_files']:verify(pin)
        dump(out/'result.json',result)
    except BaseException as error:
        dump(out/'scientific-failure.json',{'type':type(error).__name__,'message':str(error),
                                'completed_orbits':results,'certificate_passed':False})
        raise
    if not both_closed:raise SystemExit(1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--admission',required=True)
    a=ap.parse_args();card,admission=preflight(a.card,a.out,a.admission)
    out=Path(a.out);out.mkdir()
    dump(out/'startup.json',{'source_commit':admission['source_commit'],'card_sha256':sha(a.card),
         'admission_sha256':sha(a.admission),'target':card['target'],'pid':os.getpid(),
         'scientific_imports_started':False,'model_loaded':False,'prefix_evaluations_started':0})
    try:
        run_science(a,card,admission)
    except BaseException as error:
        dump(out/'failure.json',{'type':type(error).__name__,'message':str(error),
             'source_commit':admission['source_commit'],'certificate_passed':False,
             'scope':'Attempt failed or remained scientifically unqualified; completed raw files retained.'})
        raise

if __name__=='__main__':main()
