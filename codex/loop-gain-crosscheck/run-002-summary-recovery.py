"""Recover Task18 run002 summary from completed immutable events, not a rerun.

No model/producer import, matrix product, eigen/SVD evaluation or Git command.
Original exit1, truncated status and error stream remain authoritative failures.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

BASE=Path('/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck')
ORIGINAL=BASE/'run-002'
OUT=BASE/'run-002-summary-recovery-001'
SOURCE='7918af949fa60cb8bb123504e78c2dc959943d9c'
PLAN='98a01706d756e3271f3f566dedf792719bf088055ba084c753b199b649b198c4'
RECEIPT='4a3153f174bec42f034e245b63d5fe48978962908cbf82478ace3498b632b4fa'
CLOSURE='2d3a679a6dabde4379f95c56c66b303d9a82a5e1ddad6b76f4a3d7296c66b047'
SCIENCE={'codex/loop-gain-crosscheck/analysis.py':'a5553396985f947bb0d8e628f6f1190a683141b341b46538fdc06ceaeec9faf2',
 'codex/loop-gain-crosscheck/run.py':'80bf95a99738cc05e4d3480994051fc684666230a4e06aebca5719af1b1d23c0',
 'codex/crosscheck/qwen_map.py':'493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'}
CAPS=dict(cpu_threads=1,nice=10,memory_gib=1,timeout_seconds=120,maximum_output_mib=8,
          minimum_start_available_gib=20,minimum_runtime_available_gib=16)
R=dict(status='saved_summary_recovery_started',recovery_completed=False,checks=[],reconciliation=[],
       original_exit_success=False,certificate_passed=False,model_rerun=False,matrix_eigen_svd_rerun=False)
START=None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    def bad(s):raise ValueError('nonfinite JSON constant '+s)
    return json.loads(Path(path).read_text(),parse_constant=bad)


def check(value,name):
    R['checks'].append(dict(name=name,passed=bool(value)))
    if not value:raise ValueError(name)


def available():return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024


def tick():
    check(time.monotonic()-START<120,'recovery wall budget')
    check(available()>=16*1024**3,'runtime16GiB headroom')


def compare(left,right,name):
    a,b=np.asarray(left),np.asarray(right)
    check(a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),name+' finite shape')
    d=a-b;R['reconciliation'].append(dict(name=name,nonzero=int(np.count_nonzero(d)),max_absolute=float(np.max(np.abs(d)))if d.size else 0.))
    check(bool(np.all(np.abs(d)<=5e-15+5e-13*np.abs(b))),name+' saved scalar reconciliation')


def inventory(root,rows,extras=()):
    names=[r['path']for r in rows];check(len(names)==len(set(names)),'unique inventory '+str(root))
    for r in rows:
        p=root/r['path'];check(not Path(r['path']).is_absolute()and '..'not in Path(r['path']).parts and not p.is_symlink()and p.resolve()==p,'literal member '+r['path'])
        check(p.stat().st_size==r['bytes']and sha(p)==r['sha256'],'original member '+r['path'])
    check({str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}==set(names)|set(extras),'complete original inventory '+str(root))


def authenticate(card):
    check(card['admission']=='reviewed'and card['schema']=='task18-saved-summary-recovery-v1'and card['caps']==CAPS,'exact reviewed recovery card')
    check(card['script_sha256']==sha(__file__),'reviewed recovery source')
    check(card['source_commit']==SOURCE and card['plan_sha256']==PLAN and card['receipt_sha256']==RECEIPT and card['closure_sha256']==CLOSURE,'literal original authorities')
    check(sha(ORIGINAL/'receipt.json')==RECEIPT,'original failed receipt')
    receipt=read(ORIGINAL/'receipt.json');inventory(ORIGINAL,receipt['artifacts'],('receipt.json',))
    check(receipt['exit_code']==1 and receipt['status']=='loop_gain_failed'and receipt['timed_out']is False and receipt['worker_termination_verified']is True and receipt['uncertain_ownership']==[],'original failure and owned worker closure')
    check(receipt['source_commit']==SOURCE and receipt['plan_sha256']==PLAN and receipt['numerical_gain_result_available']is False,'original unsuccessful disposition retained')
    closure=BASE/'run-002-closure-001/report.json';check(sha(closure)==CLOSURE,'sealed original outer closure')
    authority=read(closure)
    check(authority['passed']is True and authority['task']==18 and authority['root']==str(ORIGINAL)and authority['tool_session_reaped_exit_code']==1 and authority['receipt_sha256']==RECEIPT and authority['source_commit']==SOURCE and authority['plan_sha256']==PLAN and authority['original_waiter_launcher_worker_pids_groups_sessions_absent']is True and authority['uncertain_ownership']==[],'original coordinator source/Git/process authority')
    inventory(BASE/'run-002-supervision',authority['supervision_files'])
    manifest=read(ORIGINAL/'manifest.json');check(sha(ORIGINAL/'manifest.json')==authority['manifest_sha256'],'original command manifest')
    check(manifest['source']==authority['source_files']and len(manifest['source'])==7,'seven original captured/Git source bindings')
    for row in manifest['source']:
        path=ORIGINAL/'source'/row['path'];check(sha(path)==row['sha256']and path.stat().st_size==row['bytes'],'captured source '+row['path'])
    for path,h in SCIENCE.items():check(sha(ORIGINAL/'source'/path)==h,'unchanged original scientific source '+path)
    planpath=ORIGINAL/'source/codex/loop-gain-crosscheck/run-002-plan.json';plan=read(planpath)
    check(sha(planpath)==PLAN and plan['source_sha256']==SCIENCE and plan['admission']=='reviewed','original admitted plan')
    check((ORIGINAL/'run/original-plan.json').read_bytes()==planpath.read_bytes(),'original worker plan bytes')
    binding=read(ORIGINAL/'run/input-binding.json');check(binding['source_commit']==SOURCE and binding['plan_sha256']==PLAN and binding['source_sha256']==SCIENCE,'original source/input binding')
    check(manifest['command']==['/data2/neural-shrimp/.venv/bin/python','-I','-B','-u',str(ORIGINAL/'source/codex/loop-gain-crosscheck/run.py'),'--plan',str(planpath),'--output',str(ORIGINAL/'run')],'exact original native command')
    partial=(ORIGINAL/'run/status.json').read_bytes();stderr=(ORIGINAL/'stderr.log').read_bytes()
    try:json.loads(partial)
    except json.JSONDecodeError:pass
    else:raise ValueError('expected truncated original status, not a completed status')
    check(partial.rstrip().endswith(b'"covariance_pass":'),'expected original covariance bool serialization position')
    check(all(marker in stderr.decode()for marker in card['expected_stderr_markers']),'original serialization and exclusive-file failure markers')
    check(len(card['expected_stderr_markers'])==3 and card['expected_stderr_markers'][1:] == ['FileExistsError','/run/status.json'],'literal disclosed failure class/path')
    check(len(partial)+len(stderr)<=512*1024,'failure-copy closing allowance')
    (OUT/'original-partial-status.json.bytes').write_bytes(partial);(OUT/'original-stderr.log').write_bytes(stderr)
    R['original_authority']=dict(source_commit=SOURCE,plan_sha256=PLAN,receipt_sha256=RECEIPT,closure_path=str(closure),closure_sha256=CLOSURE,
        original_exit_code=1,original_elapsed_seconds=receipt['elapsed_seconds'],
        original_peak_rss_kib=None,peak_note='Not recorded by accepted launcher or completed worker status; registration RSS is not peak.',
        original_backend_identity=None,backend_note='Final worker backend fields were after the truncated phase summary and are unavailable; current recovery versions are not substituted.',
        original_failure_preserved=True,partial_status_sha256=sha(ORIGINAL/'run/status.json'),stderr_sha256=sha(ORIGINAL/'stderr.log'),
        inherited_git_and_outer_closure='Inference original metadata authority accepted by Root, not new Git/process queries')
    R['original_input_binding']=binding
    return plan


def event_table():
    names=['model-loaded','prefix','fixed-start']+['transient-progress']*10+['orbit','orbit-gates']
    for i in range(8):names += ['phase'+str(i)+'-'+n for n in ['readout','covariance']+['basis']*56+['dense','factorized-dense']+['direction']*8+['spectra','phase-summary']]
    names+=['monodromy-phase'+str(i)for i in range(8)]+['composed-direction']*3+['composed-controls-summary']
    paths=sorted((ORIGINAL/'run').glob('[0-9][0-9][0-9][0-9]-*.json'))
    check([p.name for p in paths]==[f'{i:04d}-{name}.json'for i,name in enumerate(names)],'exact587 original event names/order')
    return {name:[p for p in paths if p.name[5:-5]==name]for name in set(names)}


def event(table,name):
    paths=table[name];check(len(paths)==1,'one completed event '+name)
    return read(paths[0])


def spectrum(ref,shape,dtype):
    allowed=(dtype,)if isinstance(dtype,str)else dtype
    check(set(ref)=={'path','dtype','shape','sha256'}and ref['shape']==list(shape)and ref['dtype']in allowed,'native spectrum reference schema')
    path=ORIGINAL/'run'/ref['path'];check(Path(ref['path']).name==ref['path']and sha(path)==ref['sha256'],'native spectrum member hash')
    a=np.load(path,allow_pickle=False,mmap_mode='r');check(a.shape==shape and str(a.dtype)==ref['dtype'] and np.isfinite(a).all(),'complete original native spectrum')
    return a


def recover(card):
    plan=authenticate(card);tick();table=event_table();orbit_event=event(table,'orbit');check(orbit_event['transient']==1000 and orbit_event['period']==8 and orbit_event['states']['shape']==[9,896]and orbit_event['states']['dtype']=='float64','original full orbit metadata');orbit=event(table,'orbit-gates')
    check(orbit['closure_pass']==(orbit['relative_closure']<=1e-10)and orbit['distinct_phases_pass']==(orbit['minimum_relative_phase_separation']>1e-8),'literal original orbit gates')
    phases=[];loops=[]
    for i in range(8):
        row=event(table,f'phase{i}-phase-summary');spectra=event(table,f'phase{i}-spectra')
        sv=spectrum(spectra['map_singular_values'],(896,),'float64');ev=spectrum(spectra['covariance_eigenvalues'],(896,),'float64')
        check(np.all(sv[:-1]>=sv[1:])and np.all(sv>=0)and np.all(ev[:-1]>=ev[1:]),'original ordered phase spectra')
        check((row['dimension'],row['vocabulary'],row['batch'],row['seed'])==(896,151936,16,18001+i),'full original phase definition')
        compare(row['first_two_step_singular_values'],sv[:2],'phase step singular summary'+str(i));compare(row['first_two_covariance_eigenvalues'],ev[:2],'phase covariance summary'+str(i))
        compare(row['step_second_first_ratio'],sv[1]/sv[0],'phase singular ratio'+str(i));compare(row['covariance_second_first_ratio'],ev[1]/ev[0],'phase covariance ratio'+str(i))
        dirs=row['directions'];check([d['index']for d in dirs]==list(range(8)),'all8 phase controls')
        raw_dirs=[read(p)for p in table[f'phase{i}-direction']]
        check(dirs==[{k:d[k]for k in ('index','factor_relative','dense_relative','hidden_dense_relative')}for d in raw_dirs],'native direction summary association')
        c=row['checks'];compare(c['factor_dense_relative'],event(table,f'phase{i}-factorized-dense')['relative_error'],'original factorized-dense scalar association'+str(i));compare(c['covariance_min_eigenvalue'],ev[-1],'phase covariance minimum'+str(i))
        expected=dict(factorization_pass=c['factor_dense_relative']<=1e-9 and all(d['factor_relative']<=1e-9 for d in dirs),
            dense_direction_pass=all(max(d['dense_relative'],d['hidden_dense_relative'])<=1e-9 for d in dirs),
            covariance_pass=bool(c['covariance_symmetry_relative']<=1e-12 and ev[-1]>=-1e-12*max(1.,ev[0])))
        check(all(type(c[k])is bool and c[k]==v for k,v in expected.items()),'converted native phase booleans'+str(i))
        phases.append(dict(index=i,**row));tick()
        loop=event(table,f'monodromy-phase{i}');eig=spectrum(loop['monodromy_eigenvalues'],(896,),('float64','complex128'));sing=spectrum(loop['monodromy_singular_values'],(896,),'float64')
        steps=spectrum(loop['step_singular_values'],(8,896),'float64')
        check(loop['start_phase']==i and np.all(np.abs(eig[:-1])>=np.abs(eig[1:]))and np.all(sing[:-1]>=sing[1:])and np.all(sing>=0),'original ordered cyclic-start spectra')
        compare(loop['product_step_norms'],np.prod(steps[:,0]),'saved scalar step-norm product'+str(i))
        loops.append(dict(start_phase=i,event_seconds=loop['seconds'],
            leading_multipliers=[dict(real=float(z.real),imag=float(z.imag),modulus=float(abs(z)))for z in eig[:8]],
            first_two_singular_values=sing[:2].tolist(),product_step_norms=loop['product_step_norms'],
            event_seconds_note='Original monodromy event timing; later intended final-summary timing is not recoverable exactly.'))
    composed=event(table,'composed-controls-summary');raw=[read(p)for p in table['composed-direction']]
    check(composed['rows']==[{k:r[k]for k in ('index','seed','period','derivative_relative','primal_relative','seconds')}for r in raw],'all3 composed native summaries')
    check([r['index']for r in raw]==[0,1,2]and all(r['seed']==18009 and r['period']==8 for r in raw),'original composed definition')
    check(composed['derivative_tolerance']==1e-8 and composed['primal_tolerance']==1e-12 and composed['passed']==all(r['derivative_relative']<=1e-8 and r['primal_relative']<=1e-12 for r in raw),'original composed precision gate')
    gates=bool(orbit['closure_pass']and orbit['distinct_phases_pass']and composed['passed']and all(p['checks']['forward_relative']<=1e-12 and all(p['checks'][k]for k in ('factorization_pass','dense_direction_pass','covariance_pass'))for p in phases))
    R.update(status='saved_summary_recovered',recovery_completed=True,original_exit_success=False,scientific_event_set_complete=True,
        saved_numerical_gates_pass=gates,orbit_gates=orbit,phases=phases,cyclic_monodromies=loops,composed_controls=composed,
        limitations=['Original attempt remains failed exit1; this separate recovery does not replace its status or receipt.',
        'Only completed JSON events and retained spectra summarized; no dense product, model, JVP, eigenvalue or SVD computation repeated.',
        'Historical asset postchecks are supported by original control flow reaching final serialization; large assets are not reloaded/rehashed here.',
        'Original backend and peak memory unavailable; launcher elapsed and recorded native timings retained under their actual definitions.',
        'Producer comparisons are not included; no producer numerical artifacts read. Numerical gates are not interval certification.'])
    tick()


def main():
    global START,np
    p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();card=read(a.card)
    check(card['admission']=='reviewed','prepared recovery refuses execution');check(a.output==OUT,'exclusive recovery output')
    a.output.mkdir(exist_ok=False);START=time.monotonic()
    try:
        check(os.getpriority(os.PRIO_PROCESS,0)==10,'external nice10');hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        check(hard==resource.RLIM_INFINITY or hard>=1024**3,'inherited1GiB AS');resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        import numpy as np
        check(np.__version__=='2.5.3','recovery NumPy2.5.3');check(available()>=20*1024**3,'startup20GiB headroom');recover(card)
    except BaseException as e:R.update(status='saved_summary_recovery_failed',recovery_completed=False,error=repr(e),traceback=traceback.format_exc())
    R.update(recovery_seconds=time.monotonic()-START,recovery_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        recovery_backend=dict(python=sys.version,numpy=np.__version__ if 'np'in globals()else None),recovery_source_sha256=sha(__file__),recovery_card_sha256=sha(a.card))
    raw=(json.dumps(R,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>7*1024**2:R['recovery_completed']=False;raw=json.dumps(dict(status='saved_summary_recovery_failed',recovery_completed=False,error='summary exceeds reserved7MiB')).encode()
    (OUT/'report.json').write_bytes(raw)
    files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(OUT.iterdir())if p.is_file()]
    (OUT/'manifest.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
    return 0 if R['recovery_completed']else 1

if __name__=='__main__':raise SystemExit(main())
