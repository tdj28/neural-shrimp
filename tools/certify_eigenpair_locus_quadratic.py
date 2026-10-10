#!/usr/bin/env python3
"""One profile-bound quadratic native eigenpair target at fixed32x width."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews'),str(ROOT/'tools')]
import certify_eigenpair_locus as B
import certify_eigenpair_locus_weighted as W
import profile_eigenpair_path as QP
from neural_shrimp.rigorous_eigenpair_path import quadratic_two_charts
from neural_shrimp.rigorous_eigenpair_locus import require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
SOURCES=list(dict.fromkeys(B.SOURCES+W.EXTRA+QP.SOURCES+[
 'tools/certify_eigenpair_locus_quadratic.py','tools/launch_eigenpair_locus_quadratic.py',
 'tests/test_eigenpair_locus_quadratic_adapter.py','docs/plans/eigenpair-quadratic-target.md',
 'docs/reviews/eigenpair-quadratic-source-review.md']))
PROFILE_ROOT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-quadratic-profile')
PROFILE_COMMIT='7aa2398ba429bc46beb63b3be395d27fd0076b9e'
PROFILE_PIN='810e8134971edca575034fe8e4cf01c0d60ef4bfdd242aab093a61ff90cda1c3'
PROFILE_RUN='b21736eba45d7373b049e705a7747774353b62611edf80d3eb7f776259934937'
SETTINGS={'period':18,'sigma':1,'halfwidth_exponent':-15,'radius_exponent':-18,'newton_steps':2,'precision':256}
entry=B.entry;read=B.read


def profile_gate(outcome,runtime,launcher):
    require(outcome['passed'] is True and outcome['target_acquired'] is False and runtime['mode']=='synthetic_only','quadratic profile success/scope')
    require(runtime['commit']==launcher['commit']==PROFILE_COMMIT and launcher['exit_code']==0 and not launcher['owned_live_pids'] and launcher['stop_reason'] is None,'profile source/exit')
    require(runtime['precision']==256 and runtime['nice']>=10 and runtime['threads']==dict.fromkeys(THREADS,'1') and runtime['tick_cap']==300,'profile settings')
    require(runtime['python']==sys.version and runtime['numpy']==np.__version__ and runtime['flint']==flint.__version__ and runtime['flint_library']==flint.__FLINT_VERSION__,'profile libraries')
    require(B.valid_resource(outcome['elapsed_seconds'],120) and B.valid_resource(outcome['peak_rss_kib'],2*1024**2) and B.valid_resource(launcher['elapsed_seconds'],142),'profile finite resources')
    require(outcome['ticks']==278 and outcome['event_counts']=={'chart_return':2,'curvature_proposal':2,'path_assembly_return':18,'path_phase_return':120,'phase_return':144,'preconditioner_return':2,'proposal':4},'profile exact operations')
    projection=QP.projection(outcome['chart_times'],outcome['mixed_times'],outcome['path_times'],outcome['peak_rss_kib'],outcome['stored_bytes_before_final'])
    require(projection==outcome['projection'] and projection['admitted'],'quadratic measured projection')
    require(projection['raw_bytes']<376*1024**2,'projection fits ordinary allocation plus8MiBfinalreserve')
    return projection


def bind_inputs():
    inputs,proof,bindings=B.bind_inputs()
    weights,weighted=W.bind_proposal(B.SOURCES)
    bindings['weighted_amendment']=weighted
    bindings['quadratic_profile']=B.inventory(PROFILE_ROOT,PROFILE_PIN)
    bindings['quadratic_profile_run']=B.inventory(PROFILE_ROOT/'run',PROFILE_RUN)
    rt=read(PROFILE_ROOT/'run/runtime.json')
    for name in QP.SOURCES:
        digest=hashlib.sha256(subprocess.check_output(['git','show',PROFILE_COMMIT+':'+name],cwd=ROOT)).hexdigest()
        require(entry(ROOT/name)==entry(PROFILE_ROOT/'run/source'/name)==rt['sources'][name] and digest==rt['sources'][name]['sha256'],'same quadratic-profile source '+name)
    bindings['quadratic_cost']=profile_gate(read(PROFILE_ROOT/'run/outcome.json'),rt,read(PROFILE_ROOT/'launcher-result.json'))
    bindings['adaptation']='fixed halfwidth2^-15,32x weighted-affine width; identical470radii and original whole fold root anchor; fresh quadratic residual and whole-domain MixedJet Jacobian'
    return inputs,proof,bindings,weights


def main(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];events={};serial=[0];total=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    def save(name,value,final=False):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);cap=(384 if final else 376)*1024**2;before=total[0]
        try:
            with path.open('xb') as raw:
                handle=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
                try:
                    buffer=[];size=0
                    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
                        buffer.append(piece);size+=len(piece)
                        if size>=65536:handle.write(''.join(buffer).encode());buffer=[];size=0;require(before+raw.tell()<cap,'aggregate output cap')
                    if buffer:handle.write(''.join(buffer).encode())
                finally:
                    if handle is not raw:handle.close()
                raw.flush();os.fsync(raw.fileno())
            require(before+path.stat().st_size<cap,'aggregate output cap')
        finally:
            if path.exists():total[0]=before+path.stat().st_size
    def tick():
        ticks[0]+=1
        if stop[0] or time.monotonic()-start>=120 or ticks[0]>288:raise TimeoutError('fixed target time/operation cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB monitored RSS cap')
    def observe(kind,value):
        events[kind]=events.get(kind,0)+1;save(f'events/{serial[0]:04d}-{kind}.json.gz',value);serial[0]+=1
        if stop[0] or time.monotonic()-start>=120:raise TimeoutError('deadline after saved return')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':'target','pid':os.getpid(),'settings':SETTINGS,'sources':{name:entry(ROOT/name) for name in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':{key:os.environ[key] for key in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'seconds_cap':120,'tick_cap':288,'rss_cap_kib':2*1024**2,'hard_virtual_address_bytes':4*1024**3,'ordinary_raw_cap_bytes':376*1024**2,'final_raw_cap_bytes':384*1024**2})
    for name in SOURCES:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'two quadratic normalized primitive18 +1 eigenpair charts; no attraction/genericity/projected-rate regularity or distant connection'}
    try:
        inputs,proof,bindings,weights=bind_inputs();save('bindings.json',bindings);save('parent-proof.json.gz',proof);save('exact-map-inputs.json',inputs['map_inputs'])
        with ctx.workprec(256):
            model=TwoRateCuspMap(inputs['map_inputs']);initial,anchor,witness=B.propagate_anchor(model,proof,inputs['map_inputs']['lr_output'],tick,observe);save('anchor.json.gz',witness)
            settings=dict(SETTINGS,weights=weights,normal=['0']*469+['1'],origin=initial,parameter='0',anchor=anchor)
            save('chart-inputs.json',{'initial':initial,**{k:v for k,v in settings.items() if k!='anchor'},'anchor_file':'anchor.json.gz'})
            result=quadratic_two_charts(model,initial,**settings,tick=tick,observe=observe);save('result.json.gz',result)
            outcome.update(passed=result['passed'],status='two_charts_passed' if result['passed'] else 'proof_gate_inconclusive',charts=[{'gates':r['gates'],'contraction':r['contraction'],'parameter_endpoints':r['parameter_endpoints'],'inclusions':sum(r['coordinate_inclusions'])} for r in result['charts']])
    except BaseException as exc:outcome.update(status='attempt_interrupted_or_failed',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0])
    if outcome['elapsed_seconds']>=120 or outcome['peak_rss_kib']>=2*1024**2:outcome['passed']=False
    save('outcome.json',outcome,True);save('manifest.json',{'files':{str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()}},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
