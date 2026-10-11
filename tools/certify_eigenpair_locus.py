#!/usr/bin/env python3
"""First two tiny native eigenpair-locus charts, anchored to the reviewed fold."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for key in THREADS:os.environ[key]='1'
import gzip,hashlib,json,math,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np,flint
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews'),str(ROOT/'tools')]
import profile_eigenpair_locus as PROFILE
from review_clark_rnn_c1_math import restore
from neural_shrimp.rigorous_eigenpair_locus import phase,two_charts,exact_mid,require
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,vector_json,matrix_json
SOURCES=PROFILE.SOURCES+['tools/certify_eigenpair_locus.py','tools/launch_eigenpair_locus.py',
 'tools/reviews/review_clark_rnn_c1_math.py','tests/test_eigenpair_locus_adapter.py','docs/plans/eigenpair-locus-first-charts.md',
 'certificates/s2-period18-fold-receipt.json','docs/reviews/s2-period18-fold-certificate-independent-review.json']
PROFILE_ROOT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-profile')
PROFILE_PIN='e2027e69fa4f33f80cb43cca407abdc54ab4c79e6fb128d8307c44bd5e0819e6'
PROFILE_RUN='92c12714fce63b0c42e8b4780357b9a556ff95ea356a201ade219d5f7bcb5760'
PROFILE_COMMIT='23b0a4629a62d91cbf5953e98a5a8dd2f571dac1'
PARENT=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-fold-2026-10-09/attempt-002-target')
PARENT_PIN='4473641efda1f5541fbaaa1126f70316df521618179a3faafd7d2cab8b6a97b3'
PARENT_RUN='9677cd6817d3e3cdfb45dcc10656a5660b79fec2b98aa7365ab268c705c36e5b'
PARENT_CERT='e894dedb29bafa2c054ff6eb5dfd6040cdc10711960c5321c341a32af92d8067'
PARENT_INPUT='4a5a8c53b51cd9d79c2f5ed760a53b3092c0003978033ab290486d75fcaed470'
PARENT_EXEC='c37dcb4e56a64fa7567cc2e03fe5917195a9f0fc'
RECEIPT='certificates/s2-period18-fold-receipt.json';RECEIPT_PIN='4a67a7bfb67155846e8cc3b4accf15fc9348c1218eb3057f783c5486cb84bf4c'
REVIEW='docs/reviews/s2-period18-fold-certificate-independent-review.json';REVIEW_PIN='87cfca558719d4e240703b5a59f027b27ab1aa7a9f7a0f8a70341d20e30062e0'
SETTINGS={'period':18,'sigma':1,'halfwidth_exponent':-20,'radius_exponent':-24,'newton_steps':2,'precision':256}

def entry(p):return PROFILE.entry(p)
def read(p):return json.loads(p.read_text())
def inventory(base,pin):
    require(entry(base/'manifest.json')['sha256']==pin,'manifest pin')
    recorded=read(base/'manifest.json')['files']
    actual={str(p.relative_to(base)):p for p in base.rglob('*') if p.is_file() and p!=base/'manifest.json'}
    require(set(recorded)==set(actual),'complete inventory')
    for name,path in actual.items():
        require(not path.is_symlink() and not path.stat().st_mode&0o222 and entry(path)==recorded[name],'immutable input '+name)
    return {'sha256':pin,'files':len(actual),'bytes':sum(v['bytes'] for v in recorded.values())}

def valid_resource(x,cap):return type(x) in (int,float) and math.isfinite(x) and 0<=x<cap

def profile_gate(outcome,runtime,launcher):
    require(outcome['passed'] is True and outcome['target_acquired'] is False and runtime['mode']=='synthetic_only','profile success/scope')
    require(runtime['commit']==launcher['commit']==PROFILE_COMMIT and launcher['exit_code']==0 and not launcher['owned_live_pids'],'profile source/exit')
    require(runtime['precision']==256 and runtime['nice']>=10 and runtime['threads']==dict.fromkeys(THREADS,'1'),'profile arithmetic/resource identity')
    require(runtime['python']==sys.version and runtime['numpy']==np.__version__ and runtime['flint']==flint.__version__ and runtime['flint_library']==flint.__FLINT_VERSION__,'profile library identity')
    require(valid_resource(outcome['elapsed_seconds'],120) and valid_resource(outcome['peak_rss_kib'],1024**2) and valid_resource(launcher['elapsed_seconds'],142),'profile finite resources')
    require(outcome['ticks']==194 and outcome['event_counts']=={'phase_return':180,'proposal':4,'preconditioner_return':2,'chart_return':2},'profile counts')
    p=PROFILE.projection(outcome['chart_times'],outcome['phase_times'],outcome['peak_rss_kib'],outcome['stored_bytes_before_final'])
    require(p==outcome['projection'] and p['resource_admitted'],'fixed cost projection')
    return p

def bind_inputs():
    bindings={'profile':inventory(PROFILE_ROOT,PROFILE_PIN),'profile_run':inventory(PROFILE_ROOT/'run',PROFILE_RUN)}
    runtime=read(PROFILE_ROOT/'run/runtime.json');outcome=read(PROFILE_ROOT/'run/outcome.json');launcher=read(PROFILE_ROOT/'launcher-result.json')
    for name in PROFILE.SOURCES:
        require(runtime['sources'][name]==entry(ROOT/name)==entry(PROFILE_ROOT/'run/source'/name),'same profiled source '+name)
        require(entry(ROOT/name)['sha256']==hashlib.sha256(subprocess.check_output(['git','show',PROFILE_COMMIT+':'+name],cwd=ROOT)).hexdigest(),'committed profile bytes')
    bindings['cost']=profile_gate(outcome,runtime,launcher)
    require(entry(ROOT/RECEIPT)['sha256']==RECEIPT_PIN and entry(ROOT/REVIEW)['sha256']==REVIEW_PIN,'canonical reviewed anchor pins')
    receipt,review=read(ROOT/RECEIPT),read(ROOT/REVIEW)
    require(receipt['passed'] and review['status']=='passed' and receipt['execution_commit']==review['producer_adaptive_execution_commit']==PARENT_EXEC,'prior proof/review')
    bindings['parent']=inventory(PARENT,PARENT_PIN);bindings['parent_run']=inventory(PARENT/'run',PARENT_RUN)
    bindings['review']=inventory(Path(review['raw_root']),review['raw_manifest_sha256'])
    bindings['review_run']=inventory(Path(review['raw_root'])/'run',review['run_manifest_sha256'])
    require(entry(PARENT/'run/certificate.json')['sha256']==PARENT_CERT and entry(PARENT/'run/inputs.json')['sha256']==PARENT_INPUT,'parent proof/input pins')
    certificate=read(PARENT/'run/certificate.json');inputs=read(PARENT/'run/inputs.json');proof=certificate['result']
    require(certificate['inputs_sha256']==PARENT_INPUT and inputs['period']==18 and proof['passed'] and all(proof['gates'].values()),'original primitive fold gates')
    require(certificate['runtime']['source_commit']==PARENT_EXEC,'original source execution')
    for name,digest in certificate['runtime']['source_sha256'].items():
        require(entry(PARENT/'run/source'/name)['sha256']==digest==hashlib.sha256(subprocess.check_output(['git','show',PARENT_EXEC+':'+name],cwd=ROOT)).hexdigest(),'parent source archive '+name)
    for name in ('src/neural_shrimp/rigorous_orbits.py','src/neural_shrimp/rigorous_folds.py'):
        require(entry(ROOT/name)['sha256']==certificate['runtime']['source_sha256'][name],'same exact native objective '+name)
    require(inputs['map_inputs']['lr_output']==receipt['fixed_output_rate']=='1848862009810323/2251799813685248','exact fixed output rate')
    require(len(proof['box'])==27 and len(inputs['map_inputs']['x'])==24,'fixed anchor dimensions/data')
    bindings.update(parent_receipt_sha256=RECEIPT_PIN,independent_review_sha256=REVIEW_PIN,parent_certificate_sha256=PARENT_CERT,parent_inputs_sha256=PARENT_INPUT)
    return inputs,proof,bindings


def propagate_anchor(model,proof,output_rate,tick,observe):
    """Whole proved root box, not midpoint: correlation loss only enlarges."""
    box=[restore(v) for v in proof['box']];n=model.n
    require(n==13 and len(box)==2*n+1,'original normalized root dimensions')
    state,hidden,mode=box[:n],box[n],box[n+1:];output=exact(output_rate)
    require(output.is_exact() and output>0 and hidden>0,'exact positive rates')
    initial_state,initial_mode=state[:],mode[:];full=[]
    for k in range(18):
        full+=state+mode;tick();value,block,rates=phase(model,state,mode,[hidden,output])
        observe('anchor_phase_return',{'phase':k,'input':vector_json(state+mode+[hidden,output]),'value':matrix_json(value),'block':matrix_json(block),'rates':matrix_json(rates)})
        require(all(value[i,0].is_finite() for i in range(2*n)),'finite propagated anchor')
        state=[value[i,0] for i in range(n)];mode=[value[n+i,0] for i in range(n)]
    require(all((state[i]-initial_state[i]).contains(0) and (mode[i]-initial_mode[i]).contains(0) for i in range(n)),'prior orbit/eigenpair closure consistent with propagation')
    full += [hidden,output]
    anchor={'box':full,'period':18,'sigma':1,'dimension':13,'exact_parameter':'0'}
    return [exact_mid(v) for v in full],anchor,{'full_root_box':vector_json(full),'terminal_state':vector_json(state),'terminal_mode':vector_json(mode),'source_box':proof['box'],'scope':'same known exact root; whole-box propagation loses correlations conservatively; no midpoint substitution'}


def main(out):
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stop=[False];ticks=[0];events={};serial=[0];total=[0]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    def save(name,value,final=False):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);cap=(256 if final else 248)*1024**2
        # Stream JSON: retained partial files remain inventoried after cap/interruption.
        before=total[0]
        try:
            with path.open('xb') as raw:
                handle=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) if name.endswith('.gz') else raw
                try:
                    buffer=[];size=0
                    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
                        buffer.append(piece);size+=len(piece)
                        if size>=65536:
                            handle.write(''.join(buffer).encode());buffer=[];size=0
                            require(before+raw.tell()<cap,'aggregate output cap')
                    if buffer:handle.write(''.join(buffer).encode())
                finally:
                    if handle is not raw:handle.close()
                raw.flush();os.fsync(raw.fileno())
            require(before+path.stat().st_size<cap,'aggregate output cap')
        finally:
            if path.exists():total[0]=before+path.stat().st_size
    def tick():
        ticks[0]+=1
        if stop[0] or time.monotonic()-start>=120 or ticks[0]>216:raise TimeoutError('fixed target time/operation cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=2*1024**2:raise MemoryError('2GiB monitored RSS cap')
    def observe(kind,value):
        events[kind]=events.get(kind,0)+1
        save(f'events/{serial[0]:04d}-{kind}.json.gz',value);serial[0]+=1
        if stop[0] or time.monotonic()-start>=120:raise TimeoutError('deadline after saved return')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save('runtime.json',{'commit':commit,'mode':'target','pid':os.getpid(),'settings':SETTINGS,'sources':{name:entry(ROOT/name) for name in SOURCES},'python':sys.version,'numpy':np.__version__,'flint':flint.__version__,'flint_library':flint.__FLINT_VERSION__,'threads':{key:os.environ[key] for key in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0),'seconds_cap':120,'tick_cap':216,'rss_cap_kib':2*1024**2,'hard_virtual_address_bytes':4*1024**3})
    for name in SOURCES:
        p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());total[0]+=p.stat().st_size
    outcome={'passed':False,'status':'unresolved','scope':'two tiny charts of the normalized primitive18 +1 eigenpair locus; no attraction/genericity/distant connection'}
    try:
        inputs,proof,bindings=bind_inputs();save('bindings.json',bindings);save('parent-proof.json.gz',proof);save('exact-map-inputs.json',inputs['map_inputs'])
        with ctx.workprec(256):
            model=TwoRateCuspMap(inputs['map_inputs'])
            initial,anchor,witness=propagate_anchor(model,proof,inputs['map_inputs']['lr_output'],tick,observe)
            save('anchor.json.gz',witness)
            settings=dict(SETTINGS,normal=['0']*469+['1'],origin=initial,parameter='0',anchor=anchor)
            save('chart-inputs.json',{'initial':initial,**{k:v for k,v in settings.items() if k!='anchor'},'anchor_file':'anchor.json.gz'})
            result=two_charts(model,initial,**settings,tick=tick,observe=observe)
            save('result.json.gz',result)
            outcome.update(passed=result['passed'],status='two_charts_passed' if result['passed'] else 'proof_gate_inconclusive',charts=[{'gates':r['gates'],'contraction':r['contraction'],'parameter_endpoints':r['parameter_endpoints'],'inclusions':sum(r['coordinate_inclusions'])} for r in result['charts']])
    except BaseException as exc:outcome.update(status='attempt_interrupted_or_failed',failure={'type':type(exc).__name__,'message':str(exc)})
    outcome.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks[0],event_counts=events,stored_bytes_before_final=total[0])
    if outcome['elapsed_seconds']>=120 or outcome['peak_rss_kib']>=2*1024**2:outcome['passed']=False
    save('outcome.json',outcome,True)
    inv={str(p.relative_to(out)):entry(p) for p in sorted(out.rglob('*')) if p.is_file()};save('manifest.json',{'files':inv},True)
    for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(outcome));return 0 if outcome['passed'] else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
