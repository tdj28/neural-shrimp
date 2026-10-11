"""Four fixed historical saved action-space diagnostics; no model acquisitions."""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,hashlib,io,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
from neural_shrimp.cifar_krylov_conditioning import diagnose,SETTINGS
import review_cifar_stationary_merit as U
OUT=Path('/data2/neural-shrimp-data/audits/cifar-krylov-conditioning/attempt001/run')
CARD='docs/plans/cifar-krylov-conditioning-inputs.json'
FILES=['tools/diagnose_cifar_krylov.py','tools/launch_cifar_krylov.py','tools/reviews/review_cifar_stationary_merit.py','src/neural_shrimp/cifar_krylov_conditioning.py','src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py','tests/test_cifar_krylov_conditioning.py','tests/test_cifar_krylov_driver.py','docs/plans/cifar-krylov-conditioning.md',CARD]
need=U.need;entry=U.entry;read=U.read
LIMITS=dict(seconds=180,rss_kib=2*1024**2,as_bytes=6*1024**3,raw_bytes=64*1024**2)

def events_index(events,inventory):
    result={'hvp':{},'direct_gradient':{}};i=0
    def record(row):
        name=row['file'];need(Path(name).name==name and inventory['run/'+name]==row['identity'],'exact event inventory array');return name
    while i<len(events):
        e=events[i]
        if e['kind']!='native_input':need(e['kind']in ['stationary_iteration','stationary_linear','stationary_trial'],'no stray native event');i+=1;continue
        block=events[i:i+4];need(len(block)==4 and [x['kind']for x in block]==['native_input','operation_started','operation_returned','native_output'],'full native event order');op=e['operation'];idx=e['index'];need(op in result and idx==len(result[op])+1,'complete ordered native indices')
        need(all(x['operation']==op and x.get('index',idx)==idx for x in block),'same operation/index throughout');need(np.isfinite(block[2]['seconds'])and block[2]['seconds']>=0,'native duration')
        row=dict(state=record(e['state']),output=record(block[3]['result']));need(row['output']==f'native-{op}-{idx:04d}-return.npy','literal output index')
        if op=='hvp':
            row['direction']=record(e['direction']);need(row['direction']==f'native-hvp-{idx:04d}-vector.npy','literal direction index')
        else:need('direction'not in e,'gradient not an action')
        result[op][idx]=row;i+=4
    need({k:len(v)for k,v in result.items()}==dict(hvp=1030,direct_gradient=23),'all original1053 actual calls')
    return result

def bind(card,tick,save):
    p=Path(card['producer']['root']);a=Path(card['independent']['root']);need(p==Path('/data2/neural-shrimp-data/explore/cifar/stationary-seed/continuation001')and a==Path('/data2/neural-shrimp-data/audits/cifar-stationary-continuation/attempt002'),'literal four-step parent/audit')
    inv=U.inventory(p,card['producer']['outer'],tick);U.inventory(p/'run',card['producer']['worker'],tick);U.inventory(a,card['independent']['outer'],tick);U.inventory(a/'run',card['independent']['worker'],tick)
    row=read(p/'run/result.json');U.source(p,row,card['producer']['source']);rt=read(a/'run/runtime.json');proof=read(a/'run/outcome.json');ib=read(a/'run/input-binding.json')
    need(rt['source_commit']==card['independent']['source']and proof['passed']and proof['accepted_steps']==4 and proof['native_calls']==0,'completed independent four-step acceptance')
    need(ib['parent']==str(p)and ib['pins']==[card['producer']['outer'],card['producer']['worker']]and ib['source']==card['producer']['source'],'audit of literal selected four-step source')
    for n,v in rt['files'].items():need(entry(a/'run/source'/n)==v and hashlib.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'literal independent audit source '+n)
    for path,source in [(p,card['producer']['source']),(a,card['independent']['source'])]:
        owner=read(path/'launcher-result.json');need(owner['source_commit']==source and owner['exit_code']==0 and owner['failure']is None and not owner['owned_live_pids']and owner['worker_pid_absent'],'successful owned predecessor')
        for key in ['launcher_pid','owned_group_pid']:need(not Path('/proc',str(owner[key])).exists(),'actual predecessor exited')
    need(row['status']=='stationary_continuation_complete'and row['comparison_qualified']and not row['seed_qualified']and row['mode']is None and not row['accepted_orbit']and not row['rigorous_certificate'],'preserve numerical nonstationary scope')
    need(row['settings']==card['original_settings']and row['input_identity']==card['original_input_identity']and row['runtime']['numpy']==np.__version__,'literal original objective/settings and same array arithmetic runtime')
    need(entry(p/'run/minres-source.py')==card['minres_source']and card['minres_source']['sha256']=='fc5ff3fad43999a4f956439a1e783356a686021337809ac26b6aca74b72374b3','literal unpreconditioned saved SciPy recurrence')
    need(row['operation_attempted']==row['operation_returned']==dict(action=0,j2vp=0,hvp=1030,direct_gradient=23),'complete original counts')
    events=[json.loads(s)for s in(p/'run/events.jsonl').read_text().splitlines()];index=events_index(events,inv)
    expected=[dict(state=f'state-{s:03d}.npy',gradient=g,first_hvp=h,true_hvp=h+256)for s,g,h in [(0,1,3),(10,11,260),(14,15,517),(18,19,774)]];need(card['operators']==expected,'fixed four historical operators')
    for unit in expected:
        need(index['direct_gradient'][unit['gradient']]['state']==unit['state'],'gradient belongs to same exact state')
        for j in range(unit['first_hvp'],unit['true_hvp']+1):need(index['hvp'][j]['state']==unit['state'],'every saved action uses exact current point')
    need(all(x['state']!='state-022.npy'for x in index['hvp'].values()),'no action authority at final state022')
    save('input-binding.json',dict(card=entry(ROOT/CARD),producer=card['producer'],independent=card['independent'],original_input_identity=row['input_identity'],minres_source=card['minres_source'],operators=expected,native_event_count=len(events),scope='All recorded derivative outputs shared; four exact old states and no current-final-state action.'))
    return p,index,proof

def profile(tick,save):
    start=time.monotonic();rng=np.random.default_rng(47021);n,m=98666,256
    save('profile-input.json',dict(seed=47021,rows=n,columns=m,formula='V=standard_normal/sqrt(n);W=linspace(.5,2,n)[:,None]*V;g=-V[:,0];p=V[:,1];hp=W[:,1]',settings=SETTINGS))
    V=rng.standard_normal((n,m))/np.sqrt(n);W=V*np.linspace(.5,2,n)[:,None];g=-V[:,0].copy();p=V[:,1].copy();hp=W[:,1].copy();r,small,span=diagnose(V,W,g,p,hp,tick);seconds=time.monotonic()-start;forecast=8*seconds+30
    save('profile.json',dict(passed=forecast<180,seconds=seconds,forecast_seconds=forecast,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_shape=[n,m],target_arrays_unopened=True));need(forecast<180,'twofold complete four-unit measured admission');del V,W,g,p,hp,r,small,span;gc.collect()

def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean nice10 source')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0)
    def tick():need(time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'180s2GiB owned budget')
    def stop(*_):raise TimeoutError('saved conditioning deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,180)
    def raw(name,b,closing=False):
        nonlocal used
        need(used+len(b)<64*1024**2-(0 if closing else 2*1024**2),'64MiB final reserve');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        used+=len(b)
    def save(name,v,closing=False):raw(name,(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode(),closing)
    def array(name,v):b=io.BytesIO();np.save(b,v,allow_pickle=False);raw(name,b.getvalue())
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();card=read(ROOT/CARD);need(card['status']=='reviewed'and card['settings']==SETTINGS and card['limits']==LIMITS,'fixed reviewed scientific/cost card');save('runtime.json',dict(source_commit=source,python=sys.version,numpy=np.__version__,pid=os.getpid(),threads={k:os.environ[k]for k in THREADS},files={n:entry(ROOT/n)for n in FILES}))
        for n in FILES:
            b=(ROOT/n).read_bytes();need(b==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact own source');raw('source/'+n,b)
        profile(tick,save);parent,index,audit=bind(card,tick,save);rows=[]
        for k,op in enumerate(card['operators']):
            tick();save(f'unit{k}/selection.json',op);V=np.empty((98666,256));W=np.empty_like(V)
            for j,i in enumerate(range(op['first_hvp'],op['true_hvp'])):
                tick();V[:,j]=U.vector(parent/'run'/index['hvp'][i]['direction']);W[:,j]=U.vector(parent/'run'/index['hvp'][i]['output'])
            g=U.vector(parent/'run'/index['direct_gradient'][op['gradient']]['output']);p=U.vector(parent/'run'/index['hvp'][op['true_hvp']]['direction']);hp=U.vector(parent/'run'/index['hvp'][op['true_hvp']]['output']);r,small,span=diagnose(V,W,g,p,hp,tick)
            U.close(r['actual_forcing'],audit['steps'][k]['forcing'],'independently audited true forcing');save(f'unit{k}/diagnostic.json',r)
            for name,v in small.items():array(f'unit{k}/{name}.npy',v)
            array(f'unit{k}/span-prediction-direction.npy',span);rows.append(dict(unit=k,state=op['state'],diagnostic=entry(OUT/f'unit{k}/diagnostic.json'),actual_forcing=r['actual_forcing'],span_predicted_forcing=r['span_prediction']['forcing'],qualified_ritz_prefixes=[x['columns']for x in r['prefixes']if x['ritz_qualified']],direction_action_qualified=r['actual_direction_representation']['qualified']))
            del V,W,g,p,hp,r,small,span;gc.collect()
        result.update(passed=True,units=rows,scope='Saved numerical conditioning only; no new derivative, no currentstate022 authority, no new step/seed/orbit. All span predictions require fresh physical actions before any application.')
    except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
