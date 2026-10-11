"""One saved-only mean/alternating residual diagnostic; no model imports/calls."""
import hashlib,json,math,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.cifar_phase_scaling import transform,scales,diagonal,dot
OUT=Path('/data2/neural-shrimp-data/explore/cifar/phase-scaling/diagnostic001/run')
SPECS=[('attempt-001','657eb4b4de1811ff23785125e170f522eee083d244626753f3129c4dcb3e2d32','920b829a459459903b9d45491243c008fa1a1c6178c00ee435736850a29b3f73','aad92e8a21f6c8f62d2397f7ec276ef8d3e20145',400),('attempt-002','6cbaa8aa34330a870c5cb135787325e7601a75c6fd91d8c663900a744e3ec661','2997623cd378c4726a6e710f6d0ef3a8146893cb9690833e3235170fc4f9c0fe','1e8d9abf07381e9bce200fe4ac77d7c29743abb1',1600)]
FILES=['src/neural_shrimp/cifar_phase_scaling.py','tools/diagnose_cifar_phase_scaling.py','tools/launch_cifar_phase_diagnostic.py','tests/test_cifar_phase_scaling.py','tests/test_cifar_phase_diagnostic.py','docs/plans/cifar-phase-scaling.md']

def need(v,m):
    if not v:raise ValueError(m)
def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):return json.loads(p.read_text())
def verify(root,pin,tick):
    need(entry(root/'manifest.json')['sha256']==pin,'literal immutable manifest');inv=read(root/'manifest.json')['files'];need(set(inv)=={p.name for p in root.iterdir()if p.is_file()and p.name!='manifest.json'},'complete flat inventory')
    for n,v in inv.items():tick();p=root/n;need(Path(n).name==n and not p.is_symlink()and not p.stat().st_mode&0o222 and entry(p)==v,'exact immutable inventory')
    return dict(root=str(root),manifest=pin,files=len(inv))
def norm(v):return math.sqrt(dot(v,v))
def parts(v):
    v=np.asarray(v,dtype=np.float64);q=transform(v);n=len(v)//2;total=dot(v,v);mean=dot(q[:n],q[:n]);alt=dot(q[n:],q[n:]);need(math.isclose(mean+alt,total,rel_tol=1e-12,abs_tol=1e-30),'orthogonal energy reconstruction')
    return dict(norm=math.sqrt(total),mean_norm=math.sqrt(mean),alternating_norm=math.sqrt(alt),mean_energy_fraction=mean/total if total else None,alternating_energy_fraction=alt/total if total else None)
def summary(z,az,r,eta,lam):
    need(z.shape==az.shape==(2*len(r),)and all(np.isfinite(v).all()for v in [z,az,r]),'complete finite two-phase arrays');w,scale=scales(eta,lam);rhs=np.r_[np.zeros(len(r)),-r];err=az-rhs
    return dict(scaling=scale,direction=parts(z),action=parts(az),rhs=parts(rhs),residual=parts(err),physical_shooting_relative_residual=norm(err)/norm(rhs),scaled_saved_residual=norm(diagonal(transform(err),w))/norm(diagonal(transform(rhs),w)),scope='Saved direction at its original state only. Phase residual decomposition, not separate R-I/S/R+I blocks, preconditioned convergence or new-state operator action.')

def main():
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed nice10');resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;result=dict(passed=False,native_calls=0,rows=[])
    def stop(*_):raise TimeoutError('30s diagnostic deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    def tick():need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<448*1024,'30s448MiB cap')
    def save(n,v,closing=False):
        nonlocal used
        b=(json.dumps(v,indent=2,allow_nan=False)+'\n').encode();need(used+len(b)<(2*1024**2 if closing else 1536*1024),'2MiB raw cap/final reserve');p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
        used+=len(b)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identities={n:entry(ROOT/n)for n in FILES};save('runtime.json',dict(source_commit=source,sources=identities,pid=os.getpid(),native_calls=0,python=sys.version,numpy=np.__version__))
        for n,v in identities.items():
            b=(ROOT/n).read_bytes();need(hashlib.sha256(subprocess.check_output(['git','show',source+':'+n],cwd=ROOT)).hexdigest()==v['sha256'],'exact committed source');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);used+=len(b)
        # Full-size unrelated array profile precedes all target arrays.
        before=time.monotonic();x=np.linspace(-1,1,197332);trial=summary(x,x/2,np.linspace(-.1,.1,98666),.00827753,.01);seconds=time.monotonic()-before;forecast=8*seconds+15;save('profile.json',dict(seconds=seconds,forecast_seconds=forecast,target_arrays_unopened=True));need(forecast<30,'measured diagnostic gate');del x,trial
        base=Path('/data2/neural-shrimp-data/explore/cifar/symmetric-shooting');prior_state=None;identity=None
        for name,pin,lpin,execution,iters in SPECS:
            root=base/name;launcher=base/(name+'-launcher');inventories=[verify(root,pin,tick),verify(launcher,lpin,tick)];old=read(root/'result.json');launch=read(launcher/'receipt.json')
            need(old['source_commit']==launch['source_commit']==execution and launch['successful']and not launch['owned_live_pids_remaining']and old['status']=='shooting_pilot_complete'and old['linear']['iterations']==iters and old['accepted_index']is None and not old['linear']['forcing_pass'],'literal retained failed Newton attempt')
            for pid in launch['observed_owned_pids']:need(not Path('/proc',str(pid)).exists(),'old owned process absent')
            for n,h in old['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',execution+':'+n],cwd=ROOT)).hexdigest()==h,'exact historical source')
            def array(n,size):
                tick();v=np.load(root/(n+'.npy'),allow_pickle=False);need(v.shape==(size,)and v.dtype==np.float64 and np.isfinite(v).all(),'same finite full vector');return v
            theta=array('initial-theta',98666);r=array('initial-residual',98666);z=array('shooting-direction',197332);az=array('shooting-action',197332);one=array('initial-map1',98666);two=array('initial-map2',98666);need(np.array_equal(r,two-theta),'literal original residual')
            ident=old['input_identity'];need(ident['eta']==.00827753 and ident['lam']==.01,'fixed actual binary64 ridge/rate')
            if prior_state is not None:need(np.array_equal(theta,prior_state)and ident==identity,'exact common original linearization')
            prior_state,identity=theta.copy(),ident;answer=summary(z,az,r,ident['eta'],ident['lam']);need(math.isclose(answer['physical_shooting_relative_residual'],old['linear']['shooting_relative_residual'],rel_tol=5e-12),'saved shooting norm replay')
            result['rows'].append(dict(attempt=name,iterations=iters,inventories=inventories,source=execution,summary=answer));save(name+'.json',result['rows'][-1])
        current=Path('/data2/neural-shrimp-data/explore/cifar/saved-shooting-merit/attempt-001');result['current_inventory']=verify(current,'d036dabfc82130137c4cdaad5249c836de2f960b82497c9740f4610b04668004',tick);done=read(current/'result.json');need(done['source_commit']=='56efc51180599bcfc31b306a4f9283d450f2675b'and done['accepted_index']==4 and done['last_accepted_state']=='trial-4-theta.npy','latest actual accepted state')
        current_x=np.load(current/'trial-4-theta.npy',allow_pickle=False);current_two=np.load(current/'trial-4-map2.npy',allow_pickle=False);need(current_x.shape==current_two.shape==(98666,)and current_x.dtype==current_two.dtype==np.float64 and np.isfinite(current_x).all()and np.isfinite(current_two).all(),'finite latest state')
        result.update(passed=True,current_state=dict(root=str(current),file=done['last_accepted_state'],entry=entry(current/done['last_accepted_state']),relative_closure=norm(current_two-current_x)/max(1,norm(current_x))),scope='Saved conditioning diagnosis only. Neither failed solve becomes accepted, no native derivative acquired or moved between states, no orbit claimed.')
    except BaseException as e:result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
