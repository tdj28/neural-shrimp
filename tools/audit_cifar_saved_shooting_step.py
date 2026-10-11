"""Saved-only independent check of the accepted CIFAR residual-merit step."""
import json,hashlib,os,resource,signal,subprocess,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import audit_cifar_merit_saved as A
PARENT=Path('/data2/neural-shrimp-data/explore/cifar/saved-shooting-merit/attempt-001')
PIN='d036dabfc82130137c4cdaad5249c836de2f960b82497c9740f4610b04668004'
LPIN='4afb9c18d98472e029f1e3fd4f528c610cb278fc3fc7624a04b38a7bb60ed18f'
EXEC='56efc51180599bcfc31b306a4f9283d450f2675b'
OUT=Path('/data2/neural-shrimp-data/audits/cifar-saved-shooting-merit/attempt001/run')
SOURCES=['tools/audit_cifar_merit_saved.py','tools/audit_cifar_saved_shooting_step.py','tools/launch_cifar_saved_shooting_step_audit.py','tests/test_cifar_saved_step_audit.py','docs/plans/cifar-saved-step-audit.md']

def direction(theta,r,p,ap):
    forcing=A.norm(r+ap)/A.norm(r);slope=A.dot(r,ap);prediction=(A.dot(r,r)-A.dot(r+ap,r+ap))/2;size=A.norm(p)/max(1,A.norm(theta))
    return dict(true_forcing=forcing,true_slope=slope,full_predicted_reduction=prediction,relative_direction_norm=size,old_newton_forcing_pass=bool(forcing<=.5),passed=bool(slope<0 and prediction>0 and size<=.1))

def main():
    A.need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');A.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0
    def stop(*_):raise TimeoutError('30s saved audit')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,30)
    def check():A.need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<448*1024,'30s448MiB')
    def save(name,value):
        nonlocal used
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();used+=len(raw);A.need(used<2*1024**2,'2MiB output')
        with (OUT/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
    result=dict(passed=False,native_calls=0,trials=[])
    try:
        sources={n:A.sha(ROOT/n)for n in SOURCES}
        for n,h in sources.items():A.need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==h,'committed source')
        save('runtime.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=sources,pid=os.getpid(),native_calls=0))
        result['inventories']=[A.verify(PARENT,PIN,check),A.verify(PARENT.with_name(PARENT.name+'-launcher'),LPIN,check)]
        source=A.read(PARENT/'result.json');launch=A.read(PARENT.with_name(PARENT.name+'-launcher')/'receipt.json')
        A.need(source['source_commit']==launch['source_commit']==EXEC and launch['successful']and launch['owned_live_pids_remaining']==[]and source['status']=='saved_merit_complete'and source['comparison_qualified'],'exact completed unit')
        for name,h in source['source_sha256'].items():A.need(hashlib.sha256(subprocess.check_output(['git','show',EXEC+':'+name],cwd=ROOT)).hexdigest()==h,'historical source')
        counts={k:Counter()for k in ['operation_started','operation_returned']}
        for ev in map(json.loads,(PARENT/'events.jsonl').read_text().splitlines()):
            if ev['kind']in counts:
                counts[ev['kind']][ev['operation']]+=1;A.need(ev['index']==counts[ev['kind']][ev['operation']],'complete operation order')
        expected=dict(hvp=2,direct_gradient=12)
        A.need(all(dict(v)==expected for v in counts.values()),'complete twoHVP twelvegradient inventory')
        for field in ['operation_attempted','operation_returned']:A.need({k:v for k,v in source[field].items()if v}==expected,'saved operation agreement')
        def read(n):
            check();v=np.load(PARENT/(n+'.npy'),allow_pickle=False);A.need(v.shape==(98666,)and v.dtype==np.float64 and np.isfinite(v).all(),'native vector');return v
        theta,one,two,r,p,ap=[read(n)for n in ['initial-theta','initial-map1','initial-map2','residual','direction','actual-action']]
        A.need(np.array_equal(r,two-theta),'actual residual')
        prior=Path(source['input_identity']['parent_root']);result['inventories'].append(A.verify(prior,source['input_identity']['parent_manifest'],check))
        for name,v in zip(['initial-theta','initial-map1','initial-map2','direction','actual-action'],[theta,one,two,p,ap]):A.need(np.array_equal(v,np.load(prior/(name+'.npy'),allow_pickle=False)),'exact retained parent '+name)
        gate=direction(theta,r,p,ap)
        for key,v in gate.items():A.need(source['direction_gate'][key]is v if isinstance(v,bool)else A.close(source['direction_gate'][key],v),'independent direction '+key)
        A.need(gate['passed']and not gate['old_newton_forcing_pass']and source['accepted_index']==4 and len(source['trials'])==5,'failedNewton fixed merittest')
        for i,saved in enumerate(source['trials']):
            check();alpha=2.**(-i);x,y,z=[read(f'trial-{i}-'+n)for n in ['theta','map1','map2']]
            A.need(saved['alpha']==alpha and saved['index']==i and np.array_equal(x,theta+alpha*p),'literal trial order/update')
            fresh=A.metrics(x,y,z,r,ap,alpha)
            for key,v in fresh.items():A.need(saved[key]is v if isinstance(v,bool)else A.close(saved[key],v),'independent trial '+key)
            A.need(fresh['accepted_local_step']==(i==4)and not saved['accepted_orbit'],'exact firstacceptance no orbit')
            result['trials'].append(dict(index=i,alpha=alpha,**fresh));save(f'trial-{i}.json',result['trials'][-1])
        closure=A.norm(z-x)/max(1,A.norm(x));sep=A.norm(y-x)/max(1,A.norm(x))
        A.need(A.close(closure,source['final_relative_residual'])and A.close(sep,source['final_relative_separation'])and source['last_accepted_state']=='trial-4-theta.npy'and not source['accepted_orbit']and not source['spectrum_requested'],'final state and negative orbit labels')
        result.update(passed=True,direction_gate=gate,final_relative_residual=closure,final_relative_separation=sep,accepted_alpha=1/16,closure_screen=closure<1e-8,operation_counts=expected)
    except BaseException as e:
        import traceback;result.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Independent saved fsum norms/merit/forcing/trial-order/action bindings; shared acquired native maps/HVP, no model call or exact orbit proof.');save('result.json',result);save('manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=A.sha(p))for p in OUT.iterdir()if p.is_file()}))
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
