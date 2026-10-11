#!/usr/bin/env python3
"""One resource-screened fixed-g native p7 boundary localization pilot."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('refinement',ROOT/'tools/refine_clark_rnn.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);S=R.S
from neural_shrimp.clark_rnn import iterate
from neural_shrimp.clark_rnn_boundaries import augmented,qualify_boundary
PIN='ec4b2f7083b812d2cee38b287046ecde1f68ea5625ca7fb9d56da9eeadb47af2'
EXTRA=R.EXTRA+['src/neural_shrimp/clark_rnn_boundaries.py','tools/explore_clark_rnn_p7_boundaries.py','tools/launch_clark_rnn_p7_boundaries.py','docs/plans/clark-rnn-p7-boundaries.md']

def phase_match(a,b):
    scale=max(1,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    matches=[(float(np.max(np.abs(a-sign*np.roll(b,k,axis=0)))/scale),sign,k) for sign in [1,-1] for k in range(len(a))]
    error,sign,shift=min(matches);native=float(np.max(np.abs(a-b))/scale)
    return {'best_error':error,'best_sign':sign,'best_shift':shift,'native_error':native,'passed':sign==1 and shift==0 and native<.05}

class Worker:
    def __init__(self,out):self.out=out;self.start=time.monotonic();self.calls=0;self.seq=0;self.interrupted=False
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>90:raise TimeoutError('90s internal')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('worker448MiB')
        if self.calls>=9000:raise RuntimeError('9000 native return cap')
    def emit(self,kind,value):
        self.seq+=1;name=f'{self.seq:05d}-{kind}'
        arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
        S.save_npz(self.out/(name+'.npz'),**arrays);meta.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(self.out/(name+'.npz')));S.save(self.out/(name+'.json'),meta)
    def it(self,x,j,g,d,tag):
        self.check();self.calls+=1;t=time.monotonic();end,m,c,p=iterate(x,j,g,d,7);self.emit('native',{'tag':tag,'theta':np.array(x),'gain':g,'delta':d,'end':end,'monodromy':m,'cycle':c,'phase_jacobians':p});return end,m,c,p,time.monotonic()-t
    def aug(self,z,j,g,sign,tag):
        self.check();self.calls+=1;t=time.monotonic();h,dh,a=augmented(z,j,g,7,sign);self.emit('augmented',dict(a,h=h,dh=dh,gain=g,sign=sign,tag=tag));return h,dh,a,time.monotonic()-t
    def correct(self,x,j,g,d,tag):
        x=x.copy()
        for k in range(7):
            end,m,c,p,_=self.it(x,j,g,d,tag);res=end-x;scale=max(1,float(np.max(np.abs(c))));error=float(np.max(np.abs(res))/scale)
            if error<=1e-11:
                sep=float(np.max(np.abs(c[1]-c[0]))/scale)
                e,v=np.linalg.eig(m);self.emit('native_eigen',{'tag':tag,'gain':g,'delta':d,'theta':x,'monodromy':m,'eigenvalues':e,'eigenvectors':v,'closure':error,'divisor1':sep})
                if sep<=1e-6:return None,'collapsed'
                return {'theta':x,'cycle':c,'delta':d,'eigenvalues':e,'eigenvectors':v,'radius':float(np.max(np.abs(e))),'closure':error},'closed'
            if k==6:return None,'iteration_limit'
            try:dx=np.linalg.solve(m-np.eye(len(x)),-res)
            except np.linalg.LinAlgError:return None,'linear_failure'
            self.emit('correction_solve',{'tag':tag,'matrix':m-np.eye(len(x)),'rhs':-res,'direction':dx,'iteration':k})
            if not np.all(np.isfinite(dx)) or np.max(np.abs(dx))>.1*scale:return None,'step_guard'
            x=x+dx
    def localize(self,point,j,g,sign,tag):
        e,v=point['eigenvalues'],point['eigenvectors'];idx=int(np.argmin(np.abs(e-sign)))
        if abs(e[idx].imag)>1e-8 or abs(e[idx]-sign)>.7:return {'status':'no_close_real_multiplier','sign':sign,'tag':tag}
        q=v[:,idx].real;q/=np.linalg.norm(q);z=np.r_[point['theta'],point['delta'],q];center=z.copy()
        for k in range(12):
            h,dh,a,_=self.aug(z,j,g,sign,tag)
            if np.max(np.abs(h))<1e-10:
                self.check();self.calls+=1
                arrays,summary=qualify_boundary(z,j,g,7,sign,lambda kind,val:self.emit(kind,dict(val,tag=tag)))
                pm=phase_match(point['cycle'],arrays['cycle']);self.emit('boundary_identity',dict(pm,tag=tag,parent_cycle=point['cycle'],child_cycle=arrays['cycle']));summary['native_identity']=pm
                if not pm['passed']:summary['status']='boundary_identity_unresolved'
                self.emit('boundary_result',dict(arrays,**summary,gain=g,tag=tag));return dict(summary,gain=g,delta=float(z[4]),tag=tag)
            if k==11:break
            try:dx=np.linalg.solve(dh,-h)
            except np.linalg.LinAlgError:return {'status':'augmented_linear_failure','sign':sign,'tag':tag}
            self.emit('augmented_solve',{'tag':tag,'iteration':k,'matrix':dh,'rhs':-h,'direction':dx})
            new=z+dx
            if not np.all(np.isfinite(new)) or abs(new[4]-center[4])>.01 or not .5<new[4]<.95 or np.max(np.abs(new[:4]-center[:4]))>.2*max(1,float(np.max(np.abs(center[:4])))):
                return {'status':'augmented_guard','sign':sign,'tag':tag}
            z=new
        return {'status':'augmented_iteration_limit','sign':sign,'tag':tag}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.THREADS):raise ValueError('CPU1 nice10')
    a.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=a.out;S.RAW_CAP=128*1024**2;w=Worker(a.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    S.save(a.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__,'thread_environment':{k:os.environ.get(k) for k in R.THREADS},'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024})
    result={'status':'incomplete','slices':[]}
    try:
        if int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024<1024**3:raise MemoryError('less than1GiB headroom')
        # Non-target synthetic measurements; timings include durable returned arrays.
        j=np.eye(4)*.2+np.ones((4,4))*.1;x=np.arange(1,5)/10;z=np.r_[x,.5,[1.,0,0,0]];ta=[];tb=[]
        for i in range(10):ta.append(w.it(x,j,2.,.5,'synthetic_profile')[4]);tb.append(w.aug(z,j,2.,-1,'synthetic_profile')[3])
        projection=8000*max(ta)+96*max(tb)+10
        S.save(a.out/'profile.json',{'native_seconds':ta,'augmented_seconds':tb,'projected_seconds':projection,'gate':projection<=90,'target_inputs_unopened_before_profile':True})
        if not np.isfinite(projection) or projection>90:raise RuntimeError('resource profile failed')
        raw=R.RAW/'refine-p5-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('refinement pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('input hash')
        with np.load(raw/'run/p7/inputs.npz') as f:j=f['j'];gains=f['gains'];deltas=f['deltas']
        with np.load(raw/'run/p7/batch.npz') as f:periods=f['period_proposals'].reshape(7,4,2);final=f['final'].reshape(7,4,2,4)
        S.save(a.out/'input-binding.json',{'refinement_manifest':PIN,'source_input_sha256':S.sha(raw/'run/p7/inputs.npz'),'source_batch_sha256':S.sha(raw/'run/p7/batch.npz'),'gain_indices':[1,3,5],'delta_step':.001,'maximum_steps_per_direction':96,'selection':'middle sorted consensus7 delta index at each fixed gain; original start0; longest retained comparator horizon'})
        for gi in [1,3,5]:
            g=float(gains[gi]);inds=np.flatnonzero(np.all(periods[gi]==7,axis=1))
            if not len(inds):result['slices'].append({'gain':g,'status':'no_consensus7_seed'});continue
            di=int(inds[len(inds)//2]);d=float(deltas[di]);point,status=w.correct(final[gi,di,0],j,g,d,f'g{gi}-seed')
            if point is None:result['slices'].append({'gain':g,'status':status});continue
            report={'gain':g,'gain_index':gi,'delta_index':di,'seed_delta':d,'directions':[]}
            for direction in [-1,1]:
                last=point;endstatus='step_limit';tag=f'g{gi}-dir{direction}';boundary=None;steps=0
                for k in range(1,97):
                    dn=d+direction*k*.001
                    if not .5<dn<.95:endstatus='domain_edge';break
                    candidate,status=w.correct(last['theta'],j,g,dn,tag+f'-step{k}');steps=k
                    if candidate is None:
                        endstatus='correction_'+status;boundary=w.localize(last,j,g,1,tag+'-fold');break
                    pm=phase_match(last['cycle'],candidate['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag+f'-step{k}',parent_cycle=last['cycle'],child_cycle=candidate['cycle']))
                    if not pm['passed']:endstatus='phase_sign_identity';break
                    reverse,rs=w.correct(candidate['theta'],j,g,last['delta'],tag+f'-step{k}-reverse')
                    if reverse is None:endstatus='reverse_'+rs;break
                    scale=max(1,float(np.max(np.abs(last['cycle']))));err=float(np.max(np.abs(reverse['theta']-last['theta']))/scale)
                    w.emit('reverse_check',{'tag':tag+f'-step{k}','old':last['theta'],'reverse':reverse['theta'],'relative_error':err})
                    if err>1e-7:endstatus='reverse_mismatch';break
                    if candidate['radius']>=1:
                        # First loss of attraction on the carried branch: only the
                        # nearest +/-1 real multiplier is localized, not an NS surrogate.
                        ee=candidate['eigenvalues'];dist=[float(np.min(np.abs(ee-s))) for s in [1,-1]];sign=[1,-1][int(np.argmin(dist))]
                        endstatus='first_unstable';boundary=w.localize(candidate,j,g,sign,tag+'-boundary');break
                    last=candidate
                report['directions'].append({'direction':direction,'steps':steps,'stop':endstatus,'boundary':boundary,'last_closed_delta':last['delta'],'last_closed_radius':last['radius']});S.save(a.out/f'progress-g{gi}.json',report)
            result['slices'].append(report)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_native_calls=w.calls,scope='adapted numerical local boundary discovery; no interval or complete shrimp geometry claim');S.save(a.out/'result.json',result,final=True);S.manifest(a.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
