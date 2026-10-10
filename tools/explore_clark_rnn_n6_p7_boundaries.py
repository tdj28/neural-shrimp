#!/usr/bin/env python3
"""One resource-screened fixed-g N6 native p7 boundary localization pilot."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('refinement',ROOT/'tools/refine_clark_rnn_n6_p7.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);S=R.S
from neural_shrimp.clark_rnn import iterate
from neural_shrimp.clark_rnn_boundaries import augmented,qualify_boundary
PIN='8a783fe0150f4d531372b73aec839478d455880cf0846e908a13d37c0e37ccf0'
EXTRA=R.EXTRA+['src/neural_shrimp/clark_rnn_boundaries.py','tools/explore_clark_rnn_n6_p7_boundaries.py','tools/launch_clark_rnn_n6_p7_boundaries.py','docs/plans/clark-rnn-n6-p7-boundaries.md','tests/test_clark_rnn_n6_p7_boundaries.py']

def phase_match(a,b):
    scale=max(1,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    matches=[(float(np.max(np.abs(a-sign*np.roll(b,k,axis=0)))/scale),sign,k) for sign in [1,-1] for k in range(len(a))]
    error,sign,shift=min(matches);native=float(np.max(np.abs(a-b))/scale)
    return {'best_error':error,'best_sign':sign,'best_shift':shift,'native_error':native,'passed':sign==1 and shift==0 and native<.05}

def initial_attracting(point):
    return bool(np.isfinite(point['radius']) and point['radius']<1-1e-6)

def leading_boundary_sign(e):
    e=np.asarray(e)
    if not np.all(np.isfinite(e)):return None
    critical=e[int(np.argmax(np.abs(e)))]
    if abs(critical.imag)>1e-8:return None
    return 1 if critical.real>=0 else -1

class Worker:
    def __init__(self,out):self.out=out;self.start=time.monotonic();self.calls=0;self.seq=0;self.interrupted=False
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s internal')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('worker448MiB')
        if self.calls>=15000:raise RuntimeError('15000 native return cap')
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
            if not all(np.all(np.isfinite(a))for a in(end,m,c,p))or not np.isfinite(error):return None,'nonfinite_native'
            if error<=1e-11:
                separations={str(d):float(np.max(np.abs(c[d]-c[0]))/scale) for d in (1,)};sep=min(separations.values())
                e,v=np.linalg.eig(m);self.emit('native_eigen',{'tag':tag,'gain':g,'delta':d,'theta':x,'monodromy':m,'eigenvalues':e,'eigenvectors':v,'closure':error,'proper_divisor_separations':separations})
                if sep<=1e-6:return None,'collapsed'
                residual=float(np.max(np.abs(m@v-v*e[None,:]))/max(1,np.max(np.abs(m))))
                if not np.all(np.isfinite(e))or not np.all(np.isfinite(v))or not np.isfinite(residual)or residual>1e-10:return None,'eigensystem_unqualified'
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
        q=v[:,idx].real.copy();qnorm=float(np.linalg.norm(q))
        if not np.isfinite(qnorm) or qnorm<=1e-12:return {'status':'critical_vector_unqualified','sign':sign,'tag':tag}
        q/=qnorm;z=np.r_[point['theta'],point['delta'],q];center=z.copy()
        for k in range(12):
            h,dh,a,_=self.aug(z,j,g,sign,tag)
            if not np.all(np.isfinite(h)) or not np.all(np.isfinite(dh)):return {'status':'nonfinite_augmented','sign':sign,'tag':tag}
            if np.max(np.abs(h))<1e-10:
                self.check();self.calls+=1
                arrays,summary=qualify_boundary(z,j,g,7,sign,lambda kind,val:self.emit(kind,dict(val,tag=tag)))
                pm=phase_match(point['cycle'],arrays['cycle']);self.emit('boundary_identity',dict(pm,tag=tag,parent_cycle=point['cycle'],child_cycle=arrays['cycle']));summary['native_identity']=pm
                reverse,rs=self.correct(z[:len(j)],j,g,point['delta'],tag+'-reverse')
                scale=max(1,float(np.max(np.abs(point['cycle']))));err=float(np.max(np.abs(reverse['theta']-point['theta']))/scale) if reverse is not None else None
                self.emit('boundary_reverse',{'tag':tag,'seed':point['theta'],'reverse':reverse['theta'] if reverse is not None else np.full(len(j),np.nan),'status':rs,'relative_error':err})
                summary.update(reverse_status=rs,reverse_error=err,flip_cubic_criticality='not_evaluated' if sign==-1 else 'not_applicable')
                if not pm['passed'] or err is None or err>1e-7:summary['status']='boundary_identity_unresolved'
                self.emit('boundary_result',dict(arrays,**summary,gain=g,tag=tag));return dict(summary,gain=g,delta=float(z[len(j)]),tag=tag)
            if k==11:break
            try:dx=np.linalg.solve(dh,-h)
            except np.linalg.LinAlgError:return {'status':'augmented_linear_failure','sign':sign,'tag':tag}
            self.emit('augmented_solve',{'tag':tag,'iteration':k,'matrix':dh,'rhs':-h,'direction':dx})
            new=z+dx
            if not np.all(np.isfinite(new)) or abs(new[len(j)]-center[len(j)])>.01 or not .48<new[len(j)]<.67 or np.max(np.abs(new[:len(j)]-center[:len(j)]))>.2*max(1,float(np.max(np.abs(center[:len(j)])))):
                return {'status':'augmented_guard','sign':sign,'tag':tag}
            z=new
        return {'status':'augmented_iteration_limit','sign':sign,'tag':tag}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    a.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=a.out;S.RAW_CAP=128*1024**2;w=Worker(a.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    S.save(a.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__,'thread_environment':{k:os.environ.get(k) for k in R.M.THREADS},'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024})
    result={'status':'incomplete','slices':[]}
    try:
        if int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024<1024**3:raise MemoryError('less than1GiB headroom')
        # Non-target synthetic measurements; timings include durable returned arrays.
        j=np.eye(6)*.2+np.ones((6,6))*.1;x=np.arange(1,7)/10;z=np.r_[x,.5,[1.,0,0,0,0,0]];ta=[];tb=[]
        for i in range(10):ta.append(w.it(x,j,2.,.5,'synthetic_profile')[4]);tb.append(w.aug(z,j,2.,-1,'synthetic_profile')[3])
        projection=14000*max(ta)+96*max(tb)+10
        S.save(a.out/'profile.json',{'native_seconds':ta,'augmented_seconds':tb,'projected_seconds':projection,'gate':projection<=120,'target_inputs_unopened_before_profile':True})
        if not np.isfinite(projection) or projection>120:raise RuntimeError('resource profile failed')
        raw=R.RAW/'n6-p7-refinement-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('refinement pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('input hash')
        with np.load(raw/'run/inputs.npz') as f:j=f['j'];gains=f['gains'];deltas=f['deltas']
        with np.load(raw/'run/batch.npz') as f:periods=f['period_proposals'].reshape(51,51,2);final=f['final'].reshape(51,51,2,6)
        S.save(a.out/'input-binding.json',{'refinement_manifest':PIN,'source_input_sha256':S.sha(raw/'run/inputs.npz'),'source_batch_sha256':S.sha(raw/'run/batch.npz'),'gain_indices':[16,25,34],'delta_step':.001,'maximum_steps_per_direction':128,'selection':'middle sorted delta index within fixed largest component4/380cells at gain16/25/34; original start0'})
        comp=next(c for c in json.loads((raw/'run/components.json').read_text())['components'] if c['id']==4)
        if comp['period']!=7 or comp['count']!=380:raise ValueError('fixed refined component')
        for gi in [16,25,34]:
            g=float(gains[gi]);inds=np.array(sorted(i%51 for i in comp['cells'] if i//51==gi),dtype=int)
            if any(not np.all(periods[gi,di]==7) for di in inds):raise ValueError('component period join')
            if not len(inds):result['slices'].append({'gain':g,'status':'no_consensus7_seed'});continue
            di=int(inds[len(inds)//2]);d=float(deltas[di]);point,status=w.correct(final[gi,di,0],j,g,d,f'g{gi}-seed')
            if point is None:result['slices'].append({'gain':g,'status':status});continue
            if not initial_attracting(point):result['slices'].append({'gain':g,'status':'initial_seed_not_qualified_attracting','radius':point['radius']});continue
            report={'gain':g,'gain_index':gi,'delta_index':di,'seed_delta':d,'directions':[]};result['slices'].append(report)
            for direction in [-1,1]:
                last=point;endstatus='step_limit';tag=f'g{gi}-dir{direction}';boundary=None;steps=0
                for k in range(1,129):
                    dn=d+direction*k*.001
                    if not .48<dn<.67:endstatus='domain_edge';break
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
                        # First sampled loss at step .001 on the carried branch: only the
                        # leading real +/-1 multiplier is localized, not an NS surrogate.
                        sign=leading_boundary_sign(candidate['eigenvalues'])
                        if sign is None:endstatus='first_unstable_complex_unresolved';break
                        endstatus='first_unstable';boundary=w.localize(candidate,j,g,sign,tag+'-boundary');break
                    last=candidate
                report['directions'].append({'direction':direction,'steps':steps,'stop':endstatus,'boundary':boundary,'last_closed_delta':last['delta'],'last_closed_radius':last['radius']});S.save(a.out/f'progress-g{gi}.json',report)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_native_calls=w.calls,scope='adapted first-sampled-loss native7 discovery at delta step.001; narrow pockets unexcluded, no interval or shrimp claim');S.save(a.out/'result.json',result,final=True);S.manifest(a.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
