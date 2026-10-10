#!/usr/bin/env python3
"""One resource-screened fixed-g N12 native p6 boundary localization pilot."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('refinement',ROOT/'tools/refine_clark_rnn_n12_p6.py');R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R);S=R.S
from neural_shrimp.clark_rnn import iterate
from neural_shrimp.clark_rnn_boundaries import augmented,qualify_boundary
PIN='ed58f64bbfa4513775f2ed2e9e0935da6902b7b6f151e8d72ca176210b6e7674'
EXTRA=R.EXTRA+['src/neural_shrimp/clark_rnn_boundaries.py','tools/explore_clark_rnn_n12_p6_boundaries.py','tools/launch_clark_rnn_n12_p6_boundaries.py','docs/plans/clark-rnn-n12-p6-boundaries.md','tests/test_clark_rnn_n12_p6_boundaries.py']

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

def resource_projection(native,augmented_times,eigen,solve,sizes,peak_kib):
    timings=native+augmented_times+eigen+solve
    if not timings or not all(np.isfinite(t)and t>0 for t in timings)or not np.isfinite(peak_kib)or peak_kib<=0:raise ValueError('finite positive profile')
    if set(sizes)!={'native','augmented','eigen','solve'}or any(not vals or any(v<=0 for v in vals)for vals in sizes.values()):raise ValueError('positive profile bytes')
    seconds=2*(14000*(max(native)+max(eigen)+max(solve))+96*(max(augmented_times)+4*max(eigen)+max(solve)))+20
    raw=14000*(max(sizes['native'])+max(sizes['eigen'])+max(sizes['solve'])+1024)+96*(max(sizes['augmented'])+4*max(sizes['eigen'])+max(sizes['solve'])+4096)+16*1024**2
    peak=peak_kib*1024+288*1024**2+64*1024**2
    return {'projected_seconds':seconds,'projected_raw_bytes':raw,'projected_peak_bytes':peak,'safety_factor':2,'gate':bool(seconds<=120 and raw<=384*1024**2 and peak<=448*1024**2)}

class Worker:
    def __init__(self,out):self.out=out;self.start=time.monotonic();self.calls=0;self.native_calls=0;self.augmented_calls=0;self.seq=0;self.interrupted=False
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s internal')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('worker448MiB')
        if self.calls>=15000:raise RuntimeError('15000 native return cap')
    def emit(self,kind,value):
        self.seq+=1;name=f'{self.seq:05d}-{kind}'
        arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
        S.save_npz(self.out/(name+'.npz'),**arrays);meta.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(self.out/(name+'.npz')));S.save(self.out/(name+'.json'),meta)
        return (self.out/(name+'.npz')).stat().st_size+(self.out/(name+'.json')).stat().st_size
    def it(self,x,j,g,d,tag):
        self.check()
        if self.native_calls>=14000:raise RuntimeError('14000 native cap')
        self.calls+=1;self.native_calls+=1;t=time.monotonic();end,m,c,p=iterate(x,j,g,d,6);self.emit('native',{'tag':tag,'theta':np.array(x),'gain':g,'delta':d,'end':end,'monodromy':m,'cycle':c,'phase_jacobians':p});return end,m,c,p,time.monotonic()-t
    def aug(self,z,j,g,sign,tag):
        self.check()
        if self.augmented_calls>=96:raise RuntimeError('96 augmented cap')
        self.calls+=1;self.augmented_calls+=1;t=time.monotonic();h,dh,a=augmented(z,j,g,6,sign);self.emit('augmented',dict(a,h=h,dh=dh,gain=g,sign=sign,tag=tag));return h,dh,a,time.monotonic()-t
    def correct(self,x,j,g,d,tag):
        x=x.copy()
        for k in range(7):
            end,m,c,p,_=self.it(x,j,g,d,tag);res=end-x;scale=max(1,float(np.max(np.abs(c))));error=float(np.max(np.abs(res))/scale)
            if not all(np.all(np.isfinite(a))for a in(end,m,c,p))or not np.isfinite(error):return None,'nonfinite_native'
            if error<=1e-11:
                separations={str(d):float(np.max(np.abs(c[d]-c[0]))/scale) for d in (1,2,3)};sep=min(separations.values())
                e,v=np.linalg.eig(m);self.emit('native_eigen',{'tag':tag,'gain':g,'delta':d,'theta':x,'monodromy':m,'eigenvalues':e,'eigenvectors':v,'closure':error,'proper_divisor_separations':separations})
                if sep<=1e-6:return None,'collapsed'
                residual=float(np.max(np.abs(m@v-v*e[None,:]))/max(1,np.max(np.abs(m))))
                if not np.all(np.isfinite(e))or not np.all(np.isfinite(v))or not np.isfinite(residual)or residual>1e-10:return None,'eigensystem_unqualified'
                end1,m1,c1,p1,_=self.it(c[1],j,g,d,tag+'-adjacent')
                if not all(np.all(np.isfinite(a))for a in(end1,m1,c1,p1)):return None,'nonfinite_adjacent'
                adjacent_error=float(np.max(np.abs(end1-c[1]))/scale)
                adjacent_separations={str(k):float(np.max(np.abs(c1[k]-c1[0]))/scale) for k in (1,2,3)}
                e1,v1=np.linalg.eig(m1);self.emit('native_adjacent_eigen',{'tag':tag,'gain':g,'delta':d,'theta':c[1],'monodromy':m1,'eigenvalues':e1,'eigenvectors':v1,'closure':adjacent_error,'proper_divisor_separations':adjacent_separations})
                residual1=float(np.max(np.abs(m1@v1-v1*e1[None,:]))/max(1,np.max(np.abs(m1))))
                if adjacent_error>1e-10 or min(adjacent_separations.values())<=1e-6:return None,'adjacent_closure_or_primitive'
                if not np.all(np.isfinite(e1))or not np.all(np.isfinite(v1))or not np.isfinite(residual1)or residual1>1e-10:return None,'adjacent_eigensystem_unqualified'
                return {'theta':x,'cycle':c,'delta':d,'eigenvalues':e,'eigenvalues_adjacent':e1,'eigenvectors':v,'radius':float(max(np.max(np.abs(e)),np.max(np.abs(e1)))),'closure':error,'closure_adjacent':adjacent_error},'closed'
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
            if np.max(np.abs(h))<1e-12:
                self.check()
                if self.augmented_calls>=96:raise RuntimeError('96 augmented cap')
                self.calls+=1;self.augmented_calls+=1
                arrays,summary=qualify_boundary(z,j,g,6,sign,lambda kind,val:self.emit(kind,dict(val,tag=tag)))
                pm=phase_match(point['cycle'],arrays['cycle']);self.emit('boundary_identity',dict(pm,tag=tag,parent_cycle=point['cycle'],child_cycle=arrays['cycle']));summary['native_identity']=pm
                reverse,rs=self.correct(z[:len(j)],j,g,point['delta'],tag+'-reverse')
                scale=max(1,float(np.max(np.abs(point['cycle']))));err=float(np.max(np.abs(reverse['theta']-point['theta']))/scale) if reverse is not None else None
                self.emit('boundary_reverse',{'tag':tag,'seed':point['theta'],'reverse':reverse['theta'] if reverse is not None else np.full(len(j),np.nan),'status':rs,'relative_error':err})
                summary.update(reverse_status=rs,reverse_error=err,flip_cubic_criticality='not_evaluated' if sign==-1 else 'not_applicable')
                adjacent,adj_status=self.correct(arrays['cycle'][1],j,g,float(z[len(j)]),tag+'-boundary-adjacent')
                summary['adjacent_native_status']=adj_status
                adjacent_error=float(np.max(np.abs(adjacent['theta']-arrays['cycle'][1]))/max(1,float(np.max(np.abs(arrays['cycle'])))))if adjacent is not None else None
                summary['adjacent_native_identity_error']=adjacent_error
                if adjacent_error is None or adjacent_error>1e-7:summary['status']='boundary_adjacent_unresolved'
                if not pm['passed'] or err is None or err>1e-7:summary['status']='boundary_identity_unresolved'
                self.emit('boundary_result',dict(arrays,**summary,gain=g,tag=tag));return dict(summary,gain=g,delta=float(z[len(j)]),tag=tag)
            if k==11:break
            try:dx=np.linalg.solve(dh,-h)
            except np.linalg.LinAlgError:return {'status':'augmented_linear_failure','sign':sign,'tag':tag}
            self.emit('augmented_solve',{'tag':tag,'iteration':k,'matrix':dh,'rhs':-h,'direction':dx})
            new=z+dx
            if not np.all(np.isfinite(new)) or abs(new[len(j)]-center[len(j)])>.01 or not .60<new[len(j)]<1.00 or np.max(np.abs(new[:len(j)]-center[:len(j)]))>.2*max(1,float(np.max(np.abs(center[:len(j)])))):
                return {'status':'augmented_guard','sign':sign,'tag':tag}
            z=new
        return {'status':'augmented_iteration_limit','sign':sign,'tag':tag}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,)*2)
    a.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=a.out;S.RAW_CAP=384*1024**2;w=Worker(a.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
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
        # Unrelated deterministic N12/p6 synthetic profile; includes emitted arrays.
        j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;z=np.r_[x,.8,np.eye(12)[0]];ta=[];tb=[];te=[];ts=[];sizes={'native':[],'augmented':[],'eigen':[],'solve':[]}
        def disk_bytes():return sum(p.stat().st_size for p in a.out.iterdir()if p.is_file())
        for i in range(10):
            before=disk_bytes();ret=w.it(x,j,2.,.8,'synthetic_profile');ta.append(ret[4]);sizes['native'].append(disk_bytes()-before)
            before=disk_bytes();tb.append(w.aug(z,j,2.,-1,'synthetic_profile')[3]);sizes['augmented'].append(disk_bytes()-before)
            t=time.monotonic();e,v=np.linalg.eig(ret[1]);sz=w.emit('profile_eigen',{'tag':'synthetic_profile','gain':2.,'delta':.8,'theta':x,'monodromy':ret[1],'eigenvalues':e,'eigenvectors':v,'closure':1.,'proper_divisor_separations':{'1':1.,'2':1.,'3':1.}});te.append(time.monotonic()-t);sizes['eigen'].append(sz)
            t=time.monotonic();mat=ret[1]-np.eye(12);rhs=-ret[0];dx=np.linalg.solve(mat,rhs);sz=w.emit('profile_solve',{'tag':'synthetic_profile','matrix':mat,'rhs':rhs,'direction':dx,'iteration':0});ts.append(time.monotonic()-t);sizes['solve'].append(sz)
        gate=resource_projection(ta,tb,te,ts,sizes,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        S.save(a.out/'profile.json',dict(gate,n=12,period=6,native_seconds=ta,augmented_seconds=tb,eigen_seconds=te,solve_seconds=ts,unit_bytes=sizes,target_inputs_unopened_before_profile=True))
        if not gate['gate']:raise RuntimeError('resource profile failed')
        raw=R.RAW/'n12-p6-refinement-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('refinement pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if p.stat().st_size>288*1024**2:raise MemoryError('input hashing buffer cap')
            if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('input hash')
        with np.load(raw/'run/inputs.npz') as f:j=f['j'];gains=f['gains'];deltas=f['deltas']
        with np.load(raw/'run/batch.npz') as f:periods=f['period_proposals'].reshape(51,51,2);final=f['final'].reshape(51,51,2,12)
        S.save(a.out/'input-binding.json',{'refinement_manifest':PIN,'source_input_sha256':S.sha(raw/'run/inputs.npz'),'source_batch_sha256':S.sha(raw/'run/batch.npz'),'gain_indices':[16,25,34],'delta_step':.001,'maximum_steps_per_direction':128,'selection':'middle sorted delta index within fixed component49/729cells at gain16/25/34; original start0 traced, both original starts qualified and retained as context'})
        comp=next(c for c in json.loads((raw/'run/components.json').read_text())['components'] if c['id']==49)
        if comp['period']!=6 or comp['count']!=729 or comp['grid_edge_margin']!=3:raise ValueError('fixed refined component')
        for gi in [16,25,34]:
            g=float(gains[gi]);inds=np.array(sorted(i%51 for i in comp['cells'] if i//51==gi),dtype=int)
            if any(not np.all(periods[gi,di]==6) for di in inds):raise ValueError('component period join')
            if not len(inds):result['slices'].append({'gain':g,'status':'no_consensus6_seed'});continue
            di=int(inds[len(inds)//2]);d=float(deltas[di]);points=[];context=[]
            for initial in [0,1]:
                pt,st=w.correct(final[gi,di,initial],j,g,d,f'g{gi}-seed{initial}');points.append(pt)
                context.append({'initial':initial,'status':st,'attracting':initial_attracting(pt)if pt is not None else False,'radius':pt['radius']if pt is not None else None})
            report={'gain':g,'gain_index':gi,'delta_index':di,'seed_delta':d,'seed_context':context,'directions':[]};result['slices'].append(report)
            point=points[0]
            if point is None:report['status']='original_start0_'+context[0]['status'];continue
            if not all(c['attracting']for c in context):report['status']='both_start_seed_not_qualified_attracting';continue
            report['between_start_context']=phase_match(points[0]['cycle'],points[1]['cycle'])
            w.emit('between_start_context',dict(report['between_start_context'],gain=g,delta=d,initial0_cycle=points[0]['cycle'],initial1_cycle=points[1]['cycle']))
            for direction in [-1,1]:
                last=point;endstatus='step_limit';tag=f'g{gi}-dir{direction}';boundary=None;steps=0
                for k in range(1,129):
                    dn=d+direction*k*.001
                    if not .60<dn<1.00:endstatus='domain_edge';break
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
                        sign=leading_boundary_sign(np.r_[candidate['eigenvalues'],candidate['eigenvalues_adjacent']])
                        if sign is None:endstatus='first_unstable_complex_unresolved';break
                        endstatus='first_unstable';boundary=w.localize(candidate,j,g,sign,tag+'-boundary');break
                    last=candidate
                report['directions'].append({'direction':direction,'steps':steps,'stop':endstatus,'boundary':boundary,'last_closed_delta':last['delta'],'last_closed_radius':last['radius']});S.save(a.out/f'progress-g{gi}.json',report)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_native_calls=w.native_calls,attempted_augmented_calls=w.augmented_calls,total_counted_calls=w.calls,scope='adapted first-sampled-loss native6 discovery at delta step.001; narrow pockets unexcluded, p12/24 branch joins untested, no route, interval or shrimp claim');S.save(a.out/'result.json',result,final=True);S.manifest(a.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
