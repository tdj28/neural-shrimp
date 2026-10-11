#!/usr/bin/env python3
"""Fixed short numerical pseudoarclength traces of the compact native p12 fold and flip."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('boundary_driver',ROOT/'tools/explore_clark_rnn_compact_p12_boundaries.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B);S=B.S;R=B.R
from neural_shrimp.clark_rnn_boundaries import family_augmented,oriented_null
PINS={'compact-p12-boundary-slices-001':'b5a8e4ee85008dfb84b29032c42718b7df42924829e8c7ee2d72af92e12bd6ad'}
SEEDS=[('flip','compact-p12-boundary-slices-001','g25-dir-1-boundary',-1)]
COL=np.array([10.]*4+[5.,.1]+[1.]*4);ROW=np.array([.05]*4+[1.]*5)
EXTRA=B.EXTRA+['tools/continue_clark_rnn_compact_p12.py','tools/launch_clark_rnn_compact_p12_continuation.py','docs/plans/clark-rnn-compact-p12-continuation.md','tests/test_clark_rnn_compact_p12_continuation.py']

class Work(B.Worker):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=10000:raise RuntimeError('10000 family-evaluation cap')
    def full(self,y,j,sign,tag):
        self.check();self.calls+=1;t=time.monotonic();h,b,a=family_augmented(y*COL,j,12,sign)
        self.emit('family',dict(a,h=h,b=b,y=y.copy(),sign=sign,tag=tag));return ROW*h,ROW[:,None]*b*COL[None,:],a,time.monotonic()-t
    def tangent(self,b,ref,tag):
        self.check();t,r=oriented_null(b,ref);self.emit('null_svd',dict(r,tag=tag))
        if not np.isfinite(t).all() or not np.isfinite(r['s']).all() or not np.isfinite(r['null_residual']) or r['s'][-1]<=1e-8 or r['null_residual']>1e-10:raise ValueError('tangent numerical rank/residual')
        return t
    def correct(self,start,plane,tan,j,sign,tag):
        y=start.copy()
        for k in range(8):
            h,b,a,_=self.full(y,j,sign,tag)
            if not np.isfinite(h).all() or not np.isfinite(b).all() or not np.isfinite(a['endpoint']).all():raise ValueError('nonfinite corrector return')
            plane_res=float(tan@(y-plane));r=np.r_[h,plane_res]
            if np.max(np.abs(r))<1e-10:
                if np.max(np.abs(a['endpoint']-a['theta']))>1e-9:return None,'physical_closure'
                return (y,b,a),'closed'
            if k==7:return None,'iteration_limit'
            mat=np.vstack([b,tan])
            try:dx=np.linalg.solve(mat,-r)
            except np.linalg.LinAlgError:return None,'linear_failure'
            self.emit('pal_solve',{'tag':tag,'iteration':k,'matrix':mat,'rhs':-r,'solution':dx})
            if np.max(np.abs(dx))>.1 or not np.all(np.isfinite(dx)):return None,'correction_guard'
            y+=dx
            if not in_domain(y):return None,'domain_edge'
        return None,'iteration_limit'
    def node(self,y,b,a,tangent,sign,tag):
        self.check();m=a['monodromy'];e,v=np.linalg.eig(m);self.emit('node_eigen',{'tag':tag,'matrix':m,'eigenvalues':e,'eigenvectors':v});le,lv=np.linalg.eig(m.T);self.emit('node_left_eigen',{'tag':tag,'matrix':m.T,'eigenvalues':le,'eigenvectors':lv})
        if not all(np.isfinite(z).all() for z in (m,e,v,le,lv)):raise ValueError('nonfinite node eigensystem')
        index=int(np.argmin(np.abs(e-sign)));idx=int(np.argmin(np.abs(le-sign)));left=lv[:,idx];q=y[6:]
        sep=proper_separations(a['cycle'])
        if np.max(np.abs(left.imag))>1e-8 or abs(left.real@q)<1e-8:raise ValueError('node left/right normalization')
        left=left.real/(left.real@q);other=np.delete(e,index);bqq=np.einsum('ijk,j,k->i',a['second'][:,:4,:4],q,q);coefficient=float(left@bqq) if sign==1 else None
        alpha=None
        if sign==-1:
            mat=m-np.eye(4);rhs=-a['first'][:,4:6]
            xs=np.linalg.solve(mat,rhs);self.emit('branch_derivative_solve',{'tag':tag,'matrix':mat,'rhs':rhs,'solution':xs})
            alpha=total_crossing(a['second'],q,left,xs)
        scale=max(1,float(np.max(np.abs(a['cycle']))))
        eigen_res=float(np.max(np.abs(m@v-v*e[None,:]))/max(1,np.max(np.abs(m))))
        r={'tag':tag,'gain':float(y[4]*5),'delta':float(y[5]*.1),'sign':sign,'critical_error':float(abs(e[index]-sign)),'other_distance':float(np.min(np.abs(other-sign))),'complement_radius':float(np.max(np.abs(other))),'proper_divisor_separations':sep,'fold_c':coefficient,'fold_parameter_coupling':(left@a['first'][:,4:6]).tolist() if sign==1 else None,'flip_total_parameter_crossing':alpha.tolist() if alpha is not None else None,'gain_tangent':float(tangent[4]),'delta_tangent':float(tangent[5]),'odd_companion_min_phase_distance':float(min(np.max(np.abs(a['cycle']+np.roll(a['cycle'],k,axis=0)))/scale for k in range(12))),'eigen_residual':eigen_res}
        good=r['critical_error']<1e-7 and r['other_distance']>1e-4 and all(x>1e-6 for x in sep.values()) and eigen_res<1e-10
        r['qualified_primitive_simple_boundary']=bool(good);r['attracting_complement']=r['complement_radius']<1-1e-5
        r['numerically_generic_fold']=bool(abs(coefficient)>1e-6 and np.max(np.abs(r['fold_parameter_coupling']))>1e-6) if sign==1 else None
        r['transverse_multiplier_crossing']=bool(np.max(np.abs(alpha))>1e-6) if sign==-1 else None
        r['flip_cubic_criticality']='not_evaluated' if sign==-1 else 'not_applicable'
        self.emit('node',dict(r,y=y.copy(),physical=y*COL,tangent=tangent.copy(),left_vector=left,monodromy=m,bqq=bqq))
        if not good:raise ValueError('primitive/simple boundary qualification')
        return r

def in_domain(y):return 2.6<y[4]*5<5.6 and .715<y[5]*.1<.91

def proper_separations(cycle):
    if cycle.shape!=(12,4):raise ValueError('native12 shape')
    return {str(d):float(np.max(np.abs(cycle[d]-cycle[0]))/max(1,np.max(np.abs(cycle)))) for d in (1,2,3,4,6)}

def total_crossing(second,q,left,xs):
    return np.array([left@(second[:,:4,4+i]@q+np.einsum('ijk,j,k->i',second[:,:4,:4],q,xs[:,i])) for i in range(2)])

def native_correct(w,x,j,g,d,tag):
    x=x.copy()
    for k in range(14):
        end,m,c,p,_=w.it(x,j,g,d,tag);scale=max(1.,float(np.max(np.abs(c))));res=end-x
        if not all(np.isfinite(a).all()for a in(end,m,c,p,res)):return None,'nonfinite_native'
        if np.max(np.abs(res))/scale<=1e-11:
            if min(proper_separations(c).values())<=1e-6:return None,'collapsed'
            return dict(theta=x,cycle=c,delta=d),'closed'
        if k==13:return None,'iteration_limit'
        try:dx=np.linalg.solve(m-np.eye(4),-res)
        except np.linalg.LinAlgError:return None,'linear_failure'
        w.emit('association_solve',dict(tag=tag,matrix=m-np.eye(4),rhs=-res,direction=dx,iteration=k))
        if not np.isfinite(dx).all()or np.max(np.abs(dx))>.1*scale:return None,'step_guard'
        x=x+dx

def associate_fold(w,j,raw,tag):
    def record(kind):
        matches=[p for p in raw.glob('*-'+kind+'.json')if json.loads(p.read_text()).get('tag')==tag]
        if len(matches)!=1:raise ValueError('unique retained '+kind)
        p=matches[0];v=json.loads(p.read_text());a=p.with_name(v['array_file'])
        if S.sha(a)!=v['array_sha256']:raise ValueError('retained array')
        with np.load(a)as f:arrays={k:f[k]for k in f.files}
        return v,arrays
    root,ra=record('boundary_result');_,rev=record('boundary_reverse');_,pa=record('boundary_identity')
    if not(root['residual']<1e-8 and root['critical_error']<1e-7 and root['other_distance']>1e-4 and abs(root['fold_a_delta'])>1e-5 and abs(root['fold_c'])>1e-5 and min(root['proper_divisor_separations'].values())>1e-6):raise ValueError('retained local fold gates')
    result=json.loads((raw/'result.json').read_text());old=next(d for row in result['slices']for d in row['directions']if d['boundary']is not None and d['boundary']['tag']==tag);base=old['last_closed_delta'];target=root['delta'];g=root['gain'];x=rev['seed'];last=dict(theta=x,cycle=pa['parent_cycle'],delta=base);report=dict(tag=tag,passed=False,base_delta=base,root_delta=target,levels=[])
    for k in range(1,9):
        delta=target+(base-target)*2.**(-k);candidate,status=native_correct(w,last['theta'],j,g,delta,tag+f'-fine{k}')
        if candidate is None:report['failure']='forward_'+status;return report
        match=B.phase_match(last['cycle'],candidate['cycle']);w.emit('association_phase',dict(match,tag=tag,level=k,parent_cycle=last['cycle'],child_cycle=candidate['cycle']))
        if not match['passed']:report['failure']='phase_identity';return report
        reverse,status=native_correct(w,candidate['theta'],j,g,last['delta'],tag+f'-fine{k}-reverse');scale=max(1.,float(np.max(np.abs(last['cycle']))));err=float(np.max(np.abs(reverse['theta']-last['theta']))/scale)if reverse is not None else None
        w.emit('association_reverse',dict(tag=tag,level=k,seed=last['theta'],reverse=reverse['theta']if reverse is not None else np.full(4,np.nan),error=err,status=status))
        report['levels'].append(dict(level=k,delta=delta,native_error=match['native_error'],reverse_error=err))
        if err is None or err>1e-7:report['failure']='reverse_'+status;return report
        last=candidate
    match=B.phase_match(last['cycle'],ra['cycle']);reverse,status=native_correct(w,ra['theta'],j,g,last['delta'],tag+'-root-reverse');scale=max(1.,float(np.max(np.abs(last['cycle']))));err=float(np.max(np.abs(reverse['theta']-last['theta']))/scale)if reverse is not None else None
    w.emit('association_root_join',dict(tag=tag,**match,root_cycle=ra['cycle'],last_cycle=last['cycle'],seed=last['theta'],reverse=reverse['theta']if reverse is not None else np.full(4,np.nan),reverse_error=err,reverse_status=status));report.update(root_phase_match=match,root_reverse_error=err,root_reverse_status=status,passed=bool(match['passed']and err is not None and err<=1e-7),scope='numerical finer-step phase/reverse association of retained root only; no interval connection')
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=256*1024**2;w=Work(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    # Sequential immutable record names: reserve a conservative upper byte count,
    # avoiding an O(number-of-files²) walk while still prechecking every write.
    booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite reserved byte cap')
        booked[0]=new
    S.reserve=reserve
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ.get(k) for k in R.M.THREADS},'mem_available_bytes':available,'row_scale':ROW.tolist(),'column_scale':COL.tolist()})
    result={'status':'incomplete','traces':[],'fold_associations':[]}
    try:
        if available<1024**3:raise MemoryError('host headroom')
        tj=np.eye(4)*.2+np.ones((4,4))*.1;ty=np.r_[np.arange(1,5)/10,2.,.5,[1.,0,0,0]]/COL;times=[]
        for k in range(10):times.append(w.full(ty,tj,1,'synthetic_profile')[3])
        projected=9700*max(times)+20;S.save(args.out/'profile.json',{'times':times,'projected_seconds':projected,'gate':np.isfinite(projected) and projected<=120,'target_unopened':True})
        if not np.isfinite(projected) or projected>120:raise RuntimeError('resource screen')
        for name,pin in PINS.items():
            raw=R.RAW/name
            if S.sha(raw/'manifest.json')!=pin:raise ValueError('manifest pin')
            inv=json.loads((raw/'manifest.json').read_text())['files']
            if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('input inventory')
            for name,d in inv.items():
                p=raw/name
                if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        ref=R.RAW/'compact-p12-refinement-001/run/inputs.npz';bind=json.loads((R.RAW/'compact-p12-boundary-slices-001/run/input-binding.json').read_text())
        if S.sha(ref)!=bind['source_input_sha256']:raise ValueError('matrix identity')
        with np.load(ref) as f:j=f['j']
        S.save(args.out/'input-binding.json',{'manifests':PINS,'source_input':str(ref),'source_input_sha256':S.sha(ref),'seeds':SEEDS})
        # Fixed finer native association of both old unresolved roots, no root search.
        for tag in ['g16-dir-1-fold','g34-dir-1-fold']:
            try:association=associate_fold(w,j,R.RAW/'compact-p12-boundary-slices-001/run',tag)
            except (ValueError,np.linalg.LinAlgError)as e:association={'tag':tag,'passed':False,'failure':type(e).__name__+': '+str(e)}
            result['fold_associations'].append(association);S.save(args.out/'association-progress.json',result['fold_associations'])
        for branch,rawname,tag,sign in SEEDS:
            raw=R.RAW/rawname
            files=[p for p in (raw/'run').glob('*-boundary_result.json') if json.loads(p.read_text())['tag']==tag]
            if len(files)!=1:raise ValueError('unique fixed boundary seed')
            p=files[0];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
            if S.sha(arr)!=meta['array_sha256'] or meta['status']!='numerical_boundary_candidate' or not meta['native_identity']['passed']:raise ValueError('qualified boundary array identity')
            with np.load(arr) as f:z=np.r_[f['theta'],meta['gain'],meta['delta'],f['right_vector']]
            y=z/COL;h,b,a,_=w.full(y,j,sign,tag+'-seed');t=w.tangent(b,None,tag+'-seed')
            S.save(args.out/f'seed-{branch}.json',{'tag':tag,'file':str(p),'sha256':S.sha(p),'array_sha256':S.sha(arr),'residual':float(np.max(np.abs(h)))})
            if np.max(np.abs(h))>1e-9:raise ValueError('seed root')
            for direction in [-1,1]:
                cur=y.copy();tan=direction*t;trace={'branch':branch,'sign':sign,'direction':direction,'step':.01,'nodes':[],'stop':'node_cap'}
                result['traces'].append(trace);oldcycle=a['cycle']
                try:
                    trace['nodes'].append(w.node(cur,b,a,tan,sign,f'{branch}:{direction}:seed'))
                    for k in range(1,193):
                        pred=cur+.01*tan;tag2=f'{branch}:{direction}:{k}'
                        if not in_domain(pred):trace['stop']='domain_edge';break
                        c,status=w.correct(pred,pred,tan,j,sign,tag2)
                        if c is None:trace['stop']='forward_'+status;break
                        nxt,nb,na=c;nt=w.tangent(nb,tan,tag2)
                        pm=B.phase_match(oldcycle,na['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag2,parent_cycle=oldcycle,child_cycle=na['cycle']))
                        if not pm['passed']:trace['stop']='phase_sign_identity';break
                        reverse,rs=w.correct(nxt-.01*nt,cur,tan,j,sign,tag2+'-reverse')
                        if reverse is None:trace['stop']='reverse_'+rs;break
                        mismatch=float(np.max(np.abs(reverse[0]-cur)));w.emit('reverse_check',{'tag':tag2,'old':cur,'reverse':reverse[0],'mismatch':mismatch})
                        if mismatch>1e-7:trace['stop']='reverse_mismatch';break
                        nr=w.node(nxt,nb,na,nt,sign,tag2);nr['reverse_mismatch']=mismatch;nr['phase_sign_match']=pm;trace['nodes'].append(nr);cur=nxt;tan=nt;oldcycle=na['cycle']
                        if k%8==0:S.save(args.out/'progress.json',result)
                except (ValueError,np.linalg.LinAlgError) as e:
                    trace['stop']='qualification_or_arithmetic: '+type(e).__name__+': '+str(e)
                S.save(args.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_family_calls=w.calls,scope='fixed native primitive12 fold/flip eigenpair loci with reverse, proper-divisor and phase/odd-sign diagnostics; generic/attracting labels separate. No validated curve/common orbit at projected crossing/whole shrimp/ODE claim');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
