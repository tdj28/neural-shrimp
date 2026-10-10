#!/usr/bin/env python3
"""Fixed short numerical pseudoarclength traces of the native N6 p7 two fold arms and flip."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('boundary_driver',ROOT/'tools/explore_clark_rnn_n6_p7_boundaries.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B);S=B.S;R=B.R
from neural_shrimp.clark_rnn_boundaries import family_augmented,oriented_null
PINS={'n6-p7-boundaries-001':'a24758997c8ca85d4df17e31784c06978949ea9ee9fb0434041a2a396edf8178','n6-p7-fold-association-001':'b3e6f05e4699deb9b6df8e436d0717de845e0dce26934fb21a783a312fc038ea'}
SEEDS=[('lower-fold','g16-dir-1-fold',1),('upper-fold','g16-dir1-fold',1),('lower-flip','g25-dir-1-boundary',-1)]
COL=np.array([10.]*6+[5.,.1]+[1.]*6);ROW=np.array([.05]*6+[1.]*7)
EXTRA=B.EXTRA+['tools/continue_clark_rnn_n6_p7.py','tools/launch_clark_rnn_n6_p7_continuation.py','docs/plans/clark-rnn-n6-p7-continuation.md','tests/test_clark_rnn_n6_p7_continuation.py']
class DurabilityError(RuntimeError):pass

def seed_qualified(meta,association):
    if not meta.get('native_identity',{}).get('passed'):return False
    if meta.get('status')=='numerical_boundary_candidate':return True
    return bool(meta.get('status')=='boundary_identity_unresolved' and meta.get('reverse_status')=='step_guard' and association is not None and association['status']=='numerically_associated_fold' and association['parent_check']['passed'] and any(a['association']['passed'] for a in association['attempts']))

class Work(B.Worker):
    def emit(self,kind,value):
        try:super().emit(kind,value)
        except Exception as exc:raise DurabilityError("durable return failed: "+str(exc))from exc
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=10000:raise RuntimeError('10000 family-evaluation cap')
    def full(self,y,j,sign,tag):
        self.check();self.calls+=1;t=time.monotonic();h,b,a=family_augmented(y*COL,j,7,sign)
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
        index=int(np.argmin(np.abs(e-sign)));idx=int(np.argmin(np.abs(le-sign)));left=lv[:,idx];q=y[8:]
        sep=proper_separations(a['cycle'])
        if np.max(np.abs(left.imag))>1e-8 or abs(left.real@q)<1e-8:raise ValueError('node left/right normalization')
        left=left.real/(left.real@q);other=np.delete(e,index);bqq=np.einsum('ijk,j,k->i',a['second'][:,:6,:6],q,q);coefficient=float(left@bqq) if sign==1 else None
        alpha=None
        if sign==-1:
            mat=m-np.eye(6);rhs=-a['first'][:,6:8]
            xs=np.linalg.solve(mat,rhs);self.emit('branch_derivative_solve',{'tag':tag,'matrix':mat,'rhs':rhs,'solution':xs})
            alpha=total_crossing(a['second'],q,left,xs)
        scale=max(1,float(np.max(np.abs(a['cycle']))))
        eigen_res=float(np.max(np.abs(m@v-v*e[None,:]))/max(1,np.max(np.abs(m))))
        r={'tag':tag,'gain':float(y[6]*5),'delta':float(y[7]*.1),'sign':sign,'critical_error':float(abs(e[index]-sign)),'other_distance':float(np.min(np.abs(other-sign))),'complement_radius':float(np.max(np.abs(other))),'proper_divisor_separations':sep,'fold_c':coefficient,'fold_parameter_coupling':(left@a['first'][:,6:8]).tolist() if sign==1 else None,'flip_total_parameter_crossing':alpha.tolist() if alpha is not None else None,'gain_tangent':float(tangent[6]),'delta_tangent':float(tangent[7]),'odd_companion_min_phase_distance':float(min(np.max(np.abs(a['cycle']+np.roll(a['cycle'],k,axis=0)))/scale for k in range(7))),'eigen_residual':eigen_res}
        r['left_eigen_residual']=float(np.max(np.abs(left@m-sign*left))/max(1,np.max(np.abs(left))))
        if not np.isfinite(r['left_eigen_residual']) or not np.isfinite(eigen_res):raise ValueError('nonfinite node residual')
        good=r['left_eigen_residual']<1e-10 and r['critical_error']<1e-7 and r['other_distance']>1e-4 and all(x>1e-6 for x in sep.values()) and eigen_res<1e-10
        r['qualified_primitive_simple_boundary']=bool(good);r['attracting_complement']=r['complement_radius']<1-1e-5
        r['numerically_generic_fold']=bool(abs(coefficient)>1e-6 and np.max(np.abs(r['fold_parameter_coupling']))>1e-6) if sign==1 else None
        r['transverse_multiplier_crossing']=bool(np.max(np.abs(alpha))>1e-6) if sign==-1 else None
        r['flip_cubic_criticality']='not_evaluated' if sign==-1 else 'not_applicable'
        self.emit('node',dict(r,y=y.copy(),physical=y*COL,tangent=tangent.copy(),left_vector=left,monodromy=m,bqq=bqq))
        if not good:raise ValueError('primitive/simple boundary qualification')
        return r

def in_domain(y):return 5.5<y[6]*5<12. and .48<y[7]*.1<.67

def proper_separations(cycle):
    if cycle.shape!=(7,6):raise ValueError('native7 shape')
    return {'1':float(np.max(np.abs(cycle[1]-cycle[0]))/max(1,np.max(np.abs(cycle))))}

def total_crossing(second,q,left,xs):
    return np.array([left@(second[:,:6,6+i]@q+np.einsum('ijk,j,k->i',second[:,:6,:6],q,xs[:,i])) for i in range(2)])

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
    result={'status':'incomplete','traces':[]}
    try:
        if available<1024**3:raise MemoryError('host headroom')
        tj=np.eye(6)*.2+np.ones((6,6))*.1;ty=np.r_[np.arange(1,7)/10,2.,.5,[1.,0,0,0,0,0]]/COL;times=[]
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
        ref=R.RAW/'n6-p7-refinement-001/run/inputs.npz';bind=json.loads((R.RAW/'n6-p7-boundaries-001/run/input-binding.json').read_text())
        if S.sha(ref)!=bind['source_input_sha256']:raise ValueError('matrix identity')
        with np.load(ref,allow_pickle=False) as f:j=f['j']
        assoc=json.loads((R.RAW/'n6-p7-fold-association-001/run/result.json').read_text())
        if assoc['status']!='completed':raise ValueError('complete fixed association unit')
        associations={v['tag']:v for v in assoc['folds']}
        S.save(args.out/'input-binding.json',{'manifests':PINS,'source_input':str(ref),'source_input_sha256':S.sha(ref),'seeds':SEEDS,'scope':'two fold roots have separately retained quadratic reverse associations; previous singular-start failure preserved'})
        for branch,tag,sign in SEEDS:
            raw=R.RAW/'n6-p7-boundaries-001'
            files=[p for p in (raw/'run').glob('*-boundary_result.json') if json.loads(p.read_text())['tag']==tag]
            if len(files)!=1:raise ValueError('unique fixed boundary seed')
            p=files[0];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
            if S.sha(arr)!=meta['array_sha256'] or not seed_qualified(meta,associations.get(tag)):raise ValueError('qualified associated boundary array identity')
            with np.load(arr,allow_pickle=False) as f:z=np.r_[f['theta'],meta['gain'],meta['delta'],f['right_vector']]
            y=z/COL;h,b,a,_=w.full(y,j,sign,tag+'-seed');t=w.tangent(b,None,tag+'-seed')
            S.save(args.out/f'seed-{branch}.json',{'tag':tag,'file':str(p),'sha256':S.sha(p),'array_sha256':S.sha(arr),'residual':float(np.max(np.abs(h)))})
            if np.max(np.abs(h))>1e-9:raise ValueError('seed root')
            for direction in [-1,1]:
                cur=y.copy();tan=direction*t;trace={'branch':branch,'sign':sign,'direction':direction,'step':.01,'nodes':[],'stop':'node_cap'}
                result['traces'].append(trace);oldcycle=a['cycle']
                try:
                    trace['nodes'].append(w.node(cur,b,a,tan,sign,f'{branch}:{direction}:seed'))
                    for k in range(1,129):
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
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_family_calls=w.calls,scope='fixed six native primitive7 fold/flip eigenpair loci with reverse, proper-divisor and phase/odd-sign diagnostics; generic/attracting labels separate. No validated curve/common orbit at projected crossing/whole shrimp/ODE claim');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
