#!/usr/bin/env python3
"""Fixed short numerical pseudoarclength traces of the native p5 fold/flip."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('boundary_driver',ROOT/'tools/explore_clark_rnn_boundaries.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B);S=B.S;R=B.R
from neural_shrimp.clark_rnn_boundaries import family_augmented,oriented_null
PIN='20d1171d8ccfcc41743d85b8757c5c8affbd15b1c393d6f5209af971b363f77f'
COL=np.array([20.]*4+[10.,.1]+[1.]*4);ROW=np.array([.05]*4+[1.]*5)
EXTRA=B.EXTRA+['tools/continue_clark_rnn_boundaries.py','tools/launch_clark_rnn_continuation.py','docs/plans/clark-rnn-boundary-continuation.md']

class Work(B.Worker):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=4000:raise RuntimeError('4000 family-evaluation cap')
    def full(self,y,j,sign,tag):
        self.check();self.calls+=1;t=time.monotonic();h,b,a=family_augmented(y*COL,j,5,sign)
        self.emit('family',dict(a,h=h,b=b,y=y.copy(),sign=sign,tag=tag));return ROW*h,ROW[:,None]*b*COL[None,:],a,time.monotonic()-t
    def tangent(self,b,ref,tag):
        self.check();t,r=oriented_null(b,ref);self.emit('null_svd',dict(r,tag=tag))
        if r['s'][-1]<=1e-8 or r['null_residual']>1e-10:raise ValueError('tangent numerical rank/residual')
        return t
    def correct(self,start,plane,tan,j,sign,tag):
        y=start.copy()
        for k in range(8):
            h,b,a,_=self.full(y,j,sign,tag);plane_res=float(tan@(y-plane));r=np.r_[h,plane_res]
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
            if not 1<y[4]*10<40 or not .05<y[5]*.1<1:return None,'domain_edge'
        return None,'iteration_limit'
    def node(self,y,b,a,tangent,sign,tag):
        self.check();m=a['monodromy'];e,v=np.linalg.eig(m);self.emit('node_eigen',{'tag':tag,'matrix':m,'eigenvalues':e,'eigenvectors':v});le,lv=np.linalg.eig(m.T);self.emit('node_left_eigen',{'tag':tag,'matrix':m.T,'eigenvalues':le,'eigenvectors':lv})
        index=int(np.argmin(np.abs(e-sign)));idx=int(np.argmin(np.abs(le-sign)));left=lv[:,idx];q=y[6:];scale=max(1,float(np.max(np.abs(a['cycle']))));sep=float(np.max(np.abs(a['cycle'][1]-a['cycle'][0]))/scale)
        if np.max(np.abs(left.imag))>1e-8 or abs(left.real@q)<1e-8:raise ValueError('node left/right normalization')
        left=left.real/(left.real@q);other=np.delete(e,index);bqq=np.einsum('ijk,j,k->i',a['second'][:,:4,:4],q,q);coefficient=float(left@bqq) if sign==1 else None
        r={'tag':tag,'gain':float(y[4]*10),'delta':float(y[5]*.1),'sign':sign,'critical_error':float(abs(e[index]-sign)),'other_distance':float(np.min(np.abs(other-sign))),'complement_radius':float(np.max(np.abs(other))),'divisor1':sep,'fold_c':coefficient,'fold_parameter_coupling':(left@a['first'][:,4:6]).tolist() if sign==1 else None,'gain_tangent':float(tangent[4]),'delta_tangent':float(tangent[5])}
        good=r['critical_error']<1e-7 and r['other_distance']>1e-4 and sep>1e-6
        r['qualified_primitive_simple_boundary']=good;r['attracting_complement']=r['complement_radius']<1-1e-5
        self.emit('node',dict(r,y=y.copy(),physical=y*COL,tangent=tangent.copy(),left_vector=left,monodromy=m,bqq=bqq))
        if not good:raise ValueError('primitive/simple boundary qualification')
        return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=64*1024**2;w=Work(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True))
    # Sequential immutable record names: reserve a conservative upper byte count,
    # avoiding an O(number-of-files²) walk while still prechecking every write.
    booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite reserved byte cap')
        booked[0]=new
    S.reserve=reserve
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ.get(k) for k in R.THREADS},'mem_available_bytes':available,'row_scale':ROW.tolist(),'column_scale':COL.tolist()})
    result={'status':'incomplete','traces':[]}
    try:
        if available<1024**3:raise MemoryError('host headroom')
        tj=np.eye(4)*.2+np.ones((4,4))*.1;ty=np.r_[np.arange(1,5)/10,2.,.5,[1.,0,0,0]]/COL;times=[]
        for k in range(10):times.append(w.full(ty,tj,1,'synthetic_profile')[3])
        projected=3500*max(times)+20;S.save(args.out/'profile.json',{'times':times,'projected_seconds':projected,'gate':np.isfinite(projected) and projected<=120,'target_unopened':True})
        if not np.isfinite(projected) or projected>120:raise RuntimeError('resource screen')
        raw=R.RAW/'p5-boundary-slices-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('manifest pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('input inventory')
        for name,d in inv.items():
            p=raw/name
            if p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        # Exact same matrix is inherited through the bound refinement input record.
        ref=R.RAW/'refine-p5-001/run/p5/inputs.npz';bind=json.loads((raw/'run/input-binding.json').read_text())
        if S.sha(ref)!=bind['source_input_sha256']:raise ValueError('matrix identity')
        with np.load(ref) as f:j=f['j']
        for sign,tag in [(1,'g16-dir1-fold'),(-1,'g16-dir-1-boundary')]:
            files=[p for p in (raw/'run').glob('*-boundary_result.json') if json.loads(p.read_text())['tag']==tag]
            if len(files)!=1:raise ValueError('unique fixed boundary seed')
            p=files[0];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
            if S.sha(arr)!=meta['array_sha256']:raise ValueError('boundary array pin')
            with np.load(arr) as f:z=np.r_[f['theta'],meta['gain'],meta['delta'],f['right_vector']]
            y=z/COL;h,b,a,_=w.full(y,j,sign,tag+'-seed');t=w.tangent(b,None,tag+'-seed')
            S.save(args.out/f'seed-{sign}.json',{'tag':tag,'file':str(p),'sha256':S.sha(p),'array_sha256':S.sha(arr),'residual':float(np.max(np.abs(h)))})
            if np.max(np.abs(h))>1e-9:raise ValueError('seed root')
            for direction in [-1,1]:
                cur=y.copy();tan=direction*t;trace={'sign':sign,'direction':direction,'step':.015,'nodes':[],'stop':'node_cap'}
                trace['nodes'].append(w.node(cur,b,a,tan,sign,f'{sign}:{direction}:seed'))
                for k in range(1,49):
                    pred=cur+.015*tan;tag2=f'{sign}:{direction}:{k}'
                    if not 1<pred[4]*10<40 or not .05<pred[5]*.1<1:trace['stop']='domain_edge';break
                    c,status=w.correct(pred,pred,tan,j,sign,tag2)
                    if c is None:trace['stop']='forward_'+status;break
                    nxt,nb,na=c;nt=w.tangent(nb,tan,tag2)
                    reverse,rs=w.correct(nxt-.015*nt,cur,tan,j,sign,tag2+'-reverse')
                    if reverse is None:trace['stop']='reverse_'+rs;break
                    mismatch=float(np.max(np.abs(reverse[0]-cur)));w.emit('reverse_check',{'tag':tag2,'old':cur,'reverse':reverse[0],'mismatch':mismatch})
                    if mismatch>1e-7:trace['stop']='reverse_mismatch';break
                    nr=w.node(nxt,nb,na,nt,sign,tag2);nr['reverse_mismatch']=mismatch;trace['nodes'].append(nr);cur=nxt;tan=nt
                result['traces'].append(trace);S.save(args.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_family_calls=w.calls,scope='fixed short floating pseudoarclength traces with local reverse checks; not a validated curve, whole shrimp or continuous-time result');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
