#!/usr/bin/env python3
"""One fixed low-g endpoint continuation, with phase/sign and cubic diagnostics."""
import argparse,importlib.util,json,os,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pal',ROOT/'tools/continue_clark_rnn_boundaries.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);S=P.S;R=P.R;COL=P.COL
from neural_shrimp.clark_rnn_boundaries import third_directional
PIN='8f56a841ca05e2f81d8ebbdcd1e812da03c8309af6568170dc27980352a24fd0'
EXTRA=P.EXTRA+['tools/extend_clark_rnn_lowgain.py','tools/launch_clark_rnn_lowgain.py','docs/plans/clark-rnn-lowgain-extension.md']

def phase_match(a,b):
    scale=max(1,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    options=[(float(np.max(np.abs(a-s*np.roll(b,k,axis=0)))/scale),s,k) for s in [1,-1] for k in range(len(a))]
    error,sign,shift=min(options);native=float(np.max(np.abs(a-b))/scale)
    return {'best_error':error,'best_sign':sign,'best_shift':shift,'native_error':native,'passed':sign==1 and shift==0 and native<.05}

class Work(P.Work):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>120:raise TimeoutError('120s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=7000:raise RuntimeError('7000 family-call cap')
    def node(self,y,b,a,tangent,sign,tag):
        r=super().node(y,b,a,tangent,sign,tag)
        if sign==1:
            r['nondegenerate_fold']=abs(r['fold_c'])>1e-5 and max(abs(x) for x in r['fold_parameter_coupling'])>1e-5
        else:
            m=a['monodromy'];q=y[6:];le,lv=np.linalg.eig(m.T);self.emit('coefficient_left_eigen',{'tag':tag,'matrix':m.T,'eigenvalues':le,'eigenvectors':lv});idx=int(np.argmin(np.abs(le+1)));left=lv[:,idx].real;left/=left@q
            h=a['second'][:,:4,:4];bqq=np.einsum('ijk,j,k->i',h,q,q);rhs=np.column_stack([-a['first'][:,4:6],bqq]);sol=np.linalg.solve(m-np.eye(4),rhs);self.emit('coefficient_solve',{'tag':tag,'matrix':m-np.eye(4),'rhs':rhs,'solution':sol})
            c,stages=third_directional(a['theta'],q,self.j,float(y[4]*10),float(y[5]*.1),5);self.emit('third_directional',{'tag':tag,'theta':a['theta'],'direction':q,'stages':stages,'third':c})
            alpha=left@(np.einsum('ijk,j,km->im',h,q,sol[:,:2])+np.einsum('ijk,j->ik',a['second'][:,:4,4:6],q))
            cubic=float(left@c/6-left@np.einsum('ijk,j,k->i',h,q,sol[:,2])/2)
            r.update(flip_alpha_gain=float(alpha[0]),flip_alpha_delta=float(alpha[1]),flip_cubic=cubic,nondegenerate_flip=bool(max(abs(alpha))>1e-5 and abs(cubic)>1e-5))
        self.emit('node_nondegeneracy',r);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=128*1024**2;w=Work(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite reserved cap')
        booked[0]=new
    S.reserve=reserve;available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__,'threads':{k:os.environ.get(k) for k in R.THREADS},'mem_available_bytes':available,'scales':COL.tolist()})
    result={'status':'incomplete','traces':[]}
    try:
        if available<1024**3:raise MemoryError('headroom')
        j=np.eye(4)*.2+np.ones((4,4))*.1;y=np.r_[np.arange(1,5)/10,2.,.5,[1.,0,0,0]]/COL;ts=[]
        for k in range(10):ts.append(w.full(y,j,1,'synthetic_profile')[3])
        projected=6500*max(ts)+20;S.save(args.out/'profile.json',{'times':ts,'projection':projected,'gate':np.isfinite(projected) and projected<=120,'target_unopened':True})
        if not np.isfinite(projected) or projected>120:raise RuntimeError('profile gate')
        raw=R.RAW/'p5-boundary-curves-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('manifest pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        ref=R.RAW/'refine-p5-001/run/p5/inputs.npz';old=R.RAW/'p5-boundary-slices-001/run/input-binding.json'
        oldroot=old.parents[1]
        if S.sha(oldroot/'manifest.json')!=P.PIN:raise ValueError('slice ancestor manifest')
        oldinv=json.loads((oldroot/'manifest.json').read_text())['files']
        if S.sha(old)!=oldinv['run/input-binding.json']['sha256']:raise ValueError('slice binding identity')
        binding=json.loads(old.read_text())
        # Bind original input descriptor transitively to the prior curve's frozen
        # family source, and retain its exact bytes in this output.
        if S.sha(ref)!=binding['source_input_sha256']:raise ValueError('matrix identity')
        with np.load(ref) as f:j=f['j']
        w.j=j;S.save(args.out/'input-binding.json',{'curve_manifest':PIN,'matrix_inputs':str(ref),'matrix_inputs_sha256':S.sha(ref),'slice_binding':str(old),'slice_binding_sha256':S.sha(old),'endpoint_tags':['1:-1:48','-1:-1:48']})
        for sign in [1,-1]:
            tag=f'{sign}:-1:48';paths=[p for p in (raw/'run').glob('*-node.json') if json.loads(p.read_text())['tag']==tag]
            if len(paths)!=1:raise ValueError('unique endpoint')
            path=paths[0];meta=json.loads(path.read_text());arr=path.with_name(meta['array_file'])
            if S.sha(arr)!=meta['array_sha256']:raise ValueError('endpoint array')
            with np.load(arr) as f:cur=f['y'];tan=f['tangent']
            h,b,a,_=w.full(cur,j,sign,tag+'-endpoint');trace={'sign':sign,'starting_tag':tag,'source_node_sha256':S.sha(path),'step':.015,'nodes':[],'stop':'incomplete'};result['traces'].append(trace)
            if np.max(np.abs(h))>1e-9:raise ValueError('endpoint closure')
            trace['nodes'].append(w.node(cur,b,a,tan,sign,tag+'-endpoint'));oldcycle=a['cycle'];trace['stop']='node_cap'
            for k in range(1,193):
                tag2=f'extend{sign}:{k}';pred=cur+.015*tan
                if not 1<pred[4]*10<40 or not .05<pred[5]*.1<1:trace['stop']='domain_edge';break
                c,status=w.correct(pred,pred,tan,j,sign,tag2)
                if c is None:trace['stop']='forward_'+status;break
                nxt,nb,na=c;nt=w.tangent(nb,tan,tag2);pm=phase_match(oldcycle,na['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag2,parent_cycle=oldcycle,child_cycle=na['cycle']))
                if not pm['passed']:trace['stop']='phase_sign_identity';break
                rev,rs=w.correct(nxt-.015*nt,cur,tan,j,sign,tag2+'-reverse')
                if rev is None:trace['stop']='reverse_'+rs;break
                err=float(np.max(np.abs(rev[0]-cur)));w.emit('reverse_check',{'tag':tag2,'old':cur,'reverse':rev[0],'error':err})
                if err>1e-7:trace['stop']='reverse_mismatch';break
                nr=w.node(nxt,nb,na,nt,sign,tag2);nr.update(reverse_mismatch=err,phase_sign_match=pm);trace['nodes'].append(nr);cur=nxt;tan=nt;oldcycle=na['cycle']
                if k%8==0:S.save(args.out/'progress.json',result)
            S.save(args.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_family_calls=w.calls,scope='adapted fixed endpoint extension; numerical coefficients and tracking only, no interval or complete shrimp claim');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
