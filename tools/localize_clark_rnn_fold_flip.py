#!/usr/bin/env python3
"""Two preselected native period7 fold--flip bracket localizations, numerical."""
import argparse,importlib.util,json,os,resource,signal,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('endpoint',ROOT/'tools/extend_clark_rnn_p7.py');E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E);S=E.S;R=E.R
from neural_shrimp.clark_rnn_codim2 import fold_flip_augmented
PIN='01297f353bdcc889bee8ffe497fa3cea28186836f45596b120ddc8d12a088d93'
COL=np.array([20.]*4+[10.,.1]+[1.]*8);ROW=np.array([.05]*4+[1.]*10)
EXTRA=E.EXTRA+['src/neural_shrimp/clark_rnn_codim2.py','tools/localize_clark_rnn_fold_flip.py','tools/launch_clark_rnn_fold_flip.py','docs/plans/clark-rnn-p7-fold-flip.md']

class Work(E.Work):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>45:raise TimeoutError('45s internal')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=128:raise RuntimeError('128 augmented returns')
    def full(self,z,j,tag):
        self.check();self.calls+=1;t=time.monotonic();h,dh,a=fold_flip_augmented(z,j,7)
        self.emit('fold_flip_augmented',dict(a,z=z.copy(),h=h,dh=dh,tag=tag));return h,dh,a,time.monotonic()-t
    def localize(self,z,j,tag):
        center=z.copy();seedcycle=None
        for iteration in range(12):
            h,dh,a,_=self.full(z,j,tag)
            if seedcycle is None:seedcycle=a['cycle'].copy()
            if np.max(np.abs(h))<1e-10:
                m=a['monodromy'];ev,vec=np.linalg.eig(m);self.emit('qualification_eigen',dict(tag=tag,matrix=m,eigenvalues=ev,eigenvectors=vec))
                le,lv=np.linalg.eig(m.T);self.emit('qualification_left_eigen',dict(tag=tag,matrix=m.T,eigenvalues=le,eigenvectors=lv))
                scaled=ROW[:,None]*dh*COL[None,:];u,sv,vh=np.linalg.svd(scaled);self.emit('qualification_svd',dict(tag=tag,matrix=scaled,u=u,s=sv,vh=vh))
                ip=int(np.argmin(abs(ev-1)));im=int(np.argmin(abs(ev+1)));others=np.delete(ev,[ip,im])
                q=z[6:10];left=lv[:,int(np.argmin(abs(le-1)))];overlap=float(left.real@q)
                if np.max(abs(left.imag))>1e-8 or abs(overlap)<1e-8:return {'status':'left_normalization_unresolved','tag':tag}
                left=left.real/overlap;bqq=np.einsum('ijk,j,k->i',a['second'][:,:4,:4],q,q);c=float(left@bqq);coupling=left@a['first'][:,4:6]
                scale=max(1,float(np.max(abs(a['cycle']))));sep=float(np.max(abs(a['cycle'][1]-a['cycle'][0]))/scale);pm=E.phase_match(seedcycle,a['cycle'])
                errors=[float(abs(ev[ip]-1)),float(abs(ev[im]+1))];gaps=[float(np.min(abs(np.delete(ev,ip)-1))),float(np.min(abs(np.delete(ev,im)+1)))];rho=float(max(abs(others)))
                checks={'augmented_residual':float(np.max(abs(h))),'critical_errors':errors,'critical_gaps':gaps,'other_radius':rho,'divisor1':sep,'scaled_jacobian_smin':float(sv[-1]),'scaled_jacobian_smax':float(sv[0]),'fold_c':c,'fold_coupling':coupling.tolist(),'native_identity':pm}
                good=ip!=im and max(errors)<1e-7 and min(gaps)>1e-4 and rho<1-1e-5 and sep>1e-6 and sv[-1]>1e-10*max(1,float(sv[0])) and abs(c)>1e-5 and pm['passed']
                status='numerical_fold_flip_candidate' if good else 'unqualified_fold_flip'
                self.emit('fold_flip_result',dict(a,z=z.copy(),right_plus=q,right_minus=z[10:],left_plus=left,augmented_residual=h,augmented_jacobian=dh,tag=tag,status=status,checks=checks))
                return dict(tag=tag,status=status,gain=float(z[4]),delta=float(z[5]),checks=checks)
            if iteration==11:return {'tag':tag,'status':'iteration_limit'}
            scaled=ROW[:,None]*dh*COL[None,:]
            try:dx=np.linalg.solve(scaled,-ROW*h)
            except np.linalg.LinAlgError:return {'tag':tag,'status':'linear_failure'}
            self.emit('newton_solve',dict(tag=tag,iteration=iteration,matrix=scaled,rhs=-ROW*h,scaled_direction=dx))
            nxt=z+COL*dx
            if not np.all(np.isfinite(nxt)) or np.max(abs(dx))>.1 or abs(nxt[4]-center[4])>.05 or abs(nxt[5]-center[5])>.01 or not 1<nxt[4]<5 or not .5<nxt[5]<.95:return {'tag':tag,'status':'trust_guard'}
            z=nxt
        raise AssertionError('unreachable')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.THREADS):raise ValueError('CPU1 nice10')
    args.out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=args.out;S.RAW_CAP=16*1024**2;w=Work(args.out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite reserved cap')
        booked[0]=new
    S.reserve=reserve;available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(args.out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'nice':os.getpriority(os.PRIO_PROCESS,0),'python':sys.version,'numpy':np.__version__,'threads':{k:os.environ.get(k) for k in R.THREADS},'mem_available_bytes':available,'scales':COL.tolist()})
    result={'status':'incomplete','roots':[]}
    try:
        if available<1024**3:raise MemoryError('headroom')
        j=np.eye(4)*.2+np.ones((4,4))*.1;z=np.r_[np.arange(1,5)/10,2.,.7,[1.,0,0,0],[0.,1.,0,0]];times=[]
        for i in range(10):times.append(w.full(z,j,'synthetic_profile')[3])
        projected=64*max(times)+5;S.save(args.out/'profile.json',{'times':times,'projection':projected,'gate':np.isfinite(projected) and projected<=45,'target_unopened':True})
        if not np.isfinite(projected) or projected>45:raise RuntimeError('profile gate')
        raw=R.RAW/'p7-lowgain-extension-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('manifest pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        binding=json.loads((raw/'run/input-binding.json').read_text());ref=Path(binding['matrix_inputs'])
        if ref!=R.RAW/'refine-p5-001/run/p7/inputs.npz' or S.sha(ref)!=binding['matrix_inputs_sha256']:raise ValueError('matrix identity')
        with np.load(ref) as f:j=f['j']
        S.save(args.out/'input-binding.json',{'raw_manifest':PIN,'matrix_input':str(ref),'matrix_sha256':S.sha(ref),'selected_tags':['p7extend-lower:130','p7extend-upper:28'],'period':7})
        for tag in ['p7extend-lower:130','p7extend-upper:28']:
            paths=[p for p in (raw/'run').glob('*-node.json') if json.loads(p.read_text())['tag']==tag];es=[p for p in (raw/'run').glob('*-node_eigen.json') if json.loads(p.read_text())['tag']==tag]
            if len(paths)!=1 or len(es)!=1:raise ValueError('unique fixed bracket seed')
            meta=json.loads(paths[0].read_text());arr=paths[0].with_name(meta['array_file']);em=json.loads(es[0].read_text());ea=es[0].with_name(em['array_file'])
            if S.sha(arr)!=meta['array_sha256'] or S.sha(ea)!=em['array_sha256']:raise ValueError('seed arrays')
            with np.load(arr) as f:base=f['physical']
            with np.load(ea) as f:ev=f['eigenvalues'];vec=f['eigenvectors']
            index=int(np.argmin(abs(ev+1)));rv=vec[:,index]
            if max(abs(rv.imag))>1e-8:raise ValueError('real minus-one seed')
            rv=rv.real/np.linalg.norm(rv.real);z=np.r_[base,rv]
            w.emit('selected_seed',dict(tag=tag,z=z,source_descriptor=str(paths[0]),source_sha256=S.sha(paths[0]),source_array_sha256=S.sha(arr),source_eigen_sha256=S.sha(ea)))
            record={'tag':tag,'status':'incomplete'};result['roots'].append(record);record.update(w.localize(z,j,tag));S.save(args.out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as exc:result.update(status='failed_or_interrupted',error=type(exc).__name__+': '+str(exc))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_augmented_calls=w.calls,scope='numerical local fold-flip candidates only; no interval theorem, branch connection or full shrimp claim');S.save(args.out/'result.json',result,final=True);S.manifest(args.out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
