#!/usr/bin/env python3
"""Resume only unresolved p10 endpoints, with explicit native symmetry-tie evidence."""
import argparse,importlib.util,json,os,resource,signal,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p10boundary',ROOT/'tools/explore_clark_rnn_new_p10_boundaries.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B);S=B.S;R=B.R
PIN='18fa07a209e4697aeb3ec4159fff57867a1d1bfa3f4f1f5042dc6288cc308785'
STARTS=[(42,-1,'g42-seed'),(42,1,'g42-seed'),(44,-1,'g44-dir-1-step47'),(44,1,'g44-dir1-step3'),(46,-1,'g46-dir-1-step3')]
SYMMETRY_TOL=1e-10
EXTRA=B.EXTRA+['tools/extend_clark_rnn_new_p10_slices.py','tools/launch_clark_rnn_new_p10_extension.py','docs/plans/clark-rnn-new-p10-slice-extension.md']
old_phase_match=B.phase_match

def phase_match(a,b):
    r=old_phase_match(a,b)
    # Keep both stored native cycles unchanged. Only resolve the argmin tie
    # when odd symmetry plus five phase steps acts nearly as identity on BOTH.
    if a.shape!=(10,4) or b.shape!=(10,4):raise ValueError('native period10 shape')
    scale=max(1,float(np.max(np.abs(a))),float(np.max(np.abs(b))))
    pa=float(np.max(np.abs(a+np.roll(a,5,axis=0)))/scale);pb=float(np.max(np.abs(b+np.roll(b,5,axis=0)))/scale);gap=abs(r['native_error']-r['best_error'])
    tie=(r['best_sign']==-1 and r['best_shift']==5 and pa<SYMMETRY_TOL and pb<SYMMETRY_TOL and gap<SYMMETRY_TOL and r['native_error']<.05)
    r.update(original_argmin_passed=r['passed'],antipodal_half_period_tie=bool(tie),parent_odd_self_defect=pa,child_odd_self_defect=pb,native_best_gap=gap,symmetry_tolerance=SYMMETRY_TOL,passed=bool(r['passed'] or tie),state_transform_applied=False)
    return r
# A source-bound adapter override; old module bytes and old raw labels stay unchanged.
B.phase_match=phase_match

class Work(B.Worker):
    def check(self):
        if self.interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-self.start>90:raise TimeoutError('90s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
        if self.calls>=18000:raise RuntimeError('18000 native return cap')

def load_point(raw,tag):
    es=[p for p in raw.glob('*-native_eigen.json') if json.loads(p.read_text())['tag']==tag]
    ns=[p for p in raw.glob('*-native.json') if json.loads(p.read_text())['tag']==tag]
    if len(es)!=1 or not ns:raise ValueError('unique retained endpoint')
    ep=es[0];np_=sorted(ns)[-1];em=json.loads(ep.read_text());nm=json.loads(np_.read_text())
    with np.load(ep.with_name(em['array_file'])) as f:e={k:f[k].copy() for k in f.files}
    with np.load(np_.with_name(nm['array_file'])) as f:n={k:f[k].copy() for k in f.files}
    if e['theta'].tobytes()!=n['theta'].tobytes():raise ValueError('endpoint native/eigen join')
    if any(v<=1e-6 for v in B.primitive_separations(n['cycle']).values()):raise ValueError('endpoint primitive10')
    return {'theta':e['theta'],'cycle':n['cycle'],'delta':em['delta'],'eigenvalues':e['eigenvalues'],'eigenvectors':e['eigenvectors'],'radius':float(np.max(np.abs(e['eigenvalues']))),'closure':em['closure']},em['gain'],{'native':str(np_),'native_sha256':S.sha(np_),'eigen':str(ep),'eigen_sha256':S.sha(ep)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in R.M.THREADS):raise ValueError('CPU1 nice10')
    out=args.out;out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;S.RAW_CAP=256*1024**2;w=Work(out);signal.signal(signal.SIGTERM,lambda *_:setattr(w,'interrupted',True));booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    S.save(out/'runtime-start.json',{'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':__import__('sys').version,'numpy':np.__version__,'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in R.M.THREADS},'mem_available_bytes':available})
    result={'status':'incomplete','directions':[]}
    try:
        if available<1024**3:raise MemoryError('host headroom')
        j=np.eye(4)*.2+np.ones((4,4))*.1;x=np.arange(1,5)/10;z=np.r_[x,.5,[1.,0,0,0]];ta=[];tb=[]
        for k in range(10):ta.append(w.it(x,j,2.,.5,'synthetic_profile')[4]);tb.append(w.aug(z,j,2.,-1,'synthetic_profile')[3])
        cost=16000*max(ta)+120*max(tb)+10;S.save(out/'profile.json',{'native_seconds':ta,'augmented_seconds':tb,'projected_seconds':cost,'gate':np.isfinite(cost) and cost<=90,'target_unopened':True})
        if not np.isfinite(cost) or cost>90:raise RuntimeError('resource screen')
        raw=R.RAW/'new-p10-boundary-slices-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('manifest pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for name,d in inv.items():
            p=raw/name
            if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('input bytes')
        bind=json.loads((raw/'run/input-binding.json').read_text());inp=R.RAW/'new-p10-refinement-001/run/inputs.npz'
        if S.sha(inp)!=bind['source_input_sha256']:raise ValueError('matrix input')
        with np.load(inp) as f:j=f['j']
        S.save(out/'input-binding.json',{'previous_manifest':PIN,'source_input':str(inp),'source_input_sha256':S.sha(inp),'starts':STARTS,'step':.0000625,'max_steps':384,'symmetry_tolerance':SYMMETRY_TOL})
        for gi,direction,tag0 in STARTS:
            point,g,pointer=load_point(raw/'run',tag0);d=point['delta'];last=point;report={'gain_index':gi,'gain':g,'direction':direction,'starting_tag':tag0,'starting_delta':d,'pointer':pointer,'steps':0,'stop':'step_limit','boundary':None};result['directions'].append(report)
            for k in range(1,385):
                tag=f'extend-g{gi}-dir{direction}-step{k}';dn=d+direction*k*.0000625
                if not .415<dn<.495:report['stop']='domain_edge';break
                candidate,status=w.correct(last['theta'],j,g,dn,tag);report['steps']=k
                if candidate is None:report['stop']='correction_'+status;report['boundary']=w.localize(last,j,g,1,tag+'-fold');break
                pm=phase_match(last['cycle'],candidate['cycle']);w.emit('phase_sign_match',dict(pm,tag=tag,parent_cycle=last['cycle'],child_cycle=candidate['cycle']))
                if not pm['passed']:report['stop']='phase_sign_identity';break
                reverse,rs=w.correct(candidate['theta'],j,g,last['delta'],tag+'-reverse')
                if reverse is None:report['stop']='reverse_'+rs;break
                scale=max(1,float(np.max(np.abs(last['cycle']))));err=float(np.max(np.abs(reverse['theta']-last['theta']))/scale);w.emit('reverse_check',{'tag':tag,'old':last['theta'],'reverse':reverse['theta'],'relative_error':err})
                if err>1e-7:report['stop']='reverse_mismatch';break
                if candidate['radius']>=1:
                    e=candidate['eigenvalues'];sign=[1,-1][int(np.argmin([np.min(np.abs(e-s)) for s in [1,-1]]))];report['stop']='first_unstable';report['boundary']=w.localize(candidate,j,g,sign,tag+'-boundary');break
                last=candidate
                if k%16==0:report.update(last_closed_delta=last['delta'],last_closed_radius=last['radius']);S.save(out/'progress.json',result)
            report.update(last_closed_delta=last['delta'],last_closed_radius=last['radius']);S.save(out/'progress.json',result)
        w.check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,attempted_native_calls=w.calls,scope='Native period10 numerical endpoint extension; no cycle quotient, interval/whole shrimp/ODE claim');S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
