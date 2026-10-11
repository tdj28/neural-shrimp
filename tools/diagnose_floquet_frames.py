#!/usr/bin/env python3
"""Saved proposal conditioning only; no complementary-spectrum target trial."""
import gc,json,os,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_floquet_complement as D
from flint import arb_mat,ctx
import numpy as np
O=D.O;need=D.need;read=D.read
OUT=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/attempt004-diagnosis/run')
FILES=sorted(set(D.FILES+['tools/diagnose_floquet_frames.py','tools/launch_floquet_diagnosis.py','tests/test_floquet_frame_diagnosis.py','docs/plans/floquet-frame-diagnosis.md']))

def diagnostic(flat,proposal,tick):
    need(len(flat)==704 and len(proposal['phase_complements'])==18,'fixed full18 phase proposal')
    out=[]
    def condition(s):return None if s[-1]==0 or not np.isfinite(s[0]/s[-1])else float(s[0]/s[-1])
    for j in range(18):
        tick();off=39*j;q=flat[off+13:off+26];r=flat[off+26:off+39]
        K=np.array([[float(q[i].mid()),float(r[i].mid())]for i in range(13)])
        E=D.U.R.matrix(proposal['phase_complements'][j]);e=np.array([[float(E[i,k].mid())for k in range(11)]for i in range(13)])
        u,s,_=np.linalg.svd(K,full_matrices=True);orth=D.exact_float_matrix(u[:,2:]);F=np.column_stack([K,e]);G=np.column_stack([K,u[:,2:]])
        old_s=np.linalg.svd(F,compute_uv=False);orth_s=np.linalg.svd(G,compute_uv=False)
        frame=arb_mat([[q[i],r[i]]+[orth[i,k]for k in range(11)]for i in range(13)])
        try:inverse=frame.inv();inverts=all(v.is_finite()for row in inverse.tolist()for v in row)
        except ZeroDivisionError:inverts=False
        out.append({'phase':j,'critical_singular_values':s.tolist(),'critical_condition_float':condition(s),'failed_basis_singular_values':old_s.tolist(),'failed_basis_condition_float':condition(old_s),'orthogonal_basis_singular_values':orth_s.tolist(),'orthogonal_basis_condition_float':condition(orth_s),'orthogonal_whole_frame_inversion_succeeded':inverts,'critical_max_radius':max(float(v.rad())for v in q+r),'orthogonal_proposal':D.U.M.matrix_json(orth)})
    return out

def main():
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip(),'clean source')
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in O.THREADS),'CPU1nice10')
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=[0];stop=[False];result={'passed':False,'status':'incomplete','native_calls':0,'complementary_stability_evaluated':False}
    def save(name,value):
        raw=D.canonical_bytes(value);need(used[0]+len(raw)<16*1024**2,'16MiB raw cap');p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        used[0]+=len(raw)
    def tick():need(not stop[0]and time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'30s/2GiB cap')
    signal.signal(signal.SIGALRM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.setitimer(signal.ITIMER_REAL,30)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save('runtime.json',{'source_commit':source,'pid':os.getpid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':dict.fromkeys(O.THREADS,'1'),'sources':{n:O.entry(ROOT/n)for n in FILES},'precision':384,'python':D.platform.python_version(),'numpy':np.__version__,'scipy':D.scipy.__version__,'python_flint':D.flint.__version__,'flint':D.flint.__FLINT_VERSION__})
    for n in FILES:p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);data=(ROOT/n).read_bytes();p.write_bytes(data);used[0]+=len(data)
    try:
        with ctx.workprec(384):
            card=read(ROOT/D.CARD);flat,blocks,binding=D.bind(card,tick);save('bindings.json',binding)
            previous=card['failed_basis'];pr=Path(previous['root']);O.verify(pr,previous['outer']);O.verify(pr/'run',previous['worker']);D.B.archived(pr/'run',previous['source'],'runtime.json','source')
            po=read(pr/'run/outcome.json');pl=read(pr/'launcher-result.json');need(po['status']=='failed_or_interrupted'and po['error']=='ZeroDivisionError: matrix is singular'and not po['passed']and po['native_calls']==0 and pl['exit_code']==2 and not pl['owned_live_pids'],'exact retained first-frame failure')
            inp=read(pr/'run/inputs.json');need(inp['whole_box']==flat and inp['native_blocks']==blocks,'same already proved saved domains')
            proposal=read(pr/'run/events/001-proposal.json');rows=diagnostic([D.U.R.original_ball(v)for v in flat],proposal,tick);save('diagnosis.json',{'rows':rows,'prior':previous,'proposal_identity':O.entry(pr/'run/events/001-proposal.json'),'scope':'Numerical conditioning diagnostics and conditional frame-inversion feasibility only. No complementary-product bound or stability trial.'});tick()
            result.update(passed=True,status='saved_frame_diagnosis_completed',phase_count=len(rows),orthogonal_inversion_count=sum(v['orthogonal_whole_frame_inversion_succeeded']for v in rows),first_failed_basis_condition=rows[0]['failed_basis_condition_float'],first_orthogonal_condition=rows[0]['orthogonal_basis_condition_float'])
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);save('manifest.json',{'files':{str(p.relative_to(OUT)):O.entry(p)for p in sorted(OUT.rglob('*'))if p.is_file()}})
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
