"""Exactly240 HVPs using only two observed parameter states per forecast."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import explore_tiny_lm_response as B
from neural_shrimp.tiny_lm_two_snapshot import forecast,scores,candidate_gate
from neural_shrimp.tiny_lm import TinyTransformer,LMProblem
BASE=Path('/data2/neural-shrimp-data/explore/tiny-lm-response')
OUT=Path('/data2/neural-shrimp-data/explore/tiny-lm-two-snapshot/attempt001/run')
OWN=['src/neural_shrimp/tiny_lm_two_snapshot.py','tools/explore_tiny_lm_two_snapshot.py','tools/launch_tiny_lm_two_snapshot.py','tests/test_tiny_lm_two_snapshot.py','docs/plans/tiny-lm-two-snapshot-forecast.md']

def main():
    B.need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    B.CAP=64*1024**2;store=B.Store(OUT);start=time.monotonic();events=[];rows=[]
    def stop(*_):raise TimeoutError('fixed180s worker')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,180)
    def check():
        B.need(time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'180s/4GiB worker')
    answer=dict(passed=False,rows=rows)
    try:
        source={n:B.entry(ROOT/n)for n in B.SOURCES+OWN}
        for n,v in source.items():
            B.need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed source');store.write('source/'+n,(ROOT/n).read_bytes())
        B.need(source[B.SOURCES[0]]['sha256']==B.NATIVE_SHA,'same native source')
        store.json('runtime.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sources=source,numpy=np.__version__,torch=torch.__version__,cpu_threads=1,protected_data_evaluated=False))
        target=BASE/'attempt001-target';run=target/'run'
        B.inventory(target,'fc81a70174bddf0c1dcf9a5e5cec27888baf7ab13b0610e118c8b36b43808ce5');B.inventory(run,'cb57135497afa72437651b7eaab807e58387a304fd0f0d2609113ffcff073aad')
        old=B.load(run/'result.json');B.need(old['passed'] and old['operation_returned']==dict(gradient=580,hvp=256,ce=96),'complete parent')
        audit=Path('/data2/neural-shrimp-data/audits/tiny-lm-response-independent/attempt001');B.inventory(audit,'296d015f24900d8db9e72b9dc89c71786999a7885aa7267a25856f4d621a9948')
        B.need(B.load(audit/'run/result.json')['passed'],'independent parent audit')
        profile=BASE/'attempt001-profile/run';B.inventory(profile,'246cbba006bd85ade50a5060587e9ba5f0b56da5f8f35bfb86960273c738a84f');pr=B.load(profile/'result.json')
        for record in (old,pr):
            B.need(record['torch']==torch.__version__ and record['numpy']==np.__version__ and record['source_identity']=={n:source[n]for n in B.SOURCES},'original source/library-qualified operation costs')
        B.need(pr['passed'] and pr['operation_returned']==dict(gradient=4,hvp=4,ce=4),'complete original profile')
        costs=pr['profile_costs'];B.need([x['arm']for x in costs]==[8,256],'two fixed arms')
        for x in costs:B.need(len(x['times']['hvp'])==2 and all(np.isfinite(v)and v>0 for v in x['times']['hvp']),'measured finite HVP costs')
        estimate=240*sum(max(x['times']['hvp'])for x in costs)+20;B.need(estimate<180,'measured180s feasibility')
        store.json('bindings.json',dict(parent=str(target),profile=str(profile),audit=str(audit),forecast_seconds=estimate,operation_cap=240))
        parent,pin=B.PARENTS['target'];B.inventory(parent,pin)
        for seed in (28161,28162):
            for arm in (8,256):
                label=f'seed-{seed}-arm-{arm}';folder=run/label
                theta,x,y,eta,binding=B.select(parent,'target',seed,arm,600)
                reference=np.load(folder/'reference.npy',allow_pickle=False)
                B.need(reference.dtype==np.float64 and reference.shape==(17,15776) and np.array_equal(reference[0],theta),'same promoted initial state')
                problem=LMProblem(TinyTransformer(66),torch.tensor(x),torch.tensor(y),ridge=.001)
                def hvp(t,v):
                    check();B.need(len(events)<240,'fixed operation cap')
                    B.need(np.array_equal(t,reference[0])or np.array_equal(t,reference[1]),'no future state predictor access')
                    event=dict(case=label,index=len(events),snapshot=0 if np.array_equal(t,reference[0])else 1,returned=False);events.append(event);beg=time.monotonic()
                    result=problem.hvp(torch.as_tensor(t,dtype=torch.float64),torch.as_tensor(v,dtype=torch.float64)).detach().numpy().copy()
                    event.update(returned=True,seconds=time.monotonic()-beg);return result
                outcome=B.load(folder/'outcome.json');readout=np.load(folder/'readout-gradients.npy',allow_pickle=False)
                for direction in range(2):
                    initial=np.load(folder/f'direction-{direction}-transported.npy',allow_pickle=False)[1]
                    predictions=forecast(initial,eta,reference[0],reference[1],hvp,lambda n,v:store.array(label+f'/direction-{direction}-'+n,v),check)
                    plus=np.load(folder/f'direction-{direction}-epsilon-1-plus.npy',allow_pickle=False);minus=np.load(folder/f'direction-{direction}-epsilon-1-minus.npy',allow_pickle=False)
                    responses={h:(plus[h]-minus[h])/(2*outcome['epsilon_values'][1])for h in (4,16)}
                    name='gradient'if direction==0 else'random';prior={v['horizon']:v for v in outcome['rows']if v['direction']==name}
                    measured=scores(predictions,responses,{h:readout[[1,4,16].index(h)]for h in (4,16)},{h:prior[h]['training_ce_slopes'][1]for h in (4,16)})
                    for row in measured:row.update(seed=seed,arm=arm,direction=name,mechanism_qualified=prior[row['horizon']]['mechanism_qualified'],persistence_vector_error=prior[row['horizon']]['persistence_vector_error'],frozen_vector_error=prior[row['horizon']]['frozen_vector_error'])
                    rows.extend(measured);store.json(label+f'/direction-{direction}-outcome.json',measured)
        B.need(len(events)==240 and all(x['returned']for x in events)and len(rows)==16,'complete operation/row inventory')
        answer.update(passed=True,candidate_gate=candidate_gate(rows),forecast_seconds=estimate,scope='Adaptive consumed-development two-snapshot forecast; no later Hessian enters predictors. No training intervention, fresh confirmation, quality or chaos claim.')
    except BaseException as e:
        import traceback;answer.update(error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);answer.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,hvp_attempted=len(events),hvp_returned=sum(x['returned']for x in events));answer['passed']=bool(answer['passed']and answer['seconds']<180 and answer['peak_rss_kib']<4*1024**2);store.json('events.json',events);store.finish(answer)
    print(json.dumps({k:v for k,v in answer.items()if k!='rows'}));return 0 if answer['passed']else 2
if __name__=='__main__':raise SystemExit(main())
