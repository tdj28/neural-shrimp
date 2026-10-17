"""Prospective manufactured-only E5 cost measurement. No target data or model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent

def load(name):
    s=importlib.util.spec_from_file_location('e5_profile_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

A=load('arithmetic')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,obj):
    with open(p,'x') as f:json.dump(A.clean(obj),f,sort_keys=True,separators=(',',':'),allow_nan=False)


def workload():
    V=151936; ix=np.arange(V,dtype=float)
    # Fixed synthetic logits, fully declared before either profile or actual outcomes.
    vectors=[-25*ix/(V-1),-.001*ix-.02*np.sin(.01*ix)]
    mask=np.ones(V,bool);mask[list(A.MASKED)]=False;grid=np.linspace(-1.,.5,24)
    start=time.perf_counter();entropy=[]
    for i,z in enumerate(vectors):
        full=A.entropy_root(z);masked=A.entropy_root(z[mask]);curves={}
        for tag,v in [('full',z),('masked',z[mask])]:curves[tag]=[A.entropy(v,10.**t) for t in grid]
        entropy.append(dict(vector=i,full=full,masked=masked,curves=curves,entropy_at_T1=A.entropy(z,1.),max_probability_T1=float(1/np.exp(z-z.max()).sum()),entropy_middle=A.entropy(z,10**-.25)))
    entropy_seconds=time.perf_counter()-start
    if not all(e['full']['defined'] and e['masked']['defined'] for e in entropy):raise ValueError('manufactured entropy root undefined')
    pids=[8*s+o for s in range(2,8) for o in range(8) if (s+o)%2==0];rows=[]
    for i,p in enumerate(pids):
        labels=['periodic' if j<4+i%12 else 'chaotic' if j<21 else 'irregular' for j in range(24)]
        fractions=np.clip((4+(i*5)%16-np.arange(24))/3,0,1)
        rows.append(dict(prompt=p,system_index=p//8,opener_index=p%8,labels=labels,T_chaos=A.label_onset(grid,labels,'chaotic'),T_nonperiodic=A.label_onset(grid,['periodic' if v=='periodic' else 'non-periodic' for v in labels],'non-periodic'),T_rep=A.rep_onset(grid,fractions),log10_T_H=-.7+.1*(i%7)+.003*i,repeat_fraction=fractions.tolist()))
    start=time.perf_counter();stats,draws=A.statistics(rows,grid,1000);stat_seconds=time.perf_counter()-start
    return dict(schema='e5-manufactured-workload-v1',entropy=entropy,rows=rows,statistics=stats,draws=draws,measurement=dict(entropy_seconds=entropy_seconds,statistics_seconds=stat_seconds),shape=dict(logit_vectors=2,vocabulary=V,full_and_masked_roots_per_vector=2,bisections_per_root=64,entropy_grid_temperatures=24,prompts=24,bootstrap_draws=1000,bootstrap_streams=7,permutations=1000),extrapolation_scope=dict(entropy_multiplier_to_actual=16,statistical_multiplier_to_actual=20,actual_loop_state_and_token_IO_hashing_and_period_processing_unmeasured=True,actual_comparison_and_output_lengths_unmeasured=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args();plan=json.loads(Path(args.card).read_text())
    if plan.get('status')!='reviewed':raise ValueError('profile plan unadmitted')
    if set(plan['source_sha256'])!={'profile','arithmetic','controls'}:raise ValueError('profile source inventory')
    for n,h in plan['source_sha256'].items():
        if sha(HERE/(n+'.py'))!=h:raise ValueError('source pin '+n)
    out=Path(args.out)
    if str(out)!=plan['output']:raise ValueError('output contract')
    out.mkdir(parents=True,exist_ok=False)
    try:
        start=time.perf_counter();C=load('controls');checks=C.checks();write(out/'controls.json',checks)
        if checks['passed']!=10 or checks['failed']!=0:raise ValueError('manufactured controls failed')
        data=workload();write(out/'workload.json',data)
        elapsed=time.perf_counter()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir())]
        report=dict(schema='e5-manufactured-profile-v1',computational_status='complete',source_sha256=plan['source_sha256'],controls=checks,shape=data['shape'],measurement=data['measurement'],runtime=dict(seconds_through_workload_serialization=elapsed,observed_peak_kib_before_report=rss),outputs=outputs,extrapolation_scope=data['extrapolation_scope'],limits='No target files/model were read. Timings omit actual input hashes/decompression/token period processing, actual comparison/output lengths, final report and external lifecycle. Linear extrapolation is a sizing estimate, not a guarantee.')
        write(out/'report.json',report)
    except BaseException as e:
        write(out/'failure.json',dict(type=type(e).__name__,message=str(e)));raise

if __name__=='__main__':main()
