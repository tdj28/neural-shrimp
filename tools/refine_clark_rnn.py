#!/usr/bin/env python3
"""Adapted, bounded longer-transient p5 patch and fixed p7 comparison."""
import argparse,importlib.util,json,os,platform,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('clark_store',ROOT/'tools/explore_clark_rnn.py')
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
from neural_shrimp.clark_rnn import inputs,simulate_batch,qualify_cycle
EXTRA=['tools/refine_clark_rnn.py','tools/launch_clark_rnn_refinement.py','docs/plans/clark-rnn-p5-refinement.md']
PANEL_PIN='27514e8a0473b75fa95eac8aac3656b8d457170b9ac9e315761b366c8d17f8f5'
PROFILE_PIN='422e0e8b37d43b4162088c069181a1a6770b01aaa026db1008d3a9fa2a91a2dd'
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn')
THREADS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']

def resource_projection(profile):
    a=profile['batch_seconds'];b=profile['period64_unit_seconds'];rss=profile['peak_rss_kib']
    if not all(np.isfinite(x) and x>0 for x in [a,b,rss]):raise ValueError('finite positive profile resources required')
    seconds=2*((2601+28)/81)*(34192/11096)*a+128*8*b+20
    peak=rss*1024+3*(513*5202*4*8)+64*1024**2
    if seconds>300 or peak>512*1024**2:raise ValueError('adapted resource gate')
    return {'projected_seconds':seconds,'conservative_peak_bytes':peak,'matrix_dimension_discount':False,'basis':'same core N16 measured resource profile; changed horizon/grid arithmetic, no new profile'}

def spaced(cells,limit):
    cells=np.asarray(cells,dtype=int)
    if len(cells)<=limit:return cells
    return cells[np.linspace(0,len(cells)-1,limit,dtype=int)]

def selection(periods,kind):
    p=np.asarray(periods).reshape(-1,2);same=p[:,0]==p[:,1]
    if kind=='p5':
        groups=[spaced(np.flatnonzero(same&(p[:,0]==5)),32),spaced(np.flatnonzero(same&np.isin(p[:,0],[10,20,40])),16)]
        cells=np.sort(np.concatenate(groups))
    elif kind=='p7':cells=spaced(np.flatnonzero(same&(p[:,0]>=3)),16)
    else:raise ValueError('unknown panel')
    return np.array([2*int(c)+i for c in cells for i in range(2)],dtype=int)

def bound_original(identity):
    panel=RAW/'scan-001'
    if S.sha(panel/'manifest.json')!=PANEL_PIN:raise ValueError('panel manifest pin')
    inv=json.loads((panel/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(panel)) for p in panel.rglob('*') if p.is_file() and p!=panel/'manifest.json'}:raise ValueError('panel inventory')
    for name,d in inv.items():
        p=panel/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('panel binding')
    with np.load(panel/'run/n4-s28171/inputs.npz') as f:saved={k:f[k] for k in f.files}
    j,x,v=inputs(4,28171)
    for key,actual in [('j',j),('initials',x),('tangents',v)]:
        if saved[key].dtype!=actual.dtype or saved[key].shape!=actual.shape or saved[key].tobytes()!=actual.tobytes():raise ValueError('fixed input identity')
    S.profile_gate(RAW/'profile-001/run',PROFILE_PIN,{'sha256':{k:identity['sha256'][k] for k in S.SOURCES}})
    gate=resource_projection(json.loads((RAW/'profile-001/run/result.json').read_text()))
    return j,x,v,saved,gate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('CPU1 nice>=10 required')
    # Input verification is inside the bounded worker and retained if it fails.
    out=args.out;out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;start=time.monotonic();interrupted=False
    def term(_sig,_frame):
        nonlocal interrupted;interrupted=True
    signal.signal(signal.SIGTERM,term)
    def check():
        if interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-start>300:raise TimeoutError('300s internal cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker cap')
        if S.used(out)>300*1024**2:raise OSError('300MiB storage cap')
    runtime={'source':identity,'pid':os.getpid(),'mode':'adapted_p5_refinement','started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ.get(k) for k in THREADS},'load_average':list(os.getloadavg()),'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024}
    S.save(out/'runtime-start.json',runtime)
    result={'status':'incomplete','panels':[],'scope':'adapted exploratory finite-horizon discrete-map evidence, no geometry/chaos/continuous-time proof'}
    seq=[0]
    def emit(folder,index,kind,value):
        seq[0]+=1;name=f'cycle-{index:05d}-event-{seq[0]:05d}'
        arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
        S.save_npz(folder/(name+'.npz'),**arrays);meta.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(folder/(name+'.npz')));S.save(folder/(name+'.json'),meta)
    try:
        if runtime['mem_available_bytes']<1024**3:raise MemoryError('less than1GiB host headroom')
        j,x,v,old,gate=bound_original(identity);check();S.save(out/'resource-gate.json',gate)
        panels=[('p5',np.linspace(6.5,17.5,51),np.linspace(.90,1.,51)),('p7',old['gains'][8:15],old['deltas'][14:18])]
        S.save(out/'design.json',{'adapted_after':'scan-001 spatial outcomes; lowest-period strictly interior component with>=3cells and>=3finite-positive neighboring cells','original_panel_manifest':PANEL_PIN,'profile_manifest':PROFILE_PIN,'n':4,'seed':28171,'burn':24000,'align':2000,'measured':8192,'tail':513,'block':256,'selection':'p5:32 consensus5 cells +16 consensus10/20/40 cells; p7:16 consensus>=3 cells; evenly spaced sorted indices, both starts','panels':[{ 'kind':k,'gains':g.tolist(),'deltas':d.tolist()} for k,g,d in panels],'maximum_corrections':128})
        for kind,gains,deltas in panels:
            check();folder=out/kind;folder.mkdir();S.save_npz(folder/'inputs.npz',j=j,initials=x,tangents=v,gains=gains,deltas=deltas)
            def checkpoint(t,a,b):S.save_npz(folder/f'checkpoint-{t:05d}.npz',step=np.array(t),state=a,tangent=b)
            t=time.monotonic();data=simulate_batch(j,x,v,gains,deltas,burn=24000,align=2000,measured=8192,check=check,checkpoint=checkpoint);elapsed=time.monotonic()-t
            S.save_npz(folder/'batch.npz',**data)
            inds=selection(data['period_proposals'],kind);S.save(folder/'selection.json',{'indices':inds.tolist(),'rule':'fixed before this local outcome; no ranking by corrected stability'})
            records=[]
            for index in inds:
                check();i=int(index);g,d=map(float,data['grid'][i]);p=int(data['period_proposals'][i])
                S.save_npz(folder/f'cycle-{i:05d}-start.npz',theta=data['final'][i],period=np.array(p),grid=data['grid'][i])
                q=qualify_cycle(data['final'][i],j,g,d,p,check,emit=lambda k,val:emit(folder,i,k,val))
                arrays={k:v for k,v in q.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in q.items() if not isinstance(v,np.ndarray)}
                S.save_npz(folder/f'cycle-{i:05d}.npz',**arrays);meta.update(index=i,gain=g,delta=d,initial=i%2);S.save(folder/f'cycle-{i:05d}.json',meta);records.append(meta)
            periods=data['period_proposals'];halves=data['growth_blocks'].reshape(2,16,-1).sum(axis=1)/4096/data['grid'][:,1]
            report={'kind':kind,'batch_seconds':elapsed,'trajectories':len(periods),'period_counts':{str(p):int(np.sum(periods==p)) for p in np.unique(periods)},'two_start_period_disagreements':int(np.sum(periods[::2]!=periods[1::2])),'qualified_seed_cases':sum(r['status']=='numerical_attracting_primitive' for r in records),'qualification_attempts':len(records),'qualification_statuses':{s:sum(r['status']==s for r in records) for s in sorted(set(r['status'] for r in records))},'positive_both_half_time_estimates':int(np.sum((data['tangent_collapse_count']==0)&np.all(halves>.002,axis=0))),'tangent_collapse_cases':int(np.sum(data['tangent_collapse_count']>0))}
            result['panels'].append(report);S.save(folder/'summary.json',report);S.save(out/'progress.json',result);del data
        check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':sys.exit(main())
