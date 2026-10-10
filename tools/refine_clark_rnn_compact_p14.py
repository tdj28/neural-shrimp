#!/usr/bin/env python3
"""One fixed longer-transient patch for the compact new-matrix p14 component."""
import argparse,importlib.util,json,os,platform,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('matrix_panel',ROOT/'tools/explore_clark_rnn_matrix_panel.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);S=M.S
from neural_shrimp.clark_rnn import inputs,simulate_batch,qualify_cycle
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PIN='2d07cad00c7fbbd1055262917ba907f57862bbd35afcf52a89f4254167547e7d';PROFILE='eb7655f6cf176f3d908d235cbf79b485ec3c9555ae1aea2ec7b6483041e3878b'
EXTRA=M.EXTRA+['tools/refine_clark_rnn_compact_p14.py','tools/launch_clark_rnn_compact_p14.py','docs/plans/clark-rnn-compact-p14-refinement.md','tests/test_clark_rnn_compact_p14.py']
def selection(p):
    p=np.asarray(p).reshape(-1,2);same=p[:,0]==p[:,1];groups=[]
    for periods,count in [([14],32),([28,56],16)]:
        inds=np.flatnonzero(same&np.isin(p[:,0],periods))
        if len(inds)>count:inds=inds[np.linspace(0,len(inds)-1,count,dtype=int)]
        groups.extend(inds.tolist())
    return np.array([2*k+i for k in sorted(set(groups)) for i in [0,1]],dtype=int)
def projection(r):
    a,b,rss=r['batch_seconds'],r['period64_unit_seconds'],r['peak_rss_kib']
    if not all(np.isfinite(z) and z>0 for z in [a,b,rss]):raise ValueError('finite profile')
    sec=2*(2601/81)*(34192/21192)*a+96*8*b+20
    peak=rss*1024+3*(513*5202*4*8)+64*1024**2
    raw=513*5202*4*8+18*5202*4*8*2+32*5202*8+96*8*(64*(4+16)*8+16384)+8*1024**2
    if sec>240 or peak>512*1024**2 or raw>256*1024**2:raise ValueError('resource gate')
    return {'projected_seconds':sec,'conservative_peak_bytes':peak,'conservative_raw_bytes':raw,'trajectory_factor':5202/162,'horizon_factor':34192/21192,'safety_factor':2,'maximum_qualifications':96}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in M.THREADS):raise ValueError('CPU1 nice10')
    out=args.out;out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;S.RAW_CAP=256*1024**2;start=time.monotonic();booked=[0];interrupt=[False]
    signal.signal(signal.SIGTERM,lambda *_:interrupt.__setitem__(0,True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    def check():
        if interrupt[0]:raise InterruptedError('SIGTERM')
        if time.monotonic()-start>240:raise TimeoutError('240s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
    rt={'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in M.THREADS},'load_average':list(os.getloadavg()),'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024};S.save(out/'runtime-start.json',rt)
    result={'status':'incomplete','records':[],'scope':'Adapted longer-transient exploratory discrete-map patch; no shrimp/chaos/ODE proof'};seq=[0]
    def emit(i,kind,value):
        seq[0]+=1;name=f'cycle-{i:05d}-event-{seq[0]:05d}';ar={k:v for k,v in value.items() if isinstance(v,np.ndarray)};me={k:v for k,v in value.items() if not isinstance(v,np.ndarray)};S.save_npz(out/(name+'.npz'),**ar);me.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(out/(name+'.npz')));S.save(out/(name+'.json'),me)
    try:
        if rt['mem_available_bytes']<1024**3:raise MemoryError('host headroom')
        profile=RAW/'new-matrices-profile-001/run';M.profile_gate(profile,PROFILE,{'sha256':{p:identity['sha256'][p] for p in S.SOURCES+M.EXTRA}})
        gate=projection(json.loads((profile/'result.json').read_text()));S.save(out/'resource-gate.json',dict(gate,profile_manifest=PROFILE));check()
        raw=RAW/'new-matrices-panel-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('panel pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('panel inventory')
        for n,d in inv.items():
            p=raw/n
            if Path(n).is_absolute() or '..' in Path(n).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('panel bytes')
        folder=raw/'run/n4-s28175';comp=next(c for c in json.loads((folder/'components.json').read_text())['components'] if c['id']==24)
        if comp['period']!=14 or comp['count']!=7 or comp['grid_edge_margin']!=5:raise ValueError('fixed component')
        if comp['cells']!=[642,673,674,705,706,737,738]:raise ValueError('fixed component membership')
        with np.load(folder/'inputs.npz') as f:old={k:f[k] for k in f.files}
        j,x,v=inputs(4,28175)
        for k,a in [('j',j),('initials',x),('tangents',v)]:
            if old[k].dtype!=a.dtype or old[k].shape!=a.shape or old[k].tobytes()!=a.tobytes():raise ValueError('input replay')
        # Fixed two-coarse-cell margin around all seven compact component cells.
        gains=np.geomspace(old['gains'][18],old['gains'][25],51);deltas=np.linspace(old['deltas'][20],old['deltas'][27],51)
        S.save_npz(out/'inputs.npz',j=j,initials=x,tangents=v,gains=gains,deltas=deltas)
        S.save(out/'input-binding.json',{'panel_manifest':PIN,'original_input':str(folder/'inputs.npz'),'original_input_sha256':S.sha(folder/'inputs.npz'),'component':comp,'grid_indices':[18,25,20,27],'burn':24000,'align':2000,'measured':8192,'selection':'32 evenly spaced consensus14 cells plus16 consensus28/56 cells; both original starts'})
        def checkpoint(t,a,b):S.save_npz(out/f'checkpoint-{t:05d}.npz',step=np.array(t),state=a,tangent=b)
        t=time.monotonic();data=simulate_batch(j,x,v,gains,deltas,burn=24000,align=2000,measured=8192,check=check,checkpoint=checkpoint);result['batch_seconds']=time.monotonic()-t;S.save_npz(out/'batch.npz',**data)
        cs,_=M.components(data['period_proposals'],data['growth_blocks'],data['tangent_collapse_count'],data['grid'],51);inds=selection(data['period_proposals']);S.save(out/'components.json',{'components':cs,'qualification_indices':inds.tolist()})
        for ind in inds:
            check();i=int(ind);g,d=map(float,data['grid'][i]);p=int(data['period_proposals'][i]);S.save_npz(out/f'cycle-{i:05d}-start.npz',theta=data['final'][i],period=np.array(p),grid=data['grid'][i])
            q=qualify_cycle(data['final'][i],j,g,d,p,check,emit=lambda k,value:emit(i,k,value));ar={k:v for k,v in q.items() if isinstance(v,np.ndarray)};me={k:v for k,v in q.items() if not isinstance(v,np.ndarray)};S.save_npz(out/f'cycle-{i:05d}.npz',**ar);me.update(index=i,gain=g,delta=d,initial=i%2);S.save(out/f'cycle-{i:05d}.json',me);result['records'].append(me);S.save(out/'progress.json',result)
        periods=data['period_proposals'];result.update(period_counts={str(p):int(np.sum(periods==p)) for p in np.unique(periods)},two_start_disagreements=int(np.sum(periods[::2]!=periods[1::2])),consensus_components=len(cs),qualified_seed_cases=sum(q['status']=='numerical_attracting_primitive' for q in result['records']),qualification_attempts=len(result['records']));check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
