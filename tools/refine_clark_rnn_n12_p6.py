#!/usr/bin/env python3
"""One fixed longer-transient patch for the leading N12 new-matrix p6 component."""
import argparse,importlib.util,json,os,platform,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('matrix_panel',ROOT/'tools/explore_clark_rnn_n12_panel.py');M=importlib.util.module_from_spec(sp);sp.loader.exec_module(M);S=M.S
from neural_shrimp.clark_rnn import inputs,simulate_batch,qualify_cycle
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PIN='44edb885b2ba884fdcb229eeabf4d3ae133e8cffc7a6fe289af5a530138f87d3';N=12;SEED=28192

EXTRA=M.EXTRA+['tools/refine_clark_rnn_n12_p6.py','tools/launch_clark_rnn_n12_p6.py','docs/plans/clark-rnn-n12-p6-refinement.md','tests/test_clark_rnn_n12_p6.py','tools/profile_clark_rnn_n12_p6.py','tools/launch_clark_rnn_n12_p6_profile.py']
def selection(p):
    p=np.asarray(p).reshape(-1,2);same=p[:,0]==p[:,1];groups=[]
    for periods,count in [([6],32),([12,24],16)]:
        inds=np.flatnonzero(same&np.isin(p[:,0],periods))
        if len(inds)>count:inds=inds[np.linspace(0,len(inds)-1,count,dtype=int)]
        groups.extend(inds.tolist())
    return np.array([2*k+i for k in sorted(set(groups)) for i in [0,1]],dtype=int)
def projection(r):
    a,b,rss=r['batch_seconds'],r['period64_unit_seconds'],r['peak_rss_kib']
    if not all(np.isfinite(z) and z>0 for z in [a,b,rss]):raise ValueError('finite profile')
    sec=2*(2601/81)*a+96*8*b+20
    peak=rss*1024+3*(513*5202*N*8)+128*1024**2
    raw=513*5202*N*8+18*5202*N*8*2+32*5202*8+96*8*(64*(N+N*N)*8+16384)+8*1024**2
    if sec>360 or peak>2048*1024**2 or raw>512*1024**2:raise ValueError('resource gate')
    return {'projected_seconds':sec,'conservative_peak_bytes':peak,'conservative_raw_bytes':raw,'trajectory_factor':5202/162,'horizon_factor':1.,'safety_factor':2,'maximum_qualifications':96}
def profile_gate(path,pin,identity):
    if S.sha(path/'manifest.json')!=pin:raise ValueError('external exact profile pin')
    inv=json.loads((path/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p!=path/'manifest.json'}:raise ValueError('complete profile inventory')
    for name,e in inv.items():
        q=path/name
        if Path(name).is_absolute() or '..' in Path(name).parts or q.is_symlink() or S.sha(q)!=e['sha256'] or q.stat().st_size!=e['bytes']:raise ValueError('profile bytes')
    rt=json.loads((path/'runtime-start.json').read_text());result=json.loads((path/'result.json').read_text());design=json.loads((path/'design.json').read_text())
    if rt['mode']!='profile' or result['status']!='completed' or rt['source']['sha256']!=identity['sha256']:raise ValueError('matching profile source/status')
    if rt['python']!=sys.version or rt['numpy']!=np.__version__ or rt['platform']!=platform.platform() or rt['nice']<10 or rt['threads']!=dict.fromkeys(M.THREADS,'1'):raise ValueError('matching profile environment')
    if design['n']!=N or design['seeds']!=[28189] or (design['burn'],design['align'],design['measured'])!=(24000,2000,8192) or len(design['gains'])!=9 or len(design['deltas'])!=9:raise ValueError('exact dimension/horizon/profile design')
    if not np.isfinite(result['elapsed_seconds']) or not 0<result['elapsed_seconds']<=45:raise ValueError('bounded profile')
    return projection(result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile-manifest',required=True);args=ap.parse_args();identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in M.THREADS):raise ValueError('CPU1 nice10')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2)
    out=args.out;out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;S.RAW_CAP=512*1024**2;start=time.monotonic();booked=[0];interrupt=[False]
    signal.signal(signal.SIGTERM,lambda *_:interrupt.__setitem__(0,True))
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw cap')
        booked[0]=new
    S.reserve=reserve
    def check():
        if interrupt[0]:raise InterruptedError('SIGTERM')
        if time.monotonic()-start>360:raise TimeoutError('360s cap')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>2048*1024:raise MemoryError('2GiB worker')
    rt={'source':identity,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in M.THREADS},'load_average':list(os.getloadavg()),'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024};S.save(out/'runtime-start.json',rt)
    result={'status':'incomplete','records':[],'scope':'Adapted longer-transient exploratory discrete-map patch; no shrimp/chaos/ODE proof'};seq=[0]
    def emit(i,kind,value):
        seq[0]+=1;name=f'cycle-{i:05d}-event-{seq[0]:05d}';ar={k:v for k,v in value.items() if isinstance(v,np.ndarray)};me={k:v for k,v in value.items() if not isinstance(v,np.ndarray)};S.save_npz(out/(name+'.npz'),**ar);me.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(out/(name+'.npz')));S.save(out/(name+'.json'),me)
    try:
        if rt['mem_available_bytes']<3*1024**3:raise MemoryError('host headroom')
        profile=RAW/'n12-p6-profile-001/run';gate=profile_gate(profile,args.profile_manifest,identity);S.save(out/'resource-gate.json',dict(gate,profile_manifest=args.profile_manifest));check()
        raw=RAW/'n12-matrices-panel-001'
        if S.sha(raw/'manifest.json')!=PIN:raise ValueError('panel pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('panel inventory')
        for n,d in inv.items():
            p=raw/n
            if Path(n).is_absolute() or '..' in Path(n).parts or p.is_symlink() or S.sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('panel bytes')
        folder=raw/f'run/n{N}-s{SEED}';comp=next(c for c in json.loads((folder/'components.json').read_text())['components'] if c['id']==9)
        if comp['period']!=6 or comp['count']!=48 or comp['grid_edge_margin']!=2:raise ValueError('fixed component')
        if comp['cells']!=[429, 430, 458, 459, 460, 461, 462, 489, 490, 491, 492, 493, 519, 520, 521, 522, 523, 524, 550, 551, 552, 553, 580, 581, 582, 583, 584, 585, 610, 611, 612, 613, 614, 615, 641, 642, 643, 644, 646, 647, 671, 672, 674, 678, 702, 703, 733, 734]:raise ValueError('fixed component membership')
        for index in [858,859,1468,1469]:
            record=json.loads((folder/f'cycle-{index:05d}.json').read_text())
            if record['status']!='numerical_attracting_primitive' or record['period']!=6:raise ValueError('qualified original native6 parent')
        with np.load(folder/'inputs.npz') as f:old={k:f[k] for k in f.files}
        j,x,v=inputs(N,SEED)
        for k,a in [('j',j),('initials',x),('tangents',v)]:
            if old[k].dtype!=a.dtype or old[k].shape!=a.shape or old[k].tobytes()!=a.tobytes():raise ValueError('input replay')
        # Fixed two-coarse-cell margin around all48 selected component cells.
        gains=np.geomspace(old['gains'][11],old['gains'][25],51);deltas=np.linspace(old['deltas'][18],old['deltas'][30],51)
        S.save_npz(out/'inputs.npz',j=j,initials=x,tangents=v,gains=gains,deltas=deltas)
        S.save(out/'input-binding.json',{'panel_manifest':PIN,'original_input':str(folder/'inputs.npz'),'original_input_sha256':S.sha(folder/'inputs.npz'),'component':comp,'grid_indices':[11,25,18,30],'burn':24000,'align':2000,'measured':8192,'selection':'32 evenly spaced consensus6 cells plus16 consensus12/24 cells; both original starts'})
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
