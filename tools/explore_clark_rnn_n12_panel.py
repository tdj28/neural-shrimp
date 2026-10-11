#!/usr/bin/env python3
"""Finite new-matrix N12 panel; all components retained, candidates qualified."""
import argparse,importlib.util,json,os,platform,resource,signal,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('store',ROOT/'tools/explore_clark_rnn.py');S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S)
from neural_shrimp.clark_rnn import inputs,simulate_batch,qualify_cycle,iterate
N=12;PROFILE_SEED=28189;SEEDS=list(range(28191,28195));AXES=31;BURN=12000;ALIGN=1000;MEASURED=8192;TOTAL=BURN+ALIGN+MEASURED
THREADS=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']
EXTRA=['tools/explore_clark_rnn_n12_panel.py','tools/launch_clark_rnn_n12_panel.py','docs/plans/clark-rnn-n12-panel.md','tests/test_clark_rnn_n12_panel.py']

def components(periods,growth,collapse,grid,axes):
    p=np.asarray(periods).reshape(axes,axes,2)
    half=np.asarray(growth).reshape(2,16,-1).sum(axis=1)/4096/np.asarray(grid)[:,1]
    pos=((np.asarray(collapse)==0)&(np.asarray(periods)==0)&np.all(half>.002,axis=0)).reshape(axes,axes,2)
    consensus=np.where(p[:,:,0]==p[:,:,1],p[:,:,0],-1)
    visited=set();out=[]
    for i in range(axes):
        for k in range(axes):
            if (i,k) in visited or consensus[i,k]<3:continue
            period=int(consensus[i,k]);stack=[(i,k)];visited.add((i,k));cells=[]
            while stack:
                a,b=stack.pop();cells.append((a,b))
                for u,v in [(a-1,b),(a+1,b),(a,b-1),(a,b+1)]:
                    if 0<=u<axes and 0<=v<axes and (u,v) not in visited and consensus[u,v]==period:visited.add((u,v));stack.append((u,v))
            cells=sorted(cells);inside=set(cells);halo=set()
            for a,b in cells:
                for di in range(-2,3):
                    for dj in range(-2,3):
                        u,v=a+di,b+dj
                        if 0<=u<axes and 0<=v<axes and (u,v) not in inside:halo.add((u,v))
            margin=min(min(a,b,axes-1-a,axes-1-b) for a,b in cells)
            both=sum(bool(np.all(pos[a,b])) for a,b in halo);anypos=sum(bool(np.any(pos[a,b])) for a,b in halo)
            inds=[a*axes+b for a,b in cells]
            out.append({'id':len(out),'period':period,'cells':inds,'count':len(cells),'grid_edge_margin':margin,'halo_both_start_positive_growth':both,'halo_any_start_positive_growth':anypos,'qualification_cells':sorted(set([inds[0],inds[-1]])),'scope':'finite-horizon consensus component, not verified window or chaos'})
    ranked=sorted(out,key=lambda c:(-(c['grid_edge_margin']>=2),-(c['halo_both_start_positive_growth']>=3),-c['count'],c['period'],c['cells'][0]))
    selected=[c['id'] for c in ranked[:8]]
    for c in out:c['selected_for_native_qualification']=c['id'] in selected
    return out,np.array([2*k+i for c in out if c['id'] in selected for k in c['qualification_cells'] for i in [0,1]],dtype=int)

def projection(r):
    a=r['batch_seconds'];b=r['period64_unit_seconds'];rss=r['peak_rss_kib']
    if not all(np.isfinite(x) and x>0 for x in [a,b,rss]):raise ValueError('finite positive profile')
    # Both profile and target use identical total steps; explicit factors retained.
    seconds=1.8*len(SEEDS)*(AXES**2/81)*(TOTAL/TOTAL)*a+128*8*b+20
    peak=rss*1024+3*(513*AXES**2*2*N*8)+64*1024**2
    raw=len(SEEDS)*(513*AXES**2*2*N*8+(TOTAL//2048+2)*AXES**2*2*N*8*2+32*AXES**2*2*8+2*1024**2)+128*8*(64*(N+N*N)*8+16384)+8*1024**2
    if seconds>360 or peak>512*1024**2 or raw>512*1024**2:raise ValueError('measured resource gate failed')
    return {'projected_seconds':seconds,'conservative_peak_bytes':peak,'conservative_raw_bytes':raw,'batch_factor':len(SEEDS)*AXES**2/81,'step_factor':1.,'safety_factor':1.8,'max_qualifications':128}

def profile_gate(path,pin,identity):
    path=Path(path)
    if S.sha(path/'manifest.json')!=pin:raise ValueError('external profile pin')
    inv=json.loads((path/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'}:raise ValueError('profile inventory')
    for n,d in inv.items():
        p=path/n
        if Path(n).is_absolute() or '..' in Path(n).parts or p.is_symlink() or p.stat().st_size!=d['bytes'] or S.sha(p)!=d['sha256']:raise ValueError('profile bytes')
    rt=json.loads((path/'runtime-start.json').read_text());r=json.loads((path/'result.json').read_text())
    if rt['mode']!='profile' or r['status']!='completed' or rt['source']['sha256']!=identity['sha256']:raise ValueError('profile source/status')
    if rt['python']!=sys.version or rt['numpy']!=np.__version__ or rt['platform']!=platform.platform():raise ValueError('profile environment')
    if rt['nice']<10 or set(rt['threads'])!=set(THREADS) or any(v!='1' for v in rt['threads'].values()):raise ValueError('profile resource ownership')
    if not np.isfinite(r['elapsed_seconds']) or not 0<r['elapsed_seconds']<=45:raise ValueError('profile elapsed')
    return dict(projection(r),manifest_sha256=pin)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['profile','panel'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--profile',type=Path);ap.add_argument('--profile-sha256');args=ap.parse_args()
    identity=S.source();identity['sha256'].update({p:S.sha(ROOT/p) for p in EXTRA})
    if os.getpriority(os.PRIO_PROCESS,0)<10 or any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('CPU1 nice10')
    out=args.out;out.mkdir(parents=True,exist_ok=False);S.ACTIVE_ROOT=out;S.RAW_CAP=512*1024**2;started=time.monotonic();interrupted=False;booked=[0]
    def reserve(path,size,final=False):
        old=Path(path).stat().st_size if Path(path).exists() else 0;new=booked[0]-old+size
        if new>(S.RAW_CAP if final else S.RAW_CAP-S.FINAL_RESERVE):raise OSError('prewrite raw reservation')
        booked[0]=new
    S.reserve=reserve
    def term(*_):
        nonlocal interrupted;interrupted=True
    signal.signal(signal.SIGTERM,term)
    def check():
        if interrupted:raise InterruptedError('SIGTERM')
        if time.monotonic()-started>(45 if args.mode=='profile' else 360):raise TimeoutError('internal deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>448*1024:raise MemoryError('448MiB worker')
    rt={'source':identity,'mode':args.mode,'pid':os.getpid(),'started_unix':time.time(),'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'nice':os.getpriority(os.PRIO_PROCESS,0),'threads':{k:os.environ[k] for k in THREADS},'load_average':list(os.getloadavg()),'mem_available_bytes':int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024}
    S.save(out/'runtime-start.json',rt);result={'status':'incomplete','networks':[],'scope':'Exploratory float64 finite-horizon map activity only. No shrimp/chaos/ODE/utility proof.'};seq=[0]
    def emit(folder,index,kind,value):
        seq[0]+=1;name=f'cycle-{index:05d}-event-{seq[0]:05d}';arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in value.items() if not isinstance(v,np.ndarray)}
        S.save_npz(folder/(name+'.npz'),**arrays);meta.update(kind=kind,array_file=name+'.npz',array_sha256=S.sha(folder/(name+'.npz')));S.save(folder/(name+'.json'),meta)
    try:
        if rt['mem_available_bytes']<1024**3:raise MemoryError('host headroom')
        if args.mode=='panel':
            if args.profile is None or args.profile_sha256 is None:raise ValueError('profile required')
            S.save(out/'resource-gate.json',profile_gate(args.profile,args.profile_sha256,identity));check()
        axes=9 if args.mode=='profile' else AXES;seeds=[PROFILE_SEED] if args.mode=='profile' else SEEDS
        gains=np.geomspace(.6,24,axes);deltas=np.linspace(.05,1,axes)
        S.save(out/'design.json',{'n':N,'seeds':seeds,'gains':gains.tolist(),'deltas':deltas.tolist(),'burn':BURN,'align':ALIGN,'measured':MEASURED,'tail':513,'block':256,'max_period':64,'qualification_rule':'At most8 components by fixed margin/context/size/period/index order; first/last grid cell, both starts. All components retained.','no_odd_symmetry_deduplication_of_raw':True,'reserved_seeds_untouched':True})
        for seed in seeds:
            check();folder=out/f'n{N}-s{seed}';folder.mkdir();j,x,v=inputs(N,seed);S.save_npz(folder/'inputs.npz',j=j,initials=x,tangents=v,gains=gains,deltas=deltas)
            def checkpoint(t,a,b):S.save_npz(folder/f'checkpoint-{t:05d}.npz',step=np.array(t),state=a,tangent=b)
            t=time.monotonic();data=simulate_batch(j,x,v,gains,deltas,burn=BURN,align=ALIGN,measured=MEASURED,check=check,checkpoint=checkpoint);batch_seconds=time.monotonic()-t
            S.save_npz(folder/'batch.npz',**data)
            cs,inds=components(data['period_proposals'],data['growth_blocks'],data['tangent_collapse_count'],data['grid'],axes);S.save(folder/'components.json',{'components':cs,'qualification_indices':inds.tolist()})
            if args.mode=='profile':
                t=time.monotonic()
                for k in range(10):
                    end,m,cy,js=iterate(np.zeros(N),j,.8,.5,64);e,ev=np.linalg.eig(m)
                    S.save_npz(folder/f'profile64-{k:02d}.npz',end=end,monodromy=m,cycle=cy,phase_jacobians=js,eigenvalues=e,eigenvectors=ev)
                result.update(batch_seconds=batch_seconds,period64_unit_seconds=(time.monotonic()-t)/10)
                inds=np.array([],dtype=int)
            records=[]
            for ind in inds:
                check();i=int(ind);g,d=map(float,data['grid'][i]);p=int(data['period_proposals'][i]);S.save_npz(folder/f'cycle-{i:05d}-start.npz',theta=data['final'][i],period=np.array(p),grid=data['grid'][i])
                q=qualify_cycle(data['final'][i],j,g,d,p,check,emit=lambda k,val:emit(folder,i,k,val));arr={k:v for k,v in q.items() if isinstance(v,np.ndarray)};meta={k:v for k,v in q.items() if not isinstance(v,np.ndarray)}
                S.save_npz(folder/f'cycle-{i:05d}.npz',**arr);meta.update(index=i,gain=g,delta=d,initial=i%2);S.save(folder/f'cycle-{i:05d}.json',meta);records.append(meta)
            periods=data['period_proposals'];report={'seed':seed,'batch_seconds':batch_seconds,'trajectories':len(periods),'period_counts':{str(p):int(np.sum(periods==p)) for p in np.unique(periods)},'consensus_components':len(cs),'interior_margin2_components':sum(c['grid_edge_margin']>=2 for c in cs),'margin2_positive_context_components':sum(c['grid_edge_margin']>=2 and c['halo_both_start_positive_growth']>=3 for c in cs),'qualification_attempts':len(records),'qualified_seed_cases':sum(q['status']=='numerical_attracting_primitive' for q in records)}
            result['networks'].append(report);S.save(folder/'summary.json',report);S.save(out/'progress.json',result);del data
        check();result['status']='completed'
    except BaseException as e:result.update(status='failed_or_interrupted',error=type(e).__name__+': '+str(e))
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);S.save(out/'result.json',result,final=True);S.manifest(out)
    return 0 if result['status']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
