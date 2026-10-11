"""Bounded saved-evidence audit of the Clark RNN panel, refinement and slices."""
import gc,hashlib,json,math,os,resource,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
import review_clark_rnn_math as M
BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
THREADS=['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']
UNITS=[('scan-001','3056d0f50978a060a3c99b03ddb6cb2d21a6d62b','27514e8a0473b75fa95eac8aac3656b8d457170b9ac9e315761b366c8d17f8f5','3fe0f6838ae82a4e5f0258a8c5440add43ec6055eae02926653fe13a419b84de'),('refine-p5-001','7c43f004c0300975e9cc26765af32d36f688cbeb','ec4b2f7083b812d2cee38b287046ecde1f68ea5625ca7fb9d56da9eeadb47af2','0d12bd810e92d3469e6f52e85625cd1ec05d0fafe48f685a2ed4ee97f50ed1e7'),('p5-boundary-slices-001','a623ef362a5c01331f1e0dccf5d80d77ff37947a','20d1171d8ccfcc41743d85b8757c5c8affbd15b1c393d6f5209af971b363f77f','16529a77dee6246fc1ee8c7129919f615fed3e459413fadd501bb16fc49ba9ef')]
FILES=['tools/reviews/review_clark_rnn_math.py','tools/reviews/review_clark_rnn_outcomes.py','tools/reviews/launch_clark_rnn_review.py','tests/reviews/test_clark_rnn_saved_review.py','docs/plans/clark-rnn-independent-review.md']

def entry(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(1024**2):h.update(b)
 return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def read(p):return json.loads(p.read_bytes())
def arrays(p):
 with np.load(p,allow_pickle=False)as f:return {k:f[k]for k in f.files}
def verify(root,pin):
 M.need(entry(root/'manifest.json')['sha256']==pin,'manifest pin '+str(root));inv=read(root/'manifest.json')['files']
 actual={str(p.relative_to(root)):p for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'}
 M.need(set(actual)==set(inv),'complete inventory')
 for n,p in actual.items():M.need(not p.is_symlink()and not(p.stat().st_mode&0o222)and entry(p)==inv[n],'immutable bytes '+n)
 return {'files':len(inv),'bytes':sum(x['bytes']for x in inv.values()),'manifest':pin}
def event(path):
 meta=read(path);p=path.parent/meta['array_file'];M.need(p.name==meta['array_file']and entry(p)['sha256']==meta['array_sha256'],'event array identity');return meta,arrays(p)

def cycle_record(path,j,batch):
 q=read(path);a=arrays(path.with_suffix('.npz'));i=q['index'];M.near(batch['grid'][i],[q['gain'],q['delta']],'cycle controls')
 start=arrays(path.with_name(path.stem+'-start.npz'));M.near(start['theta'],batch['final'][i],'unchanged selected start',rtol=0,atol=0);M.need(int(start['period'])==q['period']==int(batch['period_proposals'][i]),'selected period')
 M.near(start['grid'],batch['grid'][i],'start controls',rtol=0,atol=0)
 saved_events=[event(p)for p in sorted(path.parent.glob(path.stem+'-event-*.json'))]
 adjacent=[a for meta,a in saved_events if meta['kind']=='adjacent_iterate_return'];M.need(len(adjacent)==1,'one adjacent native qualification')
 b=adjacent[0];v=M.native(a['cycle'],j,q['gain'],q['delta'],a['phase_jacobians'],a['monodromy']);w=M.native(b['cycle'],j,q['gain'],q['delta'],b['phase_jacobians'],b['monodromy'],b['end'])
 M.near(b['theta'],a['cycle'][1%q['period']],'actual adjacent native start',rtol=0,atol=0)
 M.near(a['monodromy_adjacent'],b['monodromy'],'adjacent product identity',rtol=0,atol=0)
 er=max(M.spectrum(a['monodromy'],a['eigenvalues'],a['eigenvectors']),M.spectrum(a['monodromy_adjacent'],a['eigenvalues_adjacent'],a['eigenvectors_adjacent']))
 rho=float(max(np.max(np.abs(a['eigenvalues'])),np.max(np.abs(a['eigenvalues_adjacent']))))
 # Adjacent closure uses the original phase's scale in the frozen producer.
 scale=max(1.,float(np.max(np.abs(a['cycle']))));cl1=float(np.max(np.abs(b['end']-b['theta']))/scale)
 for k,x in [('closure',v['closure']),('closure_adjacent',cl1),('radius',rho),('eigen_residual',er)]:M.near(x,q[k],k)
 M.need(set(v['separations'])==set(q['proper_divisor_separations']),'proper divisor inventory')
 for k,x in v['separations'].items():M.near(x,q['proper_divisor_separations'][k],'proper divisor separation')
 good=v['closure']<=1e-11 and cl1<=1e-10 and rho<1-1e-6 and er<=1e-10 and all(x>1e-6 for x in v['separations'].values())
 M.need(good==(q['status']=='numerical_attracting_primitive'),'cycle qualification unchanged')
 return {'index':i,'period':q['period'],'qualified':bool(good),'closure':v['closure'],'radius':rho,'eigen_residual':er}

def selected(periods,kind):
 if kind=='scan':return np.r_[np.flatnonzero(periods>=3)[:32],np.flatnonzero(periods==2)[:4]]
 p=periods.reshape(-1,2);same=p[:,0]==p[:,1]
 def spaced(ids,cap):return ids if len(ids)<=cap else ids[np.linspace(0,len(ids)-1,cap,dtype=int)]
 ids=(np.sort(np.r_[spaced(np.flatnonzero(same&(p[:,0]==5)),32),spaced(np.flatnonzero(same&np.isin(p[:,0],[10,20,40])),16)])if kind=='p5'else spaced(np.flatnonzero(same&(p[:,0]>=3)),16))
 return np.array([2*i+k for i in ids for k in (0,1)])

def batch_record(folder,kind,tick):
 inp=arrays(folder/'inputs.npz');b=arrays(folder/'batch.npz');g,d=inp['gains'],inp['deltas'];grid=np.array([[gg,dd]for gg in g for dd in d for _ in range(2)])
 M.near(grid,b['grid'],'complete grid/order',rtol=0,atol=0);M.near(b['tail'][-1],b['final'],'saved final/tail',rtol=0,atol=0)
 p,res=M.periods(b['tail']);M.need(np.array_equal(p,b['period_proposals']),'all tail period proposals');M.need(np.array_equal(np.isnan(res),np.isnan(b['recurrence'])),'recurrence unresolved flags');M.near(res[np.isfinite(res)],b['recurrence'][np.isfinite(res)],'recurrence values')
 measured=4096 if kind=='scan'else 8192;le=np.sum(b['growth_blocks'],axis=0)/measured;le[b['tangent_collapse_count']>0]=np.nan
 M.need(np.array_equal(np.isnan(le),np.isnan(b['lambda_per_step'])),'collapsed tangent abstention');valid=np.isfinite(le);M.near(le[valid],b['lambda_per_step'][valid],'finite window LE');M.near(le[valid]/grid[valid,1],b['lambda_per_time'][valid],'time normalization')
 summary=read(folder/'summary.json');hist={str(k):int(np.sum(p==k))for k in np.unique(p)};M.need(hist==summary.get('period_proposal_counts',summary.get('period_counts')),'period counts')
 half=b['growth_blocks'].reshape(2,-1,len(p)).sum(axis=1)/(measured/2)/grid[:,1];positive=int(np.sum((b['tangent_collapse_count']==0)&np.all(half>.002,axis=0)));M.need(positive==summary['positive_both_half_time_estimates'],'finite-growth half count')
 paths=sorted(x for x in folder.glob('cycle-*.json')if '-event-'not in x.name);ids=np.array([read(x)['index']for x in paths]);sel=selected(p,kind);M.need(np.array_equal(ids,np.sort(sel)),'fixed correction selection')
 records=[]
 for path in paths:records.append(cycle_record(path,inp['j'],b));tick()
 M.need(sum(q['qualified']for q in records)==summary['qualified_seed_cases'],'qualified seed-case count')
 consensus=np.where(p[::2]==p[1::2],p[::2],-1).reshape(len(g),len(d));seen=set();components=[]
 for cell in np.ndindex(consensus.shape):
  if cell in seen or consensus[cell]!=5:continue
  todo=[cell];seen.add(cell);cells=[]
  while todo:
   u,v=todo.pop();cells.append((u,v))
   for a,bb in ((u-1,v),(u+1,v),(u,v-1),(u,v+1)):
    if 0<=a<len(g)and 0<=bb<len(d)and(a,bb)not in seen and consensus[a,bb]==5:seen.add((a,bb));todo.append((a,bb))
  components.append({'size':len(cells),'touches_grid_edge':any(u in(0,len(g)-1)or v in(0,len(d)-1)for u,v in cells),'gain_range':[float(min(g[u]for u,v in cells)),float(max(g[u]for u,v in cells))],'delta_range':[float(min(d[v]for u,v in cells)),float(max(d[v]for u,v in cells))]})
 pairs=[];paths_by_id={read(path)['index']:path for path in paths}
 for i,path in paths_by_id.items():
  if i%2 or i+1 not in paths_by_id:continue
  aa=arrays(path.with_suffix('.npz'))['cycle'];bb=arrays(paths_by_id[i+1].with_suffix('.npz'))['cycle']
  if aa.shape!=bb.shape:continue
  scale=max(1.,float(np.max(np.abs(aa))),float(np.max(np.abs(bb))))
  error,sign,shift=min((float(np.max(np.abs(aa-sg*np.roll(bb,k,axis=0)))/scale),sg,k)for sg in(-1,1)for k in range(len(aa)))
  pairs.append({'even_index':i,'error':error,'sign':sign,'shift':shift})
 value={'folder':folder.name,'trajectories':len(p),'qualified_cases':len(records),'finite_positive_cases':positive,'period_counts':hist,'cycles':records,'p5_components':components,'phase_sign_pairs':pairs,'scope':'saved tail/growth and selected cycles only; burn/measurement trajectory not replayed'}
 del b;gc.collect();tick();return value,inp

def boundary_records(run,j,tick):
 result=read(run/'result.json');expected={d['boundary']['tag']:d['boundary'] for s in result['slices']for d in s['directions']if d['boundary']is not None};rows=[];counts={}
 for path in sorted(run.glob('[0-9]*-*.json')):
  meta,a=event(path);kind=meta['kind'];counts[kind]=counts.get(kind,0)+1;tag=meta.get('tag','');synthetic=tag=='synthetic_profile';jj=np.eye(4)*.2+np.ones((4,4))*.1 if synthetic else j
  if kind=='native':M.native(a['cycle'],jj,meta['gain'],meta['delta'],a['phase_jacobians'],a['monodromy'],a['end']);M.near(a['theta'],a['cycle'][0],'native start',rtol=0,atol=0)
  elif kind=='native_eigen':M.spectrum(a['monodromy'],a['eigenvalues'],a['eigenvectors'])
  elif kind in ('correction_solve','augmented_solve','branch_solve'):
   x=a.get('direction',a.get('solution'));lhs=a['matrix']@x;rhs=a['rhs'];error=float(np.max(np.abs(lhs-rhs))/max(1.,float(np.max(np.abs(a['matrix'])))*float(np.max(np.abs(x))),float(np.max(np.abs(rhs)))));M.need(error<2e-10,'saved linear solve backward residual')
  elif kind in ('augmented','qualification_augmented'):
   a['delta']=meta['delta'];sign=meta.get('sign',1 if expected.get(tag,{}).get('kind')=='fold'else -1);M.augmented(a,jj,meta.get('gain',expected.get(tag,{}).get('gain')),sign)
  elif kind=='boundary_result':
   q=expected[tag];q=dict(q,gain=meta['gain']);a['delta']=meta['delta'];M.need(q['status']==meta['status'],'terminal boundary status');rows.append(dict(tag=tag,**M.boundary(a,q,jj)))
  tick()
 M.need(len(rows)==5 and sum(r['kind']=='fold'for r in rows)==3 and sum(r['kind']=='flip'for r in rows)==2,'all five qualified returned boundaries')
 M.need(sum(v for k,v in counts.items()if k in ('native','augmented','qualification_augmented'))==result['attempted_native_calls'],'complete returned native call counts')
 return {'records':rows,'event_counts':counts,'unresolved':[q for q in expected.values()if q['status']!='numerical_boundary_candidate'],'scope':'same saved numerical candidate equations and gates; no interval/generic flip cubic/continuous first-loss guarantee'}

def main(out):
 M.need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');M.need(os.getpriority(os.PRIO_PROCESS,0)>=10 and all(os.environ.get(k)=='1'for k in THREADS),'CPU1 nice10')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();stopped=False;result={'passed':False,'units':[]}
 def save(name,v):
  b=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();M.need(sum(p.stat().st_size for p in out.rglob('*')if p.is_file())+len(b)<7*1024**2,'7MiB worker output cap')
  with(out/name).open('xb')as f:f.write(b);f.flush();os.fsync(f.fileno())
 def tick():M.need(not stopped and time.monotonic()-start<60,'60s internal');M.need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<480*1024,'480MiB monitored RSS')
 def stop(*_):
  nonlocal stopped;stopped=True
 runtime={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_files':{n:entry(ROOT/n)for n in FILES},'pid':os.getpid(),'started_unix':time.time(),'threads':{k:os.environ[k]for k in THREADS},'nice':os.getpriority(os.PRIO_PROCESS,0)};save('runtime-start.json',runtime)
 for n in FILES:p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
 try:
  original=None
  for name,commit,outerpin,runpin in UNITS:
   root=BASE/name;run=root/'run';inv=verify(root,outerpin);verify(run,runpin);rt=read(run/'runtime-start.json');launch=read(root/'launch-end.json');done=read(run/'result.json')
   M.need(rt['python']==sys.version and rt['numpy']==np.__version__,'same numerical runtime');M.need(rt['source']['commit']==launch['source_commit']==commit and done['status']=='completed'and launch['exit_code']==0 and launch['owned_group_empty'],'execution success binding')
   for p,h in rt['source']['sha256'].items():M.need(hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest()==h,'complete declared source binding')
   for pid in (rt['pid'],launch['launcher_pid'],launch['child_pid']):M.need(not Path('/proc',str(pid)).exists(),'owned PID absent')
   ts=rt.get('threads',rt.get('thread_environment'));M.need(ts==dict.fromkeys(THREADS,'1')and rt['nice']>=10,'source resource threads');M.need(done['peak_rss_kib']<448*1024 and 0<done['elapsed_seconds']<(60 if name==UNITS[2][0]else 300),'source resource caps')
   if name!=UNITS[2][0]:
    dirs=sorted(p.parent for p in run.glob('*/batch.npz'));M.need([p.name for p in dirs]==(sorted('n%d-s%d'%(n,s)for n in(4,8,16)for s in(28170,28171))if name==UNITS[0][0]else ['p5','p7']),'complete fixed panels');records=[]
    for folder in dirs:
     rec,inp=batch_record(folder,'scan'if name==UNITS[0][0]else folder.name,tick);records.append(rec)
     if folder.name=='n4-s28171':original={k:inp[k].copy()for k in ('j','initials','tangents')}
     if name==UNITS[1][0]:
      for k in original:M.near(inp[k],original[k],'same original RNN input '+k,rtol=0,atol=0)
    row={'unit':name,'inventory':inv,'panels':records}
   else:
    binding=read(run/'input-binding.json');M.need(binding['refinement_manifest']==UNITS[1][2],'refinement parent pin')
    for key,rel in [('source_input_sha256','p5/inputs.npz'),('source_batch_sha256','p5/batch.npz')]:M.need(binding[key]==entry(BASE/'refine-p5-001/run'/rel)['sha256'],'exact boundary input '+key)
    row={'unit':name,'inventory':inv,'boundary':boundary_records(run,original['j'],tick)}
   save(name+'.json',row);result['units'].append({'unit':name,'passed':True});tick()
  result['passed']=True
 except BaseException as e:result['error']=type(e).__name__+': '+str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);result.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('outcome.json',result);inv={str(p.relative_to(out)):entry(p)for p in out.rglob('*')if p.is_file()};save('manifest.json',{'files':inv})
  for p in sorted(out.rglob('*'),reverse=True):p.chmod(0o555 if p.is_dir()else 0o444)
  out.chmod(0o555)
 print(json.dumps(result),flush=True);return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main(Path(sys.argv[1])))
