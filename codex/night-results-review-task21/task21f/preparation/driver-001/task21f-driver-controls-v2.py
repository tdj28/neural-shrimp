"""One manufactured-only driver delta/control and six-shape serialization run."""
import argparse,copy,hashlib,importlib.util,json,os,resource,time,traceback
from pathlib import Path


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--capsule',required=True);ap.add_argument('--out',required=True);ap.add_argument('--refusal-only',action='store_true');a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(25,25))
 priority=os.getpriority(os.PRIO_PROCESS,0)
 if priority>10:raise RuntimeError('nice >10')
 if priority<10:os.nice(10-priority)
 out=Path(a.out);out.mkdir(exist_ok=False)
 spec=importlib.util.spec_from_file_location('own_driver',Path(a.capsule)/'run.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
 np=d.np;controls=[];tic=time.perf_counter()
 def require(v):
  if not v:raise AssertionError('condition failed')
 def refuses(fn):
  try:fn()
  except (ValueError,KeyError,FileExistsError):return
  raise AssertionError('expected refusal')
 def test(name,fn):
  try:fn();controls.append({'name':name,'passed':True})
  except BaseException:controls.append({'name':name,'passed':False,'traceback':traceback.format_exc()})
 def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 def fixture(name,spec):
  nr,nc,count,attempt,commit=spec;folder=out/name;folder.mkdir()
  ranges={'p7-blob':(.1985,.2115),'p7-blob-lo':(.1985,.205),'p7-blob-hi':(.205,.2115),
          'p7-zoom-a':(.2048,.20555),'p7-zoom-b':(.20555,.2063),'p7-zoom-c':(.2063,.2072)}
  zoom=name.startswith('p7-zoom-');shape=(nr,nc)
  xs=np.linspace(.0692,.0708,nc) if zoom else np.linspace(.0686,.0716,nc)
  ys=np.linspace(*ranges[name],nr);xx,yy=np.meshgrid(xs,ys)
  fac=np.stack([(xx-xs.mean())/(xs[-1]-xs[0])+(k+1)*.07*(yy-ys.mean())/(ys[-1]-ys[0]) for k in range(7)],axis=-1)
  mu=np.prod(fac,axis=-1)
  g={'alpha':xs,'log10_temperature':ys,'mu':mu,'mu_imag':np.zeros(shape),'next_modulus':np.full(shape,1e-6),
     'product':mu.copy(),'newton':np.full(shape,1e-14),'min_reference_cosine':np.ones(shape),
     'ok':np.ones(shape,bool),'clean':np.abs(mu)>1e-5,'same_tokens':np.ones(shape,bool),'factors':fac.copy(),
     'seed':np.array([xs[nc//2],ys[nr//2]]),'seed_tokens':np.arange(7,dtype=np.int64)}
  if zoom:g.update(factors_transported=fac.copy(),stretchings=np.abs(fac),transport_cosine=np.ones(shape))
  indices=[list(ij) for ij in np.ndindex(shape)][:count]
  success={(i,j) for n,(i,j) in enumerate(indices) if n%73!=1}
  for i,j in np.ndindex(shape):
   if (i,j) not in success:
    for k in d.C.SCALARS+('factors',)+d.C.TRANSPORT:
     if k in g:g[k][i,j]=np.nan
    for k in d.C.FLAGS:g[k][i,j]=False
  progress=[{'seed':g['seed'].tolist(),'mu':-.01,'next_modulus':1e-6,'factors':[1.]*7,'product':1.,'tokens':list(range(7)),'seconds':.01}]
  for i,j in indices:
   rec={'i':i,'j':j,'alpha':float(xs[j]),'log10_temperature':float(ys[i]),'converged':(i,j) in success,'seconds':.01}
   if rec['converged']:
    rec.update({k:d.C.plain(g[k][i,j]) for k in d.C.SCALARS+d.C.FLAGS+('factors',)+d.C.TRANSPORT if k in g and k!='ok'})
    rec['split']=1.
   progress.append(rec)
  np.savez_compressed(folder/'grid.npz',**g)
  (folder/'progress.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in progress))
  args={'period':7,'resolution':[nr,nc],'model':'manufactured-no-model'}
  final=None
  if name!='p7-blob':
   final={'args':args,'seed_tokens':list(range(7)),'seed_text':['manufactured']*7,'cells':nr*nc,
          'solved':int(g['ok'].sum()),'clean':int((g['ok']&g['clean']).sum()),'same_tokens':int((g['ok']&g['same_tokens']).sum()),'seconds':0.}
   (folder/'grid.json').write_text(json.dumps(final))
  row={'id':name,'attempt':attempt,'source_commit':commit,'files':{'grid.npz':pin(folder/'grid.npz'),'progress.jsonl':pin(folder/'progress.jsonl'),
       'grid.json':pin(folder/'grid.json') if final else None},'declared_args':args if final else None,
       'original_tool_status_lines':['manufactured, no original process'],
       'metadata':{'npz_members':[{'member':k+'.npy','shape':list(v.shape),'dtype':v.dtype.str,'array_nbytes':v.nbytes} for k,v in g.items()],
                   'axis_and_seed_metadata':{k:g[k].tolist() for k in ('alpha','log10_temperature','seed')},
                   'progress':{'line_count':len(progress),'indices':indices}}}
  return row,g
 packet={'runs':[]};grids={}
 for name,specification in d.RUNS.items():
  r,g=fixture(name,specification);packet['runs'].append(r);grids[name]=g
 def schema_refusals():
  r=copy.deepcopy(packet['runs'][0]);r['metadata']['axis_and_seed_metadata']['alpha'][0]+=1e-12;refuses(lambda:d.load_run(r))
  r=copy.deepcopy(packet['runs'][0]);r['metadata']['progress']['indices'][0]=[0,1];refuses(lambda:d.load_run(r))
  r=copy.deepcopy(packet['runs'][0]);r['metadata']['npz_members'].append(r['metadata']['npz_members'][0]);refuses(lambda:d.load_run(r))
  r=copy.deepcopy(packet['runs'][0]);r['id']='p7-zoom-a';refuses(lambda:d.load_run(r))
 test('strict axis, BFS order, duplicate field and historical-method association refusal',schema_refusals)
 if a.refusal_only:
  result={'manufactured_only':True,'target_data_read':False,'profile_repeated':False,'controls':controls,
          'all_passed':all(r['passed'] for r in controls),'source_sha256':{'run.py':d.sha(Path(a.capsule)/'run.py'),'core.py':d.sha(Path(a.capsule)/'core.py')},
          'correction':'First fake BFS index [0,0] is now changed to [0,1]; original fixture had reassigned its existing value.',
          'elapsed_before_report':time.perf_counter()-tic,'peak_rss_kib_observed':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
  return 0 if result['all_passed'] else 1
 def mask_and_sectors():
  g={k:v[:2,:2].copy() if getattr(v,'ndim',0)>=2 else v[:2].copy() if k in ('alpha','log10_temperature') else v.copy() for k,v in grids['p7-zoom-a'].items()}
  for k in d.C.FLAGS:g[k][:]=True
  g['mu'][:]=1.;g['mu_imag'][:]=1e-9;g['next_modulus'][:]=.001;g['transport_cosine'][:]=1
  g['factors_transported'][:]=1.;g['factors_transported'][...,0]=[[-1,1],[-1,1]];g['factors_transported'][...,1]=[[-1,-1],[1,1]]
  views,raw=d.view_masks(g)
  require(views['names']['ok']==views['names']['ok_same_tokens'])
  require(views['names']['ok_clean_seed_real']!=views['names']['ok_clean_plot_real'])
  folder=out/'sectors';folder.mkdir();sink=d.Sink(folder,16*1024**2)
  summary=d.geometry(g,'factors_transported',views,sink,'manufactured')
  pair=summary['pairs'][0];vm={x['mask_id']:x for x in pair['by_mask']}
  require(vm[views['names']['ok_clean_seed_real']]['same_quad_four_sign_intersections']==1)
  require(vm[views['names']['ok_clean_plot_real']]['intersections']==0)
 test('exact mask reuse, distinct reality views and same-quad four-sector witness',mask_and_sectors)
 def output_ownership():
  folder=out/'ownership';folder.mkdir();sink=d.Sink(folder,100)
  sink.write('one.json',{'a':1});refuses(lambda:sink.write('one.json',{'a':2}))
  try:sink.write('too-big.json',{'a':'x'*200})
  except RuntimeError:pass
  else:raise AssertionError('output cap did not refuse')
  require((folder/'one.json').read_text().strip()=='{"a":1}')
 test('exclusive writes and declared output cap preserve earlier bytes',output_ownership)
 measurement=None
 try:
  folder=out/'six-shape-result';folder.mkdir();sink=d.Sink(folder,256*1024**2)
  start=time.perf_counter();report=d.calculate(packet,sink);sink.write('report.json',report);sink.write('manifest.json',{'files':sink.manifest()})
  measurement={'seconds_through_all_scientific_writes':time.perf_counter()-start,'output_bytes':sink.bytes,
               'peak_rss_kib_observed':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'runs':report['run_count'],'allocated_cells':report['allocated_cells'],
               'available_factor_methods':sum(x['available'] for r in report['runs'] for x in r['methods'].values()),
               'actual_segment_pairs':sum(p['all_ok_pairs'] for r in report['runs'] for m in r['methods'].values() if m['available'] for p in m['pairs'])}
  test('six shapes, all rows, nine method grids and eleven output files',lambda:require(report['run_count']==6 and report['allocated_cells']==3174 and measurement['available_factor_methods']==9 and len(sink.entries)==11))
 except BaseException:
  controls.append({'name':'manufactured six-shape calculation and serialization','passed':False,'traceback':traceback.format_exc()})
 result={'manufactured_only':True,'target_data_read':False,'source_sha256':{'run.py':d.sha(Path(a.capsule)/'run.py'),'core.py':d.sha(Path(a.capsule)/'core.py')},
         'controls':controls,'all_passed':all(r['passed'] for r in controls),'measurement':measurement,
         'elapsed_before_control_report':time.perf_counter()-tic,'limits':{'cpu_seconds':25,'wall_seconds_external':30,'as_bytes':1024**3,'nice':10},
         'scope':'Manufactured fields at exact declared shapes; no actual metadata/targets/producer/model loaded. Not worst-case contour/output proof.'}
 (out/'report.json').write_text(json.dumps(d.C.plain(result),indent=2,allow_nan=False)+'\n');print(json.dumps(result))
 return 0 if result['all_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
