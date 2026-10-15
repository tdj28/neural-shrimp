"""Narrow manufactured-only RM adapter controls; no core-control repetition."""
import argparse,copy,hashlib,importlib.util,json,os,resource,time,traceback
from pathlib import Path


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(25,25))
 n=os.getpriority(os.PRIO_PROCESS,0)
 if n>10:raise RuntimeError('nice >10')
 if n<10:os.nice(10-n)
 out=Path(a.out);out.mkdir(exist_ok=False);tic=time.perf_counter()
 here=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('adapter',here/'run.py');d=importlib.util.module_from_spec(sp);sp.loader.exec_module(d)
 np=d.np;checks=[]
 def require(x):
  if not x:raise AssertionError('condition failed')
 def refuses(fn):
  try:fn()
  except (ValueError,KeyError,FileExistsError):return
  raise AssertionError('expected refusal')
 def test(name,fn):
  try:fn();checks.append({'name':name,'passed':True})
  except BaseException:checks.append({'name':name,'passed':False,'traceback':traceback.format_exc()})
 coords=np.array([[i/10,i/20] for i in range(7)],dtype=np.float64)
 arrays={'cells':coords,'period':np.array(7,dtype=np.int64)}
 for c in range(7):
  for p in range(7):arrays[f'c{c}_p{p}']=np.linspace(c,p+7,1000,dtype=np.float64)
 np.savez_compressed(out/'fixture.npz',**arrays)
 saved={'args':{'period':7},'seconds':0.,'cells':[{'alpha':float(coords[c,0]),'log10_temperature':float(coords[c,1]),
        'phases':[{'phase':p,'turning_points':0,'explained_1':.9,'explained_2':.05,'spread':float(p+7-c)} for p in range(7)]} for c in range(7)]}
 (out/'fixture.json').write_text(json.dumps(saved))
 packet={'attempt':'RM1','source_commit':d.ORIGINAL,'requested_cell_count':7,'phase_count':7,'expected_sequence_count':49,'group_memberships':d.GROUPS,
         'files':{'return-map.npz':{'path':str(out/'fixture.npz')},'return-map.json':{'path':str(out/'fixture.json')}},
         'metadata':{'npz_members':[{'member':k+'.npy','dtype':v.dtype.str,'shape':list(v.shape),'array_nbytes':v.nbytes} for k,v in arrays.items()],
                     'coordinate_and_period_metadata':{'cells':coords.tolist(),'period':7},'args':saved['args']}}
 def valid_and_inventory():
  a,b=d.load_inputs(packet);require(len(a)==51 and len(b['cells'])==7)
  q=copy.deepcopy(packet);q['metadata']['npz_members'].pop();refuses(lambda:d.load_inputs(q))
  q=copy.deepcopy(packet);q['group_memberships']['body_edge']=[0,1];refuses(lambda:d.load_inputs(q))
  q=copy.deepcopy(packet);q['metadata']['coordinate_and_period_metadata']['cells'][0][0]=1.;refuses(lambda:d.load_inputs(q))
 test('all49 admission and missing member/group/coordinate refusals',valid_and_inventory)
 def phase_refusals():
  q=copy.deepcopy(saved);q['cells'][2]['phases'][0]['phase']=1;(out/'fixture-bad.json').write_text(json.dumps(q))
  p=copy.deepcopy(packet);p['files']['return-map.json']['path']=str(out/'fixture-bad.json');refuses(lambda:d.load_inputs(p))
  q=copy.deepcopy(saved);q['cells'][2]['phases'][0]['turning_points']=True;(out/'fixture-bad.json').write_text(json.dumps(q));refuses(lambda:d.load_inputs(p))
 test('saved phase order and Boolean-count refusal',phase_refusals)
 def comparisons():
  q=d.compare(2,3);require(q['signed_difference']==-1 and not q['exact'])
  q=d.compare(None,3);require(q['status']=='undefined_own' and q['signed_difference'] is None)
  q=d.compare(float('nan'),float('nan'));require(q['status']=='nonfinite' and q['both_nan'] and not q['exact'])
 test('signed/nonfinite/undefined producer comparisons remain distinct',comparisons)
 def ownership():
  folder=out/'sink';folder.mkdir();sink=d.Sink(folder,100)
  sink.write('existing.json',{'x':1});refuses(lambda:sink.write('existing.json',{'x':2}))
  try:sink.write('excess.json',{'x':'a'*200})
  except RuntimeError:pass
  else:raise AssertionError('output cap')
  require(not (folder/'excess.json').exists() and json.loads((folder/'existing.json').read_text())=={'x':1})
 test('exclusive output and cap preserve original bytes',ownership)
 result={'manufactured_only':True,'target_or_producer_read':False,'core_controls_repeated':False,'profile_repeated':False,
         'driver_sha256':d.sha(here/'run.py'),'core_sha256':d.sha(here/'core.py'),'checks':checks,'all_passed':all(q['passed'] for q in checks),
         'seconds_before_report_write':time.perf_counter()-tic,'peak_rss_kib_before_report_write':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if result['all_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
