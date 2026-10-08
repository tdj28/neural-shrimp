"""Independent saved-file arithmetic only; no producer/model imports or dynamics."""
import argparse, hashlib, json, math, os, signal, subprocess, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/audits/utility-inputs/attempt-001')
MANIFEST='2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f'
PRODUCER='a921b776953b5fc57d4a51469899022aec56d12f'
DATA=Path('/data2/neural-shrimp-data')

def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def entry(p):
 b=p.read_bytes();return dict(bytes=len(b),sha256=sha(b))
def check_id(array,ident):
 a=np.ascontiguousarray(array,dtype='<f8')
 assert np.isfinite(a).all() and ident['shape']==list(a.shape) and ident['dtype']=='<f8' and ident['order']=='C'
 assert sha(a.tobytes())==ident['sha256']
 assert ident['binary64_hex']==[format(int(x),'016x') for x in a.reshape(-1).view('<u8')]
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=3e-14,abs_tol=3e-15),(a,b)

def audit():
 assert entry(RAW/'manifest.json')['sha256']==MANIFEST
 manifest=read(RAW/'manifest.json')
 assert set(p.name for p in RAW.iterdir())==set(manifest['files'])|{'manifest.json'}
 for name,e in manifest['files'].items():assert entry(RAW/name)==e and not (RAW/name).is_symlink()
 assert sum(e['bytes'] for e in manifest['files'].values())==manifest['total_bytes']==738699
 inputs=read(RAW/'inputs.json')
 for e in inputs:assert entry(Path(e['path']))=={k:e[k] for k in ('bytes','sha256')}
 receipt=read(ROOT/'docs/reviews/utility-input-qualification-receipt.json')
 for e in receipt['launcher_files']:assert entry(Path(e['path']))=={k:e[k] for k in ('bytes','sha256')}
 outcome=read(RAW/'result.json');assert outcome['source_commit']==PRODUCER
 for name,digest in outcome['source_sha256'].items():
  assert sha(subprocess.check_output(['git','show',PRODUCER+':'+name],cwd=ROOT))==digest
 for source in read(RAW/'source-bindings.json'):
  raw=subprocess.check_output(['git','show',source['commit']+':'+source['path']],cwd=ROOT)
  assert len(raw)==source['bytes'] and sha(raw)==source['sha256']
 forks=read(RAW/'local-forks.json');assert len(forks)==4
 mean_discrepancies=[]
 warm=DATA/'explore/warmup-intervention/attempt-002'
 for point in (0,1):
  refs=read(warm/f'reference-{point}.json')['references']
  with np.load(warm/f'point-{point}-dwell-100-common.npz',allow_pickle=False) as z:
   trace={k:z[k] for k in ('state','loss','alpha','rates','distance','initial_state')}
  for origin in (0,1):
   fork=next(f for f in forks if f['point']==point and f['array_index'][1]==origin)
   first=next(i for i,v in enumerate(trace['alpha'][:,origin]) if v>=0)
   assert first==4100 and fork['array_index']==[4100,origin]
   assert fork['accepted_updates_at_fork']==2*(first+1)==8202
   assert np.isfinite(trace['state'][:first+1,origin]).all()
   assert fork['prefix_pair_indices_inclusive']==[3101,4100]
   check_id(trace['state'][first,origin],fork['state'])
   assert np.array_equal(trace['state'][first,origin],fork['state_values'])
   history=trace['loss'][3101:4101,origin];check_id(history,fork['prefix_loss'])
   mean=math.fsum(map(float,history.flat))/history.size
   mean_discrepancies.append(abs(mean-fork['prefix_mean_training_loss']));close(mean,fork['prefix_mean_training_loss'])
   assert fork['fork_training_loss']==float(trace['loss'][4101,origin,0])
   cross=next(r for r in refs if r['alpha']==float(trace['alpha'][first,origin]))
   assert fork['cross_log_rates']==[cross['h'],cross['o']]
   check_id(trace['rates'][first,origin],fork['cross_rates'])
   recovery=[10**float(refs[0]['h']),10**float(refs[0]['o'])]
   check_id(recovery,fork['recovery_rates']);check_id(refs[0]['orbit'],fork['saved_recovery_reference_orbit'])
   assert recovery==fork['recovery_rate_values'] and fork['recovery_log_rates']==[refs[0]['h'],refs[0]['o']]
   close(recovery[0]/trace['rates'][first,origin,0],fork['control_recovery_uniform_factor'])
   assert fork['crossing_distance']==float(trace['distance'][first,origin])
   assert fork['full_one_update_prefix_states_present'] is False and fork['gradient_at_fork_present'] is False
 crosses=[f['cross_log_rates'] for f in forks if f['array_index'][1]==0]
 def distance(logs):return min(max(abs(float(logs[j])-float(c[j])) for j in range(2)) for c in crosses)
 expected={}
 for name in ('s2-28011-out-0.0856','s2-28011-out-0.30'):
  folder=DATA/'explore/consequence'/name;md=read(folder/'metadata.json')
  with np.load(folder/'slice.npz',allow_pickle=False) as z:
   data={k:z[k] for k in ('log10_hidden','status','period','ftle','separation','tail_train_mean')}
   schema=next(s for s in read(RAW/'input-schemas.json') if s['source']==str(folder/'slice.npz'))
   assert schema['members']==list(z.files)
  for i,h in enumerate(data['log10_hidden']):
   kind=None
   if int(data['status'][i])==2 and int(data['period'][i])==2:kind='periodic_candidate'
   elif int(data['status'][i])==3 and all(float(v)>.005 for v in data['ftle'][i]) and float(data['separation'][i])>1e-6:kind='chaotic_candidate'
   logs=[float(h),md['log10_output']]
   if kind is not None and distance(logs)>=.2:
    expected[f'{name}:{i}']=(kind,logs,float(data['tail_train_mean'][i]))
 exp=DATA/'raw/EXP-028/attempt-002'
 maps={}
 for label,path in [('S2-28011','stage1/S2-28011.npz'),('S2-28011-zoom','stage2/S2-28011-zoom.npz')]:
  with np.load(exp/path,allow_pickle=False) as z:maps[label]={k:z[k] for k in ('final_state','tail_mean_loss','lr_hidden','lr_output')}
 for category,path in [('periodic','stage3/certificates.json'),('chaos','stage3/chaos-diagnostics.json')]:
  for i,r in enumerate(read(exp/path)):
   if r['map_id'] not in maps:continue
   if category=='periodic':
    c=r.get('certificate',{})
    passes=r['kind']=='other' and c.get('verified') and c.get('attracting') and c.get('primitive_period')==2
   else:passes=r['confirmed_chaotic'] and r['original_status']==r['rerun_status']==3 and r['original_ftle']>.005 and r['rerun_ftle']>.005
   logs=[math.log10(r['lr_hidden']),math.log10(r['lr_output'])]
   if not passes or distance(logs)<.2:continue
   cell=r['representative'] if category=='periodic' else r['cell'];grid=maps[r['map_id']]
   assert [r['lr_hidden'],r['lr_output']]==[float(grid[k].flat[cell]) for k in ('lr_hidden','lr_output')]
   ident=f'EXP028-{category}:{i}'
   expected[ident]=('periodic_candidate' if category=='periodic' else 'chaotic_candidate',logs,float(grid['tail_mean_loss'].flat[cell]))
 rows=read(RAW/'control-candidates.json');assert len(rows)==len(expected)==522
 assert {r['id'] for r in rows}==set(expected)
 for row in rows:
  kind,logs,loss=expected[row['id']]
  assert row['class']==kind and row['log_rates']==logs and row['tail_mean_training_loss']==loss
  close(distance(logs),row['distance_decades']);assert distance(logs)>=.2
  assert row['complete_qualification'] is False and row['full_prefix_losses_available'] is False
  assert row['full_prefix_states_available'] is False and row['recovery_qualified'] is False
  assert row['ordinary_branch_independently_traced'] is False
  if row['id'].startswith('EXP028-'):
   check_id(maps[row['map_id']]['final_state'].reshape(-1,13)[row['cell']],row['state'])
 counts={kind:sum(r['class']==kind for r in rows) for kind in ('periodic_candidate','chaotic_candidate')}
 assert counts==outcome['candidate_counts']==dict(periodic_candidate=284,chaotic_candidate=238)
 nominees=read(RAW/'nominees.json');used=set();selected=[];score_gaps=[]
 for nominee in nominees:
  target=[f['prefix_mean_training_loss'] for f in forks if f['point']==nominee['point']]
  pool=[]
  for row in rows:
   key=(row['class'],tuple(row['log_rates']))
   if row['class']!=nominee['class'] or key in used:continue
   loss=row['tail_mean_training_loss']
   if not math.isfinite(loss) or loss<=0:continue
   score=max(abs(math.log(loss/t)) for t in target)
   # The plan's secondary ordering is source path and numeric row.
   pool.append((score,row['source'],int(row['row']),row))
  pool.sort(key=lambda v:v[:3]);best=pool[0][3]
  assert best['id']==nominee['candidate'];used.add((best['class'],tuple(best['log_rates'])));selected.append(best['id'])
  score_gaps.append(pool[1][0]-pool[0][0])
  ratios=[max(best['tail_mean_training_loss']/t,t/best['tail_mean_training_loss']) for t in target]
  for a,b in zip(ratios,nominee['prefix_mean_loss_ratios']):close(a,b)
  assert nominee['prefix_mean_loss_screen_pass']==(max(ratios)<=2)
  assert nominee['selection_is_not_qualification'] is True
 assert min(score_gaps)>0
 assert outcome['fully_qualified_controls']==0 and outcome['dynamics_evaluations']==0
 md=read(warm/'metadata.json');cfg=md['args']
 # Enumerate cells rather than reuse the producer's combined expression.
 updates=0
 for point in cfg['outputs']:
  for dwell in cfg['dwell_pairs']:
   updates+=2*2*(cfg['pre_pairs']+cfg['levels']*dwell+cfg['top_pairs'])
   for origin in range(2):
    for arm in range(3):updates+=2*(cfg['levels']*dwell+cfg['recovery_pairs'])
 assert updates==outcome['historical_warmup']['counted_batched_cell_updates']==1232000
 return dict(verdict='pass_with_nonblocking_tie_rule_discrepancy',producer_source=PRODUCER,producer_manifest_sha256=MANIFEST,
   raw_files=len(manifest['files'])+1,input_files=len(inputs),source_bindings=len(read(RAW/'source-bindings.json')),
   local_forks=4,row_index=4100,updates_at_fork=8202,prefix_rows_inclusive=[3101,4100],
   maximum_independent_prefix_mean_discrepancy=max(mean_discrepancies),candidate_records=522,candidate_counts=counts,
   nominees=selected,nomination_runner_up_log_score_gaps=score_gaps,fully_qualified_controls=0,historical_counted_cell_updates=updates,
   scope='saved identities/indexing/arithmetic only; no dynamics, new qualification, pulse outcome or protected-array evaluation')

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 assert not args.out.exists() and os.getpriority(os.PRIO_PROCESS,0)>=10
 def stop(*_):raise TimeoutError('independent saved-file audit30s')
 signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,30)
 started=time.monotonic()
 try:
  result=audit();result['elapsed_seconds']=time.monotonic()-started
  result['review_source']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 except BaseException as error:
  result=dict(verdict='failed',error=type(error).__name__+': '+str(error));raise
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  with args.out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  args.out.chmod(0o444)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
