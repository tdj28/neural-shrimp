"""Independent saved arithmetic reductions; no model/solver/derivative calls.

Time-line producer and this checker share an agent; operator/utility producers
were separate agents. All outcome-informed scope is disclosed, no frozen test.
"""
import argparse,hashlib,json,math,os,resource,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
PINS={
 'operator-blocks-002':'3d004b84f1cc9b834d3b6969bdb00635ce560ceacce79eabcc48585076a77f57',
 'time-lines-001':'bdfdf84548f24b1cbd44ad0b395d785d1308842f5104ae39acab08f66e78283d',
 'utility':'739737a4db7d153e546b717b68bdb64ea704750dd5d55e60229a8b34054deb35'}

def need(value,message):
 if not value:raise ValueError(message)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(x):return math.sqrt(math.fsum(float(v)**2 for v in np.asarray(x).ravel()))
def dot(x,y):return math.fsum(float(a)*float(b) for a,b in zip(np.asarray(x).ravel(),np.asarray(y).ravel()))
def close(a,b,scale=1.):need(abs(a-b)<=256*np.finfo(float).eps*max(scale,abs(a),abs(b)),'independent scalar reduction')
def inventory(raw,pin):
 need(digest(raw/'manifest.json')==pin,'external manifest pin');inv=json.loads((raw/'manifest.json').read_text())['files']
 need(set(inv)=={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'},'exact file inventory')
 for name,e in inv.items():
  p=raw/name;need(p.resolve().is_relative_to(raw.resolve()) and not p.is_symlink() and digest(p)==e['sha256'] and p.stat().st_size==e['bytes'] and not p.stat().st_mode&0o222,'immutable file identity '+name)
 rt=json.loads((raw/'run/runtime-start.json').read_text())
 for name,e in rt['sources'].items():
  data=subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT)
  need(hashlib.sha256(data).hexdigest()==e['sha256'] and len(data)==e['bytes'],'historical source')
 return len(inv),rt['source_commit']

def response(name):
 raw=Path('/data2/neural-shrimp-data/explore/clark-response')/name;count,source=inventory(raw,PINS[name]);run=raw/'run';result=json.loads((run/'result.json').read_text());events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
 need([e['sequence'] for e in events]==list(range(1,len(events)+1)) and len(events)==result['events'],'complete ordered events')
 rows=[]
 for row in result['rows']:
  label=row['label'];matches=[e for e in events if e['kind']=='terminal' and e['slot']==label];need(len(matches)==1,'one terminal');e=matches[0]
  with np.load(run/e['array_bundle']['file'],allow_pickle=False) as z:x=z['state'].copy();saved=z['normal_residual'].copy()
  maps=np.load(run/f'{label}-maps.npy');oracle=np.load(run/f'{label}-saved-direct-column.npy');p,n,_=maps.shape;T=16*p;x=x.reshape(T+1,n);A=np.exp(.05)*maps
  # Componentwise fsum avoids the producer's BLAS normal-action reductions.
  edges=np.array([[x[t+1,i]-dot(A[t%p,i],x[t]) for i in range(n)] for t in range(T)]);edges[8*p,0]-=1
  physical=np.array([[.0001*x[t,i]+(edges[t-1,i] if t else 0)-(dot(A[t%p,:,i],edges[t]) if t<T else 0) for i in range(n)] for t in range(T+1)])
  rhs=np.zeros_like(x);rhs[8*p]=-A[0,0];rhs[8*p+1,0]=1
  rel=norm(physical)/norm(rhs);err=norm(x[8*p+1]-oracle)
  need(norm(physical.ravel()-saved)<1e-11*max(1,norm(physical)),'independent physical residual vector');close(rel,row['true_normal_relative_residual']);close(err,row['readout_error'])
  need(not row['endpoint_pass'] and row['status']=='action_budget_exhausted' and e['counts']==row['counts'],'unchanged negative labels')
  used=sum(row['counts'].values());need(used<=row['action_ceiling'],'actual budget')
  if name=='time-lines-001':
   need(row['counts']=={'JVP':p*n+(row['iterations']+1)*T,'VJP':(row['iterations']+2)*T},'line setup/RHS/iteration/terminal accounting')
   need(row['action_ceiling']-used<2*T and row['setup_jvps']==p*n,'insufficient next normal action')
  else:need(used==row['action_ceiling'],'fixed operator complete budget')
  rows.append(dict(label=label,normal_relative_residual=rel,readout_error=err,actions=used,endpoint_pass=False))
 return dict(source=source,raw_root=str(raw),manifest=PINS[name],files=count,events=len(events),rows=rows,scope='Independent terminal free-boundary componentwise fsum reductions and source/inventory accounting, no inner iteration or model replay.')

def decode_event(path):
 e=json.loads(path.read_text());arrays={}
 if 'arrays_file' in e:
  a=path.parent/e['arrays_file'];need(digest(a)==e['arrays_sha256'],'utility event bundle hash')
  with np.load(a,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 def decode(v):
  if isinstance(v,dict) and 'array' in v:
   a=arrays[v['array']];need(a.dtype.str==v['dtype'] and hashlib.sha256(a.tobytes()).hexdigest()==v['sha256'],'utility array bytes')
   # Existing serializer reports scalar as shape[1] through ascontiguousarray;
   # actual NPZ scalar stays0D. Admit only this exact size-one legacy case.
   need(list(a.shape)==v['shape'] or (a.shape==() and v['shape']==[1]),'utility shape')
   return a
  if isinstance(v,dict):return {k:decode(w) for k,w in v.items()}
  if isinstance(v,list):return [decode(w) for w in v]
  return v
 return e['sequence'],e['kind'],decode(e['payload'])

def utility():
 raw=Path('/data2/neural-shrimp-data/explore/utility-common-center/attempt-001');count,source=inventory(raw,PINS['utility']);run=raw/'run';result=json.loads((run/'result.json').read_text());events=[decode_event(p) for p in sorted(run.glob('[0-9]*.json'))]
 need([e[0] for e in events]==list(range(len(events))),'utility complete ordered events');rows=[]
 need(result['counts']==dict(gradient_attempts=4,gradient_returns=4,loss_attempts=8,loss_returns=8),'fixed complete operation counts')
 for i,row in enumerate(result['rows']):
  state=next(v for _,k,v in events if k=='paired_states' and v['index']==i)
  returns={v['kind']:v['value'] for _,k,v in events if k=='operation_return' and v['index']==i}
  starts={v['kind']:v['state'] for _,k,v in events if k=='operation_start' and v['index']==i}
  need(np.array_equal(starts['gradient_center'],state['center']) and np.array_equal(starts['loss_plus'],state['plus']) and np.array_equal(starts['loss_minus'],state['minus']),'same actual operation states')
  need(norm(state['plus']-state['center']-state['contrast'])<1e-14 and norm(state['minus']-state['center']+state['contrast'])<1e-14,'paired common center')
  prediction=2*dot(returns['gradient_center'],state['contrast']);difference=float(returns['loss_plus'])-float(returns['loss_minus']);residual=difference-prediction;fraction=abs(residual)/abs(difference)
  close(prediction,row['center_first_order_prediction']);close(difference,row['loss_difference_plus_minus']);close(residual,row['centered_residual']);close(fraction,row['residual_fraction_of_contrast'],1.)
  rows.append(dict(index=i,center_prediction=prediction,observed_contrast=difference,residual=residual,fraction=fraction))
 return dict(source=source,raw_root=str(raw),manifest=PINS['utility'],files=count,rows=rows,scope='Independent fsum paired-state and scalar reductions conditional on saved analytic gradient/loss returns; does not certify arithmetic or repeat model evaluations. Original post-finalization console failure retained.')

def main(out):
 resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,)*2);started=time.monotonic();need(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU1');need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10');need(not out.exists(),'exclusive output')
 result={'response':[response(k) for k in ('operator-blocks-002','time-lines-001')],'utility':utility()};need(time.monotonic()-started<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2*1024**2,'30s/2GiB cap');result.update(peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,passed=True,seconds=time.monotonic()-started,reviewer_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),scope='Post-outcome saved-only independent arithmetic, separate-agent operator/utility review; same-agent time-line review. No scientific acquisition or positive method claim.')
 out.mkdir(parents=True);(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps({'files':{'result.json':{'sha256':digest(out/'result.json'),'bytes':(out/'result.json').stat().st_size}}},indent=2)+'\n')
 for p in out.iterdir():p.chmod(0o444)
 out.chmod(0o555);print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
