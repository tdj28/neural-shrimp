"""Independent saved-output arithmetic audit; never invokes a language model."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/data2/neural-shrimp-data/explore/saved-lm-inference/attempt-001')
SOURCE='bcba5eea358640fb3f8b6672f7fdbe88aef298c6'
def entry(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def need(b,s):
 if not b:raise ValueError(s)
def audit():
 start=time.monotonic();root=RAW;run=root/'run';outer=json.loads((root/'manifest.json').read_text())['files']
 need(set(outer)=={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'manifest.json'},'exact outer inventory')
 for n,e in outer.items():
  p=root/n;need(not Path(n).is_absolute() and '..' not in Path(n).parts and not p.is_symlink() and p.stat().st_mode&0o222==0 and entry(p)==e,'immutable member '+n)
 inner=json.loads((run/'manifest.json').read_text())['files'];need(inner=={n.removeprefix('run/'):e for n,e in outer.items() if n.startswith('run/') and n!='run/manifest.json'},'worker inventory')
 rt=json.loads((run/'runtime-start.json').read_text());la=json.loads((root/'launch-end.json').read_text());result=json.loads((run/'result.json').read_text())
 need(rt['source_commit']==la['source_commit']==SOURCE and result['status']=='completed' and la['exit_code']==0 and la['owned_group_empty'],'successful exact execution')
 for pid in (rt['pid'],la['child_pid'],la['launcher_pid']):need(not Path('/proc',str(pid)).exists(),'owned PID absent')
 for n,h in rt['source_sha256'].items():need(hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+n],cwd=ROOT)).hexdigest()==entry(run/'source'/n)['sha256']==h,'archived committed source '+n)
 need(result['attempted_forward_calls']==result['returned_forward_calls']==780 and rt['training_calls']==0,'complete calls')
 need(len(list(run.glob('forward-*-attempt.json')))==len(list(run.glob('forward-*-return.npy')))==780,'durable call inventory')
 need([(v['seed'],v['factor']) for v in result['cases']]==[(s,f) for s in (28151,28152) for f in (.8,1.6,2.5)],'all six fixed cases')
 decisions=changed=rollout_changed=0;worst=0.;margin_min=float('inf');rows=[]
 for ci,row in enumerate(result['cases']):
  need(entry(run/row['array_file'])==row['array'],'case array identity')
  with np.load(run/row['array_file'],allow_pickle=False) as z:
   a=z['logits32'];repeat=z['repeat32'];b=z['logits64_matched'];free=z['logits64_free'];ta=z['tokens32'];tb=z['tokens64'];prefix=z['prefix_tokens'];same=z['same_history']
   need(a.shape==repeat.shape==b.shape==free.shape==(32,8,66) and a.dtype==np.float32 and b.dtype==free.dtype==np.float64,'exact logits dimensions/dtypes')
   need(np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(free).all() and np.array_equal(a,repeat),'finite repeat exact')
   need(np.array_equal(a.argmax(-1),ta) and np.array_equal(free.argmax(-1),tb),'actual greedy choices')
   x=prefix.copy();y=x.copy()
   for k in range(32):
    need(np.array_equal((x==y).all(1),same[k]),'same-history reconstruction')
    x=np.c_[x[:,1:],ta[k]];y=np.c_[y[:,1:],tb[k]]
    for j,(mode,v) in enumerate([('float32',a),('float32_repeat',repeat),('float64_matched',b),('float64_free',free)]):
     call=13+ci*128+k*4+j;t=json.loads((run/f'forward-{call:04d}-attempt.json').read_text())['tag']
     need(t==dict(seed=row['seed'],factor=row['factor'],step=k,mode=mode),'native attempt order')
     need(np.array_equal(np.load(run/f'forward-{call:04d}-return.npy',allow_pickle=False),v[k]),'all durable outputs equal summary array')
   aa=a.astype(np.float64);error=np.abs(aa-b).max(-1);top=aa.max(-1);chosen=aa.argmax(-1)
   rest=aa.copy();np.put_along_axis(rest,chosen[...,None],-np.inf,axis=-1);gap=top-rest.max(-1)
   matches=chosen==b.argmax(-1);screen=gap>2*error;need(np.all(matches[screen]),'empirical pairwise margin inequality')
   metric=row['matched_history'];need(metric['decisions']==256 and metric['changed']==int((~matches).sum()) and metric['measured_gap_sufficient']==int(screen.sum()),'decision counts')
   need(metric['max_logit_difference']==float(error.max()) and metric['min_reference_margin']==float(gap.min()),'extrema')
   first=[int(np.flatnonzero(ta[:,i]!=tb[:,i])[0]) if np.any(ta[:,i]!=tb[:,i]) else None for i in range(8)]
   need(row['first_greedy_divergence']==first and row['greedy_token_mismatches']==int((ta!=tb).sum()) and row['identical_prefix_decisions']==int(same.sum()),'free rollout result')
   decisions+=256;changed+=int((~matches).sum());rollout_changed+=int((ta!=tb).sum());worst=max(worst,float(error.max()));margin_min=min(margin_min,float(gap.min()))
   rows.append(dict(seed=row['seed'],factor=row['factor'],matched_changed=int((~matches).sum()),rollout_changed=int((ta!=tb).sum()),max_logit_difference=float(error.max()),minimum_margin=float(gap.min())))
 need(time.monotonic()-start<30 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<512*1024,'saved audit resource cap')
 return dict(passed=True,source_commit=SOURCE,audit_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),audit_source_sha256=entry(Path(__file__))['sha256'],raw_root=str(root),outer_manifest=entry(root/'manifest.json'),worker_manifest=entry(run/'manifest.json'),files=len(outer),bytes=sum(e['bytes'] for e in outer.values()),attempted=780,returned=780,decisions=decisions,matched_changed=changed,rollout_changed=rollout_changed,max_logit_difference=worst,minimum_margin=margin_min,cases=rows,producer_elapsed_seconds=result['elapsed_seconds'],audit_elapsed_seconds=time.monotonic()-start,scope='Independent reduction of retained outputs, greedy histories, every target forward and masks. Shared original forward implementation; no model reexecution, inference-chaos, generalization, training-causation or shrimp utility claim.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean source');v=audit()
 with a.output.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 a.output.chmod(0o444);print(json.dumps(v))
