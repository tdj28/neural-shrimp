"""Read-only exact-rational token-margin audit of existing certified domains."""
from pathlib import Path
import sys,json,hashlib
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');D=R/'codex/coexistence-task33';RAW=Path('/data2/neural-shrimp-data/codex/coexistence')
sys.path.insert(0,str(R/'codex/certify-llm'));import witness_audit as w

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def check_gap(d):
 assert d['vocabulary_count']==151936 and d['competitor_count']==151935 and type(d['candidate_index'])is int and 0<=d['candidate_index']<151936
 lower=w.point(d['candidate_lower']);upper=w.point(d['largest_competitor_upper']);lo,hi=w.endpoints(d['strict_margin'])
 assert lo<=lower-upper<=hi and d['strict_margin_established']==(lo>0)
 return lo>0

def main():
 from tokenizers import Tokenizer
 tokenizer_path=Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json');tok=Tokenizer.from_file(str(tokenizer_path))
 pair=read(D/'pair-certificate-release.json');p8=read(R/'certificates/llm-p8-certificate-final-001.json');assert pair['certificate_passed'] and p8['certificate_passed'];rows=[]
 for period in (2,3,8):
  if period in (2,3):
   orbit=pair['orbits'][period-2];attempt='assembly-p2-001'if period==2 else'assembly-p3-002';adpath=RAW/attempt/'admission.json';derpath=RAW/attempt/'derivative-admission.json';ad=read(adpath);assert ad['bindings']==orbit['domain_bindings'];assert sha(adpath)==read(RAW/'power-pair-001'/('p%d-summary.json'%period))['binding']['contract']['evidence_sha256'];assert sha(derpath)==ad['derivative_report_sha256'];refs=read(derpath)['authenticated_attempts'];definition=orbit['definition'];maphash=hashlib.sha256(json.dumps(pair['map'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
  else:
   adpath=Path(p8['assembly']['admission']['path']);assert sha(adpath)==p8['assembly']['admission']['sha256'];ad=read(adpath);refs=ad['shards'];definition=p8['domain']['definition'];maphash=p8['map_sha256']
  assert sha(definition['path'])==definition['sha256']
  phases=[]
  for phase in range(period):
   candidates=[v for v in refs if v['phase_index']==phase and (v.get('start_column')==0 or v.get('direction_indices',[None])[0]==0)];assert len(candidates)==1
   ref=candidates[0];base=Path(ref['attempt']);assert sha(base/'receipt.json')==ref['receipt_sha256'];rc=read(base/'receipt.json');assert rc['exit_code']==0 and not rc['timed_out'];arts={v['path']:v for v in rc['artifacts']}
   for name in ('run/head-diagnostics.json','run/derivative-shard.json','run/candidate-definition.json'):
    assert sha(base/name)==arts[name]['sha256'] and (base/name).stat().st_size==arts[name]['bytes']
   h=read(base/'run/head-diagnostics.json');shard=read(base/'run/derivative-shard.json');assert sha(base/'run/candidate-definition.json')==definition['sha256'];assert shard['head_diagnostics_file_sha256']==sha(base/'run/head-diagnostics.json')
   assert h['phase_index']==shard['phase_index']==phase and h['map_sha256']==shard['map_sha256']==maphash and h['design_sha256']==shard['design_sha256'] and h['actual_inputs_sha256']==shard['actual_inputs_sha256']
   assert h['head_passes']==[[kind,i,min(i+1024,151936)]for kind in ('head_logits','head_derivatives')for i in range(0,151936,1024)]
   d=h['diagnostic'];passed=check_gap(d);token=tok.decode([d['candidate_index']],skip_special_tokens=False)
   phases.append(dict(phase=phase,token_id=d['candidate_index'],decoded_token=token,strict_margin=d['strict_margin'],candidate_lower=d['candidate_lower'],largest_competitor_upper=d['largest_competitor_upper'],passed=passed,original_receipt=pin(base/'receipt.json'),original_head_diagnostic=pin(base/'run/head-diagnostics.json'),original_shard_sha256=arts['run/derivative-shard.json']['sha256'],source_commit=rc['source_commit']))
  rows.append(dict(period=period,map_sha256=maphash,definition=definition,assembly_admission=pin(adpath),certificate_passed=all(v['passed']for v in phases),phases=phases))
 result=dict(schema='codex-certified-orbit-token-labels-v1',certificate_passed=all(v['certificate_passed']for v in rows),orbit_releases=[pin(D/'pair-certificate-release.json'),pin(R/'certificates/llm-p8-certificate-final-001.json')],tokenizer=pin(tokenizer_path),orbits=rows,scope='Strict all-vocabulary argmax over each original containing receiver box, hence at the corresponding certified orbit phase. Independent Fraction margin checks; full-vocabulary extrema computation uses authenticated reviewed streamed_head source and original completed execution. No new model run or fixed-start reachability claim.',model_calls=0)
 (D/'token-labels-release.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([dict(period=v['period'],passed=v['certificate_passed'],tokens=[p['decoded_token']for p in v['phases']])for v in rows]))
if __name__=='__main__':main()
