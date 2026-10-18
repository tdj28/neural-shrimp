import sys,json,copy
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
import derivative_bridge as b
from flint import ctx
ctx.prec=1024;ctx.threads=1
count=0
for p in (2,3):
 packet=json.loads((R/f'codex/coexistence-task33/p{p}-dispatch-001.json').read_text());plan=json.loads((R/packet['plans'][0]['plan']).read_text())
 prefix=dict(attempt=plan['prefix_attempt'],receipt_sha256=plan['prefix_receipt_sha256'],map_source_sha256=plan['prefix_map_source_sha256'],cache_sha256=plan['prefix_cache_sha256'],source_commit=plan['prefix_source_commit'],receipt_path=plan['prefix_receipt'],admission_path=plan['prefix_admission_path'],admission_sha256=plan['prefix_admission_sha256'])
 definition=json.loads(Path(plan['definition_path']).read_text())
 def anchor(mp=definition['map'],backend=packet['expected_backend'],pins=plan['source_sha256'],dfhash=plan['definition_sha256']):return b.trust_anchors(plan['definition_path'],dfhash,mp,backend,pins,prefix)
 a=anchor();assert a['definition']['period']==p;count+=1
 for kind in ['map','backend','pins','definition_hash']:
  try:
   if kind=='map':m=copy.deepcopy(definition['map']);m['alpha_ratio']=[1,2];anchor(mp=m)
   elif kind=='backend':bb=dict(packet['expected_backend'],precision=768);anchor(backend=bb)
   elif kind=='pins':pp=dict(plan['source_sha256']);pp[next(iter(pp))]='0'*64;anchor(pins=pp)
   else:anchor(dfhash='0'*64)
  except ValueError:count+=1
  else:raise AssertionError(kind)
 for phase in range(p):
  _,state,_,_=b.producer.construct_inputs(definition,phase,0)
  receiver=b.producer.receiver_record(definition,phase,state)
  b._receiver_witnesses(a,phase,receiver);count+=1
  bad=copy.deepcopy(receiver);bad['actual_inputs'][0]={'mid':[0,0],'rad':[0,0]}
  try:b._receiver_witnesses(a,phase,bad)
  except ValueError:count+=1
  else:raise AssertionError('lost endpoint accepted')
print(count,'anchor and signed-receiver checks passed; no derivative output/model calls')
