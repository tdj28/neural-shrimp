import sys,json,tempfile,hashlib,copy
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/coexistence-task33'))
import assembly_core as a
name='p2-derivative-p0-c000-001'
run=dict(step='run',success=True,closure_verified=True,exit_code=0,watchdog_stop=None,killed_by_watchdog=None,alive_after_exit=[],alive_after_cleanup=[],command=f'/python -I codex/coexistence-task33/launch.py --attempt {name} --driver derivative_shard --plan {name}-plan.json --threads 1 --seconds 3600')
summary=dict(terminated=dict(terminated=True,in_inventory=False),attempts=[dict(attempt=name,receipt_sha256='original')])
cases=[('valid',None,None,True),('live-worker','alive_after_exit',[1],False),('cleanup-live','alive_after_cleanup',[1],False),('watchdog','watchdog_stop','deadline',False),('killed','killed_by_watchdog',True,False),('failed','exit_code',1,False),('bool-exit','exit_code',False,False),('unclosed','closure_verified',False,False),('wrong-attempt','command',run['command'].replace('--attempt '+name,'--attempt wrong'),False),('extended','command',run['command'].replace('--seconds 3600','--seconds 7200'),False),('duplicate-option','command',run['command']+' --threads 1',False)]
results=[]
with tempfile.TemporaryDirectory()as d:
 d=Path(d)
 def check(label,r,s,expected):
  refs={}
  for k,v in [('outer_closure',r),('termination',s)]:
   p=d/(k+'.json');p.write_text(json.dumps(v));refs[k]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  refs.update(attempt='/raw/'+name,receipt_sha256='original')
  try:a.authenticate_closure(refs);passed=True
  except ValueError:passed=False
  assert passed==expected,label
  results.append(dict(case=label,expected_admission=expected,matched=True))
 for label,key,val,expected in cases:
  r=copy.deepcopy(run)
  if key:r[key]=val
  check(label,r,summary,expected)
 for label,mutate in [('unterminated',lambda s:s['terminated'].update(terminated=False)),('inventory-live',lambda s:s['terminated'].update(in_inventory=True)),('receipt-mismatch',lambda s:s['attempts'][0].update(receipt_sha256='other')),('duplicate-receipt',lambda s:s['attempts'].append(dict(s['attempts'][0])))]:
  s=copy.deepcopy(summary);mutate(s);check(label,run,s,False)
print(json.dumps(dict(controls=results,count=len(results),model_calls=0),indent=2))
