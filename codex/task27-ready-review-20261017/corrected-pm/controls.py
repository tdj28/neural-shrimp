"""Tiny focused manufactured checks; run once integrated before actual records."""
import tempfile,json
from pathlib import Path
import numpy as np

def check(core,guard):
 checks=[]
 def add(name,fn):
  try:fn();checks.append(dict(name=name,passed=True))
  except Exception as e:checks.append(dict(name=name,passed=False,error=repr(e)))
 def extraction():
  assert core.answer('noise \\boxed{12}','pm2')=='12'
  assert core.answer('no numeric answer','pm2') is None
  assert core.answer('a 5.0','pm2')=='5'
 def stats():
  assert core.exact_mcnemar(0,0)==1
  assert core.exact_mcnemar(0,4)==.125
  lo,hi=core.wilson(1,2);assert 0<lo<.5<hi<1
 def rows():
  rr=[]
  for i in [2,0,1]:
   for l in [8,16]:rr.append(dict(index=i,layout=l,batch=i//2,row=i,prompt_len=10,correct=(l==8),answer='1'if l==8 else None,gold='1',text='1'if l==8 else 'missing',switch_step=2,forced_switch=False))
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'records.jsonl';p.write_text(''.join(json.dumps(x)+'\n'for x in rr));r,la,ex,d=core.load_rows(p,{'layouts':[8,16]},'pm2')
   assert list(r)==[2,0,1] and len(d['paired_rows'])==3 and all(x['flip']for x in r.values())
   assert list(core.group_counts(r,'flip','encounter'))==[1,0]
   assert list(core.group_counts(r,'flip','sorted'))==[0,1]
 def paired():
  a={0:dict(flip=True,cluster=0,gold='x'),1:dict(flip=False,cluster=1,gold='x')};b={0:dict(flip=False,cluster=0,gold='x'),1:dict(flip=True,cluster=1,gold='x')}
  q,d=core.pairwise(a,b,'flip',np.random.default_rng(1));assert q['discordant']==[1,1]and q['difference']==0 and q['mcnemar_p']==1
 def compare():
  d=core.compare_leaves({'a':1,'b':3},{'a':2,'c':4});assert len(d)==3 and {x['status']for x in d}=={'different','own_only','published_only'}
 def complete_guard():
  d=dict(record_count=4,duplicates_last_write_retained=[],unpaired=[],unexpected_layout_rows=[]);r={0:{},1:{}};e=dict(expected_records=4,expected_paired_problems=2)
  guard(d,r,e)
  for key in ['duplicates_last_write_retained','unpaired','unexpected_layout_rows']:
   bad={**d,key:[{'retained':True}]}
   try:guard(bad,r,e)
   except ValueError:pass
   else:raise AssertionError('missing refusal '+key)
  try:guard(d,{0:{}},e)
  except ValueError:pass
  else:raise AssertionError('missing paired-count refusal')
 for n,f in [('complete_arm_guard',complete_guard),('boxed_and_missing_extraction',extraction),('wilson_exact_mcnemar',stats),('row_identity_and_cluster_order',rows),('paired_signed_contrast',paired),('all_comparison_dispositions',compare)]:add(n,f)
 return dict(schema='task27-pmx-corrected-controls-v1',passed=sum(x['passed']for x in checks),failed=sum(not x['passed']for x in checks),groups=checks)
