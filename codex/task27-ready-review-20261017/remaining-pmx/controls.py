"""Focused new integration checks; inherited PM/twin broad suites are not repeated."""
import copy

def check(r):
 checks=[]
 def add(name,fn):
  try:fn();checks.append(dict(name=name,passed=True))
  except Exception as e:checks.append(dict(name=name,passed=False,error=repr(e)))
 def refusal(fn):
  try:fn()
  except ValueError:return
  raise AssertionError('expected refusal absent')
 def record(i,l=8,step=2,forced=False):return dict(index=i,layout=l,batch=i//2,row=i,prompt_len=10,answer='1',gold='1',correct=True,text='\\boxed{1}',switch_step=step,forced_switch=forced)
 def ending():
  a=[record(0),record(1,forced=True),dict(record(2,forced=True),cold_stop=True),dict(record(3),switch_step=None)]
  q=r.ends(a)['8'];assert [q[x]for x in ['rows','own','cold_stop','budget']]==[4,2,1,1]and q['median_switch']==2
 def early():
  old=[record(0,step=1535),record(1,step=1536),record(2,step=1535,forced=True)];new=copy.deepcopy(old);new[0]['answer_ids']=[3]
  q=r.early_equality(new,old)
  assert [q[k]['denominator']for k in ['strict_before1536','inclusive_through1536','natural_end_strict_before1536','natural_end_inclusive_through1536']]==[2,3,1,2]
  f=q['strict_before1536']['rows'][0]['fields_equal']['answer_ids'];assert f==dict(old_present=False,new_present=True,equal=None)
 def cohort():
  data=[record(i,l)for i in [1,0]for l in [8,16]];a,d=r.arm(data,dict(layouts=[8,16]));e=dict(expected_records=4,expected_problems=2,indices=[0,1])
  assert not r.complete_arm(data,d,a['rows'],e)
  assert 'cohort_indices'in r.complete_arm(data,d,a['rows'],{**e,'indices':[0,2]})
  assert 'record_count'in r.complete_arm(data,d,a['rows'],{**e,'expected_records':6})
  assert 'paired_count'in r.complete_arm(data,d,a['rows'],{**e,'expected_problems':3})
  for key in ['duplicates_last_write_retained','unpaired','unexpected_layout_rows']:assert key in r.complete_arm(data,{**d,key:[1]},a['rows'],e)
 def cold_reset():
  data=[record(i,l)for i in range(4)for l in [8,16]];cfg=dict(args=dict(layouts=[8,16],model='fixture',task='gsm8k'),gpu='fixture GPU');c=dict(arms={m:dict(records=copy.deepcopy(data),config=cfg,dir=m)for m in ['soft','plain','pm2_soft','pm2_plain']},expected=dict(expected_records=8,expected_problems=4,indices=list(range(4))))
  a,d=r.cold({'one':c,'two':c});assert a['one']==a['two']and d['one']['gpu_same']==dict(soft=True,plain=True)
  bad=copy.deepcopy(c);bad['arms']['soft']['records'].pop();refusal(lambda:r.cold({'bad':bad}))
 def twins(i):return dict(index=i,answer='1',gold='1',correct=True,switch_step=2,twins=[dict(answer='1',switch_step=2,flip=False,same_switch=True,growth_rate=0.,log10_end_think=-10.,log10_max_think=-10.)for _ in range(2)])
 def qcase():
  args=dict(model='qwq32b',offset=256,group=4,twins=2,eps0=1e-10,temperature=.6,top_n=15,top_p=.95,think_budget=1536,answer_budget=256,seed=20261016,threads=8,device='cuda')
  old=dict(records=[twins(i)for i in range(32)],config=dict(args={**args,'limit':32,'out':'old'}),dir='old');new=dict(records=[twins(i)for i in range(32,96)],config=dict(args={**args,'limit':96,'skip_groups':8,'out':'new'}),dir='new');return old,new
 def qscope():
  a,b=qcase();q,d=r.qwq(a,b);assert q['sets']['new64']['twins']==128 and q['sets']['all96']['problems']==96 and q['sets']['all96']['twins_shrank']==192
  b['records'][0]['index']=0;refusal(lambda:r.qwq(a,b))
 def qargs():
  a,b=qcase();b['config']['args']['temperature']=.7;refusal(lambda:r.qwq(a,b));a,b=qcase();b['config']['args'].pop('top_n');refusal(lambda:r.qwq(a,b))
 def diag():
  base=[twins(i)for i in range(64)];rr=[dict(record(i),answer_tokens=4,end_think_step=2,hit_budget=True)for i in range(64)]
  cfg=dict(args=dict(answer_budget=4),gpu='fixture');q,d=r.diagnosis({'x':dict(records=rr,config=cfg)},base,list(range(64)));assert q['conditions']['x']['answer_at_budget']==64 and 'answers_changed_by_decoder'not in q['conditions']['x']
  rr.append(rr[0]);refusal(lambda:r.diagnosis({'x':dict(records=rr,config=cfg)},base,list(range(64))))
 def compare():
  q=r.pm.compare_leaves({'a':1,'extra':2},{'a':1,'decoder':3});assert {x['status']for x in q}=={'exact','own_only','published_only'}
 for n,f in [('ending_precedence_and_missing_switch',ending),('strict_inclusive_natural_end_and_missing_fields',early),('complete_cohort_and_duplicate_guards',cohort),('cold_per_call_seed_reset_and_refusal',cold_reset),('qwq_32_plus_64_disjoint_and_endpoint_equality',qscope),('qwq_semantic_args_refusal',qargs),('diagnosis_complete64_budget_and_decoder_scope',diag),('all_comparison_dispositions',compare)]:add(n,f)
 return dict(schema='task27-pmx-remaining-controls-v1',passed=sum(x['passed']for x in checks),failed=sum(not x['passed']for x in checks),groups=checks)
