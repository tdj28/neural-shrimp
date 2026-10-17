"""Independent remaining PMX saved-data arithmetic. No tokenizer/model imports."""
import json,math,importlib.util,tempfile
from pathlib import Path
from collections import Counter,OrderedDict
import numpy as np

def loadmodule(name):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pm=loadmodule('pm_core');tw=loadmodule('twin_core')
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def arm(raw,cfg):
 # Reuse exact accepted loader through an owned temporary serialization; never edit original records.
 with tempfile.TemporaryDirectory()as d:
  p=Path(d)/'records.jsonl';p.write_text(''.join(json.dumps(r)+'\n'for r in raw));r,l,e,g=pm.load_rows(p,cfg,'pm2')
 return dict(rows=r,layouts=l,extra=e,cfg=cfg),g

def ends(raw):
 out=OrderedDict()
 for r in raw:
  if r['index']<0:continue
  d=out.setdefault(str(r['layout']),dict(rows=0,own=0,cold_stop=0,budget=0,switch=[]));d['rows']+=1
  key='cold_stop'if r.get('cold_stop')else 'budget'if r['forced_switch']else 'own';d[key]+=1
  if r['switch_step']is not None:d['switch'].append(r['switch_step'])
 for d in out.values():
  ss=d.pop('switch');d['median_switch']=float(np.median(ss))if ss else None
 return out

def early_equality(new,old):
 def table(rr):
  d={};dups=[]
  for r in rr:
   if r['index']<0:continue
   k=(r['index'],r['layout'])
   if k in d:dups.append(list(k))
   d[k]=r
  return d,dups
 a,ad=table(new);b,bd=table(old);out={}
 for label,include,natural in [('strict_before1536',lambda x:x<1536,False),('inclusive_through1536',lambda x:x<=1536,False),('natural_end_strict_before1536',lambda x:x<1536,True),('natural_end_inclusive_through1536',lambda x:x<=1536,True)]:
  ledger=[]
  for k,r in b.items():
   if r['switch_step']is None or not include(r['switch_step']) or (natural and r.get('forced_switch')is not False):continue
   s=a.get(k);common={f:dict(old_present=f in r,new_present=s is not None and f in s,equal=(s[f]==r[f])if s is not None and f in s and f in r else None)for f in ['switch_step','answer','correct','forced_switch','text','answer_ids','top1']}
   ledger.append(dict(index=k[0],layout=k[1],new_present=s is not None,fields_equal=common,switch_and_answer_equal=bool(s is not None and s.get('switch_step')==r.get('switch_step')and s.get('answer')==r.get('answer')),scope='Saved field equality only, not native full-state bitwise replay'))
  out[label]=dict(denominator=len(ledger),switch_and_answer_equal=sum(x['switch_and_answer_equal']for x in ledger),rows=ledger)
 out['duplicates']=dict(new=ad,old=bd);out['new_only']=[list(x)for x in sorted(a.keys()-b.keys())];out['old_only']=[list(x)for x in sorted(b.keys()-a.keys())]
 return out

class ScientificInputRefusal(ValueError):
 def __init__(self,message,ledger):super().__init__(message);self.ledger=ledger

def complete_arm(raw,audit,paired,expected):
 ids=sorted(set(r['index']for r in raw if r['index']>=0))
 bad=[]
 if len(raw)!=expected['expected_records']:bad.append('record_count')
 if len(paired)!=expected['expected_problems']:bad.append('paired_count')
 if ids!=expected['indices']:bad.append('cohort_indices')
 for field in ['duplicates_last_write_retained','unpaired','unexpected_layout_rows']:
  if audit[field]:bad.append(field)
 return bad

def cold(cells):
 summaries={};ledger={}
 for name,c in cells.items():
  roles=c['arms'];rr={k:v['records']for k,v in roles.items()};original_ids={k:sorted(set(r['index']for r in v if r['index']>=0))for k,v in rr.items()}
  keep=set(c['expected']['indices'])
  rr['pm2_soft']=[r for r in rr['pm2_soft']if r['index']in keep];rr['pm2_plain']=[r for r in rr['pm2_plain']if r['index']in keep]
  aa={};dd={};bad={}
  for mode,data in rr.items():
   aa[mode],dd[mode]=arm(data,roles[mode]['config']['args'])
   bad[mode]=complete_arm(data,dd[mode],aa[mode]['rows'],c['expected'])
  ledger[name]=dict(arm_rows=dd,complete_cohort_gates=bad,full_original_indices=original_ids,selected_indices=sorted(keep),gpu={k:x['config'].get('gpu')for k,x in roles.items()},gpu_same={mode:roles[mode]['config'].get('gpu')==roles['pm2_'+mode]['config'].get('gpu')if roles[mode]['config'].get('gpu')is not None and roles['pm2_'+mode]['config'].get('gpu')is not None else None for mode in ['soft','plain']})
  if any(bad.values()):raise ScientificInputRefusal('incomplete ColdStop cohort '+name,ledger)
  selected={'cs':dict(role='exploratory',soft=aa['soft'],plain=aa['plain']),'pm2':dict(role='exploratory',soft=aa['pm2_soft'],plain=aa['pm2_plain'])}
  res,pair=pm.analyze_pm2(selected,[])
  summaries[name]=dict(cold_stop=res['cells']['cs'],pm2_same_problems=res['cells']['pm2'],ends={mode:ends(x)for mode,x in rr.items()},dirs=[roles[k]['dir']for k in ['soft','plain','pm2_soft','pm2_plain']])
  ledger[name].update(pairing=pair,early_equality={mode:early_equality(rr[mode],rr['pm2_'+mode])for mode in ['soft','plain']},scope='Each cell resets seed, then CS soft/plain/pair calls precede restricted PM2. Budget and stop jointly changed. Full and selected cohorts, GPU metadata and absent comparison fields retained; no native bitwise claim.')
 return summaries,ledger

def twin_projection(rr,args):
 ts=[t for r in rr for t in r['twins']];end=[t['log10_end_think']for t in ts if t['log10_end_think']is not None];g=[t['growth_rate']for t in ts if t['growth_rate']is not None];eps=math.log10(args['eps0']);k=sum(r['correct']for r in rr);fl=sum(t['flip']for t in ts);anyfl=sum(any(t['flip']for t in r['twins'])for r in rr)
 return dict(problems=len(rr),twins=len(ts),accuracy_base=k/len(rr),accuracy_base_wilson95=pm.wilson(k,len(rr)),twins_grew=sum(v>eps for v in end),twins_shrank=sum(v<=eps for v in end),twin_flips=fl,twin_flip_rate=fl/len(ts),twin_flip_wilson95=pm.wilson(fl,len(ts)),problems_any_flip=anyfl/len(rr),problems_any_flip_wilson95=pm.wilson(anyfl,len(rr)),different_switch=sum(not t['same_switch']for t in ts),growth_rate_quartiles=[float(x)for x in np.quantile(g,[.25,.5,.75])]if g else None,growth_rate_n=len(g),log10_end_think_median=float(np.median(end))if end else None,indices=sorted(r['index']for r in rr))

def qwq(old,new):
 oi=[r['index']for r in old['records']];ni=[r['index']for r in new['records']]
 if len(oi)!=32 or len(ni)!=64 or len(set(oi))!=32 or len(set(ni))!=64 or set(oi)&set(ni):raise ValueError('QwQ original32/new64/disjointness gate')
 oa=old['config']['args'];args=new['config']['args'];excluded={'limit','skip_groups','out'}
 diffs={k:dict(old_present=k in oa,new_present=k in args,old=oa.get(k),new=args.get(k))for k in sorted(set(oa)|set(args))if k not in excluded and (k not in oa or k not in args or oa[k]!=args[k])}
 if diffs:raise ScientificInputRefusal('QwQ semantic argument mismatch',diffs)
 if oa.get('limit')!=32 or oa.get('skip_groups',0)!=0 or args.get('limit')!=96 or args.get('skip_groups')!=8:raise ValueError('QwQ declared selection arguments')
 sets={};details={}
 for name,rr,dirs in [('new64',new['records'],[new['dir']]),('all96',old['records']+new['records'],[old['dir'],new['dir']])]:
  sets[name]=dict(dirs=dirs,**twin_projection(rr,args));details[name]=tw.reduce_rows(rr,args)
 return dict(eps0=args['eps0'],sets=sets),dict(disjoint=True,semantic_arguments_equal=True,selection_arguments={k:dict(old=oa.get(k),new=args.get(k))for k in sorted(excluded)},gpu_metadata=dict(old=old['config'].get('gpu'),new=new['config'].get('gpu')),original_indices=oi,new_indices=ni,interpretation='limit96before skip8groups yields64new;32old+64new=96combined',diagnostics=details)

def diagnosis(conditions,f64,indices):
 keep=set(indices);base={};base_dups=[]
 for r in f64:
  if r['index']not in keep:continue
  if r['index']in base:base_dups.append(r['index'])
  base[r['index']]=r
 if len(indices)!=64 or len(keep)!=64 or len(base)!=64 or base_dups:raise ValueError('diagnosis selected64base gate')
 out=dict(problems=len(indices),f64=dict(n=len(base),accuracy=float(np.mean([r['correct']for r in base.values()])),wilson95=pm.wilson(sum(r['correct']for r in base.values()),len(base))),conditions={});ledger={}
 for name,c in sorted(conditions.items()):
  d={};dups=[]
  for r in c['records']:
   if r['index']not in keep:continue
   if r['index']in d:dups.append(r['index'])
   d[r['index']]=r
  if dups or set(d)!=keep:raise ValueError('diagnosis condition cohort '+name)
  cfg=c['config'];rr=list(d.values());n=len(rr);k=sum(r['correct']for r in rr);common=sorted(d.keys()&base.keys());b=sum(d[i]['correct']and not base[i]['correct']for i in common);cc=sum(base[i]['correct']and not d[i]['correct']for i in common);ab=cfg['args'].get('answer_budget');tt=[r.get('switch_step',r.get('end_think_step'))for r in rr]
  out['conditions'][name]=dict(n=n,complete=n==len(keep),accuracy=k/n,wilson95=pm.wilson(k,n),tokenizer=cfg.get('tokenizer_class'),roundtrip=cfg.get('tokenizer_roundtrip'),attn=cfg.get('attn_implementation'),gpu=cfg.get('gpu'),missing=sum(r['answer']is None for r in rr),answer_at_budget=sum(r['answer_tokens']>=ab for r in rr)if ab else sum(r.get('hit_budget',False)for r in rr),think_forced=sum(bool(r.get('forced_switch'))for r in rr),no_end_of_thinking=sum(t is None for t in tt),median_think_end=float(np.median([t for t in tt if t is not None]))if any(t is not None for t in tt)else None,minus_f64=dict(problems=len(common),difference=float(np.mean([d[i]['correct']-base[i]['correct']for i in common])),discordant=[b,cc],mcnemar_p=pm.exact_mcnemar(b,cc)))
  ledger[name]=dict(rows=[dict(index=r['index'],stored_correct=r['correct'],correct_from_stored_answer=r['answer']is not None and r['answer']==r['gold'],stored_answer=r['answer'],suffix_extracted_answer=pm.answer(r.get('text',''),'pm2'),saved_text_characters=len(r.get('text','')),answer_ids_present='answer_ids'in r)for r in rr],duplicates=dups,scope='No tokenizer import/re-decoding. Published answers_changed_by_decoder is inherited unavailable independent endpoint; saved suffix extraction is only a diagnostic.')
 return out,ledger
