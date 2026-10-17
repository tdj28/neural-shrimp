"""Independent corrected PM2 cells and declared pooled variants, saved records only.
No producer imports, tokenizer/model calls, outcome-dependent order, or watch.
"""
import argparse,hashlib,importlib.util,json,os,resource,time,traceback
from pathlib import Path

def module(name):
 p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def validate(q):
 a=pin(q['path'])
 if any(a[k]!=q[k] for k in ['bytes','sha256']):raise ValueError('changed input '+q['path'])
 return a

def complete_arm(ledger,rows,expected):
 if ledger["record_count"]!=expected["expected_records"] or len(rows)!=expected["expected_paired_problems"]:
  raise ValueError("declared complete arm count mismatch")
 if ledger["duplicates_last_write_retained"] or ledger["unpaired"] or ledger["unexpected_layout_rows"]:
  raise ValueError("declared complete arm has duplicate/unpaired/unexpected-layout rows")

def execute(card,out):
 start=time.monotonic();core=module('pm_core');ctl=module('controls')
 for n,h in card['source_sha256'].items():
  if pin(Path(__file__).with_name(n))['sha256']!=h:raise ValueError('scientific source pin mismatch '+n)
 validate(card['inputs']);inputs=json.loads(Path(card['inputs']['path']).read_text())
 pins=[validate(x) for x in inputs['files']]
 if sum(x['bytes'] for x in pins)>card['max_input_bytes']:raise ValueError('input byte cap')
 checks=ctl.check(core,complete_arm);(out/'controls.json').write_text(json.dumps(checks,indent=2)+'\n')
 if checks['failed']:raise ValueError('manufactured controls refused')
 files={x['id']:x for x in inputs['files']};loaded={};ledgers={}
 for name,a in inputs['arms'].items():
  cfg=json.loads(Path(files[a['config']]['path']).read_text())['args']
  rows,layouts,extra,ledger=core.load_rows(files[a['records']]['path'],cfg,'pm2')
  ledgers[name]=ledger
  try:complete_arm(ledger,rows,a)
  except ValueError:
   (out/'row-diagnostics.partial.json').write_text(json.dumps(ledgers,indent=1)+'\n');raise
  loaded[name]=dict(rows=rows,layouts=layouts,extra=extra,cfg=cfg)
 def cell(name):
  c=inputs['cells'][name];return dict(role='primary',soft=loaded[c['soft']],plain=loaded[c['plain']])
 results={};contrasts={}
 cohort_diagnostics={}
 for new,old in [('Bfix-gsm8k','B-gsm8k'),('Bfix-math','B-math')]:
  modes={}
  for mode in ['soft','plain']:
   x,y=cell(new)[mode]['rows'],cell(old)[mode]['rows'];common=sorted(x.keys()&y.keys())
   modes[mode]=dict(common=common,new_only=sorted(x.keys()-y.keys()),old_only=sorted(y.keys()-x.keys()),matched=[dict(index=i,new=x[i],old=y[i],same_flip=x[i]['flip']==y[i]['flip'])for i in common],scope='Descriptive paired intersection only; changed tokenizer cohorts are not treated as a paired causal effect.')
  cohort_diagnostics[new]=modes
 for v in inputs['variants']:
  selected={name:cell(name) for name in v['cells']}
  results[v['name']],contrasts[v['name']]=core.analyze_pm2(selected,[])
 # The earlier PM2 baseline is a published reference, not a new rerun of that completed audit.
 gs=json.loads(Path(files['published_gsm']['path']).read_text());ma=json.loads(Path(files['published_math']['path']).read_text())
 pairs=[]
 for v in inputs['variants']:
  published=(ma['pooled_primary_both_r1llama_cells_corrected'] if v['name']=='both_corrected' else gs['pooled_primary_variants'][v['name']])
  pairs+=core.compare_leaves(results[v['name']]['primary'],published,'pools/'+v['name'])
 for ownv,cn,pub,label in [('r1llama_gsm8k_corrected','Bfix-gsm8k',gs['cell_corrected'],'gsm'),('both_corrected','Bfix-math',ma['cell_corrected'],'math')]:
  pairs+=core.compare_leaves(results[ownv]['cells'][cn],pub,'corrected_cells/'+label)
 science=dict(cohort_diagnostics=cohort_diagnostics,variants=results,pairing=contrasts,published_baseline_reference=gs['pooled_primary_variants']['as_published_pm2'],baseline_scope='Published historical reference only; prior completed PM audit is not rerun here.',ordering=inputs['ordering'],source_role=inputs['source_role'])
 for name,obj in [('results.json',science),('row-diagnostics.json',ledgers),('comparisons.json',pairs)]:
  (out/name).write_text(json.dumps(core.__dict__.get('json_safe',lambda x:x)(obj),indent=1)+'\n')
 after=[validate(x) for x in inputs['files']];validate(card['inputs'])
 report=dict(schema='task27-pmx-corrected-saved-v1',computational_completed=True,scientific_claims_accepted=False,controls=checks,arm_count=len(loaded),variant_count=len(results),record_count=sum(x['record_count']for x in ledgers.values()),compared_leaves=len(pairs),comparison_status_counts={k:sum(x['status']==k for x in pairs)for k in sorted(set(x['status']for x in pairs))},input_pins=after,elapsed_before_report_manifest_exit_seconds=time.monotonic()-start,rss_observed_before_report_manifest_exit_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Saved labels/extractions and source-bound statistical streams; no native states, tokenization rerun, new model outcomes or confirmation of human-visible timing.',source_commit=card['source_expected_commit'])
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 names=['controls.json','results.json','row-diagnostics.json','comparisons.json','report.json']
 (out/'manifest.json').write_text(json.dumps(dict(schema='task27-pmx-corrected-output-v1',files=[pin(out/n) for n in names]),indent=2)+'\n')
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--card',required=True);p.add_argument('--out',required=True);a=p.parse_args();card=json.loads(Path(a.card).read_text())
 if card['admission']!='reviewed_for_execution':raise ValueError('prepared card refuses execution')
 if str(Path(a.out).resolve())!=card['output']:raise ValueError('fixed output mismatch')
 out=Path(a.out);out.mkdir(exist_ok=False)
 try:
  r=execute(card,out);print(json.dumps({k:r[k]for k in ['computational_completed','arm_count','variant_count','record_count']}))
 except BaseException:
  (out/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),science_accepted=False),indent=2)+'\n');raise
if __name__=='__main__':main()
