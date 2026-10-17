"""Cold Stop, QwQ64+32 and diagnosis saved arithmetic; no producer/model/tokenizer imports."""
import argparse,hashlib,importlib.util,json,resource,tempfile,time,traceback
from pathlib import Path

def module(name):
 p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def validate(q):
 a=pin(q['path'])
 if any(a[k]!=q[k]for k in ['bytes','sha256']):raise ValueError('changed input '+q['path'])
 return a

def execute(card,out):
 start=time.monotonic();r=module('reduce');ctl=module('controls');tempfile.tempdir=str(out)
 def write(name,obj):
  (out/name).write_text(json.dumps(r.tw.json_safe(obj),indent=1,allow_nan=False)+'\n')
 for name,h in card['source_sha256'].items():
  if pin(Path(__file__).with_name(name))['sha256']!=h:raise ValueError('source changed '+name)
 validate(card['inputs']);inv=json.loads(Path(card['inputs']['path']).read_text());before=[validate(q)for q in inv['files']]
 if sum(q['bytes']for q in before)+card['inputs']['bytes']>card['max_input_bytes']:raise ValueError('input byte cap')
 controls=ctl.check(r);write('controls.json',controls)
 if controls['failed']or controls['passed']!=8:raise ValueError('focused manufactured controls failed')
 files={q['id']:q for q in inv['files']}
 def load(k):return json.loads(Path(files[k]['path']).read_text())
 def role(spec):
  rr=r.rows(files[spec['records']]['path']);ids=[{k:x[k]for k in ident}for x,ident in zip(rr,spec['ordered_identity'])]
  if len(rr)!=spec['expected_records']or ids!=spec['ordered_identity']:raise r.ScientificInputRefusal('original record count/order mismatch',dict(role=spec,actual_ids=ids,actual_count=len(rr)))
  return dict(records=rr,config=load(spec['config']),dir=spec['dir'])
 try:
  cs,cd=r.cold({name:dict(expected=c['expected'],arms={mode:role(x)for mode,x in c['arms'].items()})for name,c in inv['cold'].items()});write('cold-stop.json',cs)
  qs,qd=r.qwq(role(inv['qwq']['old']),role(inv['qwq']['new']));write('qwq.json',qs)
  ds,dd=r.diagnosis({name:role(x)for name,x in inv['diagnosis']['conditions'].items()},role(inv['diagnosis']['baseline'])['records'],load(inv['diagnosis']['indices']));write('diagnosis.json',ds)
 except r.ScientificInputRefusal as e:write('input-refusal-ledger.json',e.ledger);raise
 pub={k:load(v)for k,v in inv['publications'].items()};comparisons=[]
 for own,published,path in [(cs,pub['pm2cs-summary']['cells'],'cold_stop'),(qs,dict(eps0=pub['pm2f64b-summary']['eps0'],sets=pub['pm2f64b-summary']['sets']),'qwq'),(ds,{k:pub['pm2x-llama-diag'][k]for k in ['problems','f64','conditions']},'diagnosis')]:comparisons+=r.pm.compare_leaves(own,published,path)
 for x in comparisons:
  if x['path'].endswith('/answers_changed_by_decoder'):x['scope']='inherited decoder field; independent tokenizer-only check is a separate source/card/receipt'
 write('comparisons.json',comparisons)
 write('evidence.json',dict(cold=cd,qwq=qd,diagnosis=dd,publication_metadata={k:{a:v for a,v in p.items()if a not in ['cells','sets','eps0','problems','f64','conditions']}for k,p in pub.items()},input_ordering=inv['ordering'],scope=inv['scope']))
 after=[validate(q)for q in inv['files']];validate(card['inputs'])
 report=dict(schema='task27-pmx-remaining-saved-v1',computational_completed=True,scientific_claims_accepted=False,controls=controls,cold_cells=len(cs),qwq_new_problems=qs['sets']['new64']['problems'],qwq_combined_problems=qs['sets']['all96']['problems'],diagnosis_conditions=len(ds['conditions']),input_pins=after,compared_leaves=len(comparisons),comparison_status_counts={k:sum(x['status']==k for x in comparisons)for k in sorted({x['status']for x in comparisons})},source_commit=card['source_expected_commit'],scope=inv['scope'],elapsed_before_report_manifest_exit_seconds=time.monotonic()-start,rss_observed_before_report_manifest_exit_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 write('report.json',report);names=['controls.json','cold-stop.json','qwq.json','diagnosis.json','evidence.json','comparisons.json','report.json'];write('manifest.json',dict(schema='task27-pmx-remaining-output-v1',files=[pin(out/n)for n in names]));return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--card',required=True);p.add_argument('--out',required=True);a=p.parse_args();card=json.loads(Path(a.card).read_text())
 if card['admission']!='reviewed_for_execution':raise ValueError('prepared card refuses')
 if str(Path(a.out).resolve())!=card['output']:raise ValueError('fixed output mismatch')
 out=Path(a.out);out.mkdir(exist_ok=False)
 try:
  r=execute(card,out);print(json.dumps({k:r[k]for k in ['computational_completed','cold_cells','qwq_new_problems','qwq_combined_problems','diagnosis_conditions']}))
 except BaseException:
  (out/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc(),scientific_claims_accepted=False),indent=2)+'\n');raise
if __name__=='__main__':main()
