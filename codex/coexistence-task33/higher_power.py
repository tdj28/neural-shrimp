"""Disclosed saved-monodromy extension; no model/derivative recomputation."""
import argparse,json,hashlib,sys,gc,time
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
sys.path[:0]=[str(R/'codex/certify-llm'),str(R/'codex/coexistence-task33')]
from flint import ctx,arb_mat
import proof_gates as p
import witness_audit as w
import power_witness

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def verify(pin):
 assert Path(pin['path']).stat().st_size==pin['bytes'] and sha(pin['path'])==pin['sha256'],pin['path']
def restore(records):
 m=arb_mat([[p.restore_ball(v) for v in row]for row in records])
 for i,row in enumerate(records):
  for j,v in enumerate(row):
   lo,hi=w.endpoints(v);nl,nh=w.endpoints(p.ball_record(m[i,j]))
   if not(nl<=lo and nh>=hi):raise ValueError('restored monodromy narrows original')
 return m

def main(card_path,out):
 card=read(card_path);out=Path(out);out.mkdir(exist_ok=False);ctx.prec=1024;ctx.threads=1
 assert card['powers']==[4,8] and card['precision']==1024
 summaries=[]
 for case in card['cases']:
  start=time.monotonic()
  for key in ('assembly_receipt','assembly_report','audit_receipt','audit_report'):verify(case[key])
  for key in ('assembly_receipt','audit_receipt'):
   rc=read(case[key]['path']);assert rc['returncode']==0 and rc['failure']is None and rc['worker_reaped'] and rc['worker_group_session_absent']
  audit=read(case['audit_report']['path']);assert audit['report_sha256']==case['assembly_report']['sha256']
  flags=audit['independent_saved_witness']['conditional_gates'];assert flags['root_existence_and_local_uniqueness'] and flags['primitive_period']
  original=read(case['assembly_report']['path']);assert original['period']==case['period'] and original['dimension']==896
  records=original['attraction']['monodromy'];digest=w.matrix_digest(records,896)
  assert digest==original['monodromy']['matrix_sha256']
  m=restore(records);binding=dict(period=case['period'],original_report_sha256=case['assembly_report']['sha256'],original_monodromy_sha256=digest,restored_monodromy_sha256=w.matrix_digest(p.matrix_record(m),896),original_audit_sha256=case['audit_report']['sha256'],contract=original['contract'],original_failed_attraction=flags['local_attraction'] is False)
  del original,records;gc.collect()
  print('P%d original authority and all signed endpoint containment checked; computing fixed powers4,8'%case['period'],flush=True)
  attr=p.attraction_gate(m,powers=(4,8));del m;gc.collect()
  target=out/('p%d-powers.json'%case['period'])
  with target.open('x')as f:json.dump(dict(binding=binding,attraction=attr),f);f.write('\n')
  checked=power_witness.audit(attr,896)
  row=dict(binding=binding,independent_saved_witness=checked,report_sha256=sha(target),powers=[{k:x[k]for k in ('power','induced_infinity_upper','margin','passed')}for x in attr['powers']],seconds=time.monotonic()-start,certificate_passed=False)
  (out/('p%d-summary.json'%case['period'])).write_text(json.dumps(row,indent=2)+'\n');summaries.append(row)
  print(json.dumps(dict(period=case['period'],audit=checked,seconds=row['seconds'])),flush=True);del attr;gc.collect()
 (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();main(a.card,a.out)
