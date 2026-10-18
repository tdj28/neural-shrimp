import json,hashlib,pathlib,subprocess,datetime,math
R=pathlib.Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def pin(p):p=pathlib.Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
configs=[('points','codex/certify-chaos/point-plan.json','codex/certify-chaos/points-completion.json'),('edge','codex/boundary-task29/edge-C-plan.json','codex/boundary-task29/edge-C-completion.json')]
for lane,pf,dest in configs:
 plan=read(R/pf);ln=plan['lanes'][lane];out=pathlib.Path(ln['output']);sup=pathlib.Path(ln['supervision']);receipt=read(sup/'receipt.json');assert receipt['returncode']==0 and receipt['failure'] is None and receipt['worker_reaped']
 pins=plan['source_files']+ln['preflight_pins']+receipt['output_files']
 for p in pins:assert sha(p['path'])==p['sha256'] and pathlib.Path(p['path']).stat().st_size==p['bytes'],p['path']
 for p in plan['source_files']:
  rel=str(pathlib.Path(p['path']).relative_to(R));b=subprocess.check_output(['git','show',receipt['source_commit']+':'+rel],cwd=R);assert hashlib.sha256(b).hexdigest()==p['sha256']
 ids=[receipt['supervisor'],receipt['worker_observed_after_exec']];groups={i['pgid'] for i in ids};sessions={i['sid'] for i in ids}
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=(p/'stat').read_text().split(') ',1)[1].split()
  except (OSError,IndexError):continue
  assert int(a[2]) not in groups and int(a[3]) not in sessions,(p,a)
 report=read(out/'report.json');assert report['completed']
 summary={k:v for k,v in report.items() if k not in ['ledger','results']}
 if lane=='points':
  for c in report['cells']:
   name=c['cell']['name'];rows=[json.loads(x) for x in (out/name/'layers.jsonl').read_text().splitlines()];assert [(r['kind'],r['layer']) for r in rows]==[(k,i) for k in ['prefix_end','step_end'] for i in range(24)]
   point=read(out/name/'point-step.json');assert len(point['value'])==len(point['tangent'])==896
   from fractions import Fraction
   def dy(v):return Fraction(v[0])*Fraction(2)**v[1]
   radii=[dy(v['rad']) for v in point['value']];assert all(x>=0 for x in radii);assert all(dy(v['mid'])==dy(v['rad'])==0 for v in point['tangent'])
   assert max(radii)==dy(c['max_output_radius']['mid']);assert c['max_output_radius']['rad'][0]==0
   for row in rows:
    assert dy(row['max_radius']['mid'])>=0 and dy(row['max_radius']['rad'])==0
   c['reviewed_layer_rows']=len(rows);c['output_radius_approx']=float(max(radii))
 else:
  summary['results']=[{k:v for k,v in x.items() if k not in ['history','rebrackets']} for x in report['results']]
  summary['ledger_rows']=len(report['ledger']);summary['all_fate_labels']=[x['labels'] for x in report['ledger']]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=pin(sup/'receipt.json'),report=pin(out/'report.json'),source_commit=receipt['source_commit'],source_and_output_pins_verified=len(pins),source_git_blobs_verified=len(plan['source_files']),original_groups_sessions_absent=True,external_seconds=receipt['external_elapsed_seconds'],child_peak_rss_kib=receipt['children_ru_maxrss_kib_at_reap'],summary=summary,scope='Source/input/output authentication and saved report consistency; not a second model execution or independent enclosure proof.')
 (R/dest).write_text(json.dumps(result,indent=2)+'\n');print(lane,json.dumps(result))
