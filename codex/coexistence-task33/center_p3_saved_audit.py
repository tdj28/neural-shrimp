import json,sys,hashlib,subprocess
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/certify-llm'))
from flint import ctx
from cache_io import unpack
ctx.prec=2048
raw=Path('/data2/neural-shrimp-data/codex/coexistence');base=raw/'center-p3-001';sup=raw/'center-p3-001-supervision';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((sup/'receipt.json').read_text());a=json.loads((sup/'admission.json').read_text());assert r['returncode']==0 and r['failure'] is None and r['worker_reaped'] and r['worker_group_session_absent']
for e in r['output_files']:assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
for e in a['source_blobs']:assert hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+e['relative_path']],cwd=R)).hexdigest()==e['sha256']
ids={r['worker_pid'],r['supervisor']['pid']}
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  s=(q/'stat').read_text().rsplit(')',1)[1].split();assert not ({int(q.name),int(s[2]),int(s[3])}&ids)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
report=json.loads((base/'report.json').read_text());wit=json.loads((base/'correction-witnesses.json').read_text());timings=[];values=[]
for li,label in enumerate(['before','after']):
 cs=json.loads((base/(label+'-centers.json')).read_text());centers=[[unpack(v) for v in row] for row in cs];images=[]
 assert len(centers)==3 and all(len(row)==896 and all(v.is_exact() for v in row) for row in centers)
 assert cs==wit['original_centers' if label=='before' else 'next_centers']
 for i in range(3):
  point=json.loads((base/f'{label}-phase-{i}.json').read_text());im=[unpack(v) for v in point['value']];assert len(im)==896 and all(unpack(v).is_exact() and unpack(v)==0 for v in point['tangent']);images.append(im);timings.append(point['seconds'])
  if label=='before':assert point['value']==wit['images'][i]
 rs=json.loads((base/(label+'-residuals.json')).read_text());calc=[[images[i][j]-centers[(i+1)%3][j] for j in range(896)] for i in range(3)]
 for row,saved in zip(calc,rs['residuals']):
  for v,w in zip(row,saved):
   u=unpack(w);assert v.mid()==u.mid() and v.rad()==u.rad()
 upper=max(v.abs_upper() for row in calc for v in row);assert upper==unpack(rs['residual_inf_upper'])==unpack(report['summaries'][li]['residual_inf_upper']);values.append(str(upper))
for i in range(3):
 for j in range(896):assert (unpack(wit['original_centers'][i][j])+unpack(wit['corrections'][i][j])).mid()==unpack(wit['next_centers'][i][j])
out=dict(source=r['source_commit'],receipt_sha256=sha(sup/'receipt.json'),report_sha256=sha(base/'report.json'),artifacts=len(r['output_files']),source_blobs=len(a['source_blobs']),host_original_groups_absent=True,session_reaped_exit=0,all5376_residual_components_exact=True,all2688_midpoint_choices_exact=True,all_point_witnesses_checked=True,residual_upper_before_after=values,point_seconds=timings,external_seconds=r['external_elapsed_seconds'],certificate_passed=False)
Path('/tmp/center-p3-completion.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
