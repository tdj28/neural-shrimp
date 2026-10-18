"""Authenticate a closed pilot and all saved state endpoints; no model replay."""
import hashlib,json,subprocess
from pathlib import Path
from fractions import Fraction as F
R=Path.cwd();raw=Path('/data2/neural-shrimp-data/codex/coexistence');out=raw/'transient-pilot-001';sup=raw/'transient-pilot-001-supervision'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((sup/'receipt.json').read_text());a=json.loads((sup/'admission.json').read_text())
assert r['returncode']==1 and r['worker_reaped'] and r['worker_group_session_absent']
for e in r['output_files']:assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
for e in a['source_blobs']:assert hashlib.sha256(subprocess.check_output(['git','show',r['source_commit']+':'+e['relative_path']])).hexdigest()==e['sha256']
ids={r['worker_pid'],r['supervisor']['pid']}
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  s=(q/'stat').read_text().rsplit(')',1)[1].split();assert not ({int(q.name),int(s[2]),int(s[3])}&ids)
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass

def dy(v):
 assert len(v)==2 and all(type(i)is int for i in v)
 return F(v[0])*F(2)**v[1]
rows=[]
files=sorted(out.glob('state-*.json'));assert len(files)==4
for i,p in enumerate(files):
 x=json.loads(p.read_text());assert x['step']==i and len(x['state'])==896
 radii=[]
 for v in x['state']:
  m=dy(v['mid']);rad=dy(v['rad']);assert rad>=0;radii.append(rad)
 if i==0:assert all(v==0 for v in radii)
 else:assert dy(x['max_radius']['mid'])+dy(x['max_radius']['rad'])>=max(radii)
 rows.append({'step':i,'max_radius_exact':[max(radii).numerator,max(radii).denominator],'max_radius_float_display_may_underflow':float(max(radii)),'nonzero_radii':sum(v>0 for v in radii),'seconds':x.get('seconds'),'sha256':sha(p)})
f=json.loads((out/'failure.json').read_text());assert f['message']=='positive RMS denominator not established'
res={'source':r['source_commit'],'receipt_sha256':sha(sup/'receipt.json'),'artifacts':len(r['output_files']),'source_blobs':len(a['source_blobs']),'host_original_groups_absent':True,'session_reaped_exit':1,'saved_steps':rows,'failure_on_step':4,'failure':f,'external_seconds':r['external_elapsed_seconds'],'all3584_serialized_components_checked':True,'live_chain_authority':'reviewed loop passes original output ball objects without midpoint or decode reset; no model replay','reachability_proved':False,'interpretation':'interval evaluation inconclusive; not evidence of divergent exact trajectory'}
Path('codex/coexistence-task33/transient-pilot-001-completion.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
