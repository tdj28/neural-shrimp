import json,hashlib,tarfile,pathlib,sys
R=pathlib.Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
B=pathlib.Path('/data2/neural-shrimp-data/runpod/soft64/20261018T065453Z/extracted/explore/llm-soft-loop/pod64/certcal/certcal-002')
sha=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((R/'codex/certificate-ladder-task34/calibration-002.json').read_text())
tar=B/(p['attempt']+'.tar'); pack=json.loads((B/'pack.json').read_text());assert sha(tar.read_bytes())==pack['archive_sha256']
out=pathlib.Path('/tmp/certcal002-admission');out.mkdir(exist_ok=False)
with tarfile.open(tar) as t:
 ms=t.getmembers(); assert all(m.isfile() or m.isdir() for m in ms)
 t.extractall(out,filter='data')
a=out/'codex/llm-p8-certificate'/p['attempt'];rec=json.loads((a/'receipt.json').read_text())
for x in rec['artifacts']:
 b=(a/x['path']).read_bytes();assert len(b)==x['bytes'] and sha(b)==x['sha256'],x['path']
sys.path.insert(0,str(R/'codex/certify-llm'))
from derivative_campaign import completed,common
plan=json.loads((a/'source'/p['plan']).read_text())
r=completed(a,{'phase_index':1,'start_column':256,'plan':{'name':pathlib.Path(p['plan']).name,'sha256':p['plan_sha256']}},common(plan),source='c6242ede708298d4db73b065c5d89798c729909b')
oldbytes=pathlib.Path(p['original_output']['path']).read_bytes();assert sha(oldbytes)==p['original_output']['sha256'];old=json.loads(oldbytes)
new=json.loads((a/'run/derivative-shard.json').read_text());diff=[k for k in old if old[k]!=new[k]]
for k in ('actual_inputs','directions','derivatives','primal'):assert old[k]==new[k],k
pre=json.loads((B/'preflight.json').read_text());assert pre['ok'] and all(x['pass'] for x in pre['checks'].values())
run=json.loads((B/'run.json').read_text());assert run['success'] and run['closure_verified'] and not run['alive_after_exit'] and not run['alive_after_cleanup'] and run['exit_code']==0
report={'scope':'Operational completed() admission and exact stored native endpoint comparison; no new certificate or scale-up authorization','archive_sha256':pack['archive_sha256'],'receipt_sha256':sha((a/'receipt.json').read_bytes()),'artifacts_verified':len(rec['artifacts']),'completed':r,'all_input_direction_derivative_primal_fields_exact':True,'different_top_level_fields':diff,'differences':{k:{'original':old[k],'remote':new[k]} for k in diff},'remote_wrapper':run,'preflight_sha256':sha((B/'preflight.json').read_bytes()),'audit_source_sha256':sha(pathlib.Path(__file__).read_bytes()),'termination_basis':'Producer logged termination; Codex makes no pod query'}
pathlib.Path('/tmp/certcal002-admission-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('remote_wrapper',)},indent=2))
