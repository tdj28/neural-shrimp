"""Synthetic Task34 adapter delta controls; no target evidence or model calls.

Builds an isolated temporary source/evidence tree. The sole test-only namespace
injection is bridge.OWNED -> the fresh synthetic /tmp evidence root; no original
artifact is read, relabelled, or admitted. All source/path/receipt gates execute.
Valid shard metadata reaches an intentional status sentinel before any endpoints.
"""
import argparse, copy, hashlib, importlib.util, json, shutil, sys, tempfile
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--repository',default='/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
ap.add_argument('--adapter',default=str(Path(__file__).with_name('derivative_bridge.py')))
args=ap.parse_args(); R=Path(args.repository); adapter=Path(args.adapter)
TEMPLATE='b49436618474ddd6074fd3006af5a806e028ca169a8fcc8480fd25d3d813531a'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(R/'codex/coexistence-task33/derivative_bridge.py')==TEMPLATE
count=0

def check_refusal(label, thunk, fragment):
 global count
 try: thunk()
 except ValueError as e:
  assert fragment in str(e),(label,str(e),fragment)
 else: raise AssertionError(label+' accepted')
 count+=1; print(label+': passed')

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+'\n');return p

def row(p,relative=None):
 return dict(path=str(p)if relative is None else relative,bytes=p.stat().st_size,sha256=sha(p))

with tempfile.TemporaryDirectory(prefix='task34-bridge-controls-')as temp:
 root=Path(temp); repo=root/'repo'; here=repo/'codex/certificate-ladder-task34'; here.mkdir(parents=True)
 for sub in ('codex/certify-llm','codex/crosscheck'):
  (repo/sub).mkdir(parents=True)
 for f in (R/'codex/certify-llm').glob('*.py'):shutil.copyfile(f,repo/'codex/certify-llm'/f.name)
 shutil.copyfile(R/'codex/crosscheck/qwen_map.py',repo/'codex/crosscheck/qwen_map.py')
 for name in ('derivative_shard.py','prefix_import.py'):shutil.copyfile(R/'codex/certificate-ladder-task34'/name,here/name)
 shutil.copyfile(adapter,here/'derivative_bridge.py')
 spec=importlib.util.spec_from_file_location('synthetic_task34_bridge',here/'derivative_bridge.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 b.ctx.prec=1024;b.ctx.threads=1
 # Explicit synthetic fixture namespace; the shipped adapter keeps its hardcoded owned root.
 owned=root/'evidence';owned.mkdir();b.OWNED=owned
 pins={n:sha(repo/n)for n in b.producer.SOURCE_PATHS}
 mp=b.producer.map_definition({'synthetic_asset':'1'*64},[1,2])
 backend=dict(b.producer.backend_definition('synthetic-torch'),precision=1024,threads=1)
 prefix=owned/'prefix';prefix.mkdir();psource='a'*40
 meta={k:mp[k]for k in ('pinned_assets','prefix_ids','steering_layer','alpha_ratio','log10_temperature_ratio','temperature','rotary')}
 meta.update(cell='cascade-p3',source_commit=psource,precision=2048,versions={k:backend[k]for k in ('python_flint','flint')})
 admission=write(owned/'prefix-admission.json',dict(source_commit=psource,source_blobs=[dict(relative_path='codex/certify-llm/ball_qwen.py',sha256=pins['codex/certify-llm/ball_qwen.py'])]))
 cache=prefix/'prefix-cache.json.gz';cache.write_bytes(b'synthetic-cache-bytes-not-a-model')
 def prefix_anchor(m):
  inputs=write(prefix/'inputs.json',m)
  receipt=write(owned/'prefix-receipt.json',dict(source_commit=psource,returncode=0,failure=None,worker_reaped=True,worker_group_session_absent=True,output_files=[row(inputs),row(cache)]))
  return dict(attempt=str(prefix),receipt_sha256=sha(receipt),map_source_sha256=pins['codex/certify-llm/ball_qwen.py'],cache_sha256=sha(cache),source_commit=psource,receipt_path=str(receipt),admission_path=str(admission),admission_sha256=sha(admission))
 pre=prefix_anchor(meta)
 zero={'mid':[0,0],'rad':[0,0]};radius={'mid':[1,-640],'rad':[0,0]}
 definition=dict(schema=b.producer.DEFINITION_SCHEMA,role=b.coverage.ROLE,period=3,dimension=896,centers=[[copy.deepcopy(zero)for _ in range(896)]for _ in range(3)],radii=[copy.deepcopy(radius)for _ in range(3)],map=mp,map_sha256=b.canonical(mp),selection_provenance={'synthetic':True})
 df=write(owned/'definition.json',definition)
 def anchor(d=definition,m=mp,pp=pins,pfx=None):
  write(df,d)
  return b.trust_anchors(str(df),sha(df),m,backend,pp,pre if pfx is None else pfx)
 a=anchor();assert a['definition']['period']==3;count+=1;print('synthetic valid cascade anchor: passed')
 b.authenticate_prefix(a);count+=1;print('synthetic valid named prefix: passed')
 for p in (2,6):
  d=copy.deepcopy(definition);d['period']=p;check_refusal('wrong definition period '+str(p),lambda d=d:anchor(d=d),'Task34 cascade period3')
 wrong=copy.deepcopy(mp);wrong['alpha_ratio']=list(float(.06468).as_integer_ratio())
 check_refusal('Task33 cell map',lambda:anchor(m=wrong),'reviewed map recipe differs')
 wrongpins=dict(pins);wrongpins['codex/certificate-ladder-task34/derivative_shard.py']='0'*64
 check_refusal('changed scientific source bytes',lambda:anchor(pp=wrongpins),'reviewed immutable evaluator source')
 wrongpins=dict(pins);wrongpins['codex/coexistence-task33/derivative_shard.py']=wrongpins.pop('codex/certificate-ladder-task34/derivative_shard.py')
 check_refusal('Task33 source namespace',lambda:anchor(pp=wrongpins),'all externally reviewed evaluator dependencies')
 badmeta=dict(meta,cell='cascade-p6');badpre=prefix_anchor(badmeta);badanchor=anchor(pfx=badpre)
 check_refusal('wrong original prefix named cell',lambda:b.authenticate_prefix(badanchor),'prefix named exact cell differs')
 pre=prefix_anchor(meta);a=anchor();prefix_report=b.authenticate_prefix(a)
 attempt=owned/'synthetic-shard';attempt.mkdir();source='b'*40
 plan=dict(cell='cascade-p3',schema=b.producer.SCHEMA,definition_path=str(df),definition_sha256=a['design_sha256'],map_sha256=a['map_sha256'],backend_sha256=a['backend_sha256'],source_sha256=pins,evaluator_sha256=a['evaluator_sha256'],prefix_attempt=str(prefix),prefix_receipt_sha256=pre['receipt_sha256'],prefix_map_source_sha256=pre['map_source_sha256'],prefix_cache_sha256=pre['cache_sha256'],period=3,phase_index=0,start_column=0,precision=1024,token_chunk=1024,envelope=b.producer.ENVELOPE,prefix_source_commit=psource,prefix_output=str(prefix),prefix_receipt=pre['receipt_path'],prefix_admission_path=pre['admission_path'],prefix_admission_sha256=pre['admission_sha256'])
 driver='codex/certificate-ladder-task34/derivative_shard.py';planrel='codex/certificate-ladder-task34/synthetic-plan.json'
 command=[sys.executable,'-I','-B','-u',str(attempt/'source'/driver),'--output',str(attempt/'run'),'--plan',str(attempt/'source'/planrel)]
 bind={k:a[k]for k in ('design_sha256','map_sha256','evaluator_sha256','backend_sha256','phase_domain_sha256','shooting_domain_sha256','source_bundle','backend','map')}
 def materialize(pl=plan,cmd=command,driverhash=None,refsource=source):
  # Every mutated synthetic case gets freshly self-consistent inventories; thus
  # refusals target the independent cell/source/path binding, not stale hashes.
  shutil.rmtree(attempt);attempt.mkdir()
  for n in pins:
   dst=attempt/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/n,dst)
  pf=write(attempt/'source'/planrel,pl)
  write(attempt/'run/inputs.json',dict(plan=pl,plan_sha256=sha(pf),binding=bind,source_commit=source,driver_sha256=pins[driver]if driverhash is None else driverhash,precision=1024,threads=1,certificate_passed=False))
  shutil.copyfile(df,attempt/'run/candidate-definition.json')
  write(attempt/'run/status.json',{'status':'synthetic-stop-after-metadata'})
  write(attempt/'run/derivative-shard.json',{})
  srcrows=[row(p,str(p.relative_to(attempt/'source')))for p in sorted((attempt/'source').rglob('*'))if p.is_file()]
  write(attempt/'manifest.json',dict(source_commit=source,source=srcrows,command=cmd,threads=1,memory_gib=18,timeout_seconds=3600))
  files=[row(p,str(p.relative_to(attempt)))for p in sorted(attempt.rglob('*'))if p.is_file()]
  rc=write(attempt/'receipt.json',dict(source_commit=source,exit_code=0,timed_out=False,artifacts=files))
  return dict(attempt=str(attempt),receipt_sha256=sha(rc),source_commit=refsource)
 def call(**kw):return b.authenticate_shard(materialize(**kw),a,prefix_report)
 # Positive path reaches the deliberately invalid status, proving all earlier
 # original inventory/source/plan/prefix/driver/command gates passed.
 check_refusal('valid metadata reaches post-metadata sentinel',lambda:call(),'completed status status differs')
 check_refusal('wrong independent campaign source',lambda:call(refsource='c'*40),'assigned campaign source differs')
 for key,value in [('cell','band-p6'),('period',2)]:
  bad=dict(plan);bad[key]=value
  check_refusal('wrong shard '+key,lambda bad=bad:call(pl=bad),'exact cascade-p3 allocation required')
 bad=list(command);bad[8]=str(attempt/'source/codex/coexistence-task33/synthetic-plan.json')
 check_refusal('Task33 captured plan path',lambda:call(cmd=bad),'command-bound captured JSON plan')
 bad=list(command);bad[4]=str(attempt/'source/codex/coexistence-task33/derivative_shard.py')
 check_refusal('Task33 captured driver path',lambda:call(cmd=bad),'captured command is not the isolated reviewed shard driver')
 check_refusal('wrong driver input digest',lambda:call(driverhash='0'*64),'producer input driver_sha256 differs')
 print(json.dumps(dict(controls_passed=count,template_sha256=TEMPLATE,adapter_sha256=sha(adapter),target_evidence_read=False,model_constructed=False,endpoint_or_full_campaign_admitted=False)))
