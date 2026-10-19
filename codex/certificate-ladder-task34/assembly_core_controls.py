"""Synthetic Task34 assembly lifecycle/interface controls; no target evaluation.

Copies reviewed source to a temporary module tree. Test-only DATA/OWNED/RECEIVER,
archive SHA/packet SHA constants identify manufactured /tmp artifacts; original
runtime source has no such override. No model or numerical gate is executed.
"""
import copy,hashlib,importlib.util,io,json,shutil,tarfile,tempfile
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
ADAPTER=Path(__file__).with_name('assembly_core.py');POINT=Path(__file__).with_name('point_bridge.py')
count=0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,sort_keys=True)+'\n');return p
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def reject(label,fn,part):
 global count
 try:fn()
 except (ValueError,KeyError)as e:assert part in str(e),(label,str(e),part)
 else:raise AssertionError(label+' accepted')
 count+=1;print(label+': passed')

with tempfile.TemporaryDirectory(prefix='t34-assembly-controls-')as td:
 root=Path(td);repo=root/'repo';here=repo/'codex/certificate-ladder-task34';here.mkdir(parents=True)
 (repo/'codex/certify-llm').mkdir();(repo/'codex/crosscheck').mkdir()
 for p in (R/'codex/certify-llm').glob('*.py'):shutil.copyfile(p,repo/'codex/certify-llm'/p.name)
 shutil.copyfile(R/'codex/crosscheck/qwen_map.py',repo/'codex/crosscheck/qwen_map.py')
 for name in ('derivative_shard.py','prefix_import.py','derivative_bridge.py','return_operational.py','returned_shard.py'):
  shutil.copyfile(R/'codex/certificate-ladder-task34'/name,here/name)
 shutil.copyfile(POINT,here/'point_bridge.py');shutil.copyfile(ADAPTER,here/'assembly_core.py')
 spec=importlib.util.spec_from_file_location('synthetic_assembly34',here/'assembly_core.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
 b.ctx.prec=1024;b.ctx.threads=1
 original_configs=b.returns.CONFIGS
 serial=0
 def fixture(change=None):
  global serial
  serial+=1;area=root/f'case-{serial}';area.mkdir()
  b.returns.DATA=area/'data';b.returns.OWNED=b.returns.DATA/'codex/certificate-ladder';b.returns.RECEIVER=b.returns.OWNED/'return-cascade-p3-002';b.returns.RECEIVER.mkdir(parents=True)
  back=dict(python='synthetic-python',numpy='synthetic-numpy',torch='2.14.1+synthetic',python_flint='synthetic-python-flint',flint='synthetic-flint')
  versions={'python-flint':back['python_flint'],'numpy':back['numpy'],'safetensors':'synthetic-safe','threadpoolctl':'synthetic-thread','tokenizers':'synthetic-tok','torch_distribution':'2.14.1'}
  assignments=[dict(attempt=f'cascade-p3-derivative-p{phase}-c{col:03d}-001',plan=f'codex/certificate-ladder-task34/cascade-p3-derivative-p{phase}-c{col:03d}-001-plan.json',phase=phase,start_column=col)for phase in range(3)for col in range(0,896,64)]
  packet=dict(plans=assignments,expected_backend=back,required_import_versions=versions,launcher_sha256='1'*64,collector_sha256='2'*64,definition=dict(path='synthetic-definition',bytes=123,sha256='3'*64),upstream_files=[])
  packet_file=write(repo/b.returns.PACKET_REL,packet);b.returns.PACKET_SHA=sha(packet_file)
  b.returns.CONFIGS=tuple(dict(c,sha256=hashlib.sha256(('synthetic-outer-'+c['pod']).encode()).hexdigest())for c in original_configs)
  groups=[assignments[5*i:5*i+5]for i in range(9)];cfg=b.returns.CONFIGS[7];group=groups[7];names=[r['attempt']for r in group];name=names[0]
  folder=b.returns.RECEIVER/cfg['pod']/'explore/llm-soft-loop/pod64/certpack'/cfg['pod'];folder.mkdir(parents=True)
  base=b.returns.DATA/'runpod/soft64'/cfg['stamp'];base.mkdir(parents=True)
  outer=base/'results.tgz';outer.write_bytes(('synthetic-outer-'+cfg['pod']).encode())
  pod_id='successful-h';failed_id='failed-first-h'
  result=dict(commit=cfg['outer_commit'],jobset='certpack',runner_returncode=0,hash_verified=True,local_sha256=cfg['sha256'],remote_sha256=cfg['sha256'],out=str(base),pod_id=pod_id,upload={'jobset_args':' '.join([cfg['pod'],b.returns.PACKET_REL,b.returns.SCI,b.returns.PACKET_SHA,','.join(names),'92'])})
  if change=='result_source':result['upload']['jobset_args']=result['upload']['jobset_args'].replace(b.returns.SCI,'e'*40)
  rp=write(base/'result.json',result)
  first=dict(pod_id=failed_id,terminated=True,in_inventory=False,direct_lookup=None);last=dict(pod_id=pod_id,terminated=True,in_inventory=False,direct_lookup=None)
  if change=='missing_termination':last['pod_id']='foreign-pod'
  log=b.returns.DATA/'runpod/T34C-P3-H-orchestrator.log';log.write_text(json.dumps(first)+'\n'+json.dumps(last)+'\n')
  line=log.read_bytes().splitlines()[1]
  term=dict(pod_id=pod_id,terminated=True,in_inventory=False,direct_lookup=None,line_number=2,line_sha256=hashlib.sha256(line).hexdigest(),retained_termination_record_count=2)
  n=len(names);vals={k:back[bk]for k,bk in [('sys_version','python'),('numpy','numpy'),('torch','torch'),('python_flint','python_flint'),('flint','flint')]}
  actualversions=dict(versions,torch_distribution=back['torch'])
  vals.update(third_party_imports={'flint':versions['python-flint'],'numpy':versions['numpy'],'safetensors':versions['safetensors'],'threadpoolctl':versions['threadpoolctl'],'tokenizers':versions['tokenizers'],'torch':back['torch']},data_root_real=dict(symlinks_on_path=[],realpath=str(b.returns.DATA)),hashes=[],packet_sha256=b.returns.PACKET_SHA,plans={nm:dict(plan_sha256_matches=True,source_pins=12,mismatched_pins=[],command_problems=[],attempt_absent=True)for nm in names},packet_source_pins=dict(pins=12,mismatched=[]),launcher_sha256=dict(path='codex/certificate-ladder-task34/launch.py',sha256=packet['launcher_sha256']),collector_sha256=dict(path='codex/certificate-ladder-task34/returned_shard.py',sha256=packet['collector_sha256']),definition=dict(**packet['definition'],exists=True,is_symlink=False),required_import_versions=actualversions,git_head=dict(head=b.returns.SCI,origin_main_contains=True),ram_available_gib=100,disk_free_gib=100,effective_capacity=dict(admitted_workers=n,named_attempts=n,trials=[dict(workers=nr,admission=dict(admit=True,workers=nr,per_worker_gib=18.0,shared_gib=4.0,threads_per_worker=1,required_memory_bytes=(18*nr+4)*1024**3,required_cpus=nr,reasons=[]))for nr in range(1,n+1)],capacity=dict(unresolved=[],effective_memory_bytes=(18*n+4)*1024**3,effective_cpus=n)))
  pre=dict(step='preflight',ok=True,commit=b.returns.SCI,attempts=names,admitted_workers=n,checks={k:{'value':v,'pass':True}for k,v in vals.items()})
  prepath=write(folder/'preflight.json',pre);prep=write(folder/'prepare.json',dict(step='prepare',ok=True,hash_problems=[],copied_upstream=0))
  pa=dict(checks_passed=19,admitted_workers=n,effective_memory_bytes=(18*n+4)*1024**3,effective_cpus=n,scientific_source=b.returns.SCI)
  command='/root/certvenv/bin/python -I -B -u codex/certificate-ladder-task34/launch.py --attempt '+name+' --driver derivative_shard --plan '+name+'-plan.json --threads 1 --seconds 3600'
  run=dict(step='run',success=True,closure_verified=True,exit_code=0,watchdog_stop=None,killed_by_watchdog=None,alive_after_exit=[],alive_after_cleanup=[],command=command,cwd='/root/cert-src',deadline_seconds=3900.0,rss_limit_kib=18*1024**2,owned_identities=[{'synthetic':True}])
  if change=='watchdog_false':run['killed_by_watchdog']=False
  if change=='old_launcher':run['command']=command.replace('certificate-ladder-task34','coexistence-task33')
  runpath=write(folder/name/'run.json',run)
  receipt=dict(source_commit=b.returns.SCI,exit_code=0,timed_out=False)
  if change=='native_source':receipt['source_commit']='c'*40
  rb=(json.dumps(receipt,indent=2)+'\n').encode();rh=hashlib.sha256(rb).hexdigest()
  path=b.returns.OWNED/name;path.mkdir();(path/'receipt.json').write_bytes(rb)
  native=folder/(name+'.tar');prefix='codex/certificate-ladder/'+name
  with tarfile.open(native,'w')as t:
   m=tarfile.TarInfo(prefix);m.type=tarfile.DIRTYPE;t.addfile(m)
   m=tarfile.TarInfo(prefix+'/receipt.json');body=json.dumps(receipt).encode()if change=='reserialized_receipt'else rb;m.size=len(body);t.addfile(m,io.BytesIO(body))
  pack=dict(step='pack',attempt=name,attempt_present=True,archive=name+'.tar',archive_bytes=native.stat().st_size,archive_sha256=sha(native))
  if change=='wrong_pack_assignment':pack['attempt']='foreign-attempt'
  packpath=write(folder/('pack-'+name+'.json'),pack)
  closure=dict(success=True,closure_verified=True,exit_code=0,watchdog_stop=None,killed_by_watchdog=None,alive_after_exit=[],alive_after_cleanup=[])
  row=dict(attempt=name,period=3,phase_index=group[0]['phase'],start_column=group[0]['start_column'],source_commit=b.returns.SCI,original_archive_receipt_sha256=rh,operational=dict(source_commit=b.returns.SCI,receipt_sha256=rh),whole_archive_sha256=cfg['sha256'],archive_sha256=sha(native),pack=pin(packpath),pack_path=str(packpath),pack_sha256=sha(packpath),outer_closure=pin(runpath),outer_closure_path=str(runpath),outer_closure_sha256=sha(runpath),closure_admission=closure,pod_id=pod_id,termination_record=term,termination_record_path=str(log),termination_record_sha256=sha(log),termination_record_line=2,result=pin(rp),result_path=str(rp),result_sha256=sha(rp),orchestrator_log=pin(log))
  if change=='wrong_flat_pin':row['pack_sha256']='0'*64
  if change=='first_termination':row['termination_record_line']=1
  reportrows=[]
  for a in assignments:
   r=dict(attempt=a['attempt'],period=3,phase_index=a['phase'],start_column=a['start_column'],source_commit=b.returns.SCI,original_archive_receipt_sha256='a'*64,operational=dict(source_commit=b.returns.SCI,receipt_sha256='a'*64));reportrows.append(row if a['attempt']==name else r)
  pods=[dict(pod=c['pod'],attempts=[r['attempt']for r in g])for c,g in zip(b.returns.CONFIGS,groups)]
  pods[7].update(pod_id=pod_id,outer_source=cfg['outer_commit'],scientific_source=b.returns.SCI,archive=pin(outer),result=pin(rp),orchestrator_log=pin(log),termination=term,preflight=pin(prepath),preflight_admission=pa,prepare=pin(prep))
  report=dict(schema='codex-task34-original-return-admission-v1',status='all_42_original_returns_operationally_admitted',certificate_passed=False,packet_sha256=b.returns.PACKET_SHA,scientific_source=b.returns.SCI,unique_phase_column_assignments=2688,complete_signed_endpoint_coverage=False,attempts=reportrows,pods=pods)
  if change=='incomplete_receiver':report['attempts'].pop()
  if change=='receiver_source':report['scientific_source']='f'*40
  op=write(b.returns.RECEIVER/'operational.json',report)
  plan=dict(period=3,cell='cascade-p3',operational_return=pin(op),campaign_source=b.returns.SCI)
  ref=dict(attempt=str(path),receipt_sha256=rh,source_commit=b.returns.SCI,phase_index=row['phase_index'],start_column=row['start_column'])
  return plan,ref
 def closure(change=None):
  plan,ref=fixture(change);context=b.authenticate_returns(plan,repo);return b.authenticate_closure(ref,context)
 out=closure();assert out['original_native_receipt_bytes_joined']and out['producer_termination_reported'];count+=1;print('complete synthetic original lifecycle/tar receipt join: passed')
 for kind,fragment in [('watchdog_false','original nulls'),('old_launcher','exact command'),('missing_termination','exactly one matching'),('result_source','exact scientific assignment'),('native_source','native receipt not successful'),('reserialized_receipt','original tar receipt bytes differs'),('wrong_pack_assignment','pack assignment'),('wrong_flat_pin','receiver flat hash pack differs'),('first_termination','receiver termination line differs'),('incomplete_receiver','ordered receiver attempt allocation'),('receiver_source','closed operational receiver scientific_source differs')]:reject(kind,lambda kind=kind:closure(kind),fragment)
 plan,ref=fixture();context=b.authenticate_returns(plan,repo);ref['phase_index']=0;reject('wrong assembly phase assignment',lambda:b.authenticate_closure(ref,context),'assembly original assignment phase_index differs')
 plan,ref=fixture();context=b.authenticate_returns(plan,repo);ref['source_commit']='d'*40;reject('wrong assembly scientific source',lambda:b.authenticate_closure(ref,context),'assembly original assignment source_commit differs')
 plan,ref=fixture();plan['operational_return']['path']=str(Path(plan['operational_return']['path']).parent.parent/'return-cascade-p3-001/operational.json');reject('obsolete receiver001 authority',lambda:b.authenticate_returns(plan,repo),'exact Task34 receiver002 output')
 for period,cell in [(2,'cascade-p3'),(3,'band-p6')]:reject('wrong assembly scope '+str((period,cell)),lambda period=period,cell=cell:b.assemble(dict(period=period,cell=cell),root/'unused',repo),'exact cascade-p3 period3 allocation')
 assert {'codex/certificate-ladder-task34/return_operational.py','codex/certificate-ladder-task34/returned_shard.py'}<=b.REQUIRED_SOURCES
 # Full source/review interface controls bind the newly required receiver authority.
 plan,ref=fixture();reviewdir=root/'reviews';reviewdir.mkdir()
 reviews={}
 for kind in ('point_evaluator','derivative_evaluator','map_regularity','gate_mathematics'):
  reviews[kind]=pin(write(reviewdir/(kind+'.json'),{'synthetic':kind}))
 sourceplan=dict(plan,map={'synthetic':True},derivative_source_sha256={},reviews=reviews,point_plan=pin(write(reviewdir/'point-plan.json',{})),assembly_source_sha256={n:sha(repo/n)for n in b.REQUIRED_SOURCES})
 hist=pin(write(reviewdir/'historical.json',{}));binding={k:sourceplan[k]for k in ('map','derivative_source_sha256','reviews','cell','campaign_source','operational_return','assembly_source_sha256','point_plan')};binding.update(historical_evaluator_review=hist,status='source_reviewed_conditional_enclosures')
 bp=write(reviewdir/'binding.json',binding);sourceplan['review_binding']=pin(bp);b.authenticate_sources(sourceplan,repo);count+=1;print('complete assembly/source/receiver review binding: passed')
 bad=copy.deepcopy(binding);bad['operational_return']['sha256']='0'*64;write(bp,bad);sourceplan['review_binding']=pin(bp);reject('wrong reviewed receiver output',lambda:b.authenticate_sources(sourceplan,repo),'review binding operational_return differs')
 print(json.dumps(dict(controls_passed=count,assembly_sha256=sha(ADAPTER),source_template_sha256='42e2a86b379058bcf1faa7721fa8c38f3ce249f7efee1881b9016ae12b4ab842',return_operational_sha256=sha(R/'codex/certificate-ladder-task34/return_operational.py'),target_evidence_evaluated=False,model_calls=False,mathematical_gates_run=False)))
