"""Synthetic extraction/assignment/closure controls, never executes main/run."""
import copy, io, importlib.util, json, tarfile, tempfile
from pathlib import Path
p=Path(__file__).with_name('return_operational.py');spec=importlib.util.spec_from_file_location('prepared_task34_returns',p);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
count=0

def refused(label,fn,text):
 global count
 try:fn()
 except (ValueError,FileExistsError)as e:assert text in str(e),(label,str(e))
 else:raise AssertionError(label+' accepted')
 count+=1;print(label+': passed')
def passed(label,fn):
 global count
 fn();count+=1;print(label+': passed')
def archive(path,members):
 with tarfile.open(path,'w')as t:
  for name,kind,body in members:
   m=tarfile.TarInfo(name);m.type=kind
   if kind==tarfile.REGTYPE:m.size=len(body)
   if kind in (tarfile.SYMTYPE,tarfile.LNKTYPE):m.linkname='elsewhere'
   t.addfile(m,io.BytesIO(body)if kind==tarfile.REGTYPE else None)
 return path

with tempfile.TemporaryDirectory(prefix='t34-op-controls-')as td:
 root=Path(td)
 def scan(members,prefix=None):
  path=archive(root/'test.tar',members)
  with tarfile.open(path)as t:return w.scan_tar(t,prefix=prefix)
 refused('absolute path',lambda:scan([('/x',tarfile.REGTYPE,b'x')]),'relative nontraversing')
 refused('parent traversal',lambda:scan([('a/../x',tarfile.REGTYPE,b'x')]),'relative nontraversing')
 refused('canonical duplicate',lambda:scan([('a/x',tarfile.REGTYPE,b'x'),('./a/x',tarfile.REGTYPE,b'y')]),'duplicate canonical')
 refused('symlink',lambda:scan([('link',tarfile.SYMTYPE,b'')]),'only regular')
 refused('hardlink',lambda:scan([('link',tarfile.LNKTYPE,b'')]),'only regular')
 refused('file parent conflict',lambda:scan([('a',tarfile.REGTYPE,b'x'),('a/b',tarfile.REGTYPE,b'y')]),'file used as tar parent')
 refused('foreign native prefix',lambda:scan([('codex/coexistence/x/a',tarfile.REGTYPE,b'x')],w.PurePosixPath('codex/certificate-ladder/x')),'outside assigned attempt')
 path=archive(root/'good.tar',[('native',tarfile.DIRTYPE,b''),('native/sub',tarfile.DIRTYPE,b''),('native/sub/original.bin',tarfile.REGTYPE,b'original\x00bytes')])
 with tarfile.open(path)as t:
  rows,size=w.scan_tar(t,prefix=w.PurePosixPath('native'));assert size==14
  out=root/'out';w.fresh_extract(t,rows,out,prefix=w.PurePosixPath('native'));assert(out/'sub/original.bin').read_bytes()==b'original\x00bytes';count+=1;print('safe byte-preserving native extraction: passed')
  refused('existing extraction destination',lambda:w.fresh_extract(t,rows,out,prefix=w.PurePosixPath('native')),'destination already exists')
 name='cascade-p3-derivative-p0-c000-001';cfg=dict(pod='t34c-p3-a',stamp='synthetic',sha256='d'*64,outer_commit=w.OUTER_OLD)
 command='/root/certvenv/bin/python -I -B -u codex/certificate-ladder-task34/launch.py --attempt '+name+' --driver derivative_shard --plan '+name+'-plan.json --threads 1 --seconds 3600'
 run=dict(step='run',success=True,closure_verified=True,exit_code=0,watchdog_stop=None,killed_by_watchdog=None,alive_after_exit=[],alive_after_cleanup=[],command=command,cwd='/root/cert-src',deadline_seconds=3900.0,rss_limit_kib=18*1024**2,owned_identities=[{'synthetic':True}])
 passed('original null watchdog closure',lambda:w.check_run(run,name))
 for key,val,part in [('watchdog_stop',False,'original nulls'),('killed_by_watchdog',False,'original nulls'),('exit_code',False,'exact zero exit'),('alive_after_cleanup',[1],'not closed'),('command',command.replace('--threads 1','--threads 2'),'exact command')]:
  bad=dict(run);bad[key]=val;refused('closure refuses '+key,lambda bad=bad:w.check_run(bad,name),part)
 bad=dict(run);bad.pop('watchdog_stop');refused('missing watchdog field',lambda:w.check_run(bad,name),'original nulls')
 passed('matched second pod termination retains first',lambda:w.check_termination(b'{"pod_id":"first","terminated":true,"in_inventory":false,"direct_lookup":null}\n{"pod_id":"second","terminated":true,"in_inventory":false,"direct_lookup":null}\n','second'))
 refused('wrong termination pod',lambda:w.check_termination(b'{"pod_id":"first","terminated":true,"in_inventory":false,"direct_lookup":null}\n','other'),'exactly one matching')
 term=b'{"pod_id":"same","terminated":true,"in_inventory":false,"direct_lookup":null}\n';refused('duplicate matched termination',lambda:w.check_termination(term+term,'same'),'exactly one matching')
 refused('termination still in inventory',lambda:w.check_termination(b'{"pod_id":"same","terminated":true,"in_inventory":true,"direct_lookup":null}\n','same'),'termination failed')
 names=[name];base=root/'synthetic'
 result=dict(commit=w.OUTER_OLD,jobset='certpack',runner_returncode=0,hash_verified=True,local_sha256='d'*64,remote_sha256='d'*64,out=str(base),pod_id='test',upload={'jobset_args':' '.join([cfg['pod'],w.PACKET_REL,w.SCI,w.PACKET_SHA,','.join(names),'92'])})
 passed('separate outer/scientific source result binding',lambda:w.check_result(result,cfg,names,base))
 bad=copy.deepcopy(result);bad['upload']['jobset_args']=bad['upload']['jobset_args'].replace(w.SCI,'e'*40);refused('wrong result scientific source',lambda:w.check_result(bad,cfg,names,base),'exact scientific assignment')
 bad=dict(result,commit=w.OUTER_NEW);refused('wrong assigned outer source',lambda:w.check_result(bad,cfg,names,base),'outer source/jobset')
 bad=dict(result,out=str(root/'wrong'));refused('wrong returned output path',lambda:w.check_result(bad,cfg,names,base),'output assignment')
 packet={'plans':[dict(attempt=f'cascade-p3-derivative-p{phase}-c{col:03d}-001',plan=f'codex/certificate-ladder-task34/cascade-p3-derivative-p{phase}-c{col:03d}-001-plan.json',phase=phase,start_column=col)for phase in range(3)for col in range(0,896,64)]}
 passed('complete independent assignment set',lambda:w.assignments(packet))
 bad=copy.deepcopy(packet);bad['plans'][-1]=copy.deepcopy(bad['plans'][0]);refused('duplicate phase-column assignment',lambda:w.assignments(bad),'ordered phase/block')
 bad=copy.deepcopy(packet);bad['plans'][0]['plan']=bad['plans'][0]['plan'].replace('certificate-ladder-task34','coexistence-task33');refused('Task33 plan namespace',lambda:w.assignments(bad),'assigned name/plan')
 native=archive(root/'native.tar',[('file',tarfile.REGTYPE,b'bytes')]);pack=dict(step='pack',attempt=name,attempt_present=True,archive=name+'.tar',archive_bytes=native.stat().st_size,archive_sha256=w.sha(native))
 passed('native pack hash and assignment',lambda:w.check_pack(pack,name,native))
 bad=dict(pack,archive_bytes=1);refused('wrong native archive length',lambda:w.check_pack(bad,name,native),'size/name')
 bad=dict(pack,archive_sha256='0'*64);refused('wrong native archive digest',lambda:w.check_pack(bad,name,native),'archive bytes differ')

 # Manufacture the complete original preflight schema without reading target metadata.
 back=dict(python='synthetic-python',numpy='synthetic-numpy',torch='synthetic-torch',python_flint='synthetic-python-flint',flint='synthetic-flint')
 versions={'python-flint':back['python_flint'],'numpy':back['numpy'],'safetensors':'synthetic-safe','threadpoolctl':'synthetic-thread','tokenizers':'synthetic-tok','torch_distribution':'synthetic-dist'}
 authority=dict(expected_backend=back,required_import_versions=versions,launcher_sha256='1'*64,collector_sha256='2'*64,definition=dict(path='synthetic-definition',bytes=123,sha256='3'*64))
 names=[name,'cascade-p3-derivative-p0-c064-001'];n=len(names)
 vals={k:back[bk]for k,bk in [('sys_version','python'),('numpy','numpy'),('torch','torch'),('python_flint','python_flint'),('flint','flint')]}
 vals.update(third_party_imports={'flint':versions['python-flint'],'numpy':versions['numpy'],'safetensors':versions['safetensors'],'threadpoolctl':versions['threadpoolctl'],'tokenizers':versions['tokenizers'],'torch':back['torch']},data_root_real=dict(symlinks_on_path=[],realpath=str(w.DATA)),hashes=[],packet_sha256=w.PACKET_SHA,plans={nm:dict(plan_sha256_matches=True,source_pins=12,mismatched_pins=[],command_problems=[],attempt_absent=True)for nm in names},packet_source_pins=dict(pins=12,mismatched=[]),launcher_sha256=dict(path='codex/certificate-ladder-task34/launch.py',sha256=authority['launcher_sha256']),collector_sha256=dict(path='codex/certificate-ladder-task34/returned_shard.py',sha256=authority['collector_sha256']),definition=dict(**authority['definition'],exists=True,is_symlink=False),required_import_versions=versions,git_head=dict(head=w.SCI,origin_main_contains=True),ram_available_gib=100,disk_free_gib=100,effective_capacity=dict(admitted_workers=n,named_attempts=n,trials=[dict(workers=nr,admission=dict(admit=True,workers=nr,per_worker_gib=18.0,shared_gib=4.0,threads_per_worker=1,required_memory_bytes=(18*nr+4)*1024**3,required_cpus=nr,reasons=[]))for nr in range(1,n+1)],capacity=dict(unresolved=[],effective_memory_bytes=(18*n+4)*1024**3,effective_cpus=n)))
 pre=dict(step='preflight',ok=True,commit=w.SCI,attempts=names,admitted_workers=n,checks={k:dict(value=v,pass_=True)for k,v in vals.items()})
 for v in pre['checks'].values():v['pass']=v.pop('pass_')
 passed('all19 independent preflight and exact capacity',lambda:w.check_preflight(pre,names,authority))
 bad=copy.deepcopy(pre);bad['checks']['effective_capacity']['value']['capacity']['effective_memory_bytes']-=1
 refused('effective memory one byte short',lambda:w.check_preflight(bad,names,authority),'effective CPU/RAM')
 bad=copy.deepcopy(pre);bad['checks']['effective_capacity']['value']['capacity']['effective_cpus']=n-1
 refused('effective CPU short',lambda:w.check_preflight(bad,names,authority),'effective CPU/RAM')
 bad=copy.deepcopy(pre);bad['checks']['effective_capacity']['value']['capacity']['unresolved']=['hidden-cgroup']
 refused('unresolved effective capacity',lambda:w.check_preflight(bad,names,authority),'effective CPU/RAM')
 bad=copy.deepcopy(pre);bad['checks']['hashes']['pass']=False
 refused('failed original preflight check',lambda:w.check_preflight(bad,names,authority),'original preflight check failed')
 bad=copy.deepcopy(pre);bad['checks'].pop('definition')
 refused('missing original preflight check',lambda:w.check_preflight(bad,names,authority),'complete original 19-check')
 bad=copy.deepcopy(pre);bad['attempts']=list(reversed(names))
 refused('wrong original preflight assignment',lambda:w.check_preflight(bad,names,authority),'exact assignment')
 bad=copy.deepcopy(pre);bad['commit']='f'*40
 refused('wrong original preflight source',lambda:w.check_preflight(bad,names,authority),'source/status')
 bad=copy.deepcopy(pre);bad['checks']['torch']['value']='other-torch'
 refused('wrong independent backend preflight',lambda:w.check_preflight(bad,names,authority),'independent backend')
 print(json.dumps(dict(controls_passed=count,wrapper_sha256=w.sha(p),target_archives_read=False,target_extraction_performed=False,model_calls=False)))
