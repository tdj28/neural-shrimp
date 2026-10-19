"""Bounded original Task34 cascade-P3 return admission, without model evaluation.

Prepared for root's source/card/private-checkpoint/capacity review. Main performs
real extraction; importing this module does not. All input archives stay intact.
Only compact, selected nonmonetary receiver metadata is written to its report.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, resource, shlex, shutil, tarfile, time, traceback
from pathlib import Path, PurePosixPath

R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
DATA=Path('/data2/neural-shrimp-data');OWNED=DATA/'codex/certificate-ladder'
RECEIVER=OWNED/'return-cascade-p3-001'
SCI='42508b68a733acc4216b40d7b60215de36c5d543'
PACKET_REL='codex/certificate-ladder-task34/cascade-p3-dispatch-001.json'
PACKET_SHA='3167eeeec4329eeb1ed8f01ab9738a8173decaabaa021831ab1153a6dcd3c1bc'
COLLECTOR_SHA='acd8d7d796c19276c9a94bc2df0cde67c900e74dd7ce1e04d4d6f3b0606021ac'
OUTER_OLD='5b875468a3a0aa6a9cfc7b877d06dc4c531a6275'
OUTER_NEW='b23a0f20405f6a63a52523bf71e353a403414f37'
RESERVATION=12*1024**3
CONFIGS=tuple(dict(pod='t34c-p3-'+letter,stamp=stamp,sha256=h,outer_commit=OUTER_NEW if letter in 'hi' else OUTER_OLD)for letter,stamp,h in [
 ('a','20261018T230927Z','128fa5324e6b7db635b82c1cbcae79a48d4df6faed19544e8ca43aa915b63f53'),
 ('b','20261018T231244Z','6d18bffb19da1de10bccce48efd3d98cc0419abb64c9b18415282c368ef38894'),
 ('c','20261018T231500Z','edf78012e5c3117976d6a80671425b64a22078b1764117bea1266e494f1817c9'),
 ('d','20261018T231900Z','80888148b02571ee760f8328ff3f14958a04cd40cd27bc25ebcb7ffd8ba7e858'),
 ('e','20261018T232110Z','afcccaf169aaa49eef9c8b0f46071c26d862f188fc166025513911ce5443608f'),
 ('f','20261018T232322Z','55444e2b7b7b899cddd502b50b591155fff9d724f3d0d4494155913586013bde'),
 ('g','20261018T232559Z','ab5994ea2f551757af27639f89971e58230647faecb8d7870723f2b53ecb571b'),
 ('h','20261018T234622Z','d10cf34c81ba464efa54d0b5931b0132fc86e999d6b29d8e6fabdbc3123047a1'),
 ('i','20261018T234834Z','ecb140b5e0259e84d218d306dfec20f9e475ab7423daef04eddcd871943d3be0')])

def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def unique(pairs):
 d={}
 for k,v in pairs:
  require(k not in d,'duplicate JSON key');d[k]=v
 return d
def parse(raw):return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in()).throw(ValueError('nonfinite JSON constant')))
def read(path,maximum=16*1024**2):
 p=Path(path);require(p.is_file()and not p.is_symlink()and p.resolve()==p,'original regular real path required')
 require(p.stat().st_size<=maximum,'bounded JSON size required');return parse(p.read_bytes())
def pin(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def atomic(path,value):
 temp=path.with_name(path.name+'.tmp')
 with temp.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 temp.replace(path)
def canonical_member(name):
 require(isinstance(name,str)and name and '\x00'not in name and '\\'not in name,'literal POSIX member required')
 p=PurePosixPath(name);require(not p.is_absolute()and '..'not in p.parts and p.parts,'relative nontraversing member required')
 return p

def scan_tar(t,*,prefix=None):
 """Validate canonical paths/types before writing anything from this tar."""
 rows=[];seen={};total=0
 for m in t.getmembers():
  p=canonical_member(m.name)
  require(str(p)not in seen,'duplicate canonical tar member')
  require(m.isdir()or m.isfile(),'only regular files/directories allowed')
  require(type(m.size)is int and m.size>=0,'nonnegative tar member size')
  if prefix is not None:require(p.is_relative_to(prefix),'native member outside assigned attempt')
  seen[str(p)]=m;rows.append((m,p));total+=m.size if m.isfile()else 0
  require(total<=RESERVATION,'archive exceeds extraction reservation')
 for m,p in rows:
  for parent in p.parents:
   if str(parent)in seen:require(seen[str(parent)].isdir(),'file used as tar parent')
 return rows,total

def fresh_extract(t,rows,dest,*,prefix=None):
 """Exclusive destination, no link support, byte-preserving bounded copy."""
 require(not dest.exists()and not dest.is_symlink(),'extraction destination already exists')
 require(dest.parent.is_dir()and dest.parent.resolve()==dest.parent,'real existing extraction parent required')
 dest.mkdir()
 for m,p in rows:
  rel=p if prefix is None else p.relative_to(prefix)
  if not rel.parts:
   require(m.isdir(),'native root must be directory');continue
  target=dest.joinpath(*rel.parts)
  require(target.is_relative_to(dest),'destination escaped')
  target.parent.mkdir(parents=True,exist_ok=True)
  require(target.parent.resolve()==target.parent,'link in extraction parent')
  if m.isdir():
   require(not target.is_file()and not target.is_symlink(),'directory conflicts with file');target.mkdir(exist_ok=True)
  else:
   with t.extractfile(m)as src,target.open('xb')as out:
    remaining=m.size
    while remaining:
     block=src.read(min(1024**2,remaining));require(bool(block),'truncated tar member');out.write(block);remaining-=len(block)
   require(target.stat().st_size==m.size,'extracted byte count differs')
 return dest

def assignments(packet):
 rows=packet['plans'];require(type(rows)is list and len(rows)==42,'exactly 42 committed plans required')
 expected=[(p,c)for p in range(3)for c in range(0,896,64)]
 require([(r['phase'],r['start_column'])for r in rows]==expected,'ordered phase/block allocation differs')
 require(len({r['attempt']for r in rows})==42,'duplicate assigned attempt')
 for r in rows:
  name=f"cascade-p3-derivative-p{r['phase']}-c{r['start_column']:03d}-001"
  require(r['attempt']==name and r['plan']==f'codex/certificate-ladder-task34/{name}-plan.json','assigned name/plan differs')
 return rows

def check_result(result,cfg,names,base):
 require(result.get('commit')==cfg['outer_commit']and result.get('jobset')=='certpack','outer source/jobset differs')
 require(type(result.get('runner_returncode'))is int and result['runner_returncode']==0 and result.get('hash_verified')is True,'outer runner/hash gate failed')
 require(result.get('local_sha256')==cfg['sha256']==result.get('remote_sha256'),'outer returned hash pins differ')
 require(result.get('out')==str(base),'outer output assignment differs')
 require(type(result.get('pod_id'))is str and result['pod_id'],'original pod identity required')
 expected=[cfg['pod'],PACKET_REL,SCI,PACKET_SHA,','.join(names),'92']
 require(shlex.split(result['upload']['jobset_args'])==expected,'outer exact scientific assignment differs')
 return result['pod_id']

def check_termination(raw,pod_id):
 matches=[];all_records=[]
 for number,line in enumerate(raw.splitlines(),1):
  if not line.lstrip().startswith(b'{'):continue
  obj=parse(line)
  if 'terminated'in obj:
   all_records.append((number,obj))
   if obj.get('pod_id')==pod_id:matches.append((number,line,obj))
 require(len(matches)==1,'exactly one matching original pod termination required')
 number,line,obj=matches[0]
 require(obj.get('terminated')is True and obj.get('in_inventory')is False and 'direct_lookup'in obj and obj['direct_lookup']is None,'final original termination failed')
 return dict(pod_id=pod_id,terminated=True,in_inventory=False,direct_lookup=None,line_number=number,line_sha256=hashlib.sha256(line).hexdigest(),retained_termination_record_count=len(all_records))

def check_run(run,name):
 require(run.get('step')=='run'and run.get('success')is True and run.get('closure_verified')is True,'outer successful closure required')
 require(type(run.get('exit_code'))is int and run['exit_code']==0,'outer exact zero exit required')
 require('watchdog_stop'in run and run['watchdog_stop']is None and 'killed_by_watchdog'in run and run['killed_by_watchdog']is None,'outer watchdog fields must be original nulls')
 require(run.get('alive_after_exit')==[]and run.get('alive_after_cleanup')==[],'outer owned processes not closed')
 command=['/root/certvenv/bin/python','-I','-B','-u','codex/certificate-ladder-task34/launch.py','--attempt',name,'--driver','derivative_shard','--plan',name+'-plan.json','--threads','1','--seconds','3600']
 require(shlex.split(run['command'])==command and run.get('cwd')=='/root/cert-src','outer exact command/working directory differs')
 require(run.get('deadline_seconds')==3900 and run.get('rss_limit_kib')==18*1024**2,'outer resource envelope differs')
 require(type(run.get('owned_identities'))is list and bool(run['owned_identities']),'original owned identities missing')
 return dict(success=True,closure_verified=True,exit_code=0,watchdog_stop=None,killed_by_watchdog=None,alive_after_exit=[],alive_after_cleanup=[])

def check_preflight(pre,names,packet):
 require(pre.get('step')=='preflight'and pre.get('ok')is True and pre.get('commit')==SCI,'original preflight source/status differs')
 require(pre.get('attempts')==names and type(pre.get('admitted_workers'))is int and pre['admitted_workers']==len(names),'preflight exact assignment differs')
 checks=pre['checks'];expected={'sys_version','numpy','torch','python_flint','flint','third_party_imports','data_root_real','hashes','packet_sha256','plans','packet_source_pins','launcher_sha256','collector_sha256','definition','required_import_versions','git_head','ram_available_gib','disk_free_gib','effective_capacity'}
 require(type(checks)is dict and set(checks)==expected,'complete original 19-check preflight required')
 require(all(type(v)is dict and v.get('pass')is True for v in checks.values()),'original preflight check failed')
 val=lambda k:checks[k]['value']
 for k,bk in [('sys_version','python'),('numpy','numpy'),('torch','torch'),('python_flint','python_flint'),('flint','flint')]:require(val(k)==packet['expected_backend'][bk],'independent backend preflight differs')
 require(val('packet_sha256')==PACKET_SHA and val('hashes')==[],'packet/upstream preflight differs')
 require(val('data_root_real')==dict(symlinks_on_path=[],realpath=str(DATA)),'original data root differs')
 require(val('required_import_versions')==packet['required_import_versions'],'required import versions differ')
 versions=packet['required_import_versions']
 require(val('third_party_imports')=={'flint':versions['python-flint'],'numpy':versions['numpy'],'safetensors':versions['safetensors'],'threadpoolctl':versions['threadpoolctl'],'tokenizers':versions['tokenizers'],'torch':packet['expected_backend']['torch']},'actual third-party import versions differ')
 require(val('git_head')==dict(head=SCI,origin_main_contains=True),'scientific Git preflight differs')
 require(val('packet_source_pins')==dict(pins=12,mismatched=[]),'all source pins required')
 plans=val('plans');require(set(plans)==set(names),'preflight plan set differs')
 for row in plans.values():require(row==dict(plan_sha256_matches=True,source_pins=12,mismatched_pins=[],command_problems=[],attempt_absent=True),'preflight exact plan gate failed')
 for key,rel in [('launcher_sha256','launch.py'),('collector_sha256','returned_shard.py')]:require(val(key)==dict(path='codex/certificate-ladder-task34/'+rel,sha256=packet[key]),'preflight launcher/collector differs')
 definition=packet['definition'];require(val('definition')==dict(path=definition['path'],exists=True,is_symlink=False,bytes=definition['bytes'],sha256=definition['sha256']),'preflight definition differs')
 effective=val('effective_capacity');n=len(names)
 require(effective.get('admitted_workers')==n and effective.get('named_attempts')==n,'effective capacity assignment differs')
 require([r['workers']for r in effective['trials']]==list(range(1,n+1)),'effective capacity trial allocation differs')
 for trial in effective['trials']:
  w=trial['workers'];a=trial['admission']
  require(a==dict(admit=True,workers=w,per_worker_gib=18.0,shared_gib=4.0,threads_per_worker=1,required_memory_bytes=(18*w+4)*1024**3,required_cpus=w,reasons=[]),'effective capacity admission differs')
 capacity=effective['capacity']
 require(capacity.get('unresolved')==[]and type(capacity.get('effective_memory_bytes'))is int and capacity['effective_memory_bytes']>=(18*n+4)*1024**3 and type(capacity.get('effective_cpus'))is int and capacity['effective_cpus']>=n,'effective CPU/RAM bound failed')
 return dict(checks_passed=len(checks),admitted_workers=n,effective_memory_bytes=capacity['effective_memory_bytes'],effective_cpus=capacity['effective_cpus'],scientific_source=SCI)

def check_pack(pack,name,tar):
 require(pack.get('step')=='pack'and pack.get('attempt')==name and pack.get('attempt_present')is True,'native pack assignment failed')
 require(pack.get('archive')==name+'.tar'and type(pack.get('archive_bytes'))is int and pack['archive_bytes']==tar.stat().st_size,'native pack size/name differs')
 require(pack.get('archive_sha256')==sha(tar),'native archive bytes differ')
 return pack['archive_sha256']

def load_collector():
 p=R/'codex/certificate-ladder-task34/returned_shard.py';require(sha(p)==COLLECTOR_SHA,'reviewed Task34 collector source differs')
 spec=importlib.util.spec_from_file_location('task34_returned_shard',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def run(out):
 require(out==RECEIVER and out.resolve()==out,'exact fresh owned receiver path required')
 require(not out.exists()and not out.is_symlink(),'receiver directory already exists')
 require(OWNED.is_dir()and OWNED.resolve()==OWNED,'original real owned root required')
 require(shutil.disk_usage(OWNED).free>=RESERVATION,'12GiB fresh extraction reservation unavailable')
 out.mkdir();start=time.monotonic();report=dict(schema='codex-task34-original-return-admission-v1',status='running',certificate_passed=False,scientific_endpoint_audit_pending=True,packet_sha256=PACKET_SHA,scientific_source=SCI,pods=[],attempts=[],extracted_bytes=0)
 def progress(stage):report['stage']=stage;report['seconds']=time.monotonic()-start;atomic(out/'progress.json',report)
 try:
  progress('receiver-created')
  require(sha(R/PACKET_REL)==PACKET_SHA,'independent dispatch packet differs');packet=read(R/PACKET_REL);rows=assignments(packet);rs=load_collector()
  require(packet['collector_sha256']==COLLECTOR_SHA,'packet collector authority differs')
  for a in rows:require(not(OWNED/a['attempt']).exists()and not(OWNED/a['attempt']).is_symlink(),'native destination already exists')
  plans={}
  for a in rows:
   require(sha(R/a['plan'])==a['plan_sha256'],'committed plan bytes differ');p=read(R/a['plan'])
   require(p.get('cell')=='cascade-p3'and p.get('period')==3,'wrong exact cell allocation');plans[a['attempt']]=p
  first=rs.common(plans[rows[0]['attempt']])
  for p in plans.values():rs.same(rs.common(p),first,'single independent campaign allocation')
  for i,cfg in enumerate(CONFIGS):
   group=rows[5*i:5*i+5];names=[a['attempt']for a in group];base=DATA/'runpod/soft64'/cfg['stamp'];archive=base/'results.tgz';log=DATA/'runpod'/('T34C-P3-'+cfg['pod'][-1].upper()+'-orchestrator.log')
   progress('authenticating-'+cfg['pod'])
   result=read(base/'result.json');pod_id=check_result(result,cfg,names,base);termination=check_termination(log.read_bytes(),pod_id)
   require(sha(archive)==cfg['sha256'],'whole original archive bytes differ')
   dest=out/cfg['pod']
   with tarfile.open(archive,'r:gz')as t:
    members,size=scan_tar(t);require(report['extracted_bytes']+size<=RESERVATION,'aggregate extraction reservation exceeded');fresh_extract(t,members,dest);report['extracted_bytes']+=size
   folder=dest/'explore/llm-soft-loop/pod64/certpack'/cfg['pod']
   require(folder.is_dir()and folder.resolve()==folder,'original certpack folder missing')
   require({p.relative_to(folder).as_posix()for p in folder.rglob('*.tar')}=={n+'.tar'for n in names},'exact native tar assignment differs')
   require({p.relative_to(folder).as_posix()for p in folder.rglob('pack-*.json')}=={'pack-'+n+'.json'for n in names},'exact pack assignment differs')
   require({p.relative_to(folder).as_posix()for p in folder.rglob('run.json')}=={n+'/run.json'for n in names},'exact closure assignment differs')
   preflight=check_preflight(read(folder/'preflight.json'),names,packet)
   preparation=read(folder/'prepare.json');require(preparation.get('step')=='prepare'and preparation.get('ok')is True and preparation.get('hash_problems')==[]and preparation.get('copied_upstream')==len(packet['upstream_files']),'original preparation gate failed')
   pod_record=dict(pod=cfg['pod'],pod_id=pod_id,outer_source=result['commit'],scientific_source=SCI,archive=pin(archive),result=pin(base/'result.json'),orchestrator_log=pin(log),termination=termination,preflight=pin(folder/'preflight.json'),preflight_admission=preflight,prepare=pin(folder/'prepare.json'),attempts=names)
   report['pods'].append(pod_record);progress('outer-admitted-'+cfg['pod'])
   for a in group:
    name=a['attempt'];progress('authenticating-native-'+name);native=folder/(name+'.tar');pack_path=folder/('pack-'+name+'.json');native_hash=check_pack(read(pack_path),name,native)
    run_path=folder/name/'run.json';closure=check_run(read(run_path),name)
    prefix=PurePosixPath('codex/certificate-ladder')/name
    with tarfile.open(native,'r:')as t:
     members,size=scan_tar(t,prefix=prefix);require(report['extracted_bytes']+size<=RESERVATION,'aggregate extraction reservation exceeded')
     receipts=[m for m,p in members if p==prefix/'receipt.json'];require(len(receipts)==1 and receipts[0].isfile()and receipts[0].size<=16*1024**2,'exact original receipt member required')
     receipt_bytes=t.extractfile(receipts[0]).read();parse(receipt_bytes);original=hashlib.sha256(receipt_bytes).hexdigest()
     fresh_extract(t,members,OWNED/name,prefix=prefix);report['extracted_bytes']+=size
    ref=dict(phase_index=a['phase'],start_column=a['start_column'],plan=dict(name=Path(a['plan']).name,sha256=a['plan_sha256']))
    admitted=rs.completed(OWNED/name,ref,rs.common(plans[name]),source=SCI,expected_receipt=original)
    report['attempts'].append(dict(attempt=name,period=3,phase_index=a['phase'],start_column=a['start_column'],source_commit=SCI,original_archive_receipt_sha256=original,operational=admitted,whole_archive_sha256=cfg['sha256'],archive_sha256=native_hash,pack=pin(pack_path),pack_path=str(pack_path),pack_sha256=sha(pack_path),outer_closure=pin(run_path),outer_closure_path=str(run_path),outer_closure_sha256=sha(run_path),closure_admission=closure,pod_id=pod_id,termination_record=termination,termination_record_path=str(log),termination_record_sha256=pod_record['orchestrator_log']['sha256'],termination_record_line=termination['line_number'],result=pod_record['result'],result_path=str(base/'result.json'),result_sha256=pod_record['result']['sha256'],orchestrator_log=pod_record['orchestrator_log']))
    progress('operationally-admitted-'+name)
  require(len(report['attempts'])==42,'not all original attempts admitted')
  cols=[(a['phase_index'],j)for a in report['attempts']for j in range(a['start_column'],a['start_column']+64)]
  require(len(cols)==2688 and len(set(cols))==2688 and set(cols)=={(p,j)for p in range(3)for j in range(896)},'operational phase-column assignment incomplete')
  report.update(status='all_42_original_returns_operationally_admitted',unique_phase_column_assignments=2688,complete_signed_endpoint_coverage=False)
  progress('completed');atomic(out/'operational.json',report);return report
 except BaseException as e:
  report.update(status='failed_or_incomplete',failure=dict(type=type(e).__name__,message=str(e)),preserve_all_materialized_originals=True)
  try:progress('failed');atomic(out/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
  except BaseException:pass
  raise

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(240,)*2)
 result=run(Path(a.out));print(json.dumps({k:result[k]for k in ('status','unique_phase_column_assignments','certificate_passed','scientific_endpoint_audit_pending','seconds','extracted_bytes')}))
if __name__=='__main__':main()
