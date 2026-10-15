from pathlib import Path
import json,hashlib,subprocess,os,resource,shutil,time,datetime,traceback
os.nice(10-os.getpriority(os.PRIO_PROCESS,0));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(50,)*2)
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21i');D=R/'codex/night-results-review-task21/task21i';out=Path('/tmp/task21i-preflight-001.json');assert not out.exists();start=time.monotonic();record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':False}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=R,timeout=20)
try:
 source=git('rev-parse','HEAD').decode().strip();assert source=='f5fa2fd151de4f500d74b83c7176ccc9525360fb';assert not git('status','--porcelain','--untracked-files=all').strip();remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];git('merge-base','--is-ancestor',source,remote)
 pins={'run.py':'f1e6f0679f877913ad1571c8d70dc36f5d0403469a82ac6d754a9bfbdd6b3ced','statistics.py':'4985313cb433a69fc51bbfbdfe06f665fa82f372487e371897038d30271363ca','adjustment.py':'5557b4e6c6169c071548664f8e7fc0502d7a2da724e0fb5c5fe5469423709f78','run-001-card.json':'743dda8dda0534d375ec0805c850683b24d5d7c6125dcaf4d4d33b7cc44c53ae','run-001-launch.py':'55752a4cbc23a79433300ab7e1c7ec16c81e1fdee218ad342cc68f7deba92da4'}
 for name,pin in pins.items():assert sha(D/name)==pin==hashlib.sha256(git('show',source+':'+str((D/name).relative_to(R)))).hexdigest()
 packets=[]
 for name,manifest_sha,absolute in [('source-001','d2b17682977949eec3796577fe10b2729022ee5a402b34548e67d02cac943ce1',False),('transcript-supplement-001','3b5bcecc019bfb3080e5b83a83287cfa6b9852de6f4fdb42727077b23c986b49',True)]:
  packet=Path('/tmp')/('task21i-'+name);assert sha(packet/'manifest.json')==manifest_sha==sha(B/name/'manifest.json');m=json.loads((packet/'manifest.json').read_text())
  for row in m['files']:
   rel=Path(row['path']).relative_to(packet)if absolute else Path(row['path'])
   for root in [packet,B/name]:
    p=root/rel;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
  packets.append({'name':name,'manifest_sha256':manifest_sha,'files':len(m['files']),'bytes_per_root':sum(x['bytes']for x in m['files'])})
 prior=B.parent/'task21h/run-001-authority/report.json';assert sha(prior)=='8c8107d1c2c6660379eaa567d445c12b3c6abbdf3387123765ab0a16d522ba64';a=json.loads(prior.read_text());assert a['passed'] and a['original_supervisor_worker_groups_sessions_absent'] and a['tool_session_reaped_exit_code']==0
 manufactured=Path('/tmp/task21i-manufactured-001-supervision/outer-closure.json');assert sha(manufactured)=='84ec1e1f3ec44e7138959545c09ee11b53866ed3269910564b6af348c5dab39f';mr=json.loads(manufactured.read_text());assert mr['tool_reaped_exit_code']==0 and mr['original_births_groups_sessions_absent']
 ids=[(r['pid'],r['pgid'],r['sid'])for r in a['identities']]+[(r['pid'],r['pgrp'],r['session'])for r in mr['original_supervisor_and_worker']]
 for pid,pgid,sid in ids:assert not Path('/proc',str(pid)).exists()
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:fields=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=[os.fsdecode(x)for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  except(FileNotFoundError,ProcessLookupError,PermissionError):continue
  assert all(int(fields[2])!=pgid and int(fields[3])!=sid for _,pgid,sid in ids)
  if any('/codex/night-results-review-task21/'in x and x.endswith('.py')for x in cmd):active.append({'pid':int(p.name),'command':cmd})
 assert not active,active;assert not any((B/n).exists()for n in ['run-001','run-001-supervision','run-001-dispatch'])
 avail=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'));disk=shutil.disk_usage('/data2').free;assert avail>=20*1024**3 and disk>=16*1024**3
 record.update(passed=True,source_commit=source,private_remote_ancestor=remote,source_pins=pins,packets_verified_both_roots=packets,prior21h_authority_sha256=sha(prior),manufactured_outer_closure_sha256=sha(manufactured),old_owned_ids_absent=ids,active_owned_numerical_processes=active,available_bytes=avail,disk_free_bytes=disk,load_average=os.getloadavg(),cpu_affinity_count=len(os.sched_getaffinity(0)),reservation='Sole Task21i CPU1 plus supervisor; no concurrent owned numerical reservation. Packet hashing only, no target numerical payload decoding.')
except BaseException as e:record.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 record['seconds']=time.monotonic()-start;out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));print('PREFLIGHT_SHA256',sha(out))
