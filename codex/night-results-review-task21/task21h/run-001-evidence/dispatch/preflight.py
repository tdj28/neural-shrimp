from pathlib import Path
import json,hashlib,subprocess,os,resource,shutil,time,datetime,traceback
os.nice(10-os.getpriority(os.PRIO_PROCESS,0));resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,)*2);resource.setrlimit(resource.RLIMIT_CPU,(50,)*2)
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');B=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21h');D=R/'codex/night-results-review-task21/task21h';out=Path('/tmp/task21h-preflight-001.json');assert not out.exists();start=time.monotonic();record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':False}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=R,timeout=20)
try:
 source=git('rev-parse','HEAD').decode().strip();assert source=='1fe08093028db99808599ce72c5dbb79bb8e9d96';assert not git('status','--porcelain','--untracked-files=all').strip();remote=git('ls-remote','origin','refs/heads/main').decode().split()[0];git('merge-base','--is-ancestor',source,remote)
 pins={'run.py':'4935bca02150ac2428c86a669ec86745804fca8a51f57b19659b0507666c09aa','reduce.py':'150e3777a52b5e0133ce3f95bb5475d166bbc1516a4f748b5dc44085b9c43503','run-001-card.json':'5984e65aae2b2ad3a8657d7fa63cca42392ff5f895b3038df90890eb558c33ae','run-001-launch.py':'8d2b4d995a859204bef180af21c3a73bf838fca5a8375620d8b7a5672feb31cb'}
 for name,pin in pins.items():assert sha(D/name)==pin==hashlib.sha256(git('show',source+':'+str((D/name).relative_to(R)))).hexdigest()
 manifest_sha='64c515c2f11a55770f7c0efa162e85ade990ad9af97576d1c1f6bad0c3944606';packet=Path('/tmp/task21h-source-002');assert sha(packet/'manifest.json')==manifest_sha==sha(B/'source-002/manifest.json');m=json.loads((packet/'manifest.json').read_text())
 for row in m['files']:
  for root in [packet,B/'source-002']:
   p=root/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
 prior=B.parent/'task21g/run-001-authority/report.json';assert sha(prior)=='26da80d0cfa9519acfacc362cf351115d3fb93afc3227e9190bed8e5972e5daf';a=json.loads(prior.read_text());assert a['passed'] and a['original_supervisor_worker_groups_sessions_absent'] and a['tool_session_reaped_exit_code']==0
 for row in a['identities']:assert not Path('/proc',str(row['pid'])).exists()
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   fields=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=[os.fsdecode(x)for x in(p/'cmdline').read_bytes().split(b'\0')if x]
  except(FileNotFoundError,ProcessLookupError,PermissionError):continue
  for row in a['identities']:assert int(fields[2])!=row['pgid'] and int(fields[3])!=row['sid']
  if any((str(R/'codex/night-results-review-task21')in x or '/codex/night-results-review-task21/'in x)and x.endswith('.py')for x in cmd):active.append({'pid':int(p.name),'command':cmd})
 assert not active,active
 saved_read=Path('/tmp/task21g-saved-read-001-supervision/receipt.json');sr=json.loads(saved_read.read_text());assert sha(saved_read)=='b19488c5696b249b4c9ac5685147f1b85ccdacc1b6ba3454261d3027bcbdcd7d' and sr['exit_code']==0 and sr['timed_out'] is False
 manufactured=Path('/tmp/task21h-manufactured-001-supervision/receipt.json');assert sha(manufactured)=='84e6b1b1135549cd36c1efe328307bfe2da63ca95efbbedb07b15b79acbbccf2';mr=json.loads(manufactured.read_text());assert mr['returncode']==0 and mr['failure'] is None
 # Manufactured original child was reaped in its sandbox namespace; do not assert PID90 as a host identity.
 assert not any((B/n).exists()for n in ['run-001','run-001-supervision','run-001-dispatch'])
 avail=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:'));disk=shutil.disk_usage('/data2').free;assert avail>=20*1024**3 and disk>=16*1024**3
 record.update(passed=True,source_commit=source,private_remote_ancestor=remote,source_pins=pins,packet_manifest_sha256=manifest_sha,packet_files_verified_at_both_roots=m['files'],packet_bytes_per_root=m['packet_bytes'],prior21g_authority_sha256=sha(prior),prior21g_pids_absent=[r['pid']for r in a['identities']],active_owned_numerical_processes=active,available_bytes=avail,disk_free_bytes=disk,load_average=os.getloadavg(),cpu_affinity_count=len(os.sched_getaffinity(0)),reservation='Sole Task21h CPU1 plus supervisor; no concurrent owned numerical reservation. Packet hashing only, no target numerical payload decoding.')
except BaseException as e:record.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 record['seconds']=time.monotonic()-start;out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items()if k!='packet_files_verified_at_both_roots'},indent=2));print('PREFLIGHT_SHA256',sha(out))
