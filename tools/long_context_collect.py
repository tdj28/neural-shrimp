"""Exact-owned LCM1 immutable-file retrieval; acknowledge only verified checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from datetime import datetime
import runpod_pod as rp
from runpod_phase0 import ssh_base


def sha(data):
 return hashlib.sha256(data).hexdigest()


def collect_checkpoint(read, write, out, source):
 checkpoint=json.loads(read('checkpoint.json'))
 if checkpoint['source']!=source:raise ValueError('source mismatch')
 manifest_bytes=read('manifest.json');digest=sha(manifest_bytes)
 if digest!=checkpoint['manifest_sha256']:raise ValueError('checkpoint changed')
 manifest=json.loads(manifest_bytes)
 for name,h in manifest.items():
  if Path(name).name!=name or name in ('.','..') or len(h)!=64:raise ValueError('invalid manifest entry')
  target=out/name
  if target.exists():
   if sha(target.read_bytes())!=h:raise ValueError('immutable evidence mismatch')
   continue
  data=read(name);partial=out/(name+'.partial');partial.write_bytes(data)
  if sha(data)!=h:raise ValueError('transfer hash mismatch')
  partial.replace(target)
  if sha(target.read_bytes())!=h:raise ValueError('local hash mismatch')
  with (out.parent/'receipts.jsonl').open('a') as f:f.write(json.dumps(dict(file=name,sha256=h,bytes=len(data),time=time.time()))+'\n')
 # Every local file was rehashed above, including the final checkpoint.
 (out/'manifest.json').write_bytes(manifest_bytes)
 ack=json.dumps(dict(source=source,manifest_sha256=digest)).encode()
 write('ack.json',ack)
 status=dict(source=source,manifest_sha256=digest,files=len(manifest),complete=checkpoint['final'],time=time.time())
 (out.parent/'status.json').write_text(json.dumps(status)+'\n')
 return status


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',required=True);p.add_argument('--name',required=True);p.add_argument('--approval',default='LCM1-user-approved-20261024');p.add_argument('--relative-output',default='llm-soft-loop/pod64/lcm1/map-01');p.add_argument('--hard-minutes',type=int,default=780);a=p.parse_args()
 if not a.relative_output.startswith('llm-soft-loop/pod64/') or '..' in Path(a.relative_output).parts or not 0<a.hard_minutes<=780:p.error('Invalid bounded collection scope')
 out=a.root/'incremental'/'data';out.mkdir(parents=True,exist_ok=False)
 remote='/root/neural-shrimp-data/explore/'+a.relative_output
 startup=time.time();pod=None;base=None;last=None
 (a.root/'collector-ready.json').write_text(json.dumps(dict(source=a.source,time=startup))+'\n')
 while time.time()<startup+a.hard_minutes*60:
  try:
   if pod is None:
    matches=[r for r in (json.loads(x) for x in rp.LEDGER.read_text().splitlines()) if r.get('event')=='created' and r.get('name')==a.name and r.get('commit')==a.source and r.get('approval')==a.approval]
    if not matches:
     if time.time()>startup+600:raise TimeoutError('No owned creation within startup allowance')
     time.sleep(2);continue
    if len(matches)!=1:raise ValueError('ambiguous ownership')
    pod=matches[0];deadline=datetime.fromisoformat(pod['time']).timestamp()+a.hard_minutes*60
   if time.time()>=deadline:break
   if base is None:
    state=rp.pod_status(pod['pod_id'])
    if state is None:break
    if state['name']!=a.name:raise ValueError('pod name mismatch')
    for port in (state.get('runtime') or {}).get('ports') or []:
     if port.get('privatePort')==22 and port.get('isIpPublic'):base=ssh_base(port['ip'],port['publicPort'])
    if base is None:time.sleep(2);continue
   def read(name):
    return subprocess.run(base+['cat '+shlex.quote(remote+'/'+name)],capture_output=True,check=True,timeout=min(45,max(1,deadline-time.time()))).stdout
   def write(name,data):
    path=shlex.quote(remote+'/'+name);tmp=shlex.quote(remote+'/'+name+'.tmp')
    subprocess.run(base+['cat > '+tmp+' && mv '+tmp+' '+path],input=data,capture_output=True,check=True,timeout=min(15,max(1,deadline-time.time())))
   current=read('checkpoint.json')
   if current!=last:
    status=collect_checkpoint(read,write,out,a.source);last=current
    if status['complete']:return
  except Exception as e:
   with (a.root/'collector-errors.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),error=str(e)))+'\n')
   if pod is None and time.time()>startup+600:break
  time.sleep(2)
 raise RuntimeError('Collector ended without verified completion; partial evidence retained')

if __name__=='__main__':main()
