"""Read-only local receipt/archive authentication; never contacts a pod."""
import argparse,hashlib,json,tarfile,time,resource
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();root=Path('/data2/neural-shrimp-data');rows=[]
 specs=[('default','SD','20261018T015331Z'),('p3','SP3','20261018T015356Z'),('band6','SB6','20261018T015421Z'),('p10','SP10','20261018T015446Z'),('alpha','SA','20261018T015241Z'),('temp','ST','20261018T015306Z')]
 for name,tag,date in specs:
  d=root/'runpod/soft64'/date;rp=d/'result.json';j=json.loads(rp.read_text());log=root/'runpod'/('SH-'+tag+'-orchestrator.log');records=[]
  for line in log.read_text().splitlines():
   try:v=json.loads(line)
   except json.JSONDecodeError:continue
   if isinstance(v,dict):records.append(v)
  closed=[v for v in records if v.get('pod_id')==j['pod_id'] and v.get('terminated') is True and v.get('in_inventory') is False and v.get('direct_lookup') is None]
  assert closed,'missing original local termination receipt'
  assert j['runner_returncode']==0 and j['hash_verified'] is True
  archive=d/'results.tgz';pin=digest(archive);assert pin['sha256']==j['local_sha256']==j['remote_sha256']
  expected={f:digest(root/'explore/llm-soft-loop/pod64/sheets'/name/f) for f in ['sheets.npz','sheets.json','states.npz']};matched={};member_count=0
  with tarfile.open(archive,'r|*') as tar:
   for m in tar:
    member_count+=1
    if not m.isfile():continue
    f=Path(m.name).name
    if f not in expected:continue
    assert '/sheets/'+name+'/' in '/'+m.name,'unexpected duplicate candidate member'
    assert f not in matched,'duplicate original member'
    h=hashlib.sha256();n=0;stream=tar.extractfile(m)
    for b in iter(lambda:stream.read(1024*1024),b''):h.update(b);n+=len(b)
    assert n==expected[f]['bytes'] and h.hexdigest()==expected[f]['sha256']
    matched[f]={'member':m.name,**expected[f]}
  assert set(matched)==set(expected)
  # Whitelist only provenance fields; original monetary fields are not copied.
  rows.append({'name':name,'source_commit':j['commit'],'jobset':j['jobset'],'receipt':digest(rp),'orchestrator_log':digest(log),'archive':pin,'members':matched,'member_count':member_count,'local_original_termination_records':len(closed),'runner_returncode':j['runner_returncode'],'hash_verified':True,'upload_hashes':{k:j['upload'][k] for k in ['code_sha256','data_sha256']}})
  (out/(name+'.json')).write_text(json.dumps(rows[-1],indent=2)+'\n')
 (out/'report.json').write_text(json.dumps({'completed':True,'rows':rows,'elapsed':time.monotonic()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Local original receipts and retrieved archive/member bytes; no remote asset-byte attestation and no target numerical decoding.'},indent=2)+'\n')
if __name__=='__main__':main()
