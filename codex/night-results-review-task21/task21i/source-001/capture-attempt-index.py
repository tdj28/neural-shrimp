from pathlib import Path
import json,hashlib,re,datetime
D=Path('/data2/neural-shrimp-data/runpod');out=Path('/tmp/task21i-attempt-index-001.json');assert not out.exists();rows=[]
def pin(p):b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def safe(v):
 if isinstance(v,dict):return {k:safe(x)for k,x in v.items()if not any(w in k.lower()for w in ['cost','price','rate'])}
 if isinstance(v,list):return [safe(x)for x in v]
 return v
for p in sorted(D.glob('*-orchestrator.log')):
 ident=p.name.removesuffix('-orchestrator.log')
 if not (re.fullmatch(r'BZ[0-9]+',ident)or ident.startswith('BPq05')or ident.startswith('BPq7')):continue
 ev=[]
 for line in p.read_bytes().splitlines():
  try:ev.append(json.loads(line))
  except ValueError:pass
 init=next((e for e in ev if 'jobset_args'in e),None);retrieved=[e for e in ev if 'out'in e];term=[e for e in ev if 'terminated'in e]
 r={'id':ident,'orchestrator':pin(p),'initial':safe(init),'recorded_termination':safe(term[-1])if term else None,'retrieval_available':bool(retrieved)}
 if retrieved:
  b=Path(retrieved[-1]['out']);r['retrieval_dir']=str(b)
  if (b/'result.json').exists():r['retrieval']=safe(json.loads((b/'result.json').read_text()));r['retrieval_record']=pin(b/'result.json')
  if (b/'results.tgz').exists():r['archive']={'path':str(b/'results.tgz'),'bytes':(b/'results.tgz').stat().st_size,'hash_scope':'Recorded retrieval hash only at this index stage; no archive opened or rehashed.'}
  if (b/'runner-stdout.txt').exists():
   r['runner_stdout']=pin(b/'runner-stdout.txt');r['exact_status_lines']=[l for l in (b/'runner-stdout.txt').read_text().splitlines()if re.search(r' start | end | runner done',l)]
 rows.append(r)
out.write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Original BZ/BP attempt/source/command/retrieval/status/recorded-termination metadata index only. All attempts retained; no target outcomes decoded, archive reads or model execution. BPC/BPCS are later source/exposure context, not added numeric scope.','attempts':rows},indent=2)+'\n');print('INDEX',pin(out));print('ATTEMPTS',[(r['id'],r['initial'].get('commit')if r['initial']else None,r.get('archive',{}).get('bytes'),r.get('exact_status_lines'))for r in rows])
