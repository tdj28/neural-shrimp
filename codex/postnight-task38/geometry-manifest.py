"""Package exact hashes from the completed bounded reductions and local source snapshots."""
import datetime,hashlib,json,pathlib,resource
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,)*2);resource.setrlimit(resource.RLIMIT_CPU,(20,)*2)
B=pathlib.Path('/tmp/task38-geometry');pins={}
for name in ['geometry-results-001.json','geometry-curves-004.json']:
 for path,value in json.loads((B/name).read_text())['pins'].items():
  if path in pins and pins[path]!=value:raise RuntimeError('Evidence changed: '+path)
  pins[path]=value
def pin(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
files=[p for p in B.iterdir() if p.is_file() and p.name.startswith('geometry-') and p.name not in ['geometry-manifest.json']]
roots=json.loads(pathlib.Path('/tmp/task38-root/snapshots.json').read_text())
pdf='/data2/neural-shrimp-data/references/gallas_dissecting_shrimps_PhysA1994.pdf'
missing='/data2/neural-shrimp-data/explore/llm-soft-loop/kite/basin-check-stipple/basin-check.json'
obj={'schema':'task38-geometry-saved-evidence-audit-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_model_calls':True,'scripts_and_outputs':[pin(p) for p in sorted(files)],'snapshots':roots,'raw_records':pins,'original_raw_count':len(pins),'reference':pin(pdf),'reference_text_extract':pin(B/'gallas-pdf-pages20-24.txt'),'missing_record':{'path':missing,'exists':pathlib.Path(missing).exists()},'authoritative_outputs':['geometry-results-001.json','geometry-curves-004.json','geometry-details-001.json'],'retained_extractor_correction':'Initial main audit HR12 prefix comparison used flattened source without reshaping; geometry-details-001 verifies all four prefix38/tail2 merges. Curves attempts001/002/003 failed;004 completed.','bounds':'serial nice10,HFoffline,numerical threads1;2GiB AS/100CPU seconds/120wall alarm for main and curve reductions;1GiB/20CPU/30alarm details; no target runs'}
(B/'geometry-manifest.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps({'raw_count':len(pins),'manifest':pin(B/'geometry-manifest.json'),'review':pin(B/'geometry-review.md')}))
