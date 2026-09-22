"""Reviewed existing-metadata selection only; no ODE or target descriptor calls."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path('/data2/neural-shrimp')
OUT=Path(__file__).resolve().parent
OLD=ROOT/'evidence/EXP-006/attempt-001'
ANCHOR=Path('/data2/neural-shrimp-data/raw/EXP-018/attempt-001')
PINS={'selector':'e8e2584b4b296af5d7b73f983d425e384da80406f946ea02064a4bc53e2cd023',
 'tests':'610d70cb8b2b5c59117066830c4871d74b8b97277d84c3d40c326cd40941d6f0',
 'EXP-006-receipt':'cdcfc91b9ce8a444a5007e1eda35a1ed34d492810ca1a152598754df27467ac2',
 'EXP-018-receipt':'b4b56a8fbcc592285cebdbf15c2182a3173c62d3e770ebc1681cd9a142066ce1'}
SOURCES={'EXP-006':'ddb568e6cd67ef2cfbdb45e05710cd86025ef548','EXP-018':'7c9b1977f5d12e4354ff41c537778a3a284a4027'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p,root=None):
 p=Path(p);return dict(path=str(p.relative_to(root)) if root else str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def main():
 start=time.monotonic();checks=[];inputs=[]
 def check(value,name):
  if not value:raise ValueError(name)
  checks.append(name)
 check(sha(ROOT/'tools/cycle_reference_selection.py')==PINS['selector'],'held selector SHA')
 check(sha(ROOT/'tests/test_cycle_reference_selection.py')==PINS['tests'],'held tests SHA')
 sys.path.insert(0,str(ROOT/'tools'))
 import cycle_reference_selection as selector
 receipts={};indexes={}
 for exp,root in [('EXP-006',OLD),('EXP-018',ANCHOR)]:
  p=root/'receipt.json';check(sha(p)==PINS[exp+'-receipt'],exp+' receipt pin')
  receipt=read(p);receipts[exp]=receipt
  check(receipt['source_commit']==SOURCES[exp],exp+' source pin')
  check(receipt['status']=='complete',exp+' original complete receipt')
  index={x['path']:x for x in receipt['artifacts']};indexes[exp]=index
  check(len(index)==len(receipt['artifacts']),exp+' unique original inventory')
  inputs.append(dict(root=str(root),**bind(p,root)))
 def member(exp,root,rel):
  actual=bind(root/rel,root);check(actual==indexes[exp].get(rel),exp+' inventory '+rel)
  inputs.append(dict(root=str(root),**actual));return actual
 # Verify all branch/journal and anchor bytes BEFORE parsing candidate nodes.
 old_artifacts={}
 for b in selector.BRANCHES.values():
  old_artifacts[b]=[member('EXP-006',OLD,'continuation/'+b+s) for s in ('.json','-nodes.jsonl','-attempts.jsonl')]
 manifest=bind(OLD/'manifest.json',OLD)
 check(manifest['sha256']==receipts['EXP-006']['manifest_sha256'],'EXP006 manifest receipt binding')
 inputs.append(dict(root=str(OLD),**manifest))
 anchor_artifacts={}
 for s in selector.SETTINGS:
  anchor_artifacts[s]=[member('EXP-018',ANCHOR,f'references/payloads/{s}--reference-pair/packet.{ext}') for ext in ('json','npz')]
 save('input-bindings.json',inputs)
 branches={};bindings={};anchors={}
 for b,artifacts in old_artifacts.items():
  branch=read(OLD/artifacts[0]['path'])
  nodes=[json.loads(x) for x in (OLD/artifacts[1]['path']).read_text().splitlines()]
  attempts=[json.loads(x) for x in (OLD/artifacts[2]['path']).read_text().splitlines()]
  check(nodes==branch['nodes'],b+' exact node journal')
  check(attempts==branch['attempts'],b+' exact attempt journal')
  branches[b]=branch
  bindings[b]=dict(experiment_id='EXP-006',source_commit=SOURCES['EXP-006'],receipt_sha256=PINS['EXP-006-receipt'],artifacts=artifacts)
 for s,artifacts in anchor_artifacts.items():
  p=ANCHOR/artifacts[0]['path'];meta=read(p)
  check(meta['schema']=='numeric-packet-v1',s+' original packet schema')
  entry=meta['arrays'];raw=p.parent/entry['path']
  check(raw==ANCHOR/artifacts[1]['path'],s+' array path identity')
  check(entry['sha256']==artifacts[1]['sha256'] and entry['bytes']==artifacts[1]['bytes'],s+' codec array binding')
  tree=meta['tree'];canonical=tree['canonical'];marker=canonical['state']
  check(set(marker)=={'__array__','shape','dtype'},s+' canonical marker identity')
  with np.load(raw,allow_pickle=False) as arrays:state=arrays[marker['__array__']].copy()
  check(state.shape==(3,) and state.dtype==np.dtype('float64') and state.dtype.str==marker['dtype'] and list(state.shape)==marker['shape'] and np.isfinite(state).all(),s+' canonical finite float64 state')
  anchors[s]={k:tree[k] for k in ('id','setting','canonical_method','complete','qualified','status','gates','alpha','gamma')}
  anchors[s]['canonical']=dict(state=state.tolist(),period=canonical['period'],source_verification=canonical['source_verification'])
  bindings['anchor-'+s]=dict(experiment_id='EXP-018',source_commit=SOURCES['EXP-018'],receipt_sha256=PINS['EXP-018-receipt'],artifacts=artifacts)
 result=selector.select_cycles(branches,anchors,bindings=bindings)
 save('selection.json',result)
 check(result==selector.select_cycles(branches,anchors,bindings=bindings),'pure selector deterministic replay')
 check(sha(ROOT/'tools/cycle_reference_selection.py')==PINS['selector'],'selector unchanged after selection')
 check(sha(ROOT/'tests/test_cycle_reference_selection.py')==PINS['tests'],'tests unchanged after selection')
 for b in inputs:check(bind(Path(b['root'])/b['path'],Path(b['root']))=={k:b[k] for k in ('path','bytes','sha256')},'unchanged '+b['root']+'/'+b['path'])
 receipt=dict(status='complete',scope='historical unforced metadata preparation only; no new ODE, neural descriptor, finite response or decoder probability',
  selector_author='evidence_assessment; selection execution is author replay, not independent review',
  root_review_approved_before_materialization=True,root_explicit_materialization_authorized=True,
  started_utc=STARTED,ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
  python=sys.version,executable=sys.executable,numpy=np.__version__,checks_pass=len(checks),checks=checks,
  counts=result['counts'],all_slots_available=result['all_slots_available'],new_neural_computations=0,
  source=[bind(ROOT/'tools/cycle_reference_selection.py'),bind(ROOT/'tests/test_cycle_reference_selection.py'),bind(Path(__file__))],
  artifacts=[bind(OUT/'input-bindings.json',OUT),bind(OUT/'selection.json',OUT)])
 save('receipt.json',receipt)
 print(json.dumps(dict(status='complete',checks_pass=len(checks),counts=result['counts'],slots=[{k:r[k] for k in ('id','available','status','node_id','dominant_rho','alpha','gamma','period')} for r in result['slots']],receipt=bind(OUT/'receipt.json'),selection=bind(OUT/'selection.json')),indent=2))
STARTED=datetime.now(timezone.utc).isoformat()
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  save('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),at=datetime.now(timezone.utc).isoformat(),source=bind(Path(__file__))))
  raise
