import sys,json,hashlib,importlib.util
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp'); raw=Path('/data2/neural-shrimp-data/codex/coexistence');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
import derivative_shard as m
from flint import ctx
ctx.prec=1024;ctx.threads=1
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=json.loads((raw/'prefix-001/inputs.json').read_text()); cs=raw/'refine-p2-001/iteration-14-centers.json'
mp=m.map_definition(meta['pinned_assets'],meta['prefix_ids'])
d=dict(schema=m.DEFINITION_SCHEMA,role=m.coverage.ROLE,period=2,dimension=896,centers=json.loads(cs.read_text()),radii=[dict(mid=[1,-640],rad=[0,0]) for _ in range(2)],map=mp,map_sha256=m.domain.canonical_hash(mp),selection_provenance=dict(source_commit='067753b088ce82dc187588e922ade1d9248e364e',centers_path=str(cs),centers_sha256=sha(cs),receipt_sha256=sha(raw/'refine-p2-001-supervision/receipt.json'),corrections=14,radius_choice='First disclosed domain: uniform 2^-640, following accepted P8 scale, selected after closed centre refinement; sufficient interval gates remain untested.'))
p=raw/'p2-final-domain-001.json';assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
for phase in range(2):
 _,state,_,_=m.construct_inputs(d,phase,0);assert m.receiver_record(d,phase,state)['full_nominal_phase_box_contained']
print(str(p),sha(p))
