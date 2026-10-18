import sys,json,hashlib
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
from point_bridge import authenticate,sha
from cache_io import unpack
from phase_domain import pack_ball,canonical_hash
from flint import ctx
ctx.prec=1024;ctx.threads=1
raw=Path('/data2/neural-shrimp-data/codex/coexistence');folder=R/'codex/coexistence-task33'
def pin(p):return dict(path=str(p),sha256=sha(p))
for p,it in [(2,14),(3,16)]:
 sup=raw/f'refine-p{p}-001-supervision';ad=json.loads((sup/'admission.json').read_text());df=raw/f'p{p}-final-domain-001.json';definition=json.loads(df.read_text());meta=json.loads((raw/'prefix-001/inputs.json').read_text())
 plan=dict(period=p,source_commit=ad['source_commit'],source_sha256={e['relative_path']:e['sha256'] for e in ad['source_blobs']},receipt=pin(sup/'receipt.json'),admission=pin(sup/'admission.json'),definition=pin(df),card=pin(folder/f'refine-p{p}-001-card.json'),card_relative_path=f'codex/coexistence-task33/refine-p{p}-001-card.json',map=definition['map'],point_versions=meta['versions'],output=str(raw/f'refine-p{p}-001'),iteration=it)
 (folder/f'p{p}-point-bridge-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 images,report=authenticate(plan,repository=R,unpack=unpack,pack=pack_ball,precision=1024)
 report['original_images_sha256']=canonical_hash(report.pop('original_images'));report['original_centers_sha256']=canonical_hash(report.pop('original_centers'));report['decoded_images_sha256']=canonical_hash([[pack_ball(v) for v in row] for row in images]);report['adapter_source_sha256']=sha(folder/'point_bridge.py')
 (folder/f'p{p}-point-bridge-admission.json').write_text(json.dumps(report,indent=2)+'\n')
 print(p,'full saved point authority and original endpoint containment admitted')
