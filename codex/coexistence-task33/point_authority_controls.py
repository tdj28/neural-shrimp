import sys,json,copy
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
from point_bridge import authenticate
from cache_io import unpack
from phase_domain import pack_ball
from flint import ctx
ctx.prec=1024;ctx.threads=1
original=json.loads((R/'codex/coexistence-task33/p2-point-bridge-plan.json').read_text())
for label in ['receipt_pin','source','iteration','map']:
 p=copy.deepcopy(original)
 if label=='receipt_pin':p['receipt']['sha256']='0'*64
 elif label=='source':p['source_commit']='0'*40
 elif label=='iteration':p['iteration']=13
 else:p['map']['alpha_ratio']=[1,2]
 try:authenticate(p,repository=R,unpack=unpack,pack=pack_ball,precision=1024)
 except ValueError as e:print(label,'refused:',e)
 else:raise AssertionError(label+' accepted')
