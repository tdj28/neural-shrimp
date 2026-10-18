import sys
from pathlib import Path
r=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(r/'codex/coexistence-task33'),str(r/'codex/certify-llm')]
from point_bridge import decode_original,endpoints
from cache_io import unpack
from phase_domain import pack_ball
from flint import ctx
ctx.prec=1024
n=0
for m in [0,1,-1,(1<<1800)+3,-((1<<1800)+3)]:
 for rad in [[0,0],[1,-1500],[3,-900]]:
  v=dict(mid=[m,-1801],rad=rad);decode_original(v,unpack,pack_ball);n+=1
for v in [dict(mid=[1,0],rad=[-1,0]),dict(mid=[1.5,0],rad=[0,0])]:
 try:endpoints(v)
 except ValueError:n+=1
 else:raise AssertionError('bad record accepted')
try:decode_original(dict(mid=[1,0],rad=[1,-10]),lambda v:unpack(dict(mid=v['mid'],rad=[0,0])),pack_ball)
except ValueError:n+=1
else:raise AssertionError('lost radius accepted')
print(n,'point endpoint controls passed; no actual evidence/model execution')
