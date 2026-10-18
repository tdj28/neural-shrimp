import ast
from pathlib import Path
from flint import arb,ctx
ctx.prec=2048
path=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/coexistence-task33/center_refinement.py')
tree=ast.parse(path.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration');code=compile(ast.Module(body=[loop],type_ignores=[]),str(path),'exec')
import time
for p,offset,limit,goal,applied in [(2,arb(0),16,True,0),(3,arb((1,-701)),16,True,0),(2,arb((1,-700)),1,False,1),(3,arb(1),2,False,2)]:
 writes={};calls=[];corrections=[]
 class Ball:
  def forward(self,c):calls.append(1);return [v+offset for v in c],[arb(0)]*896
 def correct(c,im,A,B):corrections.append(1);return dict(centers=c,record={'synthetic':True})
 env=dict(card={'corrections':limit},p=p,centers=[[arb(0)]*896 for _ in range(p)],ball=Ball(),matrices=None,B=None,correct=correct,time=time,arb=arb,pack=lambda v:(v.mid().man_exp(),v.rad().man_exp()),write=lambda k,v:writes.__setitem__(k,v),summaries=[],goal_met=False)
 exec(code,env);assert env['goal_met']==goal and env['iteration']==applied and len(corrections)==applied and len(calls)==p*(applied+1) and len(env['summaries'])==applied+1
 assert len([x for x in writes if x.endswith('-correction-witnesses.json')])==applied
print('4 actual-loop controls passed: immediate goal, strict threshold, exhaustion, all phase/witness counts; no model calls')
