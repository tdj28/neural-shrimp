import sys,json,tempfile,copy
from pathlib import Path
sys.path.insert(0,'/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/coexistence-task33')
import returned_shard as m
count=0
for period in (2,3):
 for bad in (None,'timeout','exit','phase','columns','precision','source','extra'):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   def write(p,v):
    q=root/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(v));return m.sha(q)
   source='a'*40;rel='codex/coexistence-task33/derivative_shard.py';hs=write('source/'+rel,{'synthetic':True});c={'period':period,'source_sha256':{rel:hs}};plan=dict(c,phase_index=period-1,start_column=832);ph=write('source/codex/coexistence-task33/plan.json',plan)
   row=dict(phase_index=period-1,start_column=832,plan={'name':'plan.json','sha256':ph})
   write('manifest.json',{'source_commit':source});write('run/inputs.json',dict(source_commit=source,plan_sha256=ph,plan=plan));sh=write('run/derivative-shard.json',{'synthetic':True})
   status=dict(status='final_derivative_shard_completed',certificate_passed=False,phase_index=period-1,direction_indices=list(range(832,896)),columns_completed=64,full_state_dimension=896,full_vocabulary=151936,precision=1024,shard_sha256=sh,max_rss_kib=1)
   if bad=='phase':status['phase_index']=period
   if bad=='columns':status['columns_completed']=63
   if bad=='precision':status['precision']=768
   write('run/status.json',status)
   receipt=dict(source_commit=source if bad!='source' else 'b'*40,exit_code=1 if bad=='exit' else 0,timed_out=bad=='timeout',elapsed_seconds=1,artifacts=[dict(path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=m.sha(q)) for q in sorted(root.rglob('*')) if q.is_file()]);write('receipt.json',receipt)
   if bad=='extra':write('unlisted.json',{})
   try:m.completed(root,row,c,source=source,expected_receipt=m.sha(root/'receipt.json'))
   except ValueError:
    assert bad is not None,bad
   else:assert bad is None,bad
   count+=1
print(count,'original-return positive/negative controls passed; no target data or model calls')
