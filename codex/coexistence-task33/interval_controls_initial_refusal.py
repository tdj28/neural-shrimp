import sys,importlib.util,json,tempfile,copy
from pathlib import Path
from fractions import Fraction
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path[:0]=[str(R/'codex/coexistence-task33'),str(R/'codex/certify-llm')]
spec=importlib.util.spec_from_file_location('newshard',R/'codex/coexistence-task33/derivative_shard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from flint import ctx
from cache_io import unpack
import phase_domain as d
ctx.prec=1024;ctx.threads=1
n=0
# Original exact endpoints remain enclosed after2048-to1024 decoding.
for mant in [(1<<1800)+3,-((1<<1800)+3),0,1]:
 for rad in [[0,0],[1,-1900],[3,-900]]:
  rec={'mid':[mant,-1801],'rad':rad};v=unpack(rec);mid,r=d._record_parts(rec);gotmid,gotr=d._record_parts(d.pack_ball(v));assert gotmid-gotr<=mid-r and gotmid+gotr>=mid+r;n+=1
for p in (2,3):
 definition={'period':p,'centers':[[{'mid':[i+1,-3],'rad':[0,0]} for _ in range(896)] for i in range(p)],'radii':[[{'mid':[1,-640],'rad':[0,0]} for _ in range(896)] for i in range(p)]}
 artifact,state,directions,records=m.construct_inputs(definition,p-1,832);assert len(state)==896 and directions.ncols()==64
 assert all(directions[i,k]==int(i==832+k) for i in range(896) for k in range(64));assert m.receiver_record(definition,p-1,state)['full_nominal_phase_box_contained'];n+=1
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);df=root/'definition.json';df.write_text('{}');prefix=root/'prefix';prefix.mkdir()
 plan=dict(schema=m.SCHEMA,definition_path=str(df),definition_sha256='1'*64,map_sha256='2'*64,backend_sha256='3'*64,source_sha256={k:'4'*64 for k in m.SOURCE_PATHS},evaluator_sha256='5'*64,prefix_attempt=str(prefix),prefix_receipt_sha256='6'*64,prefix_map_source_sha256='7'*64,prefix_cache_sha256='8'*64,phase_index=1,start_column=832,precision=1024,token_chunk=1024,envelope=m.ENVELOPE,period=2,prefix_source_commit='a'*40,prefix_output=str(prefix),prefix_receipt=str(root/'receipt'),prefix_admission_path=str(root/'admission'),prefix_admission_sha256='9'*64)
 m.validate_plan(plan,owned=root);n+=1
 for key,value in [('period',8),('phase_index',2),('start_column',833),('precision',768),('prefix_output',str(root)),('prefix_source_commit','no')]:
  bad=copy.deepcopy(plan);bad[key]=value
  try:m.validate_plan(bad,owned=root)
  except ValueError:n+=1
  else:raise AssertionError(key)
print(n,'focused controls passed; no model calls')
