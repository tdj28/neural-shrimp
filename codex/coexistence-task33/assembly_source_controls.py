import sys,json,copy
from pathlib import Path
R=Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp');sys.path.insert(0,str(R/'codex/coexistence-task33'))
import assembly_core as a
p=json.loads((R/'codex/coexistence-task33/p2-assembly-001-card.json').read_text())
a.authenticate_sources(p,R);n=1
for kind in ['missing_source','changed_source','extra_source','missing_review','changed_review','changed_map','changed_derivative','changed_point','changed_binding']:
 q=copy.deepcopy(p)
 if kind=='missing_source':q['assembly_source_sha256'].pop(next(iter(q['assembly_source_sha256'])))
 if kind=='changed_source':q['assembly_source_sha256'][next(iter(q['assembly_source_sha256']))]='0'*64
 if kind=='extra_source':q['assembly_source_sha256']['unexpected.py']='0'*64
 if kind=='missing_review':q['reviews'].pop('map_regularity')
 if kind=='changed_review':q['reviews']['map_regularity']['sha256']='0'*64
 if kind=='changed_map':q['map']['steering_layer']=11
 if kind=='changed_derivative':q['derivative_source_sha256']['codex/certify-llm/ball_qwen.py']='0'*64
 if kind=='changed_point':q['point_plan']['sha256']='0'*64
 if kind=='changed_binding':q['review_binding']['sha256']='0'*64
 try:a.authenticate_sources(q,R)
 except ValueError:n+=1
 else:raise AssertionError(kind)
print(n,'source/review binding controls passed; no model or gate calls')
