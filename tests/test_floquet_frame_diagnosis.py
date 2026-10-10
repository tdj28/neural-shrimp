import importlib.util
from pathlib import Path
from flint import arb,ctx
import numpy as np

def module():
 p=Path(__file__).parents[1]/'tools/diagnose_floquet_frames.py';s=importlib.util.spec_from_file_location('frame_diagnosis',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_failed_collinear_proposal_and_orthogonal_replacement_are_distinct():
 m=module()
 with ctx.workprec(384):
  flat=[]
  for _ in range(18):flat.extend([arb(0)]*13+[arb(int(i==0))for i in range(13)]+[arb(int(i==1))for i in range(13)])
  flat.extend([arb(1),arb(1)]);E=np.eye(13)[:,2:];E[:,1]=E[:,0]+1e-12*E[:,1]
  proposal={'phase_complements':[m.D.U.M.matrix_json(m.D.exact_float_matrix(E))for _ in range(18)]}
  rows=m.diagnostic(flat,proposal,lambda:None)
  assert all(r['orthogonal_whole_frame_inversion_succeeded']for r in rows)
  assert all(r['failed_basis_condition_float']>1e12 and r['orthogonal_basis_condition_float']==1 for r in rows)
