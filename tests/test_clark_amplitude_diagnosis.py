import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import diagnose_clark_amplitude_fold as D


def test_saved_gradient_reconstruction_keeps_native_sign_rate_and_norm_columns():
 with ctx.workprec(384):
  z=[arb(i+1)for i in range(26)];serial=D.vector_json
  def jet(i):return dict(f=D.ball_json(arb(0)),g=serial([arb(100*i+k)for k in range(27)]),d=D.ball_json(arb(0)),m=serial([arb(0)]*27))
  phases=[dict(phase=k,even=[jet(i)for i in range(12)],odd=[jet(i+12)for i in range(12)])for k in range(3)]
  A,b=D.reconstruct(dict(input=serial(z)),phases);assert A.nrows()==A.ncols()==25
  assert A[2,2]==-203 and A[14,15]==-1414 and A[24,13]==14 and A[24,12]==0 and b[0,0]==-26
  with pytest.raises(ValueError):D.reconstruct(dict(input=serial(z)),phases[:2])


def test_weighted_inverse_proposal_is_independently_checked():
 with ctx.workprec(384):
  A=arb_mat([[1,arb(0,arb(2))],[arb(0,arb('1/100')),1]]);out=D.diagnose(A)
  assert D.decode(out['unweighted_bound'])>1 and out['weighted_inverse_only']and D.decode(out['rigorous_weighted_bound'])<1
  bad=D.diagnose(arb_mat([[1,arb(0,arb(2))],[arb(0,arb(2)),1]]));assert not bad['weighted_inverse_only']


def test_owned_caps_and_no_native_api():
 import launch_clark_amplitude_diagnosis as L
 assert L.OUT==D.OUT.parent and len(D.PINS)==2
 s=Path(D.__file__).read_text();assert 'A.augmented('not in s and 'paired_family('not in s
