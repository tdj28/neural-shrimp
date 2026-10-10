import numpy as np
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_floquet_complement import proposed_complements,uniform_complement
from neural_shrimp.rigorous_periodic_lyapunov import exact_float_matrix


def fixture(mu=.2):
 # Three phases, a unit critical plane and a negative real/complex complement.
 n=5;p=3;angle=.4;c,s=np.cos(angle),np.sin(angle)
 A=np.eye(n);A[2,2]=-.3;A[3:,3:]=mu*np.array([[c,s],[-s,c]])
 blocks=[arb_mat(np.eye(n).tolist()),arb_mat(np.eye(n).tolist()),exact_float_matrix(A)]
 flat=[]
 for _ in range(p):flat.extend([arb(0)]*n+[arb(int(i==0))for i in range(n)]+[arb(int(i==1))for i in range(n)])
 flat.extend([arb(1),arb(1)])
 return blocks,flat,p,n,arb_mat([[1,0],[0,1]])

def test_all_real_complex_modes_and_cyclic_sign_are_retained():
 with ctx.workprec(256):
  args=fixture();frames,proposal=proposed_complements(args[0]);assert len(frames)==3 and proposal['cyclic_final_index']==0
  assert sorted(g['type']for g in proposal['mode_groups'])==['complex','real']
  result=uniform_complement(*args);assert result['passed']and result['complement_dimension']==3
  assert len(result['quotient_phases'])==3

def test_distant_unstable_mode_cannot_be_hidden_by_proposal_selection():
 with ctx.workprec(256):
  # Arbitrary proposal selection may become singular; either result is fail-closed.
  try:out=uniform_complement(*fixture(1.2))
  except (ValueError,ZeroDivisionError):return
  assert not out['passed']

def test_whole_domain_and_critical_plane_cannot_be_silently_changed():
 with ctx.workprec(256):
  A,f,p,n,T=fixture();f[n]=arb(0)
  with pytest.raises((ValueError,ZeroDivisionError)):uniform_complement(A,f,p,n,T)
  A,f,p,n,T=fixture();A[0][2,0]=1
  with pytest.raises(ValueError,match='invariant'):uniform_complement(A,f,p,n,T)

def test_uniform_error_is_not_replaced_by_its_stable_midpoint():
 with ctx.workprec(256):
  A,f,p,n,T=fixture();A[-1][2,2]=arb('-.3','1.5')
  out=uniform_complement(A,f,p,n,T)
  assert not out['passed']and not out['metric']['complete_D_positive']['passed']

def test_periodic_balance_retains_large_individual_expansion():
 from neural_shrimp.rigorous_floquet_complement import balanced_complements
 with ctx.workprec(256):
  A,f,p,n,T=fixture()
  A[0][2,2]=100;A[1][2,2]=arb('1/100')
  bases,proposal=balanced_complements(A,f)
  assert proposal['strategy']=='periodic_lyapunov_balanced_orthogonal'and len(bases)==3
  out=uniform_complement(A,f,p,n,T)
  assert out['passed']
  assert all(row['operator_bound']['passed']for row in out['quotient_phases'])
