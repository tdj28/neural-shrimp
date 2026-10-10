import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import audit_eigenpair_adjoint_saved as M
from flint import arb,arb_mat,ctx
import pytest

def toy():
 b=arb_mat([[2,-1,1],[-1,arb(1)/2,2],[0,1,0]])
 c=arb_mat([[arb(1)/2,-arb(1)/4,arb(1)/2],[0,0,1],[arb(1)/4,arb(1)/2,0]])
 return b,c

def test_nonnormal_two_phase_border_and_last_normalization():
 j=arb_mat(6,6);x=[arb(0)]*6;x[1]=arb(1);x[3]=arb(2)
 j[0,0]=2;j[0,2]=-1;j[2,0]=-1;j[2,2]=arb(1)/2
 j[0,4]=3;j[2,4]=5;j[1,0]=7;j[3,2]=11;j[1,2]=99;j[3,0]=101
 b,r,s=M.blocks(j,x,1,2)
 assert M.serial(b)==M.serial(toy()[0]) and r[0,0]==3 and r[1,0]==5 and s==[arb(7),arb(22)]

def test_neumann_contains_known_adjoint_and_restriction():
 with ctx.workprec(256):
  b,c=toy();d,w,q,index,rhs,y0,res,f,beta,tail,y=M.border_reduce(b,c)
  assert q[index]<1 and all(z.contains(v)for z,v in zip(y,[arb(1)/2,arb(1),arb(0)]))
  initial,trace,ref=M.refine(y,y0,f,d)
  assert len(trace)==5 and ref[-1].is_exact() and ref[-1]==0
  assert all(z.contains(v)for z,v in zip(ref,[arb(1)/2,arb(1),arb(0)]))

def test_noncontracting_border_rejected():
 b,c=toy()
 with pytest.raises(ValueError,match='invertibility'):M.border_reduce(b,c*8)

def test_alpha_must_be_compatible():
 with pytest.raises(ValueError,match='alpha'):M.refine([arb(0),arb(1)],arb_mat(2,1),arb_mat(2,1),arb_mat(2,2))

def test_scalar_identity_nontrivial_and_zero_auxiliary():
 with ctx.workprec(256):
  b,c=toy();y=[arb(1)/2,arb(1),arb(0)];v=[arb(3),arb(-5),arb(0)]
  u,r,d,dual,iv=M.scalar(b,c,y,v)
  assert d==arb(-7)/2 and dual.contains(d) and iv.contains(d) and iv<0
  with pytest.raises(ValueError,match='auxiliary'):M.scalar(b,c,y,[arb(3),arb(-5),arb(1)])

def test_interval_functional_keeps_pointwise_root_values():
 b,c=toy();y=[arb(arb(1)/2,arb(1)/100),arb(1,arb(1)/100),arb(0)];v=[arb(3,arb(1)/10),arb(-5),arb(0)]
 *_,iv=M.scalar(b,c,y,v);assert iv.contains(arb(-7)/2)

def test_exact_dyadic_restore():
 with ctx.workprec(256):
  for x in [arb(0),arb(1),arb('1/3'),arb(1,arb('1e-40'))]:
   s=M.enc(x);s['display']='untrusted display';assert M.enc(M.restore(s))==M.enc(x)

def test_restore_rejects_invalid_radius():
 with pytest.raises(ValueError):M.restore({'mid':[0,0],'rad':[-1,0],'display':'x'})

def test_complete_event_inventory(tmp_path):
 (tmp_path/'events').mkdir()
 with pytest.raises(ValueError,match='event count'):M.events(tmp_path,1)
