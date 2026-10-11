import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_clark_extension_math as R
from neural_shrimp import rigorous_clark_explicit_fold as E
from neural_shrimp.rigorous_orbits import vector_json,ball_json

def test_independent_actual_mixed_assembly_unrelated_native_map():
 with ctx.workprec(384):
  n=2;j=[[arb(int(i==k))/4 for k in range(n)]for i in range(n)];z=[arb(1)/9,arb(1)/11,arb(3)/5,arb(1),arb(0),arb(7)/5];u=z+[arb(i+1)/100 for i in range(5)];b=arb(1)/64;events=[]
  f,J,h,_=E.augmented(j,u,b,lambda k,v:events.append((k,v)));got=R.explicit(dict(input=vector_json(u),squared_amplitude=ball_json(b)),[v for k,v in events if k=='native_even_odd_phase'])
  for x,y in zip(got,(f,J,h),strict=True):assert all(v.contains(0)for row in(x-y).tolist()for v in row)
  rows=[v for k,v in events if k=='native_even_odd_phase'];rows[0]['phase']=1
  with pytest.raises(ValueError,match='phase order'):R.explicit(dict(input=vector_json(u),squared_amplitude=ball_json(b)),rows)

def test_fixed_gain_column_permutation_keeps_external_b():
 J=arb_mat([[100*i+k for k in range(11)]for i in range(11)]);h=arb_mat([[1000+i]for i in range(11)]);got=R.fixed_gain_jacobian(J,h)
 for i in range(11):assert [got[i,k]for k in range(11)]==[J[i,k]for k in list(range(5))+list(range(6,11))]+[h[i,0]]

def test_saved_acquisition_direction_must_cover_own_tangent():
 # Algebra-only negative: a synthetic exact H has identity paired derivative;
 # its independent tangent cannot be silently replaced by a narrower direction.
 z=[arb(0),arb(0),arb(1)/2,arb(1),arb(0),arb(2)];N=6;K=5;rows=[]
 def enc(x):return ball_json(arb(x))
 for phase in range(3):
  native=[]
  for i in range(4):
   coord=i if i<2 else i+1;sgn=-1 if i<2 else 1
   g=[arb(0)]*7;g[coord]=sgn;g[[0,1,2,4][i]]-=1;g[6]=-1
   native.append(dict(f=enc(sgn*z[coord]),g=[enc(x)for x in g],d=enc(0),m=[enc(0)]*7))
  rows.append(dict(phase=phase,even=native[:2],odd=native[2:]))
 with pytest.raises(ValueError,match='second acquisition direction'):R.paired_curvature(z,arb(1)/64,rows,rows,[enc(0)]*K)

def test_positive_synthetic_fixed_gain_curvature_and_bzero_rejection():
 from copy import deepcopy
 z=[arb(0),arb(0),arb(1)/2,arb(1),arb(0),arb(2)];rows=[]
 def enc(x):return ball_json(arb(x))
 for phase in range(3):
  native=[]
  for i in range(4):
   coord=i if i<2 else i+1;sgn=-1 if i<2 else 1;g=[arb(0)]*7;g[coord]=sgn;g[[0,1,2,4][i]]-=1
   native.append(dict(f=enc(sgn*z[coord]),g=[enc(x)for x in g],d=enc(0),m=[enc(0)]*7))
  rows.append(dict(phase=phase,even=native[:2],odd=native[2:]))
 second=deepcopy(rows);second[-1]['odd'][0]['m'][6]=enc(2)
 got=R.paired_curvature(z,arb(1)/64,rows,second,[enc(0)]*5);assert got['passed']and got['leak_bb']==2 and got['leak_aa']==arb(1)/8
 with pytest.raises(ValueError,match='positive'):R.paired_curvature(z,arb(0),rows,second,[enc(0)]*5)
