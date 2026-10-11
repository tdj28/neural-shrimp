import copy,sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_halves_math as A


def test_closed_halves_union_and_reject_other_units():
 c,e=A.geometry(1);d,f=A.geometry(2)
 assert e[1]==f[0]==F(383,4096)and e[0]==F(319,4096)and f[1]==F(447,4096)
 for i in(0,3,True,'1'):
  with pytest.raises(ValueError):A.geometry(i)


def test_exact_translation_and_separate_cubic_taylor_bounds():
 with ctx.workprec(480):
  c=list(map(arb,[2,3]));v=list(map(arb,[1,-2]));a=list(map(arb,[4,6]));w=[arb(1)]*2;cc=arb_mat([[1,2,3,4],[2,-3,1,-2]])
  for shift in(arb('-1/128'),arb('1/128')):
   r=A.translate(c,v,a,w,arb('1/64'),arb('1/1024'),shift,arb('1/128'),cc,cc)
   for d in map(arb,['-1/128','0','1/128']):
    old=shift+d
    for i in range(2):
     assert r['c'][i]+r['v'][i]*d+a[i]*d*d/2==c[i]+v[i]*old+a[i]*old*old/2
     assert r['residual'][i,0].contains(sum(cc[i,j]*old**j for j in range(4)))
     assert r['rp'][i,0].contains(sum(j*cc[i,j]*old**(j-1)for j in range(1,4)))
  with pytest.raises(ValueError,match='closed subset'):A.translate(c,v,a,w,arb('1/64'),arb('1/1024'),arb('1/64'),arb('1/128'),cc,cc)


def jet(f,g,d,m):return dict(f=A.ball_json(arb(f)),g=[A.ball_json(arb(x))for x in g],d=A.ball_json(arb(d)),m=[A.ball_json(arb(x))for x in m])


def fixture():
 # F_i(x,rate)=rate_i*x_i²; held directional derivative2rate_i*x_i*q_i.
 x=[2,3];q=[5,7];rates=[11,13];inputs=[];outputs=[]
 for i,z in enumerate(x+rates):inputs.append(jet(z,[int(k==i)for k in range(4)],q[i]if i<2 else 0,[0]*4))
 for i in range(2):
  g=[0]*4;g[i]=2*rates[i]*x[i];g[2+i]=x[i]**2;m=[0]*4;m[i]=2*rates[i]*q[i];m[2+i]=2*x[i]*q[i]
  outputs.append(jet(rates[i]*x[i]**2,g,2*rates[i]*x[i]*q[i],m))
 return dict(kind='full_jacobian',stage='target_center',phase=0,direction='q',inputs=inputs),dict(kind='full_jacobian',stage='target_center',returns=outputs)


def test_raw_mixed_jet_block_and_rate_reconstruction():
 a,b=fixture();r=A.mixed_direction(a,b,2);J=A.R.matrix(r['block']);V=A.R.matrix(r['value']);rates=A.R.matrix(r['rates'])
 assert J==arb_mat([[44,0,0,0],[0,78,0,0],[110,0,44,0],[0,182,0,78]])
 assert V==arb_mat([[44],[117],[220],[546]])and rates==arb_mat([[4,0],[0,9],[20,0],[0,42]])


def test_corrupted_actual_mixed_seed_rejected():
 a,b=fixture()
 for key,index in [('g',0),('m',0)]:
  bad=copy.deepcopy(a);bad['inputs'][0][key][index]=A.ball_json(arb(9))
  with pytest.raises(ValueError,match='held-mode'):A.mixed_direction(bad,b,2)
 bad=copy.deepcopy(a);bad['inputs'][2]['d']=A.ball_json(arb(1))
 with pytest.raises(ValueError,match='rate directional'):A.mixed_direction(bad,b,2)
