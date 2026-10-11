import copy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_saved_cubic_endpoint as E
from neural_shrimp.rigorous_cubic_gradient_path import cubic_jacobian_coefficients
from neural_shrimp.rigorous_orbits import ball_json

class Model:
 n=2
 def cusp_step(self,x,h,o):return [h*x[0]+o*x[1],h*x[1]+o*x[0]]

def fixture():
 c=[arb('1/4'),arb('3/8'),arb(1),arb(0),arb(0),arb(1),arb(1),arb('1/4'),arb('1/64')]
 s=[arb(i+1)/128 for i in range(9)];a=[arb(i-4)/512 for i in range(9)];b=[arb(i-2)/1024 for i in range(9)];err=[arb(0,arb((1,-18)))]*9;normal=[arb(0)]*8+[arb(1)]
 return c,s,a,b,err,normal

def test_actual_gradient_inputs_replay_and_exact_restore():
 with ctx.workprec(256):
  c,s,a,b,err,n=fixture();rows=[];cursor=[]
  def observe(k,v):
   if k=='gradient_path_input':cursor.append(v)
   if k=='gradient_path_return':rows.append(dict(input=cursor[-1],return_value=v))
  expected=cubic_jacobian_coefficients(Model(),c,s,a,b,err,arb(0,arb('1/1024')),1,n,observe=observe)
  records=[{'input':x['input'],'return':x['return_value']}for x in rows]
  got=E.replay_jacobian(c,s,a,b,err,arb(0,arb('1/1024')),1,n,records)
  for old,new in zip(expected,got):
   for i in range(9):
    for j in range(9):assert ball_json(old[i,j])==ball_json(new[i,j])
  bad=copy.deepcopy(records);bad[0]['input']['inputs'][0]['value'][0][0]=ball_json(arb(0))
  with pytest.raises(ValueError):E.replay_jacobian(c,s,a,b,err,arb(0,arb('1/1024')),1,n,bad)
  bad=copy.deepcopy(records);bad[1]['return']['direction']='q'
  with pytest.raises(ValueError):E.replay_jacobian(c,s,a,b,err,arb(0,arb('1/1024')),1,n,bad)

def test_exact_saved_domain_edge_not_inflated_decoder():
 v={'mid':[0,0],'rad':[536870913,-40]}
 with ctx.workprec(480):
  out=E.original_ball(v);assert ball_json(out)['rad']==v['rad']
  E.covered(v,ball_json(out))
  with pytest.raises(ValueError):E.covered(v,ball_json(arb((536870913,-40))+arb((1,-80))))

def test_point_operator_noncommuting_and_tangent():
 with ctx.workprec(256):
  J=[arb_mat([[2,1],[0,3]]),arb_mat([[0,2],[1,0]]),arb_mat([[1,0],[0,2]])];C0=arb_mat([[arb(1)/2,-arb(1)/6],[0,arb(1)/3]]);C1=-C0*J[1]*C0;d=arb(1)/64
  C,D,q=E.endpoint_operator(J,C0,C1,d,[arb(1)]*2)
  exactJ=J[0]+J[1]*d+J[2]*d*d;assert q<arb('1/100')
  for i in range(2):
   for j in range(2):assert(D[i,j]-(arb(int(i==j))-(C*exactJ)[i,j])).contains(0)
  with pytest.raises(ValueError):E.endpoint_operator([arb_mat(2,2)]*3,C0,C1,d,[arb(1)]*2)
