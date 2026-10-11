import sys
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb,arb_mat,ctx
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'tools/reviews')]
import review_signed_output_math as H
from neural_shrimp.rigorous_orbits import ball_json as b,vector_json as v

def mixed(f,d,g):return dict(kind='mixed',f=b(f),d=b(d),g=v(g),m=v([arb(0)]*len(g)))

def test_native_signed_residual_and_corruption():
 with ctx.workprec(256):
  z=list(map(arb,[2,1,3,4]));normal=list(map(arb,[0,0,1,0]));origin=[arb(0)]*4
  inputs=[mixed(z[i],arb(int(i==0)),[arb(int(j==i))for j in range(3)])for i in(0,2,3)]
  # Native base order is x,hidden,output.
  for i,a in enumerate(inputs):a['g']=v([arb(int(i==j))for j in range(3)])
  ret=mixed(arb(5),arb(-1),[arb(2),arb(3),arb(0)])
  e=[dict(phase=0,inputs=inputs,returns=[ret])]
  a=H.point_residual(z,normal,origin,arb(3),e,1,1)
  assert [a[i,0]for i in range(4)]==list(map(arb,[3,0,0,0]))
  bad=deepcopy(e);bad[0]['inputs'][1]['f']=b(arb(9))
  with pytest.raises(ValueError,match='state/rates'):H.point_residual(z,normal,origin,arb(3),bad,1,1)
  bad=deepcopy(e);bad[0]['inputs'][0]['d']=b(arb(-1))
  with pytest.raises(ValueError,match='direction'):H.point_residual(z,normal,origin,arb(3),bad,1,1)

def test_fixed_domain_exact_edge_and_nonnormal_localization():
 with ctx.workprec(256):
  c=[arb(1),arb(2)];chart=dict(center=v(c),slope=v([arb(0)]*2),curvature=v([arb(0)]*2),parameter_center='0',parameter_endpoints=['-1','1']);domain=dict(error=v([arb(0,arb(1)/8)]*2));known=v([arb(1,arb(1)/16),arb(2,arb(1)/16)]);J=arb_mat([[1,2],[0,1]]);C=arb_mat([[1,-2],[0,1]]);point=[arb(1)+arb((1,-20)),arb(2)];g=J*arb_mat([[point[i]-c[i]]for i in range(2)])
  r=H.localize(C,J,g,point,known,chart,domain,'0',[arb(1)]*2)
  assert r['passed']and all(H.R.original_ball(r['box'][i]).contains(c[i])for i in range(2))
  bad=deepcopy(known);bad[0]=b(arb(1)+H.R.exact(H.M.interval(domain['error'][0])[1])+arb((1,-100)))
  with pytest.raises(ValueError,match='literal original'):H.fixed_domain(chart,domain,bad,v(point),'0')

def test_projected_full470_rejects_single_coordinate():
 with ctx.workprec(256):
  n=470;c=[arb(0)]*n;normal=[arb(0)]*(n-1)+[arb(1)];box=v([arb(0)]*n)
  r=H.S.D.anchor(box,c,c,c,normal,c,arb(0),arb(1),[arb(1)]*n)
  assert r['passed']and r['coordinates']==n
  box[13]=b(arb(1));r=H.S.D.anchor(box,c,c,c,normal,c,arb(0),arb(1),[arb(1)]*n)
  assert not r['passed']and r['coordinates']==n-1
