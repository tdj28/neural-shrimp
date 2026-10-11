from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_correlated_pal import correlated_tube
from neural_shrimp.rigorous_eigenpair_march import decode
from neural_shrimp.rigorous_unit_pair_pal import quadratic_tube
from neural_shrimp.rigorous_orbits import exact
from test_rigorous_unit_pair import Polynomial,Primitive
from test_rigorous_unit_pair_pal import fixture

@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_actual_full_native_coefficients_bound_and_known_endpoint(model,period):
 with ctx.workprec(256):
  ini,kw=fixture(model,period);kw.update(parameter='3/1024',halfwidth_exponent=-11)
  old,data=quadratic_tube(model,ini,**kw);N=len(ini);C=arb_mat([[decode(v)for v in row]for row in old['preconditioner']]);events=[];ticks=[]
  args=dict(period=period,normal=[exact(v)for v in kw['normal']],origin=[arb(0)]*N,parameter='3/1024',halfwidth_exponent=-11,radius_exponent=-12,weights=[arb(1)]*N,anchor={'box':data['endpoints'][0],'exact_parameter':'5/2048','dimension':model.n,'period':period})
  got=correlated_tube(model,data['center'],data['slope'],data['curvature'],C,**args,tick=lambda:ticks.append(1),observe=lambda k,v:events.append((k,v)))
  assert got['passed'],got['gates'];assert len(ticks)==8*period+2
  assert len(got['bound']['coordinate_inclusions'])==N and all(got['bound']['coordinate_inclusions'])
  domains=[v for k,v in events if k=='correlated_domain'];inputs=[v for k,v in events if k=='gradient_path_input']
  assert len(domains)==1 and len(inputs)==4*period
  # The actual fixed correction domain must occur in BOTH d acquisitions.
  assert len(domains[0]['error'])==N and all(decode(v).contains(0)for v in domains[0]['error'])
  if period==1:
   # Independently known exact native state/rates on this polynomial model.
   for d,endpoint in zip([exact('5/2048'),exact('7/2048')],got['endpoint_root_boxes']):
    with ctx.workprec(512):x=2*(d.sqrt().cos()-1);h=1-x*x/2;o=1-x
    assert decode(endpoint[0]).contains(x)and decode(endpoint[-3]).contains(h)and decode(endpoint[-2]).contains(o)
  args['anchor']['box'][model.n]+=1
  bad=correlated_tube(model,data['center'],data['slope'],data['curvature'],C,**args)
  assert not bad['gates']['anchor_identified']and not bad['passed']

def test_invalid_proposal_domain_fails_before_acquisition():
 model=Polynomial();N=9;z=[arb(0)]*N;calls=[]
 with pytest.raises(ValueError):correlated_tube(model,z,z,z,arb_mat(N,N),period=1,normal=z,origin=z,parameter='0',halfwidth_exponent=-1,radius_exponent=-1,weights=[arb(-1)]*N,tick=lambda:calls.append(1))
 assert not calls


def test_compact_evidence_keeps_native_returns_exact_inverse_and_identical_gates():
 with ctx.workprec(256):
  model=Polynomial();ini,kw=fixture(model,1);kw.update(parameter='3/1024',halfwidth_exponent=-11)
  old,data=quadratic_tube(model,ini,**kw);N=len(ini);C=arb_mat([[decode(v)for v in row]for row in old['preconditioner']])
  args=dict(period=1,normal=[exact(v)for v in kw['normal']],origin=[arb(0)]*N,parameter='3/1024',halfwidth_exponent=-11,radius_exponent=-12,weights=[arb(1)]*N,anchor={'box':data['endpoints'][0],'exact_parameter':'5/2048','dimension':2,'period':1})
  records=[];full=correlated_tube(model,data['center'],data['slope'],data['curvature'],C,**args)
  slim=correlated_tube(model,data['center'],data['slope'],data['curvature'],C,**args,compact=True,observe=lambda k,v:records.append((k,v)))
  assert full['passed']==slim['passed']is True and full['gates']==slim['gates']
  for k in ['contraction','coordinate_inclusions','radii','whole_image']:assert full['bound'][k]==slim['bound'][k]
  for i,M in enumerate(slim['inverse_polynomial_exact']):
   assert all(arb(tuple(v))==decode(full['inverse_polynomial'][i][j][k])for j,row in enumerate(M)for k,v in enumerate(row))
  assert sum(k=='gradient_path_input'for k,v in records)==sum(k=='gradient_path_return'for k,v in records)==4
  assert all('coefficients'not in v and v['shape']==[9,9]for k,v in records if k=='gradient_path_assembly_return')
  assert len(slim['bound']['image_radius_ratio_upper'])==N
