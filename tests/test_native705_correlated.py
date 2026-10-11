import importlib.util
from pathlib import Path
import pytest
from flint import arb,ctx


def module():
 p=Path(__file__).parents[1]/'tools/probe_native705_correlated.py';s=importlib.util.spec_from_file_location('native705_correlated',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture(m):
 x=[arb(0)]*702+[arb(2),arb(3),arb('1/128')];v=[arb(0)]*702+[arb(5),arb(7),arb(11)]
 return {'passed':True,'dimension':705,'weighted_neumann_bound':m.D.ball_json(arb('1/4')),'center_box':m.vector_json(x),'whole_box':m.vector_json(x),'tangent_box':[[m.D.ball_json(a)]for a in v]}


def test_full705_projection_retains_both_rates_and_excludes_only_angle():
 m=module()
 with ctx.workprec(384):
  c,w,v,T,full=m.native_domains(fixture(m))
  assert len(c)==len(w)==len(v)==704 and c[-2:]==[arb(2),arb(3)]and v[-2:]==[arb(5),arb(7)]
  assert len(full['whole'])==705 and m.U.R.original_ball(full['tangent'][-1])==11
  assert T.nrows()==T.ncols()==2 and T[0,0].contains(arb('1/128').sqrt().cos())
  assert T[1,0]<0 and T[0,1]>0


def test_bad_full705_or_angle_is_rejected():
 m=module()
 for key,value in [('dimension',704),('passed',False),('weighted_neumann_bound',m.D.ball_json(arb(1)))]:
  x=fixture(m);x[key]=value
  with pytest.raises(ValueError):m.native_domains(x)
 for t in [arb(0),arb(-1),arb(1)]:
  x=fixture(m);x['whole_box'][-1]=m.D.ball_json(t)
  with pytest.raises(ValueError):m.native_domains(x)
 x=fixture(m);x['tangent_box'].pop()
 with pytest.raises(ValueError):m.native_domains(x)


def test_unrelated_full_quotient_and_native_profile():
 m=module()
 with ctx.workprec(384):
  assert m.synthetic_cocycle(lambda:None)['passed'];inp,x,rates,v,vp=m.gd_fixture();A,Ap=m.total_two_rate_jacobian(m.TwoRateCuspMap(inp),x,rates,v,vp)
  assert A.nrows()==Ap.nrows()==13 and vp[0]!=0 and vp[1]!=0


def test_fixed_owned_packet_and_archive_closure():
 m=module();root=Path(__file__).parents[1]
 assert all((root/f).is_file()for f in m.FILES)
 text=(root/'tools/launch_native705_correlated.py').read_text()
 assert str(m.OUT.parent)in text and 'tools/probe_native705_correlated.py'in text
 assert "'internal_seconds':240"in text and "'external_seconds':255"in text
