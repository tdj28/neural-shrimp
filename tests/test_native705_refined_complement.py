import importlib.util,sys,copy
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx


def module():
 p=Path(__file__).parents[1]/'tools/probe_native705_refined_complement.py';s=importlib.util.spec_from_file_location('refined705_complement',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture(m,j=0):
 center=[arb(i)for i in range(704)];a={'kind':'center','phase':j,'state':m.vector_json(center[39*j:39*j+13]),'rates':m.vector_json(center[-2:]),'state_tangent':m.vector_json([arb(0)]*13),'rate_tangent':m.vector_json([arb(0)]*2)}
 v={'kind':'center','phase':j,'A':m.matrix_json(m.eye(13)),'Aprime':m.matrix_json(arb_mat(13,13))};return a,v,center


def test_retained_derivative_requires_exact_same_state_rates_and_zero_tangent():
 m=module()
 for j in [0,17]:
  a,v,c=fixture(m,j);assert m.retained_center(a,v,j,c)==m.eye(13)
 for field in ['state','rates','state_tangent','rate_tangent']:
  a,v,c=fixture(m);a[field][-1]=m.D.ball_json(arb(-99))
  with pytest.raises(ValueError):m.retained_center(a,v,0,c)
 a,v,c=fixture(m);v['phase']=1
 with pytest.raises(ValueError):m.retained_center(a,v,0,c)


def test_full_profile_and_source_archive_closure():
 m=module();root=Path(__file__).parents[1].resolve()
 with ctx.workprec(384):assert m.synthetic_cocycle(lambda:None)['passed']
 modules={str(Path(x.__file__).resolve().relative_to(root))for x in list(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).resolve().is_relative_to(root)and '/tests/'not in str(Path(x.__file__).resolve())}
 assert modules<=set(m.FILES)and all((root/n).is_file()for n in m.FILES)
 text=(root/'tools/launch_native705_refined_complement.py').read_text()
 assert str(m.OUT.parent)in text and 'tools/probe_native705_refined_complement.py'in text
