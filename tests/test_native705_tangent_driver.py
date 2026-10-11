import importlib.util,sys
from pathlib import Path
import pytest
from flint import arb,ctx


def module():
 p=Path(__file__).parents[1]/'tools/refine_native705_tangent.py';s=importlib.util.spec_from_file_location('native705_tangent',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def fixture(m):
 c=[arb(0)]*702+[arb(2),arb(3),arb('1/128')];z=[arb(0)]*705
 args={'center':c,'slope':z,'curvature':z,'halfwidth':arb('1/128')}
 error=[arb(0,arb('1/1024'))for _ in range(705)]
 proof={'passed':True,'dimension':705,'weighted_neumann_bound':m.D.ball_json(arb('1/4')),'whole_refinement':{'box':[[m.D.ball_json(v)]for v in error]}}
 original=[m.D.ball_json(v+arb(0,arb('1/512')))for v in c]
 return args,proof,original


def test_full_known_segment_and_balanced_directions():
 m=module()
 with ctx.workprec(384):
  args,proof,old=fixture(m);segment,error,pprime,plus,minus=m.segment_inputs(args,proof,old)
  assert len(segment)==len(error)==len(pprime)==len(plus)==len(minus)==705
  assert all(s.contains(c)for s,c in zip(segment,args['center']))
  assert all(s.contains(c+e)for s,c,e in zip(segment,args['center'],error))
  assert segment[-3]>0 and segment[-2]>0 and segment[-1]>0


def test_exact_original_domain_does_not_expand_to_admit_segment():
 m=module();a,p,old=fixture(m);old[3]=m.D.ball_json(arb(0,arb('1/2048')))
 with pytest.raises(ValueError,match='exact original domain'):m.segment_inputs(a,p,old)
 a,p,old=fixture(m);p['whole_refinement']['box'].pop()
 with pytest.raises(ValueError):m.segment_inputs(a,p,old)


def test_synthetic_native_dimensions_and_ownership_source_closure():
 m=module();model,flat,plus,minus,v,n,o,s,inp=m.synthetic_inputs()
 assert model.n==13 and len(inp['x'])==24 and len(flat)==len(plus)==len(minus)==len(v)==len(n)==len(o)==705
 assert all(x>0 for x in flat[-3:])and m.SCALE.is_exact()
 root=Path(__file__).parents[1].resolve()
 modules={str(Path(x.__file__).resolve().relative_to(root))for x in list(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).resolve().is_relative_to(root)and '/tests/'not in str(Path(x.__file__).resolve())}
 assert modules<=set(m.FILES)
 assert all((root/f).is_file()for f in m.FILES)
 text=(root/'tools/launch_native705_tangent.py').read_text()
 assert str(m.OUT.parent)in text and 'tools/refine_native705_tangent.py'in text and "'internal_seconds':240"in text
