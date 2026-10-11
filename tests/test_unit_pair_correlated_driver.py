import importlib.util,hashlib,subprocess
from pathlib import Path
from flint import ctx

def module():
 p=Path(__file__).parents[1]/'tools/probe_native_unit_pair_correlated.py';s=importlib.util.spec_from_file_location('correlated_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_full_unrelated_correlated_kernel_passes():
 m=module()
 with ctx.workprec(384):
  r=m.synthetic_cocycle(lambda:None);assert r['passed']and r['dimension']==13 and r['period']==18 and r['complement_dimension']==11

def test_unrelated_native_profile_full_dimension_and_both_rate_tangents():
 m=module();inp,x,rates,v,vp=m.gd_fixture()
 with ctx.workprec(384):
  model=m.TwoRateCuspMap(inp);a,ap=m.total_two_rate_jacobian(model,x,rates,v,vp)
  assert len(inp['x'])==24 and a.nrows()==a.ncols()==ap.nrows()==ap.ncols()==13 and vp[0]!=0 and vp[1]!=0

def test_copied_analytic_two_rate_model_is_literal_original():
 root=Path(__file__).parents[1];p='src/neural_shrimp/rigorous_cusps.py'
 assert hashlib.sha256((root/p).read_bytes()).digest()==hashlib.sha256(subprocess.check_output(['git','show','a7b5a169:'+p],cwd=root)).digest()
