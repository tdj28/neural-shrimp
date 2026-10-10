import importlib.util
from pathlib import Path
from copy import deepcopy
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import ball_json,exact
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('probe_native_unit_pair_pal');L=load('launch_native_unit_pair_pal')

def profile():
 out={'passed':True,'target_acquired':False,'ticks':590,'events':{'direction_return':348,'pal_path_phase_return':216,'pal_profile_path_return':24,'pal_chart_return':2},'elapsed_seconds':5.,'peak_rss_kib':100000,'chart_times':[1.,2.],'phase_times':[.01]*12,'path_times':[.01]*12,'stored_bytes_before_final':1000}
 out['projection']=P.P.projection(out['chart_times'],out['phase_times'],out['path_times'],out['peak_rss_kib'],out['stored_bytes_before_final']);return out

def test_profile_recomputed_counts_and_math_resources():
 assert P.admit_profile(profile())['resource_admitted']
 for mutate in [lambda x:x.update(ticks=589),lambda x:x['events'].update(pal_path_phase_return=215),lambda x:x['projection'].update(one_quadratic_chart_seconds=0),lambda x:x.update(target_acquired=True)]:
  x=profile();mutate(x)
  with pytest.raises(ValueError):P.admit_profile(x)

def test_full_independent_endpoint_and_corruptions():
 b=ball_json(arb(0));proof={'passed':True,'shooting_dimension':704,'dimension':13,'period':18,'parameter_endpoints':['0','1/32768']}
 audit={'passed':True,'inclusions':704,'anchor_coordinates':704,'precision':384,'gates':{'root':True},'recomputed_endpoints':[[b]*704,[b]*704]}
 out={'passed':True,'strict_inclusions':704,'anchor_coordinates':704,'primitive_period':18,'same_independently_connected_root':True,'parameter_endpoints':['0','1/32768']}
 assert len(P.admit_endpoint(proof,audit,out))==705 and P.admit_endpoint(proof,audit,out)[-1]==exact('1/32768')
 for mutate in [lambda x:x['recomputed_endpoints'][1].pop(),lambda x:x.update(precision=256),lambda x:x['gates'].update(root=False)]:
  a=deepcopy(audit);mutate(a)
  with pytest.raises(ValueError):P.admit_endpoint(proof,a,out)

def test_seed_retains_full_endpoint_and_fixed_angle_weight():
 with ctx.workprec(256):
  anchor=[arb(0)]*702+[arb(1),arb(1),exact('1/32768')]
  proof={'slope':[[ball_json(arb((i%3)-1))]for i in range(704)],'weights':['1/2']*704}
  initial,normal,origin,weights,proposal=P.chart_seed(anchor,proof)
  assert len(initial)==len(normal)==len(origin)==len(weights)==705
  assert weights[:-1]==proof['weights']and weights[-1]=='1/512'
  assert proposal['known_endpoint']==[ball_json(v)for v in anchor]
  assert sum(exact(a)*exact(b)for a,b in zip(normal,[P.exact_mid(v)for v in anchor])) .is_finite()

def test_unique_owned_path_and_fixed_forecast_operation_count():
 assert P.BASE==L.BASE and P.BASE.name=='pal'
 assert P.SETTINGS=={'period':18,'precision':256,'parameter':'63/4096','halfwidth_exponent':-6,'radius_exponent':-10,'newton_steps':2}
 assert P.PARENT==P.BASE.parent/'attempt002-target'
 assert 'probe_native_unit_pair_pal.py'in (ROOT/'tools/launch_native_unit_pair_pal.py').read_text()


def test_owned_launcher_help_entrypoint():
 import subprocess,sys
 result=subprocess.run([sys.executable,str(ROOT/'tools/launch_native_unit_pair_pal.py'),'--help'],capture_output=True,text=True,timeout=15)
 assert result.returncode==0,result.stderr
 assert '--profile-manifest'in result.stdout and '--profile-outer-manifest'in result.stdout


def test_higher_precision_anchor_only_rounds_arbitrary_gauge_origin():
 with ctx.workprec(384):
  high=arb(1)+arb((1,-350));anchor=[high]*704+[exact('1/32768')]
  saved=[ball_json(v)for v in anchor]
 with ctx.workprec(256):
  proof={'slope':[[ball_json(arb(1))]for _ in range(704)],'weights':['1/2']*704}
  _,_,origin,_,proposal=P.chart_seed(anchor,proof)
  assert all(exact(v).is_exact()for v in origin)
  assert origin[0]=='1' and proposal['known_endpoint']==saved
