import copy,importlib.util,subprocess,sys
from pathlib import Path
from fractions import Fraction
from flint import arb,ctx
import pytest
from neural_shrimp.rigorous_orbits import vector_json
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_native705_cubic_batch as W
import launch_native705_cubic_batch as L

def chart():
 z=vector_json([arb(0)]*705);o=vector_json([arb(1)]*705)
 return {'dimension':13,'period':18,'shooting_dimension':705,'normal':['0']*704+['1'],'origin':['0']*705,'parameter_center':'0','parameter_endpoints':['-1/16','1/16'],'settings':{'halfwidth_exponent':-4,'radius_exponent':-10},'center':copy.deepcopy(z),'slope':copy.deepcopy(z),'curvature':copy.deepcopy(z),'cubic':copy.deepcopy(z),'weights':['1']*705,'bound':{'passed':True,'coordinate_inclusions':[True]*705,'radii':vector_json([arb((1,-10))]*705)},'gates':{'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'whole_positive_angle':True,'anchor_identified':False}}

def test_full705_join_rejects_phase_mode_rate_angle_corruptions():
 with ctx.workprec(256):
  p=chart();a={'box':vector_json([arb(0)]*705),'normal':p['normal'],'origin':p['origin'],'parameter':'0'}
  assert W.join(a,p)['passed']
  for i in [0,13,26,39,702,703,704]:
   b=copy.deepcopy(a);b['box'][i]=vector_json([arb(1)])[0]
   assert not W.join(b,p)['passed']
  b=copy.deepcopy(a);b['parameter']='1/8';assert not W.join(b,p)['passed']
  b=copy.deepcopy(a);b['normal'][0]='1'
  with pytest.raises(ValueError):W.join(b,p)

def test_uniform_admission_requires_all705_and_proper_scopes():
 p=chart();assert W.uniform(p)
 for mutate in [lambda q:q['bound']['coordinate_inclusions'].pop(),lambda q:q['bound']['coordinate_inclusions'].__setitem__(200,False),lambda q:q['gates'].__setitem__('anchor_identified',True),lambda q:q['gates'].__setitem__('primitive_period',False)]:
  q=copy.deepcopy(p);mutate(q);assert not W.uniform(q)

def test_native_overlaps_advance_and_contain_prior_endpoint():
 right=Fraction(79,2048)
 for i in range(1,5):
  c=Fraction(W.settings(str(Fraction(79,2048)),i)['parameter']);assert c-Fraction(1,16)<right<c+Fraction(1,16);right=c+Fraction(1,16)
 assert right==Fraction(1099,2048)
 with pytest.raises(ValueError):W.settings('0',5)

def test_exact_complete_synthetic_profile_counts_match_native_budget():
 e={k:2*v for k,v in W.EXPECTED.items()};e['direction_return']+=36
 assert e['direction_return']==252 and e['pal_path_phase_return']==144 and e['gradient_path_return']==e['cubic_path_return']==144
 assert 252+144+144+144+48+14==746
 assert sum(W.EXPECTED[k]for k in ['direction_return','pal_path_phase_return','gradient_path_return','cubic_path_return'])+7==331

def test_sources_entrypoints_and_parent_hard_limit_are_coherent():
 assert all((ROOT/n).is_file()for n in W.SOURCES)
 assert W.BASE==L.BASE and len(W.SOURCES)==len(set(W.SOURCES))
 assert '(6*1024**3,12*1024**3)'in(ROOT/'tools/launch_native705_cubic_batch.py').read_text()
 for name in ['probe_native705_cubic_batch.py','launch_native705_cubic_batch.py']:
  out=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True);assert out.returncode==0 and 'usage:'in out.stdout

def test_seed_is_only_finite_proposal_and_uses_all_cubic_coordinates():
 p=chart();p['slope'][0]=vector_json([arb(2)])[0];p['cubic'][704]=vector_json([arb(3)])[0]
 a={'normal':p['normal'],'origin':p['origin'],'parameter':'0'}
 with ctx.workprec(256):
  s=L.seeds(p,a);assert len(s)==4 and all(len(x)==705 for x in s)
  d=Fraction(W.settings('0',1)['parameter']);assert Fraction(s[0][0])==2*d and Fraction(s[0][704])==3*d**3

def test_card_rejects_missing_coordinates_changed_gauge_and_domain():
 p=chart();anchor={'normal':p['normal'],'origin':p['origin'],'box':p['center'],'parameter':'79/2048'}
 c={'source':'fixed','batch':1,'worker_count':4,'kernel':'fresh-cubic705-v1','settings':[W.settings('79/2048',i)for i in range(1,5)],'normal':p['normal'],'origin':p['origin'],'weights':p['weights'],'initials':[['0']*705 for _ in range(4)],'anchor':anchor}
 assert W.validate_card(c,'fixed',1)==c
 for mutate in [lambda q:q['initials'][1].pop(),lambda q:q['settings'][0].__setitem__('halfwidth_exponent',-3),lambda q:q['weights'].__setitem__(0,'0')]:
  q=copy.deepcopy(c);mutate(q)
  with pytest.raises(ValueError):W.validate_card(q,'fixed',1)
