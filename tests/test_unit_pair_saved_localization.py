import importlib.util
from pathlib import Path
from flint import ctx
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/localize_native_unit_pair_saved.py';s=importlib.util.spec_from_file_location('saved_localization',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_unrelated_dense_profile_has_all_coordinates_and_no_native_calls():
 m=module()
 with ctx.workprec(384):
  r=m.synthetic_localization(12,lambda:None);assert r['passed']and len(r['center_box'])==len(r['whole_box'])==len(r['tangent_box'])==12
  assert m.U.R.original_ball(r['weighted_neumann_bound'])<1

def test_wrong_native_dimension_stops_before_acquisition_replay():
 m=module()
 with pytest.raises(ValueError,match='convention'):m.native_inputs({'schema':'neural-shrimp-native-unit-pair-tube-v1','dimension':13,'period':18,'shooting_dimension':703},[],lambda:None)

def test_wrong_original_domain_stops_before_acquisition_replay():
 m=module();r={'schema':'neural-shrimp-native-unit-pair-tube-v1','dimension':13,'period':18,'shooting_dimension':704,'settings':{'precision':256,'newton_steps':2,'halfwidth_exponent':-16,'radius_exponent':-11},'parameter_center':'1/32768','parameter_endpoints':['0','1/32768']}
 with pytest.raises(ValueError,match='geometry'):m.native_inputs(r,[],lambda:None)
