import importlib.util
from pathlib import Path
import pytest
from flint import ctx

def module():
 p=Path(__file__).parents[1]/'tools/localize_native705_saved.py';s=importlib.util.spec_from_file_location('pal_localization',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_dense_synthetic_localization_and_dimension_cap():
 m=module()
 with ctx.workprec(384):
  r=m.synthetic(lambda:None,12);assert r['passed']and len(r['tangent_box'])==12
  with pytest.raises(ValueError,match='bounded'):m.synthetic(lambda:None,706)

def test_wrong_native_dimension_fails_before_arrays():
 m=module()
 with pytest.raises(ValueError,match='native chart'):
  m.chart_inputs({'schema':'neural-shrimp-native-unit-pair-pal-v1','dimension':13,'period':18,'shooting_dimension':704})

def test_fixed_connected_domain_cannot_change():
 m=module()
 with pytest.raises(ValueError,match='quadratic interval'):
  m.chart_inputs({'schema':'neural-shrimp-native-unit-pair-pal-v1','dimension':13,'period':18,'shooting_dimension':705,'settings':{'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'parameter_center':'32/4096','parameter_endpoints':['0','64/4096']})
