import pytest
from neural_shrimp.unit_pair_pal_conditioning import diagnose

def ball(x):return {'mid':[x,0],'rad':[0,0]}
def chart():
 z=ball(0);o=ball(1)
 return {'schema':'neural-shrimp-native-unit-pair-pal-v1','shooting_dimension':705,'gates':{'anchor_identified':True,'primitive_period':True,'realified_frame_independent':True,'uniform_full_native_unit_pair_locus':False},'defect':[[z]*705 for _ in range(705)],'preconditioner':[[o if i==j else z for j in range(705)]for i in range(705)],'predictor_residual':[[z]for _ in range(705)],'coordinate_radii':[o]*705,'coordinate_inclusions':[False]*704+[True]}

def test_full705_angle_separate_and_zero_resolvent():
 r=diagnose(chart());assert r['groups']['angle']['coordinates']==1 and r['groups']['angle']['strict_inclusions']==1
 assert sum(v['coordinates']for v in r['groups'].values())==705
 assert r['resolvent_proposal']['nonnegative'] and max(r['resolvent_proposal']['radii'])==0

def test_reject_wrong_schema_dimension_or_proof_scope():
 for k,v in [('schema','other'),('shooting_dimension',704),('gates',{})]:
  c=chart();c[k]=v
  with pytest.raises(ValueError):diagnose(c)

def test_fixed_saved_paths_and_no_native_model_call():
 from pathlib import Path
 root=Path(__file__).parents[1];w=(root/'tools/diagnose_native_unit_pair_pal.py').read_text();l=(root/'tools/launch_native_unit_pair_pal_diagnosis.py').read_text()
 assert "PARENT=BASE/'attempt002-target'"in w and "attempt002-diagnosis"in w and 'attempt002-diagnosis'in l
 assert 'quadratic_tube('not in w and 'TwoRateCuspMap('not in w


def test_reused_conditioning_kernel_in_archived_source_closure():
 from pathlib import Path
 s=(Path(__file__).parents[1]/'tools/diagnose_native_unit_pair_pal.py').read_text()
 assert "'src/neural_shrimp/unit_pair_conditioning.py'"in s
 assert "'tests/test_unit_pair_conditioning.py'"in s
