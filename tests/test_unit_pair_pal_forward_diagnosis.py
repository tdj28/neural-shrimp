import numpy as np
import pytest
from neural_shrimp.unit_pair_pal_forward_diagnosis import project_geometry,diagnose

def test_geometry_retains_all705_and_projection_turn_is_only_proposal():
 p=[{'native_flat':[0.]*704,'t':x,'diagnostics':{'complement_max_modulus':.5}}for x in [1.,2.,1.]]
 n=np.zeros(705);n[-1]=1
 r=project_geometry(p,n,np.zeros(705));assert [v['gauge_projection']for v in r['points']]==[1,2,1]and r['projection_reversal_proposals']==[1]
 p[0]['native_flat']=p[0]['native_flat'][:-1]
 with pytest.raises(ValueError):project_geometry(p,n,np.zeros(705))

def test_diagnostic_rejects_missing_full_uniform_theorem():
 with pytest.raises(ValueError):diagnose({'schema':'neural-shrimp-native-unit-pair-pal-v1','shooting_dimension':705,'gates':{},'coordinate_inclusions':[True]*705})


def test_successful_full705_zero_defect_retains_all_groups():
 from test_unit_pair_pal_conditioning import chart
 c=chart();c['gates']={'anchor_identified':False,'primitive_period':True,'realified_frame_independent':True,'uniform_full_native_unit_pair_locus':True};c['coordinate_inclusions']=[True]*705
 result=diagnose(c)
 assert result['scaling']['collatz_max']==0 and result['resolvent_proposal']['nonnegative']
 assert sum(row['strict_inclusions']for row in result['groups'].values())==705
 assert result['groups']['angle']['coordinates']==1
