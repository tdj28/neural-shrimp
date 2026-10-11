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
