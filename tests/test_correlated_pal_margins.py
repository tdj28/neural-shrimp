from copy import deepcopy
import pytest
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import matrix_json,vector_json
from neural_shrimp.correlated_pal_margins import margins

def fixture():
 D=[arb_mat([[arb(1)/64,0],[0,0]]),arb_mat([[0,1],[0,0]]),arb_mat(2,2),arb_mat(2,2)]
 F=[arb_mat([[arb(1)/1024],[0]])]+[arb_mat(2,1)for _ in range(4)]
 b={'passed':True,'dimension':2,'coordinate_inclusions':[True]*2,'radii':vector_json([arb(1)/64]*2),'defect_coefficients':[matrix_json(m)for m in D],'forcing_coefficients':[matrix_json(m)for m in F],'whole_image':matrix_json(arb_mat([[arb(1)/128],[0]]))}
 return {'passed':True,'gates':{'full':True},'shooting_dimension':2,'bound':b,'settings':{'halfwidth_exponent':-3},'weights':['1','1']}

def test_actual_saved_decomposition_and_extrapolation_scope():
 r=margins(fixture());assert r['current_max_image_radius_ratio']==.5 and r['current_defect_terms']==[1/64,1/8,0,0]
 assert r['current_forcing_terms']==[1/16,0,0,0,0]
 assert r['frozen_coefficient_forecasts'][1]['sum_individual_defect_norms']==1/64+1/4
 assert 'not bounds'in r['scope']and 'fresh full acquisitions'in r['scope']

def test_failed_or_missing_full_proof_never_admitted():
 for key in ['passed','gates','weights']:
  p=fixture();p[key]=False if key=='passed'else {'full':False}if key=='gates'else ['0','1']
  with pytest.raises(ValueError):margins(p)
 p=fixture();p['bound']['coordinate_inclusions']=[True]
 with pytest.raises(ValueError):margins(p)
