import copy,sys
from pathlib import Path
from flint import arb,arb_mat
import pytest
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json
from neural_shrimp.cubic_batch_diagnostics import diagnose

def test_constant_residual_distinguished_from_quartic_width_and_gauge():
 z=vector_json([arb(0)]*3);C=[[[[1,0],[0,0],[0,0]],[[0,0],[1,0],[0,0]],[[0,0],[0,0],[1,0]]],[[[0,0]]*3 for _ in range(3)]]
 p={'shooting_dimension':3,'weights':['1']*3,'settings':{'radius_exponent':-10,'halfwidth_exponent':-4},'slope':vector_json([arb(1),arb(0),arb(0)]),'curvature':z,'cubic':z,'normal':['1','0','0'],'center':z,'inverse_polynomial_exact':C,'residual_coefficients':[matrix_json(arb_mat([[arb(1)if k==0 else arb(0)],[0],[0]]))for k in range(5)],'parameter_center':'0','bound':{'contraction':ball_json(arb(0)),'coordinate_inclusions':[False]*3,'image_radius_ratio_upper':vector_json([arb(1024)]*3)}}
 rows=[{'iteration':i,'correction':matrix_json(arb_mat([[1],[0],[0]])),'value':matrix_json(arb_mat([[1],[0],[0]])),'condensed_border':matrix_json(arb_mat([[1,0],[0,2]]))}for i in range(3)]
 d=diagnose(p,rows);assert d['forcing_orders'][0]['forcing_proposals'][0]['scaled_midpoint_max']==1024 and d['forcing_orders'][4]['forcing_proposals'][0]['scaled_midpoint_max']==0 and d['newton_steps'][-1]['weighted_correction_in_radii']==1024 and d['gauge_alignment_cosine']==1
 assert 'Noncertified'in d['scope']and 'passed'not in d
 p['weights'][0]='0'
 with pytest.raises(ValueError):diagnose(p,rows)

def test_source_closure_and_exclusive_roots():
 sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
 import diagnose_native705_cubic_batch as D
 import launch_native705_cubic_batch_diagnosis as L
 assert D.BASE==L.BASE and D.OUT==L.BASE/'attempt001-diagnosis/run'
 assert all((D.ROOT/n).is_file()for n in D.SOURCES)
