import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
from flint import arb,arb_mat,ctx
import review_eigenpair_reanchor as R
from neural_shrimp.rigorous_orbits import vector_json

def test_correlated_projection_retains_whole_box():
 c={'normal':['1','2'],'origin':['3','4']};a={'box':vector_json([arb(3,1),arb(4,2)])};p=R.E.anchor_parameter(c,a)
 assert p.contains(-5) and p.contains(5) and p.rad()>0

def test_exact_inherited_gauge_remains_exact():
 c={'normal':['1'],'origin':['0']};a={'box':vector_json([arb(1,1)]),'exact_parameter':'1/3'}
 assert R.E.anchor_parameter(c,a).contains(arb(1)/3)

def test_known_root_localization_needs_no_inverse():
 c=[arb(3)];g=arb_mat([[3]]);A=arb_mat([[1]]);C=arb_mat([[0]]);box=[arb(0,1)]
 image,_=R.local_image(c,g,A,C,box);assert image[0,0].contains(box[0])

def test_localization_center_outside_box_uses_displacement():
 c=[arb(3)];g=arb_mat([[3]]);A=arb_mat([[1]]);C=arb_mat([[arb(1)/2]]);box=[arb(0,1)]
 image,_=R.local_image(c,g,A,C,box);assert image[0,0].contains(0) and image[0,0].rad()<1

def test_localization_retains_nonzero_root():
 c=[arb(5)];g=arb_mat([[8]]);A=arb_mat([[2]]);C=arb_mat([[arb(1)/4]]);box=[arb(1,1)]
 image,_=R.local_image(c,g,A,C,box);assert image[0,0].contains(1)
