import sys
from pathlib import Path
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import vector_json,matrix_json
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools/reviews')]
import review_census13_ad_math as A

def run(period,inverse):
    with ctx.workprec(128):
        proposal=dict(center=vector_json([arb(0)]),box=vector_json([arb(0,arb((1,-20)))]),preconditioner=matrix_json(arb_mat([[inverse]])))
        return A.prove(dict(period=period),proposal,lambda x:([-x[0]/2],arb_mat([[-arb(1)/2]])))
def test_same_box_contraction_and_attraction():assert run(1,arb(-.6666666666666666))['passed']
def test_nonprimitive_return_rejected():
    r=run(2,arb(-1.3333333333333333));assert r['gates']['existence_and_local_uniqueness']and not r['gates']['primitive_period']
def test_singular_preconditioner_rejected():assert not run(1,arb(0))['gates']['existence_and_local_uniqueness']
