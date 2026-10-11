"""Independent analytic coupling controls and audit tamper checks; synthetic only."""
import importlib.util,json,hashlib
from fractions import Fraction
from pathlib import Path
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_flips import _proof
from neural_shrimp.rigorous_orbits import ball_json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('flip_review',ROOT/'tools/reviews/review_rigorous_period18_flip.py');review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)

class CoupledFlip:
    n=2
    def __init__(self,c,transform=False):self.c,self.transform=c,transform
    def jet_step(self,z,s):
        # y'=r*y+d*x². Center-manifold y=d/(1-r)*x²+..., hence
        # b=c+a²+e*d/(1-r); this contribution is absent in uncoupled controls.
        x=(z[0]-z[1])*(arb(1)/2) if self.transform else z[0]
        y=z[1]
        f=-(1+s)*x+arb(1)/2*x*x+self.c*x*x*x+arb(7)/10*x*y
        g=arb(1)/4*y+arb(3)/10*x*x
        return [2*f+g,g] if self.transform else [f,g]

@pytest.mark.parametrize('c,expected',[(1,Fraction(153,100)),(-1,Fraction(-47,100))])
@pytest.mark.parametrize('transform',[False,True])
@pytest.mark.parametrize('sign',[-1,1])
def test_center_manifold_coupling_exact_normal_form(c,expected,transform,sign):
    with ctx.workprec(256):
        out=_proof(CoupledFlip(c,transform),[arb(0),arb(0),arb(0),arb(sign),arb(0)],1,-90,0,128)
        assert out['passed']
        b=expected/(4 if transform else 1)
        assert review.stored(out['cubic_b']).contains(arb(b.numerator)/b.denominator)
        assert review.stored(out['alpha']).contains(-1)
        assert out['criticality']==('supercritical' if b>0 else 'subcritical')
        assert out['doubled_side_in_s']==('greater' if b>0 else 'less')

@pytest.mark.parametrize('value',[{'mid':[1,0],'rad':[-1,-3]},{'mid':[True,0],'rad':[0,0]},{'special':'nan'}])
def test_ball_decoder_rejects_invalid_inputs(value):
    with pytest.raises(ValueError):review.interval(value)

def test_nested_box_requires_strict_containment():
    inner={'mid':[0,0],'rad':[1,-10]};outer={'mid':[0,0],'rad':[1,-9]}
    assert review.nested([inner],[outer])
    assert not review.nested([outer],[outer])

def test_complete_inventory_and_external_hash_reject_mutations(tmp_path):
    (tmp_path/'x').write_text('one')
    original={'files':review.inventory(tmp_path),'total_bytes':3,'within_limit':True}
    (tmp_path/'manifest.json').write_text(json.dumps(original));pin=review.sha(tmp_path/'manifest.json')
    review.verify(tmp_path,pin,True)
    (tmp_path/'x').write_text('two')
    with pytest.raises(ValueError):review.verify(tmp_path,pin,True)
    original['files']=review.inventory(tmp_path);(tmp_path/'manifest.json').write_text(json.dumps(original))
    with pytest.raises(ValueError):review.verify(tmp_path,pin,True)

def test_exact_infinity_norm_does_not_ignore_radius():
    values=[[{'mid':[1,0],'rad':[1,-2]},{'mid':[-2,0],'rad':[1,-3]}]]
    assert review.exact_norm(values)==Fraction(27,8)


def test_legacy_numeric_manifest_requires_explicit_schema(tmp_path):
    (tmp_path/'x').write_text('one')
    manifest={'files':review.inventory(tmp_path),'total_bytes':3}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));pin=review.sha(tmp_path/'manifest.json')
    with pytest.raises(KeyError):review.verify(tmp_path,pin,True)
    review.verify(tmp_path,pin,True,limit_flag=False)
    manifest['total_bytes']=2;(tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):review.verify(tmp_path,review.sha(tmp_path/'manifest.json'),True,limit_flag=False)
