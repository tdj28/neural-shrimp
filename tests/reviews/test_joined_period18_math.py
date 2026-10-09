"""Synthetic exact shooting and periodic-metric controls; no target arrays."""
import copy,importlib.util
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
import pytest
from flint import arb,arb_mat
from neural_shrimp.rigorous_orbits import ball_json,matrix_json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('joined_math',ROOT/'tools/reviews/review_joined_period18_math.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def b(x):
    x=Fraction(x);return ball_json(arb(x.numerator)/x.denominator)
def matrix(rows):return [[b(x) for x in row] for row in rows]
@lru_cache(None)
def fixture():
    p,n=18,13;N=p*n;w=Fraction(1,4096);r=Fraction(1,1048576);slope=Fraction(1,128);settings={'precision':256,'halfwidth_exponent':-12,'radius_exponent':-20,'newton_steps':2,'max_power':4096}
    A=[[0]*N for _ in range(N)];C=[[0]*N for _ in range(N)]
    for j in range(p):
        for i in range(n):A[j*n+i][((j+1)%p)*n+i]=-1;C[((j+1)%p)*n+i][j*n+i]=-1
    def chart(h):
        center=[[Fraction(j if i==0 else 0)+slope*(h-1) for i in range(n)] for j in range(p)]
        boxes=[[ball_json(arb(x.numerator)/x.denominator, ) for x in row] for row in center]
        for row,vals in zip(boxes,center):
            for i,x in enumerate(vals):row[i]=ball_json(arb(arb(x.numerator)/x.denominator,arb((abs(slope)*w+r).numerator)/(abs(slope)*w+r).denominator))
        endpoints=[sum(([b(x+sign*slope*w) for x in row] for row in center),[]) for sign in (-1,1)]
        return {'settings':settings,'parameter_center':str(h),'statement':{'period':p,'phase_dimension':n,'shooting_dimension':N},'gates':{'uniform_root':True,'primitive_period':True,'anchor_identified':True},'exact_parameter_endpoints':[str(h-w),str(h+w)],'parameter_interval':ball_json(arb(arb(h.numerator)/h.denominator,arb(w.numerator)/w.denominator)),'correction_radius':b(r),'center':matrix(center),'predictor_slope':matrix([[slope]]*N),'phase_boxes':boxes,'phase_jacobians':[matrix([[0]*n for _ in range(n)]) for _ in range(p)],'divisors':[{'divisor':d} for d in [1,2,3,6,9]],'preconditioner':matrix(C),'jacobian_tube':matrix(A),'parameter_partial_tube':matrix([[slope]]*N),'residual_center':matrix([[0]]*N),'image_correction':matrix([[0]]*N),'endpoint_root_enclosures':endpoints}
    charts=[chart(Fraction(1)),chart(1+2*w)];inp={'rate':'1','anchor':{'rate':'1','phase_boxes':charts[0]['center']}}
    return charts,inp,settings

def test_exact_two_chart_join_and_all234_coordinates():
    charts,inp,settings=fixture();out=m.audit_join(charts,inp,settings)
    assert out['passed'] and out['joined_endpoint_coordinates']==234
    assert out['exact_interval']==['4095/4096','4099/4096']

@pytest.mark.parametrize('mutation',['anchor','endpoint','rate','slope','primitive'])
def test_join_mutations_rejected(mutation):
    charts,inp,settings=copy.deepcopy(fixture())
    if mutation=='anchor':inp['anchor']['phase_boxes'][17][12]=b(1)
    elif mutation=='endpoint':charts[0]['endpoint_root_enclosures'][1][-1]=b(1)
    elif mutation=='rate':charts[1]['parameter_center']='2'
    elif mutation=='slope':charts[1]['predictor_slope'][233][0]=b(1)
    else:charts[0]['phase_boxes'][9]=charts[0]['phase_boxes'][0]
    with pytest.raises(ValueError):m.audit_join(charts,inp,settings)

def positive_record(n):return {'passed':True,'basis':matrix([[int(i==j) for j in range(n)] for i in range(n)]),'inverse':matrix([[int(i==j) for j in range(n)] for i in range(n)])}

def test_periodic_metric_full18_cyclic_decreases():
    n=13;I=[[int(i==j) for j in range(n)] for i in range(n)];J=matrix([[Fraction(1,2)*x for x in row] for row in I]);P=matrix(I)
    w={'passed':True,'point_passed':True,'phase_count':18,'dimension':13,'cyclic_P_final_index':0,'interval_J':[J]*18,'phases':[{'phase':j,'P':P,'P_positive':positive_record(n),'interval_D_positive':positive_record(n)} for j in range(18)]}
    assert m.audit_periodic(w)['passed']
    w['interval_J'][-1]=matrix([[2*x for x in row] for row in I])
    with pytest.raises(ValueError):m.audit_periodic(w)

def test_nonsymmetric_metric_rejected():
    n=13;P=matrix([[int(i==j) for j in range(n)] for i in range(n)]);P[0][1]=b(1)
    w={'passed':True,'point_passed':True,'phase_count':18,'dimension':13,'cyclic_P_final_index':0,'interval_J':[matrix([[0]*n for _ in range(n)])]*18,'phases':[{'phase':j,'P':P} for j in range(18)]}
    with pytest.raises(ValueError):m.audit_periodic(w)


def driver():
    spec=importlib.util.spec_from_file_location('joined_driver',ROOT/'tools/reviews/review_joined_period18_branch.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

@pytest.mark.parametrize('value',[-1,float('nan'),float('inf'),True])
def test_nonfinite_negative_and_bool_resource_values_rejected(value):
    with pytest.raises(ValueError):driver().bounded(value,120,'resource bound')

def test_closed_slices_cover_each_chart_without_gaps():
    d=driver();charts,_,_=fixture();coverage=d.expected_coverage(charts)
    assert len(coverage)==2 and all(len(c)==16 for c in coverage)
    for chart,parts in zip(charts,coverage):
        assert parts[0][0]==chart['exact_parameter_endpoints'][0]
        assert parts[-1][1]==chart['exact_parameter_endpoints'][1]
        assert all(a[1]==b[0] for a,b in zip(parts,parts[1:]))
    assert coverage[0][-1][1]==coverage[1][0][0]

def test_inventory_rejects_rehashed_extra_file_and_changed_payload(tmp_path):
    d=driver();(tmp_path/'x.json').write_text('{}');manifest={'files':d.inventory(tmp_path)}
    (tmp_path/'manifest.json').write_bytes(d.canonical_bytes(manifest));pin=d.sha(tmp_path/'manifest.json')
    assert d.verify(tmp_path,pin)==manifest
    (tmp_path/'extra').write_text('unlisted')
    with pytest.raises(ValueError):d.verify(tmp_path,pin)
    (tmp_path/'extra').unlink();(tmp_path/'x.json').write_text('changed')
    with pytest.raises(ValueError):d.verify(tmp_path,pin)

def test_source_inventories_include_implicit_imports_and_fixed_full_sets():
    d=driver();assert len(d.PARENT_FILES)==len(set(d.PARENT_FILES))==9
    assert len(d.SLICE_FILES)==len(set(d.SLICE_FILES))==20
    for name in ('src/neural_shrimp/controls.py','src/neural_shrimp/models.py'):
        assert name in d.FILES and name in d.SLICE_FILES
