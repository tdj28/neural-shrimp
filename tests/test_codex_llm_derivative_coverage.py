"""Pure synthetic records only: no receipt reader, checkpoint or model call."""
import copy
from fractions import Fraction
from pathlib import Path
import sys

from flint import arb,arb_mat,ctx
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import derivative_coverage as collect
import phase_domain as domain
import shooting_gates as gates


def packed(mid=0,rad=0):
    return dict(mid=domain._pair(Fraction(mid)),rad=domain._pair(Fraction(rad)))


def fixture(p=2,n=3,high_center=False):
    centers=[[packed(i+j) for j in range(n)] for i in range(p)]
    if high_center:centers[0][0]=packed(1+Fraction(1,2**1700))
    radii=[packed(Fraction(1,2**30)) for _ in range(p)]
    bindings=dict(expected_design_sha256='a'*64,expected_map_sha256='b'*64,
                  expected_evaluator_sha256='c'*64,expected_backend_sha256='d'*64,expected_precision=64)
    _,_,mh,sh=collect._nominal(centers,radii,p,n)
    shards=[]
    for i in range(p):
        for indices in [list(range(0,n,2)),list(range(1,n,2))]:
            if not indices:continue
            actual=[packed(domain._fraction(c['mid']),domain._fraction(radii[i]['mid'])) for c in centers[i]]
            record=dict(role=collect.ROLE,phase_index=i,direction_indices=indices,precision=64,
                design_sha256='a'*64,map_sha256='b'*64,evaluator_sha256='c'*64,backend_sha256='d'*64,
                phase_domain_sha256=mh,shooting_domain_sha256=sh,actual_inputs=actual,
                actual_inputs_sha256=domain.canonical_hash(actual),
                directions=[[packed(int(row==j)) for j in indices] for row in range(n)],
                derivatives=[[packed((i+1)*(row+1)-2*j,Fraction(1,2**50)) for j in indices] for row in range(n)],
                primal=[packed(-row,Fraction(1,2**40)) for row in range(n)])
            shards.append(record)
    args=dict(period=p,dimension=n,**bindings)
    return centers,radii,shards,args


@pytest.fixture(autouse=True)
def cpu_context():
    threads=ctx.threads;ctx.threads=1
    with ctx.workprec(128):yield
    ctx.threads=threads


def test_complete_all_rows_columns_shuffled_shards_and_signed_values():
    c,r,shards,args=fixture()
    matrices,report=collect.collect_derivatives(c,r,reversed(shards),**args)
    assert report['coverage_complete'] and report['phase_column_counts']==[3,3]
    assert report['total_columns']==6 and report['certificate_passed'] is False
    assert report['external_authentication_performed'] is False
    assert report['point_images_supplied'] is False and report['root_gate_evaluated'] is False
    assert report['source_precision']==64 and report['assembly_precision']==128
    for i,matrix in enumerate(matrices):
        for row in range(3):
            for col in range(3):assert matrix[row,col].contains((i+1)*(row+1)-2*col)
    assert report['phase_matrix_sha256']==[gates._matrix_sha256(m) for m in matrices]
    assert all(a['original_endpoints_contain_nominal'] for a in report['shards'])


def test_exact_high_precision_centers_survive_low_precision_domain_hashing():
    with ctx.workprec(32):
        c,r,s,args=fixture(high_center=True)
        _,report=collect.collect_derivatives(c,r,s,**args)
    exact=[[arb(tuple(x['mid'])) for x in row] for row in c]
    exactr=[arb(tuple(x['mid'])) for x in r]
    assert report['phase_domain_sha256']==domain.canonical_hash(dict(period=2,dimension=3,centers=c,radii=r))
    assert report['shooting_domain_sha256']==gates.domain_sha256(exact,exactr)
    assert report['phase_domain_sha256']!=report['shooting_domain_sha256']


@pytest.mark.parametrize('fault',['prototype','missing_role','design','map','bundle','backend','precision','precision_bool',
    'mixed_hash','shooting_hash','actual_hash','phase_bool','phase_large','indices_bool','indices_duplicate','indices_unsorted',
    'indices_outside','empty_indices','missing_input','narrow_input','shifted_input','direction_rows','direction_columns',
    'negative_basis','basis_radius','derivative_rows','derivative_columns','missing_primal','negative_radius',
    'float_mid','infinite_mid','bool_mid','noncanonical_dyadic'])
def test_refuses_wrong_bindings_domains_directions_and_outputs(fault):
    c,r,s,args=fixture();v=s[0]
    if fault=='prototype':v['role']='prototype_box'
    if fault=='missing_role':v.pop('role')
    if fault in ['design','map','backend']:v[fault+'_sha256']='e'*64
    if fault=='bundle':v['evaluator_sha256']='e'*64
    if fault=='precision':v['precision']=1024
    if fault=='precision_bool':v['precision']=True
    if fault=='mixed_hash':v['phase_domain_sha256']='e'*64
    if fault=='shooting_hash':v['shooting_domain_sha256']='e'*64
    if fault=='actual_hash':v['actual_inputs_sha256']='e'*64
    if fault=='phase_bool':v['phase_index']=False
    if fault=='phase_large':v['phase_index']=2
    if fault=='indices_bool':v['direction_indices'][0]=False
    if fault=='indices_duplicate':v['direction_indices']=[0,0]
    if fault=='indices_unsorted':v['direction_indices']=[2,0]
    if fault=='indices_outside':v['direction_indices']=[0,3]
    if fault=='empty_indices':v['direction_indices']=[]
    if fault=='missing_input':v['actual_inputs'].pop()
    if fault=='narrow_input':v['actual_inputs'][0]['rad']=[1,-31]
    if fault=='shifted_input':v['actual_inputs'][0]['mid']=[1,-29]
    if fault in ['narrow_input','shifted_input']:v['actual_inputs_sha256']=domain.canonical_hash(v['actual_inputs'])
    if fault=='direction_rows':v['directions'].pop()
    if fault=='direction_columns':v['directions'][0].pop()
    if fault=='negative_basis':v['directions'][0][0]=packed(-1)
    if fault=='basis_radius':v['directions'][0][0]=packed(1,Fraction(1,2**50))
    if fault=='derivative_rows':v['derivatives'].pop()
    if fault=='derivative_columns':v['derivatives'][0].pop()
    if fault=='missing_primal':v['primal'].pop()
    if fault=='negative_radius':v['derivatives'][0][0]['rad']=[-1,-50]
    if fault=='float_mid':v['derivatives'][0][0]['mid']=[1.0,0]
    if fault=='infinite_mid':v['derivatives'][0][0]['mid']=[float('inf'),0]
    if fault=='bool_mid':v['derivatives'][0][0]['mid']=[True,0]
    if fault=='noncanonical_dyadic':v['derivatives'][0][0]['mid']=[2,-1]
    with pytest.raises(ValueError):collect.collect_derivatives(c,r,s,**args)


@pytest.mark.parametrize('fault',['duplicate_shard','missing_shard','missing_phase','empty','different_centers','different_radii'])
def test_exact_total_coverage_and_nominal_domain(fault):
    c,r,s,args=fixture()
    if fault=='duplicate_shard':s.append(copy.deepcopy(s[0]))
    if fault=='missing_shard':s.pop()
    if fault=='missing_phase':s=[v for v in s if v['phase_index']==0]
    if fault=='empty':s=[]
    if fault=='different_centers':c[0][0]=packed(2)
    if fault=='different_radii':r[0]=packed(Fraction(1,2**31))
    with pytest.raises(ValueError):collect.collect_derivatives(c,r,s,**args)


def test_original_input_containment_cannot_be_repaired_by_outward_decoding():
    c,r,s,args=fixture(p=2,n=1)
    narrow=Fraction(566971103,2**502);nominal=Fraction(566971104,2**502)
    r=[packed(nominal),packed(nominal)]
    _,_,mh,sh=collect._nominal(c,r,2,1)
    for v in s:
        v['phase_domain_sha256']=mh;v['shooting_domain_sha256']=sh
        v['actual_inputs']=[packed(v['phase_index'],nominal)]
        v['actual_inputs_sha256']=domain.canonical_hash(v['actual_inputs'])
    original=packed(0,narrow)
    assert domain._record_parts(domain.pack_ball(collect._decode(original)))[1]>=nominal
    s[0]['actual_inputs']=[original];s[0]['actual_inputs_sha256']=domain.canonical_hash([original])
    with pytest.raises(ValueError,match='ORIGINAL producer box'):collect.collect_derivatives(c,r,s,**args)


def test_output_radius_widening_safe_and_actual_matrix_hash_retained():
    c,r,s,args=fixture(p=2,n=1)
    original=packed(0,Fraction(566971103,2**502));s[0]['derivatives'][0][0]=original
    matrices,report=collect.collect_derivatives(c,r,s,**args)
    assert domain._record_parts(domain.pack_ball(matrices[0][0,0]))[1]>domain._record_parts(original)[1]
    assert report['phase_matrix_sha256'][0]==gates._matrix_sha256(matrices[0])


def test_distinct_wider_receiver_boxes_each_cover_same_nominal_domain():
    c,r,s,args=fixture()
    s[0]['actual_inputs'][0]['rad']=[1,-29]
    s[0]['actual_inputs_sha256']=domain.canonical_hash(s[0]['actual_inputs'])
    _,report=collect.collect_derivatives(c,r,s,**args)
    assert report['coverage_complete']
    assert report['shards'][0]['original_actual_inputs_sha256']!=report['shards'][1]['original_actual_inputs_sha256']


def test_never_allocates_global_shooting_matrix(monkeypatch):
    c,r,s,args=fixture(p=4,n=3);shapes=[];real=collect.arb_mat
    def checked(rows,cols):
        shapes.append((rows,cols));assert (rows,cols)==(3,3);return real(rows,cols)
    monkeypatch.setattr(collect,'arb_mat',checked)
    matrices,_=collect.collect_derivatives(c,r,s,**args)
    assert len(matrices)==4 and shapes==[(3,3)]*4


@pytest.mark.parametrize('name,value',[('period',True),('dimension',0),('expected_precision',True),
    ('expected_design_sha256','a'*63),('expected_map_sha256','Z'*64)])
def test_external_contract_types(name,value):
    c,r,s,args=fixture();args[name]=value
    with pytest.raises(ValueError):collect.collect_derivatives(c,r,s,**args)


@pytest.mark.parametrize('touching',[False,True])
def test_original_primal_intersection_consistency_and_touching(touching):
    c,r,s,args=fixture()
    s[0]['primal'][0]=packed(Fraction(1,2),Fraction(1,2)) # [0,1]
    s[1]['primal'][0]=packed(Fraction(3,2) if touching else 2,Fraction(1,2))
    if touching:
        _,report=collect.collect_derivatives(c,r,s,**args)
        assert report['original_primal_intersections'][0][0]==dict(lower=[1,0],upper=[1,0])
        assert report['original_primal_intersections_sha256']==domain.canonical_hash(report['original_primal_intersections'])
        assert report['point_images_supplied'] is False
    else:
        with pytest.raises(ValueError,match='disjoint ORIGINAL primal'):collect.collect_derivatives(c,r,s,**args)


def test_primal_consistency_checks_original_not_outward_decoded_endpoints():
    c,r,s,args=fixture()
    rad=Fraction(566971103,2**502);wide=Fraction(566971104,2**502)
    s[0]['primal'][0]=packed(0,rad)
    s[1]['primal'][0]=packed(2*wide,rad)
    assert collect._decode(s[0]['primal'][0]).overlaps(collect._decode(s[1]['primal'][0]))
    with pytest.raises(ValueError,match='disjoint ORIGINAL primal'):collect.collect_derivatives(c,r,s,**args)
