"""Independent synthetic/source controls; no S1R target or resource profile."""
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import certify_s1r_period2 as subject
from neural_shrimp.rigorous_orbits import canonical_bytes, check_orbit


def fixture_profile(folder):
    sources={'synthetic_source':'bound_hash'};inputs={'synthetic_input':'exact'}
    runtime={'source_sha256':sources,'settings':subject.SETTINGS,'action':'profile',
             'source_commit':'synthetic_review_fixture','status':'profile_passed',
             'elapsed_seconds':1.0,'max_rss_kib':123,'cpu_threads':1,
             'nice':10,'gpu':False,'paid':False,'pid':12345}
    result={'passed':True,'map_seconds':.1,'projected_seconds':12.8,'max_rss_kib':123,
            'map_calls_in_checker':14,'safety_factor':2,'overhead_allowance_seconds':10,
            'evidence_rung':0,'scope':'Resource timing only; no proof outcome or continuation.'}
    for name,value in [('runtime-start.json',runtime),('runtime-end.json',runtime),
                       ('inputs.json',inputs),('result.json',result)]:
        (folder/name).write_bytes(canonical_bytes(value))
    subject.manifest(folder,'profile_passed')
    return sources,inputs


def rewrite(folder,name,patch):
    value=json.loads((folder/name).read_bytes());value.update(patch)
    (folder/name).write_bytes(canonical_bytes(value))
    subject.manifest(folder,'profile_passed')


def test_resource_profile_rehashed_inconsistent_projection_rejected(tmp_path):
    sources,inputs=fixture_profile(tmp_path)
    subject.profile_gate(tmp_path,sources,inputs)
    rewrite(tmp_path,'result.json',{'map_seconds':500.0})
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)


@pytest.mark.parametrize('name,patch',[
    ('result.json',{'map_seconds':-1.0,'projected_seconds':-18.0}),
    ('result.json',{'max_rss_kib':-1}),
    ('result.json',{'map_calls_in_checker':1}),
    ('runtime-start.json',{'action':'collect'}),
    ('runtime-end.json',{'status':'failed'}),
    ('runtime-end.json',{'elapsed_seconds':.01}),
    ('runtime-end.json',{'max_rss_kib':-1}),
    ('runtime-end.json',{'pid':67890}),
    ('runtime-end.json',{'elapsed_seconds':21}),
    ('result.json',{'map_seconds':True}),
    ('runtime-end.json',{'source_sha256':{'changed':'hash'}}),
])
def test_resource_profile_rehashed_semantic_contradictions_rejected(tmp_path,name,patch):
    sources,inputs=fixture_profile(tmp_path);rewrite(tmp_path,name,patch)
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)


def test_resource_profile_missing_completion_record_rejected(tmp_path):
    sources,inputs=fixture_profile(tmp_path);(tmp_path/'runtime-end.json').unlink()
    subject.manifest(tmp_path,'profile_passed')
    with pytest.raises((ValueError,FileNotFoundError)):subject.profile_gate(tmp_path,sources,inputs)


def test_resource_profile_manifest_total_recomputed(tmp_path):
    sources,inputs=fixture_profile(tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_bytes());manifest['total_bytes']=-1
    (tmp_path/'manifest.json').write_bytes(canonical_bytes(manifest))
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)


def test_resource_profile_input_and_settings_identity(tmp_path):
    sources,inputs=fixture_profile(tmp_path)
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,{'different':'input'})
    rewrite(tmp_path,'runtime-start.json',{'settings':{**subject.SETTINGS,'precision':512}})
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)


def linear72(period=1):
    # Zero inputs/no bias make data loss constant. F(theta)=3theta/4 exactly.
    return {'schema':'neural-shrimp-c1-input-v1',
            'network':dict(d=8,width=8,activation='tanh',bias=False,in_scale='1/2',
                           out_scale='1/8',loss_scale='1',ridge='1/2'),
            'x':[['0']*8],'y':['0'],'lr_hidden':'1/2','lr_output':'1/2',
            'period':period,'approximation':['0']*72}


def test_72d_exact_linear_existence_and_false_period_control():
    fixed=check_orbit(linear72(),precision=128,newton_steps=0,radius_exponent=-40,max_power=1)
    assert fixed['passed'] and fixed['exact_statement']['dimension']==72
    assert fixed['exact_statement']['existence_and_local_uniqueness']
    assert fixed['attraction_witness_power']==1
    false_period=check_orbit(linear72(2),precision=128,newton_steps=0,radius_exponent=-40,max_power=1)
    assert not false_period['passed']
    assert false_period['exact_statement']['existence_and_local_uniqueness']
    assert not false_period['exact_statement']['primitive_period_excluded_divisors']


def test_72d_unstable_exact_fixed_point_cannot_pass_attraction():
    inputs=linear72();inputs['lr_hidden']=inputs['lr_output']='5'
    result=check_orbit(inputs,precision=128,newton_steps=0,radius_exponent=-40,max_power=1)
    assert result['exact_statement']['existence_and_local_uniqueness']
    assert not result['exact_statement']['attraction'] and not result['passed']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf')])
def test_resource_profile_nonfinite_measurements_rejected(tmp_path,value):
    sources,inputs=fixture_profile(tmp_path)
    path=tmp_path/'result.json';result=json.loads(path.read_bytes());result['map_seconds']=value
    path.write_text(json.dumps(result))
    subject.manifest(tmp_path,'profile_passed')
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)


def test_resource_profile_extra_file_and_symlink_rejected(tmp_path):
    sources,inputs=fixture_profile(tmp_path)
    (tmp_path/'unexpected').write_text('unhashed')
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)
    (tmp_path/'unexpected').unlink()
    original=tmp_path/'result.json';external=tmp_path.parent/(tmp_path.name+'-external.json')
    external.write_bytes(original.read_bytes());original.unlink();original.symlink_to(external)
    with pytest.raises(ValueError):subject.profile_gate(tmp_path,sources,inputs)
