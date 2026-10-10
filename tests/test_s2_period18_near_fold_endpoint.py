import importlib.util
from pathlib import Path
from flint import ctx
from neural_shrimp.rigorous_arclength_localization import localize_endpoint
from neural_shrimp.rigorous_reboxing_gradient import ReboxingMeanValueHiddenRateMap


def module():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_near_fold_endpoint.py'
    s=importlib.util.spec_from_file_location('near_fold_endpoint',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_frozen_partial_loader_and_counts():
    m=module()
    assert m.partial_input is m.parent.partial_input
    assert m.PIN=='a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2'
    assert 'tools/analyze_s2_period18_near_fold_gauge.py' in m.SOURCES
    assert 'src/neural_shrimp/rigorous_arclength_localization.py' in m.SOURCES
    assert len(m.SOURCES)==len(set(m.SOURCES))


def test_profile_does_not_claim_qualified_native_orbit():
    m=module()
    with ctx.workprec(256):
        inp=m.profile_input()
        assert not inp['previous']['passed']
        assert len(inp['previous']['endpoint_root_enclosures'][1])==235
        assert inp['previous']['exact_parameter_endpoints'][1]=='0'
