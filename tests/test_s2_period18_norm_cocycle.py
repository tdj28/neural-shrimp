from pathlib import Path
import importlib.util


def driver():
    p=Path(__file__).parents[1]/'tools/certify_s2_period18_norm_cocycle.py'
    s=importlib.util.spec_from_file_location('norm_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_two_exact_predeclared_intervals_and_source_inventory():
    m=driver();assert m.cases()==[{'role':'full','halfwidth':'1/4096','exact_endpoints':['719/2048','45/128']},
        {'role':'quarter','halfwidth':'1/16384','exact_endpoints':['5755/16384','5757/16384']}]
    for name in (m.CARD,'src/neural_shrimp/rigorous_norm_cocycle.py','tools/certify_s2_period18_correlated_cocycle.py','src/neural_shrimp/models.py','src/neural_shrimp/controls.py'):
        assert name in m.SOURCES


def test_launcher_freeze_and_caps_match_card():
    text=(Path(__file__).parents[1]/'tools/launch_s2_period18_norm_cocycle.py').read_text()
    assert 'HEAD:tools/launch_s2_period18_norm_cocycle.py' in text
    assert "'30s'" in text and "'external_seconds':30" in text and 'timeout=37' in text
    assert '33*1024**2' in text
