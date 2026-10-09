import copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_saved_root_bound_and_failed_coordinate_are_distinguished():
    fixtures=load('old_join_fixture','tests/reviews/test_joined_period18_math.py');m=load('mean_saved','tools/reviews/review_meanvalue_outcome.py')
    chart=copy.deepcopy(fixtures.fixture()[0][0]);chart['settings']=m.SETTINGS
    assert m.bounds(chart)['passed_root_bound']
    chart['residual_center'][0][0]=fixtures.b(1);chart['gates']['uniform_root']=False
    result=m.bounds(chart);assert not result['passed_root_bound'] and result['matches_producer_root_gate'] and result['failed_coordinates']
