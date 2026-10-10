import importlib.util,json,sys
from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import pytest
ROOT=Path(__file__).parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_s2_attraction_remaining as R

def fixture(chart):
    n=4 if chart==145 else 8;low=Fraction(0);high=Fraction(n,512);ends=[str(low),str(high)];slices=[[str(Fraction(k,512)),str(Fraction(k+1,512))]for k in reversed(range(n))]
    counts={'map_attempted':162*n,'map_returned':162*n,'unit_attempted':324*n,'unit_returned':324*n,'mixed_attempted':36*n,'mixed_returned':36*n}
    spec={'chart':chart,'coverage_endpoints':ends,'slices':slices,'counts':counts}
    inp={'chart':{'exact_parameter_endpoints':ends},'coverage_endpoints':ends,'slices':slices}
    done={'complete_coverage':ends,'passed':True,'completed_slices':n,'stages':161*n,'rows':[{}]*n,'counts':counts}
    return inp,done,spec

@pytest.mark.parametrize('chart',[142,143,144,145])
def test_exact_quarter_and_eighth_partitions(chart):
    assert R.validate_partition(*fixture(chart))==(4 if chart==145 else 8)
@pytest.mark.parametrize('change',['gap','ordering','counter','stage','parent'])
def test_corruption_is_not_a_complete_cover(change):
    inp,done,spec=deepcopy(fixture(142))
    if change=='gap':inp['slices'][1][0]='99';spec['slices']=inp['slices']
    if change=='ordering':inp['slices']=list(reversed(inp['slices']))
    if change=='counter':done['counts']={**done['counts'],'mixed_returned':0}
    if change=='stage':done['stages']-=1
    if change=='parent':inp['chart']['exact_parameter_endpoints']=['0','99']
    with pytest.raises(ValueError):R.validate_partition(inp,done,spec)

def test_fixed_batch_has_four_distinct_charts_and_environment_caps():
    p=ROOT/'tools/reviews/launch_s2_attraction_remaining.py';spec=importlib.util.spec_from_file_location('batch',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.CHARTS==(142,143,144,145) and len(set(m.CHARTS))==4
    assert m.OUT.name=='batch001' and set(m.THREADS)=={'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'}
