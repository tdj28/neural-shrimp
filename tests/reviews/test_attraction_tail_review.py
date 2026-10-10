import importlib.util,inspect,sys
from copy import deepcopy
from pathlib import Path
from fractions import Fraction
import pytest
ROOT=Path(__file__).parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_s2_attraction_tail as R
import launch_s2_attraction_tail as L

def fixture(chart):
    n=8 if chart==155 else 7;low=Fraction(43,128)if chart==155 else Fraction(11,32);high=low+Fraction(n,1024)
    ends=list(map(str,[low,high]));complete=ends if chart==155 else ['11/32','45/128'];slices=[[str(low+Fraction(k,1024)),str(low+Fraction(k+1,1024))]for k in reversed(range(n))]
    counts={'map_attempted':162*n,'map_returned':162*n,'unit_attempted':324*n,'unit_returned':324*n,'mixed_attempted':36*n,'mixed_returned':36*n}
    reused=None if chart==155 else {'exact_endpoints':['359/1024','45/128'],'manifest':'retained-unaudited'}
    spec={'chart':chart,'coverage_endpoints':ends,'original_complete_coverage':complete,'slices':slices,'counts':counts,'reused_unaudited':reused}
    inp={'chart':{'exact_parameter_endpoints':complete},'slices':slices,'reused':deepcopy(reused)}
    if chart==155:inp['coverage_endpoints']=ends
    done={'complete_coverage':complete,'passed':True,'completed_slices':n,'stages':161*n,'rows':[{}]*n,'counts':counts}
    return inp,done,spec

@pytest.mark.parametrize('chart',[155,156])
def test_exact_native_tail_partitions(chart):assert R.validate_partition(*fixture(chart))==(8 if chart==155 else 7)
@pytest.mark.parametrize('change',['gap','ordering','counter','stage','parent','reused','claim'])
def test_incomplete_partition_or_unaudited_reuse_is_rejected(change):
    inp,done,spec=deepcopy(fixture(156))
    if change=='gap':inp['slices'][1][0]='99';spec['slices']=inp['slices']
    if change=='ordering':inp['slices']=list(reversed(inp['slices']))
    if change=='counter':done['counts']={**done['counts'],'mixed_returned':0}
    if change=='stage':done['stages']-=1
    if change=='parent':inp['chart']['exact_parameter_endpoints']=['0','99']
    if change=='reused':inp['reused']['manifest']='wrong'
    if change=='claim':spec['coverage_endpoints']=['11/32','45/128']
    with pytest.raises(ValueError):R.validate_partition(inp,done,spec)

def test_fixed_two_distinct_workers_with_original_resource_fixes():
    assert L.CHARTS==(155,156)and L.OUT.name=='batch004'
    assert L.current_rss_kib('VmHWM: 612668 kB\nVmRSS: 18432 kB\n')==18432
    with pytest.raises(ValueError):L.current_rss_kib('VmRSS: -1 kB\n')
    s=inspect.getsource(R.worker);assert s.index('resource.setrlimit')<s.index('subprocess.check_output')
    assert "'core.preloadIndex=false'"in s and "'index.threads=1'"in s
