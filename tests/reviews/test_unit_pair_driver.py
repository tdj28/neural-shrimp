import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[2]/'tools/reviews'))
import review_unit_pair_driver as D
import launch_unit_pair_review as L

def packet():
    c={'producer':{'source':'example'},'settings':{'fixed':'design'},'exact_parameter_endpoints':['0','1/32768']}
    rt={'commit':'example','mode':'target','settings':c['settings'],'nice':10,'threads':dict.fromkeys(D.O.THREADS,'1')}
    la={'commit':'example','exit_code':0,'owned_live_pids':[],'stop_reason':None}
    o={'passed':True,'status':'qualified_unit_pair_tube','strict_inclusions':704,'parameter_endpoints':c['exact_parameter_endpoints'],'events':{'chart_return':1,'direction_return':216,'newton_proposal':2,'preconditioner_return':1},'ticks':217}
    return o,rt,la,c

def test_complete_header_and_same_exclusive_owned_root():
    D.check_native_contract(*packet())
    assert D.OUT.parent==L.OUT
    assert str(L.OUT).endswith('/native-unit-pair-independent/attempt001')
    card=D.read(D.ROOT/D.CARD)
    assert card['expected_direction_acquisitions']==216 and card['independent_precision']==384
    assert(card['worker_seconds'],card['worker_rss_bytes'],card['worker_as_bytes'],card['worker_output_bytes'])==(300,4*1024**3,6*1024**3,24*1024**2)

@pytest.mark.parametrize('which,key,value',[(0,'strict_inclusions',703),(0,'passed',False),(0,'ticks',216),(0,'parameter_endpoints',['0','1/16384']),(1,'commit','wrong'),(1,'mode','profile'),(1,'nice',0),(2,'exit_code',2),(2,'owned_live_pids',[123]),(2,'stop_reason','cap')])
def test_incomplete_wrong_source_native_or_ownership_rejected(which,key,value):
    x=list(copy.deepcopy(packet()));x[which][key]=value
    with pytest.raises(ValueError):D.check_native_contract(*x)

def test_missing_native_acquisition_or_changed_arithmetic_rejected():
    x=list(copy.deepcopy(packet()));x[0]['events']['direction_return']=215
    with pytest.raises(ValueError):D.check_native_contract(*x)
    x=list(copy.deepcopy(packet()));x[1]['threads'][D.O.THREADS[0]]='2'
    with pytest.raises(ValueError):D.check_native_contract(*x)
    x=list(copy.deepcopy(packet()));x[1]['settings']={'fixed':'changed'}
    with pytest.raises(ValueError):D.check_native_contract(*x)
