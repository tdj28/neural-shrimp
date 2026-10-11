import sys
from pathlib import Path
from copy import deepcopy
import pytest
ROOT=Path(__file__).parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews')]
import review_s2_attraction_reused as R
import launch_s2_attraction_reused as L

def fixture():
 c={'stages':{'base':{'worker_manifest':{'sha256':'base'}},'tangent':{'worker_manifest':{'sha256':'tangent'}}},'exact_endpoints':['359/1024','45/128'],'halfwidth':'1/2048'}
 b={'runtime':{'mode':'target'},'outcome':{'status':'fixed_slice_bounds_completed','result_status':'complete_slice_bounds','passed':False,'stages':85,'completed_intersections':6,'completed_physical_phases':18,'completed_products':18,'adapter_counts':{'map_attempted':162,'map_cap':162,'map_returned':162,'unit_attempted':324,'unit_cap':324,'unit_returned':324},'mixed_counts':{'attempted':18,'returned':18,'cap':18}}}
 t={'runtime':{'parent_manifest':'base','model_calls':0,'derivative_calls':0},'outcome':{'status':'saved_tangent_variation_completed','conditional_bound_passed':True}}
 r={'runtime':{'mode':'target','tangent_manifest':'tangent','model_C1_calls':0},'outcome':{'status':'fixed_refined_cocycle_completed','passed':True,'stages':75,'mixed_counts':{'attempted':18,'returned':18,'cap':18}}}
 bi={'exact_endpoints':c['exact_endpoints'],'model_inputs':{'actual':'map'}};ri={'exact_endpoints':c['exact_endpoints'],'halfwidth':'1/2048','bindings':{'original_manifest':'base','tangent_manifest':'tangent'},'model_inputs':bi['model_inputs']}
 return {'base':b,'tangent':t,'refined':r},c,[bi,ri]

def test_complete_three_stage_contract_and_owned_path():
 R.contracts(*fixture());assert R.OUT.parent==L.OUT and L.OUT.name=='attempt001'
 assert str(L.OUT).endswith('/s2-attraction-reused/attempt001')

@pytest.mark.parametrize('change',['old_pass','old_counter','dependency','new_counter','new_status','interval','model','hidden_calls'])
def test_missing_wrong_or_relabelled_history_fails(change):
 m,c,i=deepcopy(fixture())
 if change=='old_pass':m['base']['outcome']['passed']=True
 if change=='old_counter':m['base']['outcome']['adapter_counts']['map_returned']=161
 if change=='dependency':m['tangent']['runtime']['parent_manifest']='wrong'
 if change=='new_counter':m['refined']['outcome']['mixed_counts']['returned']=17
 if change=='new_status':m['refined']['outcome']['passed']=False
 if change=='interval':i[1]['exact_endpoints']=['0','1']
 if change=='model':i[1]['model_inputs']={'actual':'different'}
 if change=='hidden_calls':m['tangent']['runtime']['derivative_calls']=1
 with pytest.raises(ValueError):R.contracts(m,c,i)
