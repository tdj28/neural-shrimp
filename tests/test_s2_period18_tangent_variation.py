import importlib.util
from pathlib import Path
import copy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import matrix_json,vector_json


def driver():
    p=Path(__file__).parents[1]/'tools/analyze_s2_period18_tangent_variation.py'
    spec=importlib.util.spec_from_file_location('tangent_variation_driver',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(m):
    z=matrix_json(arb_mat(235,1));v=matrix_json(arb_mat([[int(i==234)] for i in range(235)]));box=vector_json([arb(0)]*234+[arb(1)])
    pre=matrix_json(arb_mat(235,235));chart={'passed':True,'gates':{'uniform_root':True,'primitive_period':True},'statement':{'period':18,'phase_dimension':13},
       'normal':['0']*234+['1'],'origin':['0']*235,'preconditioner':pre,'image_correction':z,'exact_weights':['1']*235}
    inp={'chart':chart,'exact_endpoints':m.acquisition.ENDPOINTS,'model_inputs':{'network':{'d':2,'width':3,'bias':True}}}
    local={'final_error':z,'final_state_box':box}
    result={'exact_endpoints':m.acquisition.ENDPOINTS,'whole_localization':copy.deepcopy(local),'center_localization':dict(copy.deepcopy(local),exact_endpoints=['719/2048']*2),
       'uniform_tangent':{'passed':True,'derivative_box':v}}
    start={'exact_endpoints':m.acquisition.ENDPOINTS,**{k:chart[k] for k in ('normal','origin','preconditioner')},'inherited_image':z}
    final={'iteration':2,'retained_error':z,'jacobian':pre}
    physical={'exact_endpoints':m.acquisition.ENDPOINTS,'center_parameter':'719/2048','actual_uniform_tangent':vector_json([arb(0)]*234+[arb(1)]),'actual_whole_box':box,'actual_center_box':box}
    phases=[{'phase':i,'whole_J_mixed_diagnostic':matrix_json(arb_mat(13,13)),'whole_total_Jprime':matrix_json(arb_mat(13,13))} for i in range(18)]
    return inp,result,start,copy.deepcopy(final),copy.deepcopy(final),physical,phases


def test_complete_saved_join_and_actual_hidden_prefix():
    m=driver()
    with ctx.workprec(256):
        args=fixture(m);result=m.joined(*args)
        assert result['hidden']==9 and result['halfwidth']==arb('1/2048') and result['rate']==1


@pytest.mark.parametrize('position,key,value',[(0,'exact_endpoints',['0','1']),(2,'normal',['wrong']),(3,'iteration',1),(4,'retained_error',[]),(5,'center_parameter','0'),(5,'actual_whole_box',[])])
def test_join_rejects_mixed_domains_or_incomplete_intersections(position,key,value):
    m=driver();args=list(fixture(m));args[position][key]=value
    with pytest.raises(ValueError):m.joined(*args)


def test_native_phase_and_packing_rejected():
    m=driver();args=list(fixture(m));args[-1][0]['phase']=1
    with pytest.raises(ValueError):m.joined(*args)
    args=list(fixture(m));args[0]['model_inputs']['network']['bias']=1
    with pytest.raises(ValueError):m.joined(*args)


def test_source_inventory_includes_actual_derivative_and_new_kernel():
    m=driver()
    for name in ('src/neural_shrimp/rigorous_folds.py','src/neural_shrimp/rigorous_tangent_variation.py','src/neural_shrimp/rigorous_gradient_enclosure.py',m.CARD):assert name in m.SOURCES


def test_returned_bound_survives_postreturn_deadline(tmp_path):
    m=driver();result={'passed':False,'reason':'synthetic retained result'}
    def stop():raise TimeoutError('post-return deadline')
    with pytest.raises(TimeoutError,match='post-return'):
        m.retain_result(tmp_path,result,stop)
    assert m.io.read(tmp_path/'tangent-result.json.gz')==result
    assert (tmp_path/'tangent-result-payload.json').is_file()
