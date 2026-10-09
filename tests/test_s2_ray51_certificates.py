"""Pure synthetic input/source controls for the two distinct ray51 exact maps."""
import copy,importlib.util,json
from fractions import Fraction
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ray51cert',ROOT/'tools/certify_s2_ray51.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
    parent=m.synthetic('period36');parent={k:parent[k] for k in ('schema','network','x','y')}
    window={'center':[-1.,-1.3],'scale':[.1,.2],'rays':[{'type':'flip','direction':[.5,-.25],'s_boundary':.1} for _ in range(52)]}
    orbit=[[0.]*13 for _ in range(18)];row={'map_id':'S2-28011-zoom','component':587,'period':18,'certificate':{'orbit':orbit}}
    ell=window['center'];direction=[a*b for a,b in zip(window['scale'],window['rays'][51]['direction'])]
    family={'schema':'neural-shrimp-exact-log-ray-v1','ell':[str(Fraction(x)) for x in ell],'direction':[str(Fraction(x)) for x in direction],'ell_float':ell,'direction_float':direction,'historical_scale':window['scale'],'historical_unit_direction':window['rays'][51]['direction']}
    point={'qualified':True,'state':[0.]*13,'s':.125,'q':[1.]+[0.]*12}
    numeric={'row_index':111,'ray_index':51,'period':18,'ray':window['rays'][51],'historical_orbit':orbit,'consumed_map':parent,'family':family}
    seed={'schema':'neural-shrimp-flip-proof-seed-v1','ray_index':51,'period':18,'consumed_map':parent,'family':family,'theta':['0']*13,'parameter':'1/8','right_vector':['1']+['0']*12}
    rates=[10.**float(a+point['s']*d) for a,d in zip(ell,direction)]
    rows=[{'qualified':True,'qualification':{'stability':'attracting'},'opposite_qualification':{'stability':'attracting'},'amplitude_requested':2.**e,'state':point['state'],'s':point['s'],'data':{'theta':point['state'],'s':point['s'],'period':36,'orbit':[[0.]*13 for _ in range(37)],'rates':rates,'rates_exact':[str(Fraction(x)) for x in rates]}} for e in (-12,-11,-10)]
    return parent,row,window,numeric,point,seed,rows

def test_exact_maps_are_explicitly_distinct():
    args=fixture();flip,fs=m.assemble('flip',*args);orbit,os=m.assemble('period36',*args)
    assert flip['period']==18 and len(flip['approximation'])==27
    assert orbit['period']==36 and len(orbit['approximation'])==13
    assert flip['map_inputs']['lr_hidden']=='1/1' and orbit['lr_hidden']!='1/1'
    assert os['amplitude_index']==2 and 'Fixed exact dyadic' in os['map_convention']
    assert os['connection_not_proved'] and fs['connection_not_proved']

@pytest.mark.parametrize('field',['direction','factor','state','rate','index','stable','amplitude'])
def test_identity_or_selection_mutations_rejected(field):
    a=list(copy.deepcopy(fixture()))
    if field=='direction':a[5]['family']['direction'][0]='1/9'
    elif field=='factor':a[2]['scale'][0]=.125
    elif field=='state':a[4]['state'][0]=1.
    elif field=='rate':a[6][2]['data']['rates_exact'][0]='1/7'
    elif field=='index':a[3]['ray_index']=13
    elif field=='stable':a[6][2]['opposite_qualification']['stability']='unstable'
    else:a[6][1]['amplitude_requested']=1.
    with pytest.raises(ValueError):m.assemble('period36',*a)

def test_proximity_uses_whole_box_not_only_center():
    inp={'approximation':['0']};r={'box':[{'mid':[0,0],'rad':[1,-10]}]}
    assert not m.proximity(inp,r)['passed']
    r['box'][0]['rad']=[1,-30];assert m.proximity(inp,r)['passed']

def test_finalization_reserve_survives_ordinary_cap(tmp_path,monkeypatch):
    m.ARTIFACT_ROOT=tmp_path;monkeypatch.setattr(m,'total',lambda _:14*1024**2)
    with pytest.raises(ValueError):m.save(tmp_path/'math.json',{'returned':True})
    m.save(tmp_path/'failure.json',{'retained':True});assert (tmp_path/'failure.json').exists()

def test_pinned_inventory_rejects_modified_or_extra_file(tmp_path):
    (tmp_path/'a').write_text('a');m.ARTIFACT_ROOT=tmp_path;m.save(tmp_path/'manifest.json',{'files':m.inventory(tmp_path),'total_bytes':1});pin=m.sha(tmp_path/'manifest.json');m.verify(tmp_path,pin)
    (tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError):m.verify(tmp_path,pin)


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf'),True])
def test_profile_resource_nonfinite_negative_and_bool_rejected(value):
    good={'elapsed_seconds':.2,'max_rss_kib':1024,'timings':[.1,.1]};launch={'elapsed_seconds':.3}
    m.profile_resources(good,launch)
    for key in ['elapsed_seconds','max_rss_kib']:
        bad=dict(good);bad[key]=value
        with pytest.raises(ValueError):m.profile_resources(bad,launch)
    with pytest.raises(ValueError):m.profile_resources(good,{'elapsed_seconds':value})
