from fractions import Fraction
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import ball_json,exact,BallTanhMap
from neural_shrimp.rigorous_connection import tube,balanced_product,shooting,HiddenRateMap,restore_ball

class Quadratic:
    n=1
    def step(self,x,h):return [x[0]*x[0]-1+(h-1)],arb_mat([[2*x[0]]]),[arb(1)]


def run(**kw):
    anchor={'rate':'1','phase_boxes':[[ball_json(arb(0))],[ball_json(arb(-1))]]}
    return tube(Quadratic(),[['0'],['-1']],'1',halfwidth_exponent=-12,radius_exponent=-12,
                anchor=kw.pop('anchor',anchor),**kw)


def test_uniform_two_cycle_and_anchor():
    result=run()
    assert result['passed'] and all(result['gates'].values())
    assert result['statement']['shooting_dimension']==2
    assert result['divisors'][0]['excluded']
    assert result['attraction_power'] is not None


def test_interval_is_not_only_endpoint_roots():
    r=run();i=restore_ball(r['parameter_interval'])
    assert i.contains(exact('4095/4096')) and i.contains(exact('4097/4096'))
    assert len(r['endpoint_root_enclosures'])==2


def test_missing_or_wrong_anchor_fails_closed():
    assert not run(anchor=None)['passed']
    a={'rate':'1','phase_boxes':[[ball_json(arb(1))],[ball_json(arb(-1))]]}
    assert not run(anchor=a)['gates']['anchor_identified']
    with pytest.raises(ValueError):run(anchor={'rate':'2','phase_boxes':[]})


def test_constant_fixed_point_not_primitive_two():
    class Linear:
        n=1
        def step(self,x,h):return [x[0]/2+(h-1)],arb_mat([['1/2']]),[arb(1)]
    a={'rate':'1','phase_boxes':[[ball_json(arb(0))]]*2}
    r=tube(Linear(),[['0'],['0']],'1',anchor=a,halfwidth_exponent=-12,radius_exponent=-12)
    assert r['gates']['uniform_root'] and not r['gates']['primitive_period'] and not r['passed']


def test_noncontracting_and_unstable_controls():
    class Unstable:
        n=1
        def step(self,x,h):return [2*x[0]+(h-1)],arb_mat([[2]]),[arb(1)]
    a={'rate':'1','phase_boxes':[[ball_json(arb(0))]]}
    r=tube(Unstable(),[['0']],'1',anchor=a,max_power=4)
    assert r['gates']['uniform_root'] and not r['gates']['uniform_attraction']
    wide=tube(Quadratic(),[['0'],['-1']],'1',halfwidth_exponent=-2,radius_exponent=-2,newton_steps=0)
    assert not wide['gates']['uniform_root']


def test_product_native_order_and_odd_length():
    m=[arb_mat([[1,2],[0,1]]),arb_mat([[2,0],[3,1]]),arb_mat([[0,1],[1,0]])]
    p=balanced_product(m)
    assert p==m[2]*m[1]*m[0]


@pytest.mark.parametrize('settings',[{'precision':1},{'newton_steps':5},{'max_power':0}])
def test_settings(settings):
    with pytest.raises(ValueError):run(**settings)


def test_nonfinite_or_negative_radius_rejected():
    with pytest.raises(ValueError):restore_ball({'finite':False})
    with pytest.raises(ValueError):restore_ball({'mid':[0,0],'rad':[-1,-10],'display':'x'})


def test_hidden_rate_adapter_encloses_reviewed_c1_values():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','bias':True,'d':1,'width':1,'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'1/100'},'x':[['1/2'],['-1/3']],'y':['1/4','-1/5'],'lr_hidden':'3/8','lr_output':'1/7'}
    with ctx.workprec(256):
        x=[exact(q) for q in ['1/8','1/9','-1/5','1/7']]
        direct,j=BallTanhMap(inp).step(x);wrapped,k,b=HiddenRateMap(inp).step(x,exact('3/8'))
        assert all(a.overlaps(z) for a,z in zip(direct,wrapped))
        assert all(j[i,l].overlaps(k[i,l]) for i in range(4) for l in range(4))
        lo=HiddenRateMap(inp).step(x,exact('3/8')-exact('1/1024'))[0]
        hi=HiddenRateMap(inp).step(x,exact('3/8')+exact('1/1024'))[0]
        assert all(((hi[i]-lo[i])/exact('1/512')).overlaps(b[i]) for i in range(4))


def test_interval_adapter_rejects_nonpositive_rate():
    inp={'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','bias':False,'d':1,'width':1,'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'0'},'x':[['1']],'y':['0'],'lr_hidden':'1','lr_output':'1'}
    with pytest.raises(ValueError):HiddenRateMap(inp).step([arb(0),arb(0)],arb(0,1))


def driver():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_connection.py'
    s=importlib.util.spec_from_file_location('connection_driver_tests',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_inventory_corruption_missing_extra(tmp_path):
    m=driver();p=tmp_path/'value.json';p.write_text('{}');entries={'value.json':m.entry(p)}
    m.verify_inventory(tmp_path,entries)
    p.write_text('[]')
    with pytest.raises(ValueError):m.verify_inventory(tmp_path,entries)
    p.unlink()
    with pytest.raises(ValueError):m.verify_inventory(tmp_path,entries)
    p.write_text('{}');(tmp_path/'extra').write_text('x')
    with pytest.raises(ValueError):m.verify_inventory(tmp_path,entries)


def test_profile_pin_rejected_before_read(tmp_path):
    m=driver();(tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match='profile manifest'):m.profile_gate(tmp_path,'0'*64)


def test_fixed_anchor_receipt_corruption(tmp_path,monkeypatch):
    m=driver();monkeypatch.setattr(m,'ROOT',tmp_path)
    p=tmp_path/m.RECEIPT;p.parent.mkdir();p.write_text('{}')
    with pytest.raises(ValueError,match='canonical anchor'):m.prepare()


def test_lossless_deterministic_compressed_evidence(tmp_path):
    m=driver();obj={'exact':{'mid':[123,-400],'rad':[789,-800]},'false_gate':False}
    a=tmp_path/'a.json.gz';b=tmp_path/'b.json.gz';m.save(a,obj);m.save(b,obj)
    assert a.read_bytes()==b.read_bytes() and m.read(a)==obj
