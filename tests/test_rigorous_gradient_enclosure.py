import itertools
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_orbits import exact,BallTanhMap
from neural_shrimp.rigorous_gradient_enclosure import mean_value_gradient,MeanValueHiddenRateMap


def inputs():
    return {'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','bias':True,'d':1,'width':1,'in_scale':'1','out_scale':'1','loss_scale':'1/2','ridge':'1/100'},'x':[['1/2'],['-1/3']],'y':['1/4','-1/5'],'lr_hidden':'3/8','lr_output':'1/7'}


def test_hessian_sign_and_nonpoint_gradient_dependency():
    with ctx.workprec(256):
        diag=list(map(exact,['1/1000','2','-1/2']))
        def unit(x):return [x[i]-diag[i]*x[i] for i in range(3)],arb_mat([[int(i==j)-(diag[i] if i==j else 0) for j in range(3)] for i in range(3)])
        x=[arb(exact(c),exact('1/100')) for c in ('1/4','-1/3','2/5')]
        gradient,hessian=mean_value_gradient(x,unit)
        assert all(gradient[i].contains(diag[i]*x[i]) for i in range(3))
        assert all(hessian[i,i].overlaps(diag[i]) for i in range(3))
        naive=x[0]-unit(x)[0][0]
        assert gradient[0].rad()<naive.rad()/100


def test_point_matches_original_map_all_coordinates():
    with ctx.workprec(256):
        inp=inputs();model=MeanValueHiddenRateMap(inp);theta=list(map(exact,['1/8','1/9','-1/5','1/7']))
        actual,jac,partial=model.step(theta,exact('3/8'));reference,jref=BallTanhMap(inp).step(theta)
        assert all(a.overlaps(b) for a,b in zip(actual,reference))
        assert all(jac[i,j].overlaps(jref[i,j]) for i in range(4) for j in range(4))
        assert all(partial[i].is_zero() for i in range(model.hidden,model.n))
        assert model.counts()=={'map_attempted':1,'map_returned':1,'unit_attempted':2,'unit_returned':2,'map_cap':180,'unit_cap':360}


def test_nonpoint_box_encloses_all_sampled_corner_maps_and_jacobians():
    with ctx.workprec(256):
        inp=inputs();model=MeanValueHiddenRateMap(inp);center=list(map(exact,['1/8','1/9','-1/5','1/7']));r=exact('1/128')
        box=[arb(v,r) for v in center];image,jac,_=model.step(box,exact('3/8'))
        original=BallTanhMap(inp)
        for signs in itertools.product((-1,1),repeat=4):
            theta=[v+s*r for v,s in zip(center,signs)];value,pointjac=original.step(theta)
            assert all(a.contains(b) for a,b in zip(image,value))
            assert all(jac[i,j].contains(pointjac[i,j]) for i in range(4) for j in range(4))


def test_caps_and_invalid_rates_stop_before_extra_unit_calls():
    with ctx.workprec(256):
        m=MeanValueHiddenRateMap(inputs(),max_calls=1);theta=[arb(0)]*4
        with pytest.raises(ValueError):m.step(theta,arb(-1))
        assert m.unit_attempted==0
        m.step(theta,exact('3/8'))
        with pytest.raises(ValueError):m.step(theta,exact('3/8'))
        assert m.unit_attempted==m.unit_returned==2


def test_coupled_affine_gradient_interval_hidden_rate_distinct_output_rate():
    with ctx.workprec(256):
        model=MeanValueHiddenRateMap(inputs());hessian=arb_mat([[1,2,0,1],[2,-3,1,0],[0,1,'1/2',2],[1,0,2,4]])
        affine=list(map(exact,['1/2','-1/4','3/8','-1/8']))
        class Coupled:
            def step(self,x):
                hx=hessian*arb_mat([[v] for v in x]);g=[hx[i,0]+affine[i] for i in range(4)]
                return [x[i]-g[i] for i in range(4)],arb_mat([[int(i==j)-hessian[i,j] for j in range(4)] for i in range(4)])
        model.unit=Coupled();center=list(map(exact,['1/8','1/4','-1/8','3/8']));r=exact('1/128')
        box=[arb(v,r) for v in center];hr=arb(exact('3/8'),exact('1/512'))
        image,jac,partial=model.step(box,hr)
        for signs in itertools.product((-1,1),repeat=4):
            theta=[v+s*r for v,s in zip(center,signs)];hx=hessian*arb_mat([[v] for v in theta]);gradient=[hx[i,0]+affine[i] for i in range(4)]
            for direction in (-1,0,1):
                rates=[exact('3/8')+direction*exact('1/512')]*2+[exact('1/7')]*2
                assert all(image[i].contains(theta[i]-rates[i]*gradient[i]) for i in range(4))
                assert all(jac[i,j].contains(int(i==j)-rates[i]*hessian[i,j]) for i in range(4) for j in range(4))
                assert all(partial[i].contains(-gradient[i]) for i in range(2))
                assert all(partial[i].is_zero() for i in range(2,4))


@pytest.mark.parametrize('field,bad',[('elapsed_seconds',-1),('elapsed_seconds',float('nan')),('elapsed_seconds',float('inf')),('max_rss_kib',-1),('max_rss_kib',float('inf')),('max_rss_kib',2*1024**2)])
def test_profile_resource_rejections(field,bad):
    m,runtime,result=profile_fixture();result[field]=bad
    with pytest.raises(ValueError):m.profile_environment(runtime,result)


def profile_fixture():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('meanvalue_driver',Path(__file__).resolve().parents[1]/'tools/certify_s2_period18_connection_meanvalue.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime={'python':m.platform.python_version(),'python_flint':m.flint.__version__,'flint':m.flint.__FLINT_VERSION__,
        'numpy':m.numpy.__version__,'scipy':m.scipy.__version__,'internal_seconds':20,'tick_cap':256,'map_cap':180,'unit_derivative_cap':360,
        'mode':'profile','threads':dict.fromkeys(m.THREADS,'1'),'nice':10}
    return m,runtime,{'elapsed_seconds':1.0,'max_rss_kib':1000,'exit_code':0}


def test_profile_library_and_thread_binding():
    m,runtime,result=profile_fixture();m.profile_environment(runtime,result)
    runtime['numpy']='different'
    with pytest.raises(ValueError):m.profile_environment(runtime,result)
    m,runtime,result=profile_fixture();runtime['threads']['OPENBLAS_NUM_THREADS']='2'
    with pytest.raises(ValueError):m.profile_environment(runtime,result)
