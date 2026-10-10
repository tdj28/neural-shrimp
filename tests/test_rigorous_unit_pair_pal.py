from fractions import Fraction
from math import factorial
import numpy as np
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_unit_pair_pal import assemble,proposal,entire_path,path_coefficients,quadratic_tube,anchor_test
from neural_shrimp.rigorous_eigenpair_path import PathJet
from neural_shrimp.rigorous_unit_pair import native_lift,entire_pair
from neural_shrimp.rigorous_orbits import exact,dyadic,eye,norm_inf_upper
from test_rigorous_unit_pair import Polynomial,Primitive,chain,floats,arr


def fixture(model,period):
    flat,_=native_lift(model,chain(model),period);flat.append(arb(0));size=len(flat)
    initial=[str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in flat]
    kw=dict(period=period,normal=['0']*(size-1)+['1'],origin=['0']*size,parameter='1/1024',halfwidth_exponent=-10,radius_exponent=-12,newton_steps=2,anchor={'box':flat,'dimension':model.n,'period':period})
    return initial,kw


def test_entire_path_nilpotent_coefficients_and_cubic_identities():
    with ctx.workprec(256):
        c,e,d=entire_path(PathJet([0,1,0,0]))
        for k in range(4):
            with ctx.workprec(512):
                rc=exact(str(Fraction((-1)**k,factorial(2*k))));re=exact(str(Fraction((-1)**k,factorial(2*k+1))))
            assert c.a[k].contains(rc)and e.a[k].contains(re)
        for t in ['-1/8','0','1/8']:
            z=PathJet([exact(t),exact('3/4'),exact('-1/3'),exact('1/8')]);c,e,d=entire_path(z)
            identity=c*c+z*e*e-1
            assert all(v.contains(0)for v in identity.a)
            zp=PathJet([z.a[1],2*z.a[2],3*z.a[3],0]);rhs=-e*zp/2
            assert all((k+1)*c.a[k+1]-rhs.a[k] == 0 or ((k+1)*c.a[k+1]-rhs.a[k]).contains(0)for k in range(3))
            assert all((d-z*e).a[k].contains(0)for k in range(4))
        with pytest.raises(ValueError):entire_path(PathJet([0,1,0,0],[1,0,0,0]))
        with pytest.raises(ValueError):entire_path(PathJet([exact('1/2'),0,0,0]))


@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_full_angle_column_gauge_and_condensed_inverse(model,period):
    with ctx.workprec(256):
        ini,kw=fixture(model,period);z=np.array([float(Fraction(v))for v in ini]);z[-1]=.001
        size=len(z);nn=arr(np.linspace(.001,.003,size));nn[-1]=arb(1);oo=[arb(0)]*size
        value,j,local=assemble(model,arr(z),period,nn,oo,arb(0))
        for k in range(size):
            v=np.eye(size)[k]*1e-6
            fd=(floats(assemble(model,arr(z+v),period,nn,oo,arb(0))[0])-floats(assemble(model,arr(z-v),period,nn,oo,arb(0))[0]))[:,0]/2e-6
            np.testing.assert_allclose(fd,floats(j)[:,k],atol=1e-9,rtol=1e-7)
        pre,small=proposal(j,local,model.n,arr(z)[-1]);assert small.nrows()==3*model.n+3
        assert norm_inf_upper(eye(size)-pre*j)<arb('1e-55')
        if period==2:assert local[0][0].det().contains(0)
        assert any(local[-1][1][i,2]!=0 for i in range(3*model.n))


@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_full_path_constant_derivative_and_quadratic_remainder(model,period):
    with ctx.workprec(256):
        ini,kw=fixture(model,period);center=[exact(v)for v in ini];center[-1]=exact('1/128');size=len(center)
        slope=[exact(str(Fraction(i+1,256)))for i in range(size)];curve=[exact(str(Fraction((-1)**i,128)))for i in range(size)]
        nn=[exact(v)for v in kw['normal']];oo=[arb(0)]*size;sc=exact('1/128')
        pc=path_coefficients(model,center,slope,curve,arb(0),period,nn,oo,sc)
        value,j,_=assemble(model,center,period,nn,oo,sc);first=j*arb_mat([[v]for v in slope]);first[size-1,0]-=1
        assert all(pc[i,0].overlaps(value[i,0])and pc[i,1].overlaps(first[i,0])for i in range(size))
        delta=exact('1/256');wc=path_coefficients(model,center,slope,curve,arb(0,delta),period,nn,oo,sc)
        for d in [-delta,delta/2,delta]:
            with ctx.workprec(512):
                point=[center[i]+slope[i]*d+curve[i]*d*d/2 for i in range(size)];v,_,_=assemble(model,point,period,nn,oo,sc+d)
            assert all((pc[i,0]+pc[i,1]*d+pc[i,2]*d*d+wc[i,3]*d*d*d).contains(v[i,0])for i in range(size))


@pytest.mark.parametrize('model,period',[(Polynomial(),1),(Primitive(),2)])
def test_connected_quadratic_tube_full_anchor_and_second_chart(model,period):
    ini,kw=fixture(model,period);result,data=quadratic_tube(model,ini,**kw)
    assert result['passed'],result['gates'];assert all(result['coordinate_inclusions']);assert result['shooting_dimension']==3*model.n*period+3
    with ctx.workprec(256):
        dd=exact('1/512');next0=[c+data['slope'][i]*dd+data['curvature'][i]*dd*dd/2 for i,c in enumerate(data['center'])]
        initial=[str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in next0]
        kw.update(parameter='3/1024',anchor={'box':data['endpoints'][1],'dimension':model.n,'period':period,'exact_parameter':'1/512'})
        second,_=quadratic_tube(model,initial,**kw);assert second['passed'],second['gates']
        # Corrupt a native mode, not only the extra angle coordinate.
        kw['anchor']['box'][model.n]+=1
        bad,_=quadratic_tube(model,initial,**kw);assert not bad['passed']and not bad['anchor_check']['passed']


def test_repeated_native_cycle_rejected_despite_root_and_parameter():
    model=Primitive();z=chain(model);z[-3:]=[v/2 for v in z[-3:]];flat,_=native_lift(model,z,4);flat.append(arb(0));size=len(flat)
    initial=[str(Fraction(int(v.mid().man_exp()[0]))*Fraction(2)**int(v.mid().man_exp()[1]))for v in flat]
    result,_=quadratic_tube(model,initial,period=4,normal=['0']*(size-1)+['1'],origin=['0']*size,parameter='1/1024',halfwidth_exponent=-10,radius_exponent=-12,anchor={'box':flat,'dimension':3,'period':4})
    assert result['gates']['uniform_full_native_unit_pair_locus']and not result['gates']['primitive_period']


def test_path_calls_preserve_both_native_directions_and_interrupt():
    model=Primitive();ini,kw=fixture(model,2);center=[exact(v)for v in ini];size=len(center);events=[];calls=[]
    path_coefficients(model,center,[arb(0)]*size,[arb(0)]*size,arb(0),2,[exact(v)for v in kw['normal']],[arb(0)]*size,arb(0),tick=lambda:calls.append(1),observe=lambda k,v:events.append((k,v)))
    assert len(calls)==4 and [(v['phase'],v['direction'])for k,v in events if k=='pal_path_phase_return']==[(0,'q'),(0,'r'),(1,'q'),(1,'r')]
    def stop():raise TimeoutError('owned cap')
    with pytest.raises(TimeoutError):path_coefficients(model,center,[arb(0)]*size,[arb(0)]*size,arb(0),2,[exact(v)for v in kw['normal']],[arb(0)]*size,arb(0),tick=stop)
