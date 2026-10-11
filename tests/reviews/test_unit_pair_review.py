import sys
from pathlib import Path
from copy import deepcopy
from fractions import Fraction as F
import numpy as np
import pytest
from flint import arb,arb_mat,ctx
sys.path[:0]=[str(Path(__file__).parents[2]/'tools/reviews')]
import review_unit_pair_math as U
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json

def a(x):return U.R.exact(F(x))
def events(flat,n=2,p=2):
    A=arb_mat([[a('1/4'),a('1/8')],[a('-1/8'),a('1/2')]]);E=arb_mat([[a('1/3'),a('1/5')],[a('2/7'),a('-1/4')]]);rows=[]
    for k in range(p):
        x=arb_mat([[v]for v in flat[6*k:6*k+2]]);rates=arb_mat([[v]for v in flat[-2:]]);fx=A*x+E*rates
        for j,name in [(1,'q'),(2,'r')]:
            d=arb_mat([[v]for v in flat[6*k+2*j:6*k+2*j+2]]);ad=A*d;B=arb_mat(4,4);Er=arb_mat(4,2)
            for i in range(2):
                for l in range(2):B[i,l]=B[2+i,2+l]=A[i,l];Er[i,l]=E[i,l]
            rows.append({'phase':k,'direction':name,'input':vector_json([v[0]for v in x.tolist()]+[v[0]for v in d.tolist()]+flat[-2:]),'value':matrix_json(arb_mat([[fx[i,0]]for i in range(2)]+[[ad[i,0]]for i in range(2)])),'block':matrix_json(B),'rates':matrix_json(Er)})
    return rows

def oracle(z,t):
    A=np.array([[.25,.125],[-.125,.5]]);E=np.array([[1/3,1/5],[2/7,-.25]]);c=np.cos(np.sqrt(t));e=np.sin(np.sqrt(t))/np.sqrt(t);d=t*e;out=[]
    for k in range(2):
        x,q,r=z[6*k:6*k+6].reshape(3,2);nx,nq,nr=z[6*((k+1)%2):6*((k+1)%2)+6].reshape(3,2)
        out.extend(A@x+E@z[-2:]-nx);out.extend(A@q-(nq if k==0 else c*nq-d*nr));out.extend(A@r-(nr if k==0 else e*nq+c*nr))
    q,r=z[2:4],z[4:6];out.extend([(q@q-1)/2,q@r]);return np.array(out)

def test_full_native_jacobian_and_parameter_derivative():
    with ctx.workprec(384):
        z=[a(F(k-5,16))for k in range(14)];t=a('1/16');H,J,Ht=U.rebuild(z,events(z),2,2,t);zz=np.array([float(v)for v in z]);eps=1e-6
        np.testing.assert_allclose([float(v[0])for v in H.tolist()],oracle(zz,1/16),atol=2e-15)
        fd=np.column_stack([(oracle(zz+eps*np.eye(14)[i],1/16)-oracle(zz-eps*np.eye(14)[i],1/16))/(2*eps)for i in range(14)])
        np.testing.assert_allclose(np.array([[float(v)for v in row]for row in J.tolist()]),fd,atol=1e-10,rtol=1e-8)
        np.testing.assert_allclose([float(v[0])for v in Ht.tolist()],(oracle(zz,1/16+eps)-oracle(zz,1/16-eps))/(2*eps),atol=1e-10,rtol=1e-8)

def test_jordan_terminal_and_derivatives_at_zero():
    with ctx.workprec(384):
        S=U.terminal(2,arb(0));T=U.terminal(2,arb(0),1);Q=U.terminal(2,arb(0),2)
        assert S[2,2]==1 and S[2,4]==0 and S[4,2]==1 and S[4,4]==1
        assert T[2,2].contains(a('-1/2'))and T[2,4]==-1 and T[4,2].contains(a('-1/6'))
        assert Q[2,2].contains(a('1/12'))and Q[2,4].contains(a('1/3'))and Q[4,2].contains(a('1/60'))

@pytest.mark.parametrize('lo,hi', [('0','1/32768'),('-1/8','1/8'),('1/8','3/16')])
def test_series_contains_exact_sampled_trigonometric_values(lo,hi):
    with ctx.workprec(384):
        lo,hi=F(lo),F(hi);t=arb(a((lo+hi)/2),a((hi-lo)/2));S=U.terminal(2,t)
        for q in [lo,(lo+hi)/2,hi]:
            x=a(q)
            if q>0:c=x.sqrt().cos();e=x.sqrt().sin()/x.sqrt()
            elif q<0:c=(-x).sqrt().cosh();e=(-x).sqrt().sinh()/(-x).sqrt()
            else:c=e=arb(1)
            assert S[2,2].contains(c)and S[4,2].contains(e)and S[2,4].contains(-x*e)

@pytest.mark.parametrize('bad',['domain','phase','df','state','shape'])
def test_corrupt_native_acquisition_fails(bad):
    z=[a(F(k-5,16))for k in range(14)];rows=events(z)
    if bad=='domain':rows[0]['input'][0]=ball_json(arb(99))
    if bad=='phase':rows[1]['phase']=1
    if bad=='df':rows[1]['block'][0][0]=ball_json(arb(99))
    if bad=='state':rows[1]['value'][0][0]=ball_json(arb(99))
    if bad=='shape':rows[0]['rates'][0].pop()
    with pytest.raises(ValueError):U.rebuild(z,rows,2,2,a('1/16'))

def test_strict_full_contraction_and_coordinate_gate():
    args=[arb_mat([[1]]),arb_mat([[a('1/2')]]),arb_mat([[a('1/1000')]]),arb_mat([[a('1/2')]]),arb_mat([[a('1/10')]]),arb_mat([[0]]),['1'],F(1,10),F(1,100)]
    assert U.contraction(*args)['passed'];args[2]=arb_mat([[a('1/5')]]);assert not U.contraction(*args)['passed']
    args[2]=arb_mat([[0]]);args[1]=arb_mat([[0]]);assert not U.contraction(*args)['passed']
    args[6]=['0']
    with pytest.raises(ValueError):U.contraction(*args)


def test_exact_boundary_and_outward_interval_domain_guard():
    with ctx.workprec(384):
        S=U.terminal(2,a('1/4'));assert S[2,2].contains(a('1/4').sqrt().cos())
        with pytest.raises(ValueError):U.terminal(2,arb(a('3/16'),a('1/16')))


def make_lift():
    with ctx.workprec(384):
        root=[arb(0),arb(0),arb(1),arb(2),arb(1),arb(0),arb(0),arb(1)]
        current=root[:2]+root[4:6]+root[6:8];rates=root[2:4];all_events=[];flat=[]
        for k in range(2):
            pair=events(current+rates,n=2,p=1)
            for e in pair:e['phase']=k
            all_events+=pair;flat+=current
            current=[U.R.original_ball(v[0])for v in pair[0]['value']]+[U.R.original_ball(v[0])for v in pair[1]['value'][2:]]
        lift={'dimension':2,'period':2,'shooting_dimension':14,'parameter':'0','box':vector_json(flat+rates)}
        return vector_json(root),lift,all_events


@pytest.mark.parametrize('bad',[None,'root','phase','rates','saved_box'])
def test_conditional_lift_requires_full_domains(bad):
    root,lift,ev=make_lift()
    if bad=='root':root[0]=ball_json(arb(99))
    if bad=='phase':ev[3]['input'][0]=ball_json(arb(99))
    if bad=='rates':ev[2]['input'][-1]=ball_json(arb(99))
    if bad=='saved_box':lift['box'][8]=ball_json(arb(99))
    if bad is None:assert U.check_lift(root,lift,ev,n=2,p=2)['native_coordinates']==14
    else:
        with pytest.raises(ValueError):U.check_lift(root,lift,ev,n=2,p=2)
