import copy
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp import junction_audit as A
from neural_shrimp.rigorous_orbits import vector_json,matrix_json,ball_json

def fixture():
    n=4;size=19;c=[arb(0)]*size
    c[4]=c[5]=c[6]=c[11]=c[17]=arb(1);c[14]=arb(1)/10;c=[v.mid()for v in c]
    box=[arb(v,arb((1,-80)))for v in c]
    return dict(dimension=n,period=2,augmented_dimension=size,center=vector_json(c),box=vector_json(box),preconditioner=matrix_json(A.eye(size)),whole_jacobian=matrix_json(A.eye(size)),value=matrix_json(arb_mat(size,1)),orbit_enclosures=[vector_json(box[:n]),vector_json([arb(2)]*n),vector_json(box[:n])])

def test_root_inclusion_and_corruption_rejections():
    with ctx.workprec(192):
        root=fixture();assert A.root_arithmetic(root)['inclusions']==19
        for key in ['value','divisor','inverse']:
            bad=copy.deepcopy(root)
            if key=='value':bad['value'][0][0]=ball_json(arb(1))
            if key=='divisor':bad['orbit_enclosures'][1]=bad['box'][:4]
            if key=='inverse':bad['preconditioner'][0][0]['rad']=[1,-100]
            with pytest.raises(ValueError):A.root_arithmetic(bad)

def test_strict55_identity_rejects_overlap():
    small=[dict(mid=[1,0],rad=[1,-84])]*55;large=[dict(mid=[1,0],rad=[1,-80])]*55
    assert all(A.strict_nested(large,small))
    for x,y in [(small,large),(small,small),(large[:-1],small)]:
        with pytest.raises(ValueError):A.strict_nested(x,y)

def test_critical_quotient_and_total_crossing():
    with ctx.workprec(192):
        root=fixture();t=A.decode(root['box'][14]);a=t.sqrt();c=a.cos();e=a.sin()/a
        M=arb_mat([[c,e,0,0],[-t*e,c,0,0],[0,0,-1,0],[0,0,0,arb(1)/4]])
        J=A.eye(19);J[4,14]=1;J[16,4]=-2
        root.update(monodromy=matrix_json(M),whole_jacobian=matrix_json(J))
        p=dict(coordinate_columns=[3],spectral_exclusion=dict(proposal_basis=[[dict(real=ball_json(arb(1)),imag=ball_json(arb(0)))]]))
        out=A.spectrum_crossing(root,p);assert out['complement_bounds'][0]<1 and out['crossing']<0
        M[3,3]=2;root['monodromy']=matrix_json(M)
        with pytest.raises(ValueError):A.spectrum_crossing(root,p)

def test_nonfinite_and_negative_ball_rejected():
    for v in [dict(mid=[1,0],rad=[-1,-80]),dict(mid=[1,0],rad=[0,0],finite=False)]:
        with pytest.raises(ValueError):A.decode(v)
