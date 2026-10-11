import numpy as np
import pytest
from neural_shrimp.cifar_krylov_conditioning import diagnose

def fixture():
    rng=np.random.default_rng(37);A=np.diag(np.linspace(.2,3,24));g=rng.normal(size=24);r1=-g;r2=r1.copy();beta=np.linalg.norm(g);oldb=0.;vs=[];ws=[]
    for i in range(8):
        v=r2/beta;w=A@v;vs.append(v);ws.append(w);y=w.copy()
        if i:y-=(beta/oldb)*r1
        a=v@y;y-=(a/beta)*r2;r1=r2;r2=y;oldb=beta;beta=np.linalg.norm(y)
    V=np.stack(vs,axis=1);W=np.stack(ws,axis=1);p=V@np.linalg.lstsq(W,-g,rcond=1e-12)[0];return A,V,W,g,p,A@p

def test_exact_lanczos_and_span_geometry():
    _,V,W,g,p,hp=fixture();r,_,_=diagnose(V,W,g,p,hp);assert r['prefixes'][0]['ritz_qualified']and r['actual_direction_representation']['qualified'];assert abs(r['span_prediction']['forcing']-r['actual_forcing'])<1e-12;assert max(r['recurrence_vector_errors'])<1e-12

def test_duplicate_basis_and_changed_action_are_not_qualified():
    _,V,W,g,p,hp=fixture();V[:,2]=V[:,1];W[:,2]=W[:,1];r,_,_=diagnose(V,W,g,p,hp);assert not r['prefixes'][0]['ritz_qualified']
    _,V,W,g,p,hp=fixture();r,_,_=diagnose(V,W,g,p,hp+1);assert not r['actual_direction_representation']['qualified']

def test_saved_span_can_improve_scaled_direction_but_requires_fresh_action():
    _,V,W,g,p,hp=fixture();r,_,_=diagnose(V,W,g,p*.01,hp*.01);assert r['span_prediction']['relative_to_actual']<.5 and r['span_prediction']['requires_fresh_physical_action']and r['actual_direction_representation']['qualified']

def test_nonfinite_or_misaligned_rejected():
    _,V,W,g,p,hp=fixture();V[0,0]=np.nan
    with pytest.raises(ValueError):diagnose(V,W,g,p,hp)
    _,V,W,g,p,hp=fixture()
    with pytest.raises(ValueError):diagnose(V,W[:-1],g,p,hp)
