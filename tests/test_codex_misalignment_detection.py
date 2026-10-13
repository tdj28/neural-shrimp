import importlib.util
from pathlib import Path
import sys
import numpy as np

SRC=Path(__file__).resolve().parents[1]/'codex/misalignment-audit'
sys.path.insert(0,str(SRC))
import detection as D


def fake_arm():
    pre=np.empty((2,6,24),np.float16)
    for r in range(2):
        for t in range(6): pre[r,t]=10*r+t
    score=np.zeros((2,6)); score[0,4]=.8
    return {'score':score,'axis':np.arange(12).reshape(2,6),'q':np.full((2,6),.25),
            'q_plus':np.full((2,6),.375),'tok':np.full((2,6,3),np.nan),
            'pre':pre,'dose':.04}


def test_feature_time_and_susceptibility_scale():
    arm=fake_arm(); tab=D.make_table([arm],2,1)
    assert tab['y'].tolist()==[0,0,1,1,0,0,0,0]
    np.testing.assert_array_equal(tab['state'][:,0],[1,2,3,4,11,12,13,14])
    np.testing.assert_array_equal(tab['extra'],np.tile([.25,.125,0],(8,1)))
    assert np.all(tab['turn']<=3)
    assert np.all(tab['base'][:,5]==.04)


def test_crossfit_training_ids_and_pca_do_not_see_test(monkeypatch):
    rng=np.random.default_rng(12); groups=np.repeat(np.arange(20),3)
    tab={'y':np.tile([0,1,0],20),'group':groups,'base':rng.normal(size=(60,6)),
         'extra':rng.normal(size=(60,3)),'state':np.column_stack((groups,np.zeros((60,23)))),
         'turn':np.tile(np.arange(3),20),'arm':np.zeros(60,int)}
    seen=[]
    class PCA:
        def __init__(self,x): self.ids=set(x[:,0]); seen.append(self.ids)
        def transform(self,x): return np.zeros((len(x),20))
    monkeypatch.setattr(D,'fit_pca',lambda x,d:PCA(x))
    monkeypatch.setattr(D,'fit_logistic',lambda x,y,d:None)
    monkeypatch.setattr(D,'score_model',lambda m,x:np.arange(len(x),dtype=float))
    out,fold=D.crossfit(tab,[])
    for i,training in enumerate(seen):
        assert training==set(groups[fold!=i])
        assert not training & set(groups[fold==i])
    assert all(np.isfinite(v).all() for v in out.values())


def test_frozen_threshold_uses_training_negatives_and_strict_flags(monkeypatch):
    y=np.array([0,0,1,1]); scores=[np.array([0.,1.,1.,2.]),np.array([1.,2.,2.,3.])]
    result=D.operating_point(y,scores,[1.,2.])
    assert result=={'baseline':.5,'with_state':.5,'false_positive_rate_baseline':0.,
                    'false_positive_rate_with_state':0.}
