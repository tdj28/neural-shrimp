import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from diagnose_cifar_phase_scaling import parts,summary


def test_saved_phase_energy_diagnoses_only_actual_residual():
    x=np.array([1.,2.,3.]);z=np.r_[x,x];row=parts(z);assert row['mean_energy_fraction']==pytest.approx(1)and row['alternating_norm']==0
    row=parts(np.r_[x,-x]);assert row['alternating_energy_fraction']==pytest.approx(1)and row['mean_norm']==0
    r=np.ones(3);rhs=np.r_[np.zeros(3),-r];s=summary(z,rhs,r,.01,.1);assert s['physical_shooting_relative_residual']==0 and s['scaled_saved_residual']==0
    with pytest.raises(ValueError):summary(z[:-1],rhs,r,.01,.1)


def test_git_preflight_raises_same_envelope_before_single_thread_status(monkeypatch):
    import diagnose_cifar_phase_scaling as m
    calls=[]
    monkeypatch.setattr(m.resource,'setrlimit',lambda kind,limits:calls.append(('limit',limits)))
    def git(argv,**kw):
        assert calls==[('limit',(1024**3,1024**3))]
        assert argv==['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain']
        calls.append(('git',None));return b''
    monkeypatch.setattr(m.subprocess,'check_output',git)
    m.preflight();assert calls[-1][0]=='git'
