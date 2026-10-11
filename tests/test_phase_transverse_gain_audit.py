"""Small independent synthetic-only arithmetic audit checks; no target reads."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/phase_transverse_gain_audit'))
import numpy as np
import pytest
import audit_gain as a


def toy():
    psi=np.broadcast_to(np.eye(3),(32,3,3)).copy()
    flow=np.tile([1.,0.,0.],(32,1))
    return dict(psi=psi,states=np.zeros((32,3)),velocity=flow.copy(),physical=flow.copy(),
                f0=np.array([1.,0.,0.]),z=np.array([1.,2.,0.]),w=flow/32)


def test_complete_census():
    cells=a.census()
    assert len(cells)==192 and len({c['id'] for c in cells})==192
    assert sum(len(a.pulse_ids(c)) for c in cells)==1152


def test_oblique_partition_and_contractions():
    common=toy();direction=np.array([.01,.02,.03]);value=a.derive(common,direction)
    assert np.allclose(value['state']['phase_algebraic']+value['state']['transverse'],value['state']['total'])
    assert value['observable']['total']==pytest.approx(.01)
    assert value['observable']['phase_algebraic']==pytest.approx(.05)
    assert value['observable']['transverse']==pytest.approx(-.04)
    assert all(v['passed'] for v in a.recorded_checks(value,direction).values())


def test_separate_defects_do_not_cancel_gates():
    common=toy();common['physical']*=2
    value=a.derive(common,np.array([.01,0.,0.]));gates=a.approximation(value)
    assert not gates['transport_defect']['passed'] and not gates['clock_flow_defect']['passed']
    assert gates['combined_ideal_clock_defect']['passed']


def test_matrix_power_and_chronological_order():
    matrix=np.array([[1.,2.,0.],[0.,.5,1.],[0.,0.,.25]])
    for k in range(9):assert np.array_equal(a.power(matrix,k),np.linalg.matrix_power(matrix,k))
    phi=np.array([[1.,0.,0.],[.3,1.,0.],[0.,0.,1.]])
    assert not np.array_equal(phi@a.power(matrix,2),a.power(matrix,2)@phi)


def test_strict_arithmetic_separate_from_evidence():
    evidence=a.Audit();arithmetic=a.Arithmetic()
    a.compare_tree({'passed':True,'residual':0.},{'passed':True,'residual':1e-10},evidence,arithmetic,'synthetic')
    assert not evidence.failures and len(arithmetic.failures)==1


def test_frozen_threshold_and_nonfinite_rejection():
    assert a.close(1e-14,0.)['passed'] and not a.close(1.01e-14,0.)['passed']
    audit=a.Arithmetic();assert not audit.near(np.array([np.nan]),np.array([np.nan]),'nonfinite')


def test_periodic_endpoint_and_elapsed_propagator_reconstruction():
    t=np.linspace(0,1,4097);x=np.zeros((4097,3));x[:,0]=np.cos(2*np.pi*t)
    dx=np.zeros_like(x);dx[:,0]=-2*np.pi*np.sin(2*np.pi*t)
    x[-1]=[123.,0.,0.]
    phi=np.broadcast_to(np.eye(3),(4097,3,3)).copy();dphi=np.zeros_like(phi)
    ref=dict(period=1.,alpha=1.,gamma=10.,orbit=dict(times=t,states=x,derivatives=dx))
    prop=dict(times=t,phi=phi,phi_derivatives=dphi,monodromy=np.eye(3))
    context=dict(phase_epoch=0.,flow=np.ones(3),phase_gradient=np.ones(3))
    clock=dict(times=7+(np.arange(32)+.5)/32,weights=np.ones((32,3)))
    actual=a.context_arrays(ref,prop,context,clock)
    assert np.max(np.abs(actual['states'][:,0]-np.cos(2*np.pi*clock['times'])))<1e-12
    assert np.array_equal(actual['psi'],phi[:32])
