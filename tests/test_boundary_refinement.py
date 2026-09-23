import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import refine_boundary_behavior as r
from hopfield_drift_probe import build_trace


def test_failure_selection_uses_every_amplitude_sign(tmp_path):
    for i in range(26):
        folder=tmp_path/f'point-{i:02d}'
        folder.mkdir()
        for k in range(2):
            for a in range(2):
                x=np.zeros((3,4,2,3))
                if i==7 and k==1 and a==1: x[2,3,1,2]=.001
                np.save(folder/f'traces-{k}-{a}.npy',x)
    assert r.selected_contexts(tmp_path)==[dict(point=7,context=2)]


def test_refinement_changes_only_selected_context_artificial_model(tmp_path):
    folder=tmp_path/'point-00'
    folder.mkdir()
    np.savez(folder/'contexts.npz',contexts=np.tile([.1,.2,.3],(3,1)))
    for k in range(2):
        for a in range(2): np.save(folder/f'traces-{k}-{a}.npy',np.zeros((3,4,61,3)))
    fn,_=build_trace(tmp_path/'build')
    r.refine_point(0,[1],dict(dt=[.0025,.00125]),
                   dict(points=[dict(alpha=0.,gamma=0.)],amplitudes=[.01,.1]),
                   tmp_path,fn,time.monotonic()+30)
    for k in range(2):
        for a in range(2):
            x=np.load(folder/f'traces-{k}-{a}.npy')
            assert np.all(x[[0,2]]==0)
            assert np.isfinite(x[1]).all() and np.max(abs(x[1]))>0
