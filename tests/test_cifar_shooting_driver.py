import importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import explore_cifar_symmetric_shooting as D
import launch_cifar_symmetric_shooting as L

def test_caps_cover_real_native_operations():
 assert D.CAPS==dict(action=0,j2vp=0,hvp=2*(400+2)+2+8+2,direct_gradient=2+12+26)
 assert 'explore_cifar_symmetric_shooting.py' in L.command(Path('/tmp/a'),'x')[8]

def test_measured_gate_and_fd_metric():
 events=[dict(kind='operation_returned',operation=k,seconds=v)for k,v in [('hvp',.09),('direct_gradient',.88)]]
 assert D.measured_gate(events)['forecast_seconds']<300
 assert D.fd_error(np.ones(2),np.ones(2))==0

def test_launcher_requires_qualified_native_calls_and_no_orbit_claim():
 r=dict(status='shooting_pilot_complete',comparison_qualified=True,accepted_orbit=False,spectrum_requested=False,operation_caps=D.CAPS,operation_attempted=dict(D.CAPS),operation_returned=dict(D.CAPS),qualification=dict(passed=True),trials=[],linear=dict(iterations=400))
 assert L.scientific_complete(r)
 r['accepted_orbit']=True;assert not L.scientific_complete(r)
