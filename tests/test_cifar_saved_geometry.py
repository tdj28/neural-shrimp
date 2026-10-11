import numpy as np
import pytest
from neural_shrimp.cifar_saved_geometry import assess

def test_optimal_actual_line_and_endpoint_secant_are_exact():
 x=np.array([1.,2.,3.]);d=np.array([.01,0,0]);r=np.array([.003,.004,0]);p=np.array([1.,0,0]);ap=np.array([-.01,0,0]);z=np.r_[p,p];az=np.r_[p,p];row=assess(x,x+d,x+r,p,ap,z,az,.1,.01)
 assert row['line']['unconstrained_beta']==pytest.approx(.3)
 assert row['line']['best_predicted_forcing']==pytest.approx(.8)
 assert not row['line']['forcing_point5_reachable']
 assert row['cycle_geometry']['residual_parallel_to_phase_fraction']==pytest.approx(.36)
 assert row['cycle_geometry']['residual_transverse_fraction']==pytest.approx(.64)
 assert row['cycle_geometry']['secant_curvature']==pytest.approx(17)
 assert row['cycle_geometry']['curvature_to_threshold']==pytest.approx(.85)

def test_non_descent_is_not_candidate_and_old_direction_mismatch_rejected():
 x=np.ones(3);d=np.array([.01,0,0]);r=np.array([.003,.004,0]);p=np.ones(3);z=np.r_[p,p];az=np.arange(6.)
 row=assess(x,x+d,x+r,p,r,z,az,.1,.01);assert row['line']['bounded_beta']==0 and row['line']['best_predicted_forcing']==pytest.approx(1)
 with pytest.raises(ValueError):assess(x,x+d,x+r,p*2,r,z,az,.1,.01)


def test_complete_archive_import_closure_in_isolated_directory(tmp_path):
 import json,os,shutil,subprocess,sys
 from pathlib import Path
 root=Path(__file__).parents[1];sys.path.insert(0,str(root/'tools'))
 import diagnose_cifar_phase_corrector as W
 for name in W.FILES:
  dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
 code="""import json,sys
from pathlib import Path
import diagnose_cifar_phase_corrector as W
root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(root)}
assert loaded<=set(W.FILES),(loaded-set(W.FILES))
assert Path(W.__file__).resolve().is_relative_to(root)
print(json.dumps(sorted(loaded)))"""
 env=dict(os.environ,PYTHONPATH=str(tmp_path/'src')+os.pathsep+str(tmp_path/'tools'),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 result=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
 assert 'src/neural_shrimp/models.py' in json.loads(result.stdout)
