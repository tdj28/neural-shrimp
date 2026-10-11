import importlib.util,json,os,resource,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import diagnose_cifar_ray_remainder as D

def test_actual_complete_metadata_without_scientific_array_read(monkeypatch):
 def reject(*a,**k):raise AssertionError('no target array reading in metadata')
 monkeypatch.setattr(np,'load',reject)
 b,a=D.metadata(lambda:None)
 assert a['counts']==dict(hvp=514,direct_gradient=10)
 assert a['target_arrays_unopened'] and a['fit_indices']==[1,2] and a['check_indices']==[0,3,4]
 assert len(b[4])==8

def test_target_state_direction_corruption_rejected(monkeypatch):
 n=98666;x=np.ones(n)*.1;g=np.ones(n)*.001;p=-g*.1;hp=-g*.75;origin=x.copy()
 arrays={'initial-point.npy':x,'initial-gradient.npy':g,'candidate-direction.npy':p,'global-original-midpoint.npy':origin,'state-000.npy':x,'g0.npy':g,'p.npy':p,'hp.npy':hp}
 calls={('direct_gradient',1):dict(state='state-000.npy',result='g0.npy'),('hvp',514):dict(state='state-000.npy',direction='p.npy',result='hp.npy')}
 trials=[dict(within_trust=False)]*3
 for i,a in enumerate(D.H.ALPHAS):
  arrays[f'x{i}']=x+a*p;arrays[f'g{i}']=g+a*hp+a*a*g
  calls['direct_gradient',i+6]=dict(state=f'x{i}',result=f'g{i}')
  trials.append(dict(within_trust=True,alpha=a,merit=D.M.dot(arrays[f'g{i}'],arrays[f'g{i}'])/2))
 monkeypatch.setattr(D.M,'vector',lambda path:arrays[Path(path).name])
 arrays['final-point.npy']=arrays['x4'];arrays['final-gradient.npy']=arrays['g4']
 binding=({},calls,[],{},trials,{},{});report,_=D.target(binding,lambda:None)
 assert report['quadratic_sufficiency_qualified'] and len(report['actual_source_references'])==5
 arrays['p.npy']=p.copy();arrays['p.npy'][1]+=1e-8
 with pytest.raises(ValueError,match='actual initial gradient'):D.target(binding,lambda:None)
 arrays['p.npy']=p;arrays['x3']=arrays['x3'].copy();arrays['x3'][1]+=1e-8
 with pytest.raises(ValueError,match='literal saved alpha'):D.target(binding,lambda:None)

def test_isolated_archive_complete_import_under_inherited_limits(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
 def limits():resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,3*1024**3))
 env=dict(os.environ,**dict.fromkeys(D.THREADS,'1'),PYTHONDONTWRITEBYTECODE='1')
 p=subprocess.run([sys.executable,str(tmp_path/'tools/diagnose_cifar_ray_remainder.py'),'--source-import-check'],cwd=tmp_path,env=env,preexec_fn=limits,capture_output=True,text=True,timeout=10)
 assert p.returncode==0,p.stderr
 assert p.stdout.strip()=='source import complete'

def test_literal_limits_profile_and_launcher():
 c=json.loads((ROOT/D.CARD).read_text());assert c['limits']==D.LIMITS and c['settings']==D.H.SETTINGS and c['independent']==D.AUDIT and c['producer']==D.PRODUCER
 launcher=(ROOT/'tools/launch_cifar_ray_remainder.py').read_text()
 assert 'available>6*1024**2'in launcher and "'external_seconds':75"in launcher and "'worker_as_bytes':3*1024**3"in launcher and "'output_cap_bytes':34*1024**2"in launcher
 assert "str(ROOT/'tools/diagnose_cifar_ray_remainder.py')"in launcher
 assert len(D.FILES)==len(set(D.FILES))and all((ROOT/n).is_file()for n in D.FILES)
