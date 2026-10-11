import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import explore_tiny_lm_temporal_mean as D


def test_lossless_vector_aliases_do_not_duplicate_full_matrices(tmp_path):
 s=D.Store(tmp_path/'run');v=np.arange(15776,dtype=float);p=s.array(v);before=s.used
 s.alias('repeated',np.repeat(v[None,:],17,axis=0))
 data=D.read(s.root/'aliases/repeated.json')
 assert len(data['rows'])==17 and all(r==p for r in data['rows'])
 assert len(list((s.root/'objects').glob('*.npy')))==1 and s.used-before<10000
 assert np.array_equal(np.load(s.root/p['path']),v)


def test_nonfinite_native_return_and_attempted_input_are_retained(tmp_path):
 s=D.Store(tmp_path/'run');c=D.Calls(s,'profile',60)
 value=c.call('hvp',lambda:np.array([np.nan]),np.array([1.]),np.array([2.]))
 with pytest.raises(ValueError):D.T.vector(value)
 assert c.attempted['hvp']==c.returned['hvp']==1
 a=D.read(s.root/'events/00001-input.json');b=D.read(s.root/'events/00001-return.json')
 assert a['direction'] and np.isnan(np.load(s.root/b['returned']['path'])[0])
 with pytest.raises(RuntimeError):c.call('gradient',lambda:(_ for _ in ()).throw(RuntimeError('native failure')),np.array([1.]))
 assert c.attempted['gradient']==1 and c.returned['gradient']==0
 assert (s.root/'events/00002-input.json').is_file() and not(s.root/'events/00002-return.json').exists()


def test_closing_reserve_and_unadmitted_card(tmp_path):
 s=D.Store(tmp_path/'run');s.used=D.CAP-D.RESERVE
 with pytest.raises(ValueError):s.json('oversize.json',dict(error='retained'))
 s.finish(dict(passed=False,error='raw cap'))
 assert (s.root/'result.json').is_file() and (s.root/'manifest.json').is_file()
 card=dict(D.read(ROOT/D.CARD),status='source_prepared')
 with pytest.raises(ValueError,match='admission'):D.card_check(card,'profile')
 card=dict(card,status='profile_admitted',source_review='synthetic')
 D.card_check(card,'profile')
 with pytest.raises(ValueError):D.card_check(dict(card,control_rate_factor=.5),'profile')


def test_exact_historical_metadata_and_source_no_target_arrays():
 # Headers, result objects and manifests only. No scientific target arrays opened.
 for root,pin,source in D.HISTORY:
  assert D.sha(root/'manifest.json')==pin
  runtime=D.read(root/'run/runtime.json');assert runtime['source_commit']==source
  end=D.read(root/('launcher-result.json' if(root/'launcher-result.json').exists() else 'launch-end.json'))
  assert end.get('exit_code',end.get('code'))==0
 for parent,pin in D.PARENTS.values():assert D.sha(parent/'manifest.json')==pin
 assert D.sha(ROOT/'src/neural_shrimp/tiny_lm.py')==D.NATIVE_SHA


def test_complete_isolated_archive_import_and_real_launcher_help(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
 code="import sys;sys.path.insert(0,'tools');import explore_tiny_lm_temporal_mean as D;from pathlib import Path;root=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(root)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(root)};assert loaded<=set(D.FILES),loaded-set(D.FILES)"
 subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=env,check=True,capture_output=True)
 r=subprocess.run([sys.executable,str(tmp_path/'tools/launch_tiny_lm_temporal_mean.py'),'--help'],cwd=tmp_path,env=env,check=True,capture_output=True,text=True)
 assert '--mode {profile,target}' in r.stdout


def test_original_comparator_h1_index_is_not_h0():
 u=np.array([1.,0.]);first=np.array([.9,.1]);values=np.repeat(first[None,:],16,axis=0);values[-1]=[.2,.3]
 aligned=D.align_snapshot(u,first,values)
 assert np.array_equal(aligned[0],u) and np.array_equal(aligned[1],first) and np.array_equal(aligned[16],values[-1])
 with pytest.raises(ValueError):D.align_snapshot(u,first,values[1:])
 with pytest.raises(ValueError):D.align_snapshot(u,u,values)

def test_actual_full_historical_binding_without_scientific_arrays():
 for row in D.HISTORY:D.history(*row)
