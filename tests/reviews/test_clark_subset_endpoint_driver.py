import copy,json,sys,subprocess,os
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_clark_subset_endpoint as W

def test_actual_metadata_without_scientific_decoding(monkeypatch):
 card=json.loads((ROOT/W.CARD).read_text());card['status']='reviewed';read=W.read
 def safe(p):
  assert p.suffix!='.gz','no scientific proof/return decode in metadata test'
  return card if p==ROOT/W.CARD else read(p)
 monkeypatch.setattr(W,'read',safe);monkeypatch.setattr(W.M,'decode',lambda v:v)
 class Work:
  def tick(self):pass
  def save(self,*a,**kw):pass
 paths,kinds,known,binding=W.metadata(Work());assert len(paths)==331 and len(known)==51 and kinds.count('context_attempt')==1

def test_changed_endpoint_authority_fails(monkeypatch):
 card=json.loads((ROOT/W.CARD).read_text());card['status']='reviewed';card['known']['endpoint']['sha256']='0'*64;read=W.read
 monkeypatch.setattr(W,'read',lambda p:card if p==ROOT/W.CARD else read(p))
 with pytest.raises(ValueError,match='literal source'):W.metadata(None)

def test_selected_radius_cannot_change_actual_error_or_coefficients():
 ball=lambda n:dict(mid=[n,0],rad=[0,0]);old={k:[]for k in('center','slope','curvature','parameter','halfwidth','inverse_coefficients','residual_coefficients','jacobian_coefficients','actual_error')};old['radii']=[ball(2)]*51;old['residual_coefficients']=[[[ball(0)]for _ in range(51)]for _ in range(4)];old['jacobian_coefficients']=[[[ball(0)for _ in range(51)]for _ in range(51)]for _ in range(3)];new=copy.deepcopy(old);new['radii']=[ball(1)]*51
 assert W.selected_domain(old,new)==[1]*51
 new['actual_error']=[ball(1)]
 with pytest.raises(ValueError,match='actual_error'):W.selected_domain(old,new)
 new=copy.deepcopy(old);new['radii'][0]=ball(3)
 with pytest.raises(ValueError,match='subset'):W.selected_domain(old,new)

def test_archive_import_and_final_resource_checks(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;sys.path.insert(0,'tools/reviews');import review_clark_subset_endpoint as W;assert sum(W.EXPECTED.values())==331;assert not any('rigorous_clark_' in n for n in sys.modules);from pathlib import Path;r=Path.cwd();loaded={str(Path(m.__file__).resolve().relative_to(r)) for m in sys.modules.values() if getattr(m,'__file__',None) and Path(m.__file__).is_file() and Path(m.__file__).resolve().is_relative_to(r)};assert loaded<=set(W.FILES),loaded-set(W.FILES)"
 p=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert p.returncode==0,p.stderr
 src=(ROOT/'tools/reviews/review_clark_subset_endpoint.py').read_text();launcher=(ROOT/'tools/launch_clark_subset_endpoint_review.py').read_text();assert "out['elapsed_seconds']<120"in src and "'external_seconds':135"in launcher and "['status']=='reviewed'"in launcher

def test_exact_384_coefficient_reserialization_and_corruption():
 from flint import ctx
 ball=lambda n:dict(mid=[n,0],rad=[0,0]);x=dict(mid=[2**383+123456789,-400],rad=[536870915,-500]);z=ball(0)
 old={k:[]for k in('center','slope','curvature','parameter','halfwidth','inverse_coefficients','actual_error')};old.update(radii=[ball(2)]*51,residual_coefficients=[[[x]for _ in range(51)]for _ in range(4)],jacobian_coefficients=[[[x if i==j else z for j in range(51)]for i in range(51)]for _ in range(3)])
 new=copy.deepcopy(old);new['radii']=[ball(1)]*51
 with ctx.workprec(384):
  for key in('residual_coefficients','jacobian_coefficients'):new[key]=[[[W.serial(W.M.decode(v))for v in row]for row in matrix]for matrix in old[key]]
 assert W.selected_domain(old,new)==[1]*51
 assert W.selected_domain(old,old,False)==[2]*51
 new['residual_coefficients'][0][0][0]=ball(3)
 with pytest.raises(ValueError,match='exact384'):W.selected_domain(old,new)
