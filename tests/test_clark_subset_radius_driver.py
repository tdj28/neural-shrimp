import copy,json,os,subprocess,sys
from fractions import Fraction as F
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import screen_clark_subset_radii as W

class Work:
 def tick(self):pass
 def save(self,*args,**kwargs):pass

def test_complete_actual_metadata_before_scientific_decoding(monkeypatch):
 read=W.read;card=read(ROOT/W.CARD);card['status']='reviewed'
 def safe(p):
  assert not p.name.endswith('.gz'),'target scientific arrays remain unopened'
  return card if p==ROOT/W.CARD else read(p)
 monkeypatch.setattr(W,'read',safe)
 paths,kinds,ib,old=W.metadata(Work());assert len(paths)==2257 and kinds.count('correlated_explicit_tube_return')==32 and old['passed']

def test_changed_authority_rejects(monkeypatch):
 read=W.read;card=read(ROOT/W.CARD);card['status']='reviewed';card['independent_extension']['endpoint']['sha256']='0'*64
 monkeypatch.setattr(W,'read',lambda p:card if p==ROOT/W.CARD else read(p))
 with pytest.raises(ValueError,match='same exact prior'):W.metadata(Work())

def ball(m,r=0):return dict(mid=[m,0],rad=[r,0])

def test_projection_uses_entire_known51_box_and_exact_quadratic():
 t=dict(center=[ball(0)]*51,slope=[ball(1)]*51,curvature=[ball(2)]*51,radii=[ball(8)]*51,parameter=ball(2),halfwidth=ball(1));known=[ball(-1,1)]*51
 a,b=W.exact_projection(known,F(1),t);assert a==[F(1,4)]*51 and b==[F(2)]*51
 changed=copy.deepcopy(known);changed[50]['rad']=[8,0];a,_=W.exact_projection(changed,F(1),t);assert a[-1]==F(9,8)
 with pytest.raises(ValueError,match='full known51'):W.exact_projection(known[:-1],F(1),t)
 with pytest.raises(ValueError,match='scalar inside'):W.exact_projection(known,F(4),t)

def test_source_archive_import_and_no_native(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;sys.path.insert(0,'tools');import screen_clark_subset_radii as W;assert W.FAILURES==list(range(3,31,3));assert not any('rigorous_clark_' in n for n in sys.modules)"
 r=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=15);assert r.returncode==0,r.stderr

def test_fixed_caps_and_final_failure_retention():
 s=(ROOT/'tools/screen_clark_subset_radii.py').read_text();launch=(ROOT/'tools/launch_clark_subset_radius_screen.py').read_text()
 assert W.CAPS['native_calls']==0 and W.CAPS['seconds']==180
 assert "out['seconds']<180"in s and "out.update(passed=False,error="in s and "observe=lambda record:w.save(tag+'/proposal.json',record)"in s
 assert "'external_seconds':195"in launch and "'worker_as_bytes':4*1024**3"in launch and "'output_cap_bytes':66*1024**2"in launch
