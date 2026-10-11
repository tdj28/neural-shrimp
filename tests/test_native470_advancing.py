import sys,json,gzip,shutil,subprocess,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
import probe_native470_advancing as D

def test_literal_old_gauge_and_advance_endpoint():
 from fractions import Fraction as F
 c=dict(passed=True,shooting_dimension=470,period=18,sigma=-1,normal=['0']*468+['1','0'],parameter_center='0',parameter_endpoints=['-1/256','1/256'],settings={'halfwidth_exponent':-8})
 new,old=D.check_geometry(c);assert F(new)+F(1,32)==F(old)and F(new)-F(1,32)==F(-17,256)
 c['normal'][0]='1'
 with pytest.raises(ValueError):D.check_geometry(c)

def test_actual_independent_schema_and_profile_counts():
 card=D.read(ROOT/D.CARD);r=D.read(Path(card['authority']['independent']['root'])/'run/independent512.json');assert r['passed']and r['inclusions']==r['anchor_coordinates']==470 and len(r['recomputed_endpoints'][0])==470
 assert card['settings']['profile_ticks']==151 and card['settings']['target_ticks']==127 and card['settings']['native_calls']==276 and card['settings']['chart']==D.SETTINGS

def test_actual_metadata_retained_authority_without_target_arrays(monkeypatch):
 class W:
  start=time.monotonic()
  def check(self):pass
  def save(self,*a):pass
 old=D.read
 def guarded(p):
  assert p.name not in ('target-inputs.json','target-certificate.json.gz','independent512.json')
  return old(p)
 monkeypatch.setattr(D,'read',guarded);D.retained(W(),old(ROOT/D.CARD))

def test_complete_archive_and_default_preserving_target_hook(tmp_path):
 import probe_native470_first_flip as C
 assert C.TARGET_SETUP is None and C.SETTINGS['halfwidth_exponent']==-8
 # Isolated import installs only inside subprocess so old default remains here.
 code="import sys;sys.path.insert(0,'tools');import probe_native470_advancing as d;d.install();from pathlib import Path;import launch_native470_advancing as l;r=Path.cwd();assert d.C.OUT.parent==l.OUT;assert d.C.SETTINGS['halfwidth_exponent']==-5;assert d.C.TARGET_SETUP is d.setup;a={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(r)};assert a<=set(d.C.FILES),a-set(d.C.FILES)"
 # Ask a clean subprocess for the installed source set.
 names=json.loads(subprocess.check_output([sys.executable,'-B','-c',"import sys,json;sys.path.insert(0,'tools');import probe_native470_advancing as d;d.install();print(json.dumps(d.C.FILES))"],cwd=ROOT))
 for n in names:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
