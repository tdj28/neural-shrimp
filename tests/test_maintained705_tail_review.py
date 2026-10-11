import copy,importlib.util,subprocess,sys,tempfile
from pathlib import Path
from fractions import Fraction as F
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_maintained705_tail as D

def test_fixed19_and_all24_actual_metadata():
 card=D.read(ROOT/D.CARD);rows=D.selection(card);root=Path(card['producer']['root']);previous=D.read(Path(card['anchor']['root'])/'run/connected-endpoint.json');rotations=[];selected=[]
 assert D.entry(root/'manifest.json')['sha256']==card['producer']['outer']
 for row in rows:
  base=root/f"attempt{row['attempt']:03d}";pc=D.read(base/'input-card.json');out=D.read(base/'run/outcome.json')
  assert D.entry(base/'input-card.json')==row['card'] and D.entry(base/'run/manifest.json')['sha256']==row['worker']
  assert out['passed']==row['root_passed'] and out['ticks']==331 and out['events']==D.B.EXPECTED and pc['settings']==row['settings']
  if row['attempt']==1 or not row['joined']:continue
  if pc['mode']=='rotated_gauge':
   rotations.append(row['attempt']);assert 'exact_parameter'not in pc['anchor']and (previous['chart']['normal']!=pc['normal']or previous['chart']['origin']!=pc['origin'])
  else:assert D.same_gauge_parameter(previous,pc)==pc['anchor']['exact_parameter']
  previous=D.read(base/'connected-endpoint.json');selected.append(row['attempt'])
 assert selected==D.SELECTED and rotations==[11]and previous['chart']['parameter_endpoints'][1]=='2995/2048'

def test_changed_count_missing_failure_and_wrong_rotation_reject():
 original=D.read(ROOT/D.CARD)
 for which in ('selected','failure','rotation'):
  card=copy.deepcopy(original)
  if which=='selected':card['selected_attempts']=card['selected_attempts'][:-1]
  elif which=='failure':card['producer']['workers'].pop(2)
  else:card['producer']['workers'][9]['mode']='rotated_gauge'
  with pytest.raises(ValueError):D.selection(card)

def test_actual_overlap_and_wrong_scalar_gauge_reject():
 c=D.read(ROOT/D.CARD);r=Path(c['producer']['root']);pc=D.read(r/'attempt002/input-card.json');ep=D.read(Path(c['anchor']['root'])/'run/connected-endpoint.json')
 assert D.same_gauge_parameter(ep,pc)=='63/1024'
 changed=copy.deepcopy(pc);changed['anchor']['exact_parameter']='125/2048'
 with pytest.raises(ValueError):D.same_gauge_parameter(ep,changed)
 changed=copy.deepcopy(pc);changed['origin'][0]='0'
 with pytest.raises(ValueError):D.same_gauge_parameter(ep,changed)

def test_archive_import_and_scalar_limits():
 with tempfile.TemporaryDirectory()as folder:
  out=Path(folder)
  for n in D.FILES:
   p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
  subprocess.run([sys.executable,'-B','-c',"import sys;sys.path[:0]=['src','tools/reviews'];import review_maintained705_tail as d;assert callable(d.A.rebuild_J);assert d.settings()['charts']==19"],cwd=out,check=True,capture_output=True)
 s=(ROOT/'tools/launch_maintained705_tail_review.py').read_text();assert 'review_maintained705_tail.py'in s and 'native705-maintained-tail-independent/attempt001'in s and '4515'in s and '14338*1024**2'in s

def test_small_degree5_math_synthetic(tmp_path,monkeypatch):
 from flint import arb,ctx
 with ctx.workprec(480):
  monkeypatch.setattr(D,'OUT',tmp_path);w=D.Work();C,J,res,h,rho=D.B.synthetic_fixture(4);proof=D.H.polynomial_root(C,J,res,[arb(1)]*4,rho,h,[arb(0,rho)]*4,w.tick,w.emit)
  assert proof['passed']and all(proof['coordinate_inclusions'])
