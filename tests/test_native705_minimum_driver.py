import copy,json,subprocess,sys,tempfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_minimum as D

def test_actual_fixed_metadata_without_target_math():
 c=D.read(ROOT/D.CARD);assert c['settings']==D.settings()
 for key in ['prior','old','root5','fallback','successor']:
  v=c[key];r=Path(v['root']);assert D.entry(r/'manifest.json')['sha256']==v['outer']and D.entry(r/'run/manifest.json')['sha256']==v['worker'];rt=D.read(r/'run/runtime.json');assert rt.get('source_commit',rt.get('commit'))==v['source']
 old=D.read(Path(c['old']['root'])/'input-card.json');native=D.read(Path(c['old']['root'])/'run/runtime.json')['sources'];model=old['map_inputs']
 for k,i,mode in [('old',4,'same_gauge'),('root5',5,'rotated_gauge'),('successor',1,'same_gauge')]:
  r=Path(c[k]['root']);rt=D.read(r/'run/runtime.json');out=D.read(r/'run/outcome.json');D.worker_contract(r,rt,out,c[k]['source'],i,mode,model,native)
  bad=copy.deepcopy(out);bad['ticks']=330
  with pytest.raises(ValueError):D.worker_contract(r,rt,bad,c[k]['source'],i,mode,model,native)
 assert all(n in native for n in D.NATIVE)


def test_isolated_archive_import_and_launch_bounds():
 with tempfile.TemporaryDirectory()as folder:
  r=Path(folder)
  for n in D.FILES:
   p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
  subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'tools/reviews');import review_native705_minimum as d;assert callable(d.H.P.rebuild);assert callable(d.fallback)"],cwd=r,check=True,capture_output=True)
 source=(ROOT/'tools/launch_native705_minimum_review.py').read_text()
 for s in ['parents[1]','tools/reviews/review_native705_minimum.py','915','2050*1024**2','10*1024**3','6*1024**2']:assert s in source
 assert D.OUT.parent.name=='attempt001'and D.settings()['native_calls']==0


def test_copied_reviewed_math_exact_and_no_producer_import():
 import hashlib
 assert hashlib.sha256((ROOT/'tools/reviews/review_cubic705_math.py').read_bytes()).hexdigest()==hashlib.sha256(subprocess.check_output(['git','show','e687be3198d92e25fb78dabec9310d002353a6fc:tools/reviews/review_cubic705_math.py'],cwd=ROOT)).hexdigest()
 source=(ROOT/'tools/reviews/review_native705_minimum.py').read_text();assert 'rigorous_saved_cubic_endpoint'not in source and 'localize_fresh_endpoint'not in source
