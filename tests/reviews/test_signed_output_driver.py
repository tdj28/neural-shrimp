import sys,copy,subprocess,os
from pathlib import Path
import pytest
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'src'),str(R/'tools/reviews')]
import review_signed_output as D
class Sink:
 def tick(self):pass
 def save(self,*a,**k):pass

def test_prepared_card_blocked():
 c=D.read(R/D.CARD);c['status']='source_prepared'
 with pytest.raises(ValueError,match='fixed reviewed'):D.bind(c,Sink(),True)

def test_actual_metadata_before_interval_arrays(monkeypatch):
 # Historical Git objects live in the common private repository, even when
 # this source-only packet is inspected from /tmp before Root copies it.
 monkeypatch.setattr(D.S.A.C,'ROOT',Path('/data2/neural-shrimp'))
 def no(*a,**k):raise AssertionError('target interval arithmetic forbidden')
 monkeypatch.setattr(D.P.R,'original_ball',no);monkeypatch.setattr(D.P.M,'point',no)
 c=D.read(R/D.CARD);c['status']='reviewed';p,fp,ap,mp=D.bind(c,Sink(),True)
 assert p.name=='attempt002'and fp.name=='attempt001'and ap.name=='attempt001'and mp.name=='attempt001'

def test_isolated_archive_and_limits(tmp_path):
 import shutil
 for n in D.FILES:
  src=R/n;assert src.is_file(),n;out=tmp_path/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
 code="import sys;from pathlib import Path;r=Path(sys.argv[1]);sys.path[:0]=[str(r/'src'),str(r/'tools/reviews')];import review_signed_output as d;assert d.LIMITS['native_calls']==0;assert len(d.FILES)>20"
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=tmp_path,env=env,check=True,timeout=30)
 assert D.LIMITS['seconds']==300 and D.LIMITS['rss_bytes']==6*1024**3
 text=(R/'tools/launch_signed_output_review.py').read_text();assert 'external315s' in text and '770*1024**2' in text
