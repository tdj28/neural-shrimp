import hashlib,importlib.util,sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import review_cubic_center as C


def test_exact_domain_diagnostic_keeps_missed_coordinates():
 old=[C.ball_json(arb(0,1)),C.ball_json(arb(2))];new=[arb(0,2),arb(2)]
 report=C.coverage_diagnostic(old,new)
 assert not report['passed']and report['coordinate_containment']==[False,True]
 assert [v['coordinate']for v in report['failures']]==[0]
 assert C.F(report['failures'][0]['lower_excess'])>0 and C.F(report['failures'][0]['upper_excess'])>0


def fixture(monkeypatch):
 n=705;zero=[arb(0)]*n;spec=dict(domain=dict(index=1,endpoints=['0','1/128']),start=144)
 td=dict(domain=spec['domain'],scale='1/256',original_offset=['0','1/128'],normal=C.vector_json(zero),origin=C.vector_json(zero),parameter=C.ball_json(arb(1)/256),segment=C.vector_json(zero),error=[[v]for v in C.vector_json(zero)],predictor_tangent=C.vector_json(zero),plus=C.vector_json(zero),minus=C.vector_json(zero))
 qd=dict(domain=spec['domain'],complements=[],metrics=[],full705=dict(whole=C.vector_json(zero),tangent=C.vector_json(zero)))
 def read(p):return td if p.name=='tangent-domain-inputs.json'else qd if p.name=='quotient-domain-inputs.json'else {}
 monkeypatch.setattr(C,'read',read);monkeypatch.setattr(C.Q,'root',lambda *a:(arb_mat(n,1),dict(passed=True,error=[[v]for v in C.vector_json(zero)])))
 class Work:
  def __init__(self):self.saved={}
  def tick(self):pass
  def save(self,name,row):self.saved[name]=row
 data=dict(c=zero,s=zero,a=zero,b=zero,sc=arb(0),normal=zero,origin=zero,ds=[],ff=[],weights=[],original=None,image=None,C=[],rp=[],E=[],metrics=[],domain=dict(whole_box=C.vector_json(zero)))
 return Work(),spec,data


def test_domain_failure_retains_root_and_whole_before_check(monkeypatch):
 w,s,d=fixture(monkeypatch)
 def fail(*a):raise ValueError('strict acquired coverage')
 monkeypatch.setattr(C.Q,'cover',fail)
 with pytest.raises(ValueError,match='strict acquired'):C.shared_domain(w,Path('/unread'),s,d)
 assert 'domains/001/root.json'in w.saved and 'domains/001/whole-domain.json'in w.saved
 assert len(w.saved['domains/001/whole-domain.json']['whole'])==705


def test_tangent_failure_retains_rhs_and_actual_comparison(monkeypatch):
 w,s,d=fixture(monkeypatch)
 def cover(old,new,label):
  if label=='actual705 tangent':raise ValueError('strict acquired tangent')
 monkeypatch.setattr(C.Q,'cover',cover);monkeypatch.setattr(C.A.A,'path_event',lambda *a:None);monkeypatch.setattr(C.H,'polarized_paths',lambda *a:arb_mat(705,1));monkeypatch.setattr(C.Q,'tangent',lambda *a:([arb(0,1)]*705,dict(passed=True,rhs=C.matrix_json(arb_mat(705,1)))))
 with pytest.raises(ValueError,match='strict acquired tangent'):C.shared_domain(w,Path('/unread'),s,d)
 assert 'domains/001/tangent.json'in w.saved and not w.saved['domains/001/tangent-domain-comparison.json']['passed']


def test_unchanged_independent_core_reducers():
 pins={'review_polynomial_center_math.py': '83a9763f457de9dcb3e2e1268f35085886d13dc032a4761d45ee29bbaf856380', 'review_native705_complement_math.py': 'e28e856290cff0ae13f8e4c736b5ed117072084485e4354ee76221f84a3b3a35', 'review_native704_complement_math.py': 'a86bb04c0f040a446b784dd4c88c8635c224c4729b8f1f941272c790c72a54bb', 'review_fresh_center.py': '6c56711d086ff6f4898a950b34e7c84a172491be133d6acbb050bcf7cf6b84cf', 'review_second_quarter.py': '8be838c89f152b100f1ef47c69b0369585743a9c1d998fe5033fdfdadf7c0f60'}
 for name,pin in pins.items():assert hashlib.sha256((ROOT/"tools/reviews"/name).read_bytes()).hexdigest()==pin
