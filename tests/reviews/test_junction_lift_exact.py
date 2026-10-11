import copy,os,subprocess,sys
from fractions import Fraction as F
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_junction_lift_exact as R

def test_exact_dyadic_arithmetic_and_bad_intervals():
 assert R.bounds(R.ball(-F(3,8),F(5,8)))==(-F(3,8),F(5,8))
 assert R.mul((-F(2),F(3)),(-F(4),F(5)))==(-F(12),F(15))
 assert not R.encloses((F(0),F(1)),(-F(1),F(1)))
 with pytest.raises(ValueError):R.bounds(dict(mid=[1,0],rad=[-1,0]))

def test_full55_exact_enclosures_reject_defect_and_intersection_corruption():
 root,record,events=R.fixture();answer,box=R.refinement(root,record,events);assert answer['passed']and len(box)==55
 for what in('defect','image','retained','finalbox'):
  rr,ee=copy.deepcopy(record),copy.deepcopy(events)
  if what=='defect':ee[0]['defect'][0][0]=R.ball(2)
  elif what=='image':ee[1]['image'][0][0]=R.ball(1)
  elif what=='retained':ee[1]['retained'][0][0]=R.ball(1)
  else:rr['box'][0]=R.ball(1)
  with pytest.raises(ValueError):R.refinement(root,rr,ee)

def native_fixture():
 zero=R.ball(0);one=R.ball(1);x=[zero]*13;q=[one]+[zero]*12;r=[zero,one]+[zero]*11;rates=[R.ball(F(1,100))]if False else[R.ball(F(1,64))]*2
 box=x+rates+q+r+[R.ball(F(1,16))]+[zero]*13;events=[];grad=[[R.ball(i==j)for j in range(15)]for i in range(15)];mixed=[[zero]*15 for _ in range(15)];block=[[R.ball(i==j)for j in range(26)]for i in range(26)]
 for phase in range(18):
  for label,mode in [('q',q),('r',r)]:
   events.append(dict(phase=phase,direction=label,state=x,mode=mode,rates=rates,base_gradients=grad,mixed=mixed))
   events.append(dict(phase=phase,direction=label,input=x+mode+rates,value=[[v]for v in x+mode],block=block,rates=[[zero]*2 for _ in range(26)]))
 report=dict(dimension=13,period=18,shooting_dimension=705,native_calls=36,box=(x+q+r)*18+rates+[box[41]])
 return box,events,report

def test_all705_chain_and_literal_held_direction_seeds():
 box,events,report=native_fixture();assert R.lift(box,events,report)['coordinates']==705
 for what in('state','seed','packing'):
  e,p=copy.deepcopy(events),copy.deepcopy(report)
  if what=='state':e[68]['state'][0]=R.ball(1)
  elif what=='seed':e[0]['mixed'][1][1]=R.ball(1)
  else:p['box'][-1]=R.ball(F(1,8))
  with pytest.raises(ValueError):R.lift(box,e,p)

def test_source_closure_and_no_native_or_producer_reducer(tmp_path):
 for n in R.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews'];import review_junction_lift_exact as r;p=Path.cwd();a={str(Path(m.__file__).resolve().relative_to(p))for m in sys.modules.values()if getattr(m,'__file__',None)and Path(m.__file__).is_file()and Path(m.__file__).resolve().is_relative_to(p)};assert a<=set(r.FILES),a-set(r.FILES)"
 z=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True);assert z.returncode==0,z.stderr
 s=Path(R.__file__).read_text();assert 'rigorous_junction_lift'not in s and 'cusp_step'not in s
 s=(ROOT/'tools/reviews/launch_junction_lift_exact.py').read_text();assert "'internal_seconds':90"in s and "'external_seconds':105"in s and "'worker_rss_kib':1024**2"in s and "'output_cap_bytes':34*1024**2"in s


def test_literal_historical_authority():
 import copy
 known=dict(commit='source',pid=999999999)
 owner=dict(commit='source',exit_code=0,owned_live_pids=[])
 R.authority(known,owner,'source',True)
 for key,bad in [('commit','wrong'),('exit_code',2),('owned_live_pids',[1])]:
  changed=copy.deepcopy(owner);changed[key]=bad
  with pytest.raises(ValueError):R.authority(known,changed,'source',True)
