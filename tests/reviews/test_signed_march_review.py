import copy,os,subprocess,sys
from pathlib import Path
from fractions import Fraction as F
import pytest
from flint import arb
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
import review_signed_march as W
import review_signed_march_math as T
from neural_shrimp.rigorous_orbits import ball_json

def history():
 current=F(-81,256);width=-3;rows=[];inputs=[];prior=None
 for i,action in enumerate(('advance','advance','retry','advance'),1):
  h=F(2)**width;ends=[str(current-2*h),str(current)];inp=dict(previous_accepted=prior,parameter=str(current-h),parameter_endpoints=ends,known=dict(exact_parameter=str(current)),settings=dict(halfwidth_exponent=width,radius_exponent=-12));inputs.append(inp)
  good=action=='advance';nextwidth=-2 if i==2 else -3
  strong=1 if i in(1,4)else 0
  rows.append(dict(index=i,native_calls=144,ticks=150,parameter_endpoints=ends,gates=dict(uniform_normalized_eigenpair_locus=good,anchor_identified=True,primitive_period=True),decision=dict(action=action,width=nextwidth,strong=strong),passed=good,strict_inclusions=470 if good else 469,anchor_inclusions=470,contraction=ball_json(arb(1)/10)))
  if good:current-=2*h;prior=i
  width=nextwidth
 out=dict(attempts=rows,accepted_indices=[1,2,4],connected=True,last_known_parameter=str(current),native_calls=180+4*144,ticks=186+4*150);return out,rows,inputs

def test_complete_failed_width_history_and_actual_endpoint_not_last_trial():
 o,r,i=history();assert W.check_history(o,r,i,'-81/256')==[1,2,4]
 for change in('prior','count','failedadvance','width'):
  oo,rr,ii=copy.deepcopy((o,r,i))
  if change=='prior':ii[-1]['previous_accepted']=3
  if change=='count':oo['native_calls']-=1
  if change=='failedadvance':oo['accepted_indices']=[1,2,3,4]
  if change=='width':ii[-1]['settings']['halfwidth_exponent']=-2
  with pytest.raises(ValueError):W.check_history(oo,rr,ii,'-81/256')

def test_successful_root_join_failure_must_stop():
 o,r,i=history();r[2]['gates']['uniform_normalized_eigenpair_locus']=True;r[2]['gates']['anchor_identified']=False
 with pytest.raises(ValueError):W.check_history(o,r,i,'-81/256')

def test_exact_variable_width_scalar_and_full_gauge():
 normal=['0']*468+['1','0'];origin=['0']*470;known=dict(normal=normal,origin=origin,box=[{}]*470,dimension=13,period=18,sigma=-1,exact_parameter='-81/256');inp=dict(normal=normal,origin=origin,known=known,parameter='-145/256',parameter_endpoints=['-209/256','-81/256'],settings=dict(halfwidth_exponent=-2,radius_exponent=-12));chart=dict(normal=normal,origin=origin,parameter_endpoints=inp['parameter_endpoints']);T.check_geometry(inp,known,chart)
 for change in ('scalar','gauge','mode','dimension'):
  ii,kk,cc=copy.deepcopy((inp,known,chart))
  if change=='scalar':kk['exact_parameter']='-80/256'
  elif change=='gauge':kk['normal'][0]='1'
  elif change=='mode':kk['sigma']=1
  else:kk['box']=kk['box'][:-1]
  with pytest.raises(ValueError):T.check_geometry(ii,kk,cc)

def test_fixed_caps_and_archive_closure(tmp_path):
 for n in W.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
 code="import sys;from pathlib import Path;sys.path[:0]=['tools/reviews','src'];import review_signed_march as w;r=Path.cwd();mods={str(Path(m.__file__).resolve().relative_to(r))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(r)};assert mods<=set(w.FILES),mods-set(w.FILES)"
 p=subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,text=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));assert p.returncode==0,p.stderr
 assert W.LIMITS['native_calls']==0 and W.LIMITS['precision']==512 and W.LIMITS['unit_seconds']==300


def test_complete_actual_metadata_no_target_arithmetic(monkeypatch):
 class W0:
  def tick(self):pass
  def save(self,*a,**kw):pass
 def no(*a,**kw):raise AssertionError('target derivative/polynomial reduction forbidden in metadata')
 monkeypatch.setattr(T.D,'jacobian',no);monkeypatch.setattr(T.H,'polynomial_root',no)
 card=W.read(ROOT/W.CARD);card['status']='reviewed';p,a=W.bind(card,W0(),True);assert p.name=='attempt001'and a.name=='attempt001'and len(card['accepted_indices'])==16
