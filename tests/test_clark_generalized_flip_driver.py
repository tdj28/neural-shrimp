import copy,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from flint import arb,ctx
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools'))
import certify_clark_rnn_generalized_flip as D


def test_exact_full_root_is_mandatory_before_conditional_proofs():
 box=[arb(0)]*26;box[12]=arb(1)/2;box[-1]=arb(4);root=dict(passed=True,coordinate_inclusions=[True]*26)
 D.exact_root_gate(root,box,arb(0))
 for bad in(dict(root,passed=False),dict(root,coordinate_inclusions=[True]*25),dict(root,coordinate_inclusions=[False]+[True]*25)):
  with pytest.raises(ValueError):D.exact_root_gate(bad,box,arb(0))
 for bad in(box[:-1],box[:12]+[arb(1)]+box[13:],box[:-1]+[arb(-1)]):
  with pytest.raises(ValueError):D.exact_root_gate(root,bad,arb(0))
 with pytest.raises(ValueError):D.exact_root_gate(root,box,arb(1))


def test_original_full_native_qualifications_and_reverse_joins():
 point=dict(passed=True,gates={'native':True});join=dict(reverse_native_passed=True,reverse={'passed':True},forward={'passed':True},reverse_q_error=0.)
 v=dict(status='completed_fixed_native_quintic_unfolding',points=[copy.deepcopy(point)for _ in range(9)],associations=[copy.deepcopy(join)for _ in range(8)],unfolding=dict(candidate=True,strict_cubic_bracket=True));D.parent_gate(v)
 for mutation in('point','reverse','count','candidate'):
  bad=copy.deepcopy(v)
  if mutation=='point':bad['points'][3]['gates']['native']=False
  if mutation=='reverse':bad['associations'][2]['reverse']['passed']=False
  if mutation=='count':bad['points'].pop()
  if mutation=='candidate':bad['unfolding']['candidate']=False
  with pytest.raises(ValueError):D.parent_gate(bad)


def test_literal_full_cycle_mode_association_and_gain_bracket():
 with ctx.workprec(128):
  old=np.r_[np.arange(12)/10,.75,np.eye(12)[0]];cycle=np.repeat(old[:12][None,:],6,axis=0);arrays=dict(physical=old,cycle=cycle);box=[D.exact_float(v)for v in old]+[arb(4)];context={'orbit_enclosures':[D.vector_json([D.exact_float(v)for v in row])for row in cycle]};bracket=[arb(3),arb(5)]
  assert D.association(box,context,arrays,bracket)['passed']
  for i in(0,12,13,25):
   bad=list(box);bad[i]+=1
   with pytest.raises(ValueError):D.association(bad,context,arrays,bracket)
  bad=copy.deepcopy(context);bad['orbit_enclosures'][3][7]=D.ball_json(arb(10))
  with pytest.raises(ValueError):D.association(box,bad,arrays,bracket)


def test_fixed_source_paths_call_ceiling_and_owned_launcher():
 assert D.CAPS==dict(augmented=9,path5=10,normal=2,context=2,unfolding=2,native_phase_returns=69,events=256)
 c=json.loads((ROOT/'docs/reviews/clark-rnn-n12-quintic-outcome.json').read_text());assert(c['outer'],c['worker'],c['source'])==(D.OUTER,D.WORKER,D.EXECUTION)
 s=(ROOT/'tools/launch_clark_rnn_generalized_flip.py').read_text();assert str(D.OUT.parent)in s and 'tools/certify_clark_rnn_generalized_flip.py'in s and "'internal_seconds':180"in s and "'worker_rss_kib':2*1024**2"in s and '258*1024**2'in s


def test_complete_isolated_archive_import(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code='import sys;sys.path.insert(0,'+repr(str(tmp_path/'tools'))+');import certify_clark_rnn_generalized_flip as d;assert d.ROOT==__import__("pathlib").Path('+repr(str(tmp_path))+');assert all((d.ROOT/n).is_file() for n in d.FILES)'
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
