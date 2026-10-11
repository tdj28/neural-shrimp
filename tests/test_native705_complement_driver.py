import copy,json,shutil,subprocess,sys
from pathlib import Path
from flint import arb,arb_mat,ctx
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_complement as D

def test_all144_literal_path_order_and_domains():
 for i in (1,36,37,72,73,108,109,144):
  stage=['profile_plus','profile_minus','target_plus','target_minus'][(i-1)//36];a=dict(stage=stage,phase=((i-1)%36)//2,direction='q'if i%2 else'r',inputs=[None]*15);b=dict(stage=stage,returns=[None]*13)
  r=D.path_event(a,b,i);assert r['d']['rad']==[0,0]and r['phase']==a['phase']
  bad=dict(a,phase=99)
  with pytest.raises(ValueError):D.path_event(bad,b,i)
  with pytest.raises(ValueError):D.path_event(a,dict(b,returns=[None]*12),i)

def test_complete_native_acquisitions_no_missing_extra_pair(tmp_path):
 p=tmp_path/'native';p.mkdir()
 for i in range(1,4):
  for kind in ('input','return'):(p/f'{i:03d}-{kind}.json').write_text('{}')
 D.native_inventory(tmp_path,3);(p/'002-return.json').unlink()
 with pytest.raises(ValueError):D.native_inventory(tmp_path,3)
 (p/'002-return.json').write_text('{}');(p/'004-input.json').write_text('{}')
 with pytest.raises(ValueError):D.native_inventory(tmp_path,3)

def test_full_actual_rate_tangent_binding_and_output_phase():
 b=D.ball_json;zero=[b(arb(0))]*704;dom=dict(center=zero,whole=zero,tangent=zero);inp=dict(kind='whole_tangent',phase=17,state=zero[:13],rates=zero[:2],state_tangent=zero[:13],rate_tangent=zero[:2]);out=dict(kind='whole_tangent',phase=17,A=D.H.matrix_json(D.M.ident(13)),Aprime=D.H.matrix_json(arb_mat(13,13)))
 D.native_return(inp,out,dom,'whole_tangent',17)
 a=copy.deepcopy(inp);a['rate_tangent'][1]=b(arb(1))
 with pytest.raises(ValueError):D.native_return(a,out,dom,'whole_tangent',17)
 with pytest.raises(ValueError):D.native_return(inp,dict(out,phase=16),dom,'whole_tangent',17)

def test_complete_fixed_card_and_launcher():
 c=json.loads((ROOT/D.CARD).read_text());assert c['inherited_sources']==dict(independent='b53ddc9ab3801b5102c66ec4a587876986d1c04c',connection='ff0372a9fc9bf658f345c02f33e61d336f87ce79')
 assert all('source'not in c['inputs']['original']['curve'][k]for k in c['inherited_sources'])
 assert c['audit']['own_intersections']==6 and c['audit']['output']==str(D.OUT)
 s=(ROOT/'tools/reviews/launch_native705_complement_review.py').read_text();assert "'internal_seconds':180"in s and "'external_seconds':195"in s and str(D.OUT.parent)in s and 'review_native705_complement.py'in s

def test_full_arithmetic_profile_and_final_failure_reserve(tmp_path):
 with ctx.workprec(480):assert D.synthetic(8,lambda:None)['passed']
 used=[62*1024**2-10]
 with pytest.raises(ValueError):D.save_bound(tmp_path,'overflow.json',dict(v='x'*100),used)
 D.save_bound(tmp_path,'failure.json',dict(passed=False),used,True)
 D.save_bound(tmp_path,'manifest.json',dict(files={'failure.json':D.entry(tmp_path/'failure.json')}),used,True)

def test_isolated_archive_import_closure(tmp_path):
 for n in D.FILES:
  p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
 code='import sys;sys.path.insert(0,'+repr(str(tmp_path/'tools/reviews'))+');import review_native705_complement as d;assert d.ROOT==__import__("pathlib").Path('+repr(str(tmp_path))+');assert all((d.ROOT/n).is_file() for n in d.FILES)'
 subprocess.run([sys.executable,'-I','-B','-c',code],cwd=tmp_path,check=True,capture_output=True)
