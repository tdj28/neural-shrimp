import copy,importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
from fractions import Fraction as F
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_march as D
import review_native705_march_math as Q

def accepted(card):return [dict(passed=True,same_root=True,complement_dimension=11,ordinal=r['ordinal'],cell=r['cell'])for p in card['packets']for r in p['rows']if r['passed']]

def test_all48_exact_closed_unions_and_no_cross_gauge_interpolation():
 card=D.read(ROOT/D.CARD);rows=accepted(card);charts=card['packets'][0]['fixed_index']['charts'];got=Q.closed_unions(rows,charts)
 assert len(got)==24 and sum(v['leaves']for v in got)==48 and [v['ordinal']for v in got if not v['same_gauge_with_predecessor']]==[7,15,23]
 assert sorted(v['leaves']for v in got)==[1]*4+[2]*18+[4]*2
 bad=copy.deepcopy(rows);bad[0]['cell']['center']='0'
 with pytest.raises(ValueError):Q.closed_unions(bad,charts)
 with pytest.raises(ValueError):Q.closed_unions(rows[:-1],charts)
 bad=copy.deepcopy(rows);bad[0]=copy.deepcopy(bad[1])
 with pytest.raises(ValueError):Q.closed_unions(bad,charts)

def test_fixed_all72_actual_metadata_contracts_before_any_math():
 card=D.read(ROOT/D.CARD)
 for packet in card['packets']:
  root=Path(packet['root']);assert D.entry(root/'manifest.json')['sha256']==packet['outer'];assert D.entry(root/'fixed-index.json')==packet['fixed_index_identity'];assert D.read(root/'fixed-index.json')==packet['fixed_index'];done=D.read(root/'launcher-result.json');assert done['failure']is None
  for row in packet['rows']:
   D.canonical_assignment(row,packet,packet['fixed_index']);outer=Path(row['root']);run=outer/'run';assert D.entry(outer/'manifest.json')==row['outer']and D.entry(run/'manifest.json')==row['worker'];job=D.read(outer/'attempt-card.json');out=D.read(run/'outcome.json');dec=D.read(outer/'decision.json');rt=D.read(run/'runtime.json')
   assert rt['sources'][packet['source_index_path']]==packet['source_index'];assert D.read(run/'source'/packet['source_index_path'])==packet['fixed_index'];assert job==out['assignment']and job['input_index']==packet['source_index']and rt['native_call_cap']==360 and out['native_calls']==360 and out['profile_native_calls']==out['target_native_calls']==180 and dec['action']==row['action']
   if row['passed']:
    b=D.read(run/'bindings.json');assert b['original']['card']==packet['fixed_index']and b['cell']==row['cell']and b['original']['index']==packet['fixed_index']['charts'][row['ordinal']-7]
    dom=D.read(run/'quotient-domain-inputs.json');assert dom['halfwidth']==row['cell']['halfwidth']and dom['parameter_endpoints']==row['cell']['endpoints']

def test_generic_restriction_formula_keeps_original_closed_support():
 from flint import arb,arb_mat,ctx
 with ctx.workprec(480):
  for dep in range(4):
   for i in range(2**dep):
    sc=F(15,2048);h=F(1,128)/2**dep;c=sc+(2*i+1-2**dep)*h;cell=dict(center=str(c),depth=dep,endpoints=list(map(str,[c-h,c+h])),halfwidth=str(h),index=i,shift=str(c-sc));Q.geometry(dict(parameter_center=str(sc),parameter_endpoints=['-1/2048','31/2048']),cell)
    out=Q.translate([arb(1)],[arb(2)],[arb(3)],[arb(1)],arb('1/128'),arb('1/1024'),Q.R.exact(c-sc),Q.R.exact(h),arb_mat([[0,0,0,0]]),arb_mat([[0,0,0,0]]));assert out['h']==Q.R.exact(h)and out['r']==[arb('1/1024')]

def test_archive_closure_and_launcher_path():
 with tempfile.TemporaryDirectory()as name:
  root=Path(name)
  for n in D.FILES:
   p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
  code="import sys;sys.path.insert(0,'tools/reviews');import review_native705_march as d;assert callable(d.main);assert callable(d.S.B.canonical_bytes)"
  subprocess.run([sys.executable,'-B','-c',code],cwd=root,check=True,capture_output=True)
 source=(ROOT/'tools/reviews/launch_native705_march_review.py').read_text();assert 'review_native705_march.py'in source and '1815'in source and '258*1024**2'in source and '6*1024**3'in source

def test_old_measured_time_storage_and_cap_are_explicit():
 card=D.read(ROOT/D.CARD);prior=Path(card['prior_measured']['root']);out=D.read(prior/'run/outcome.json');m=D.read(prior/'manifest.json');assert 36*out['elapsed_seconds']+90<1800 and 36*sum(v['bytes']for v in m['files'].values())+8*1024**2<252*1024**2
 assert card['settings']==D.settings()
