import copy,importlib.util,subprocess,sys
from pathlib import Path
from fractions import Fraction
import pytest
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[1]

def module(name):
 sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
 spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_exact_eight_domains_and_ten_batch_ceiling():
 w=module('probe_native705_rotated');prev=None
 for i in range(1,9):
  c=Fraction(w.settings(i)['parameter']);lo=c-Fraction(1,128);hi=c+Fraction(1,128)
  assert lo==(-Fraction(1,2048)if prev is None else prev);prev=hi
 assert prev==Fraction(255,2048)
 for b,u in [(0,1),(11,1),(1,0),(1,9),(True,1),(1,True)]:
  with pytest.raises(ValueError):w.paths(b,u)
 assert w.paths(10,8)[1].parts[-3:]==('batch010','chart008','run')

def test_complete_705_gates_required():
 w=module('probe_native705_rotated');v={'shooting_dimension':705,'dimension':13,'period':18,'coordinate_inclusions':[True]*705,'gates':{'uniform_full_native_unit_pair_locus':True,'primitive_period':True,'realified_frame_independent':True,'anchor_identified':False}}
 assert w.uniform_admission(v,True)
 assert not w.uniform_admission(v,False)
 for key in v['gates']:
  x=copy.deepcopy(v);x['gates'][key]=not x['gates'][key];assert not w.uniform_admission(x,True)
 for arr in [[True]*704,[True]*706,[True]*704+[False]]:
  x=copy.deepcopy(v);x['coordinate_inclusions']=arr;assert not w.uniform_admission(x,True)

def card(w):
 return {'source':'head','batch':1,'worker_count':8,'kernel':'unchanged-full705-quadratic-v1','settings':[w.settings(i)for i in range(1,9)],'normal':['1']*705,'origin':['0']*705,'weights':['1/16']*705,'anchor':[{}]*705,'initials':[['0']*705 for _ in range(8)]}

def test_literal_batch_settings_and_full_proposal_binding():
 w=module('probe_native705_rotated');v=card(w);assert w.validate_card(v,1,'head')is v
 for key,value in [('source','wrong'),('batch',2),('worker_count',7),('normal',['0']*704),('weights',['-1']*705),('initials',v['initials'][:-1]),('settings',v['settings'][:-1])]:
  x=copy.deepcopy(v);x[key]=value
  with pytest.raises(ValueError):w.validate_card(x,1,'head')

def test_measured_forecast_retains_failed_complete_attempts():
 l=module('launch_native705_rotated');row={'ticks':253,'events':l.EXPECTED_EVENTS,'status':'completed_bounds_failed','elapsed_seconds':57.3,'peak_rss_kib':1785204,'stored_bytes_before_final':152028398};rows=[copy.deepcopy(row)for _ in range(4)]
 f=l.measured_forecast(rows);assert f=={'seconds':3*57.3+20,'rss_kib':2*1785204+128*1024,'bytes':2*152028398+8*1024**2};assert f['rss_kib']<4*1024**2
 rows[3]['events']['direction_return']=143
 with pytest.raises(ValueError):l.measured_forecast(rows)

def test_rotated_join_projects_full_box_without_old_exact_parameter(monkeypatch):
 l=module('launch_native705_rotated');seen=[]
 monkeypatch.setattr(l,'anchor_test',lambda a,*args:seen.append(a)or{'passed':True})
 vec=l.vector_json([arb(0)]*705)
 v={'settings':{'precision':256,'newton_steps':2,'halfwidth_exponent':-7,'radius_exponent':-10},'normal':['1']*705,'origin':['0']*705,'center':vec,'slope':vec,'curvature':vec,'coordinate_radii':l.vector_json([arb(1)]*705),'parameter_center':'15/2048','parameter_endpoints':['-1/2048','31/2048']}
 a={'box':vec,'normal':['2']*705,'origin':['3']*705,'parameter':'447/4096'}
 assert l.join(a,v)['passed']and'exact_parameter'not in seen[-1]and len(seen[-1]['box'])==705
 with pytest.raises(ValueError):l.join(a,v,same_gauge=True)
 a.update(normal=v['normal'],origin=v['origin'],parameter='-1/2048');assert l.join(a,v,same_gauge=True)['passed'];assert seen[-1]['exact_parameter']=='-1/2048'
 a['parameter']='0'
 with pytest.raises(ValueError):l.join(a,v,same_gauge=True)

def test_archive_closure_and_actual_worker_entrypoint():
 l=module('launch_native705_rotated');w=l.W
 loaded={str(Path(x.__file__).resolve().relative_to(ROOT))for x in list(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).resolve().is_relative_to(ROOT)and '/tests/'not in str(Path(x.__file__).resolve())}
 assert loaded<=set(w.SOURCES),sorted(loaded-set(w.SOURCES))
 assert all((ROOT/n).is_file()for n in w.SOURCES)
 p=subprocess.run([sys.executable,str(ROOT/'tools/probe_native705_rotated.py'),'--help'],capture_output=True,text=True);assert p.returncode==0 and'--batch'in p.stdout and'--card-sha'in p.stdout
