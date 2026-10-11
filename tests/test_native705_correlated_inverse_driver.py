import copy,importlib.util,subprocess,sys
from pathlib import Path
from fractions import Fraction
import pytest
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=load('probe_native705_correlated_inverse');L=load('launch_native705_correlated_inverse')

def valid():
 e={'direction_return':180,'pal_path_phase_return':252,'pal_chart_return':1,'gradient_path_input':144,'gradient_path_return':144,'gradient_path_assembly_return':4,'correlated_domain':2,'correlated_chart_return':2,'profile_gradient_input':24,'profile_gradient_return':24,'pal_profile_path_return':24}
 v={'passed':True,'target_acquired':False,'ticks':629,'events':e,'elapsed_seconds':10,'peak_rss_kib':100000,'stored_bytes_before_final':10000,'chart_times':[1,2],'gradient_times':[.01]*12,'path_times':[.01]*12}
 v['projection']=C.P.projection(v['chart_times'],v['gradient_times'],v['path_times'],v['peak_rss_kib'],v['stored_bytes_before_final']);return v

def test_exact_full_profile_admission_and_missing_native_return_rejection():
 assert C.admit_profile(valid())['resource_admitted']
 for key in ['gradient_path_return','pal_path_phase_return','correlated_chart_return']:
  v=valid();v['events'][key]-=1
  with pytest.raises(ValueError):C.admit_profile(v)
 for field,value in [('ticks',628),('passed',False),('target_acquired',True),('elapsed_seconds',240),('peak_rss_kib',6*1024**2)]:
  v=valid();v[field]=value
  with pytest.raises(ValueError):C.admit_profile(v)

def test_forecast_recomputed_not_claimed_and_all_sources_present():
 v=valid();v['projection']['one_fresh_chart_seconds']/=2
 with pytest.raises(ValueError):C.admit_profile(v)
 assert len(C.SOURCES)==len(set(C.SOURCES))and all((ROOT/n).is_file()for n in C.SOURCES)
 assert set(C.P.SOURCES)<=set(C.SOURCES)
 assert 'src/neural_shrimp/rigorous_parameter_localization.py'in C.SOURCES
 assert 'src/neural_shrimp/rigorous_connection.py'in C.SOURCES

def test_fixed_provenance_and_wider_domain_preserves_known_parameter_margin():
 assert C.AUDIT.name=='attempt004'and C.APINS['endpoint']=='d56ed2992697924b767e16b76e7cd1271384899b039e52e8d416f790318ac9be'
 sc=Fraction(C.SETTINGS['parameter']);h=Fraction(2)**C.SETTINGS['halfwidth_exponent']
 assert h==4*Fraction(1,128)and sc-h<Fraction(-1,2048)<sc+h
 assert C.SETTINGS['radius_exponent']==-10 and C.SETTINGS['newton_steps']==0
 assert C.TARGET==L.BASE/'attempt001-target/run'
 assert C.TARGET!=C.P.OUT

def test_actual_launcher_entrypoint_help_and_required_pins():
 p=subprocess.run([sys.executable,str(ROOT/'tools/launch_native705_correlated_inverse.py'),'--help'],text=True,capture_output=True)
 assert p.returncode==0 and all(s in p.stdout for s in ['--out','--profile-outer','--profile-worker'])
 with pytest.raises(SystemExit):C.parser().parse_args(['--out',str(C.TARGET)])


def test_changed_native_map_gradient_or_closure_bytes_rejected():
 identity={n:{'sha256':'same','bytes':1}for n in C.NATIVE_SOURCES}
 C.same_native_source(identity,copy.deepcopy(identity))
 for n in C.NATIVE_SOURCES:
  bad=copy.deepcopy(identity);bad[n]['sha256']='changed'
  with pytest.raises(ValueError):C.same_native_source(identity,bad)
  del bad[n]
  with pytest.raises(ValueError):C.same_native_source(identity,bad)
