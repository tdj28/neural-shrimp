import copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).parents[1]
def load(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
D=load('probe_native_unit_pair');L=load('launch_native_unit_pair')
def profile():
 v={'passed':True,'target_acquired':False,'ticks':422,'events':{'direction_return':420,'chart_return':2},'elapsed_seconds':20.,'peak_rss_kib':100000,'chart_times':[1.,1.1],'phase_times':[.02]*12,'stored_bytes_before_final':10000}
 v['projection']=D.P.projection(v['chart_times'],v['phase_times'],v['peak_rss_kib'],v['stored_bytes_before_final']);return v

def test_same_exact_source_profile_settings_and_own_paths():
 assert D.SETTINGS is D.P.SETTINGS and D.BASE==L.BASE
 assert D.BRIDGE==D.BASE.parent/'native-lift/membership-bridge/unit003'
 assert set(D.P.SOURCES)<=set(D.SOURCES)and set(D.U.SOURCES)<=set(D.SOURCES)
 assert D.admit_profile(profile())['resource_admitted']
@pytest.mark.parametrize('key,value',[('passed',False),('target_acquired',True),('ticks',421),('elapsed_seconds',240),('peak_rss_kib',4*1024**2)])
def test_incomplete_or_over_cap_profile_cannot_admit(key,value):
 p=profile();p[key]=value
 with pytest.raises(ValueError):D.admit_profile(p)

def test_profile_full_calls_and_measured_forecast_not_just_pass_label():
 p=profile();p['events']['direction_return']=419
 with pytest.raises(ValueError):D.admit_profile(p)
 p=profile();p['projection']['one_chart_plus_lift_seconds']=1
 with pytest.raises(ValueError):D.admit_profile(p)

def test_bridge_full_native_membership_and_identity():
 m={'passed':True,'parameter_inside':True,'coordinate_inclusions':[True]*470}
 out={'passed':True,'status':'rigorous_chain_membership','accepted_count':2,'membership':m}
 D.admit_bridge(out,m)
 for change in ('mode','rate','state'):
  bad=copy.deepcopy(m);bad['coordinate_inclusions'][{'state':0,'mode':13,'rate':469}[change]]=False
  with pytest.raises(ValueError):D.admit_bridge(out,bad)
 bad=copy.deepcopy(out);bad['status']='neutral_event'
 with pytest.raises(ValueError):D.admit_bridge(bad,m)

def test_launcher_fixed_target_and_profile_pins():
 L.OUT=L.BASE/'attempt001-target';L.LIMIT=255
 L.ARGS=SimpleNamespace(profile_manifest='a'*64,profile_outer_manifest='b'*64)
 cmd=L.command();assert 'probe_native_unit_pair.py'in cmd[5]and '--mode'not in cmd
 assert cmd[-4:]==['--profile-manifest','a'*64,'--profile-outer-manifest','b'*64]
