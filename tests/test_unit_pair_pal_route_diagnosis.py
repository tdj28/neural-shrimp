import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.unit_pair_pal_route_diagnosis import route_lengths,diagnose

def point(x):
 a=np.zeros(704);a[0:2]=x;return {'native_flat':a.tolist(),'t':0.}

def test_full705_polygonal_length_preserves_duplicate_native_join():
 r=route_lengths([point([0,0]),point([3,4]),point([3,4]),point([0,4])])
 assert r['segment_lengths']==[5,0,3]and r['duplicate_indices']==[2]
 assert [p['polygonal_arclength']for p in r['points']]==[0,5,5,8]
 assert r['points'][-1]['chord_from_first']==4


def test_all_coordinates_and_no_nonfinite_or_empty_route():
 a=point([0,0]);b=point([0,0]);b['native_flat'][703]=12;b['t']=5
 r=route_lengths([a,b]);assert r['segment_lengths']==[13]
 for v in [[],[a], [a,{'native_flat':[0]*703,'t':0}], [a,{'native_flat':[0]*703+[float('nan')],'t':0}]]:
  with pytest.raises(ValueError):route_lengths(v)


def test_no_relabeling_success_or_joined_failure():
 v={'schema':'neural-shrimp-native-unit-pair-pal-v1','shooting_dimension':705,'gates':{'anchor_identified':False,'primitive_period':True,'realified_frame_independent':True,'uniform_full_native_unit_pair_locus':True}}
 with pytest.raises(ValueError):diagnose(v)
 v['gates'].update(anchor_identified=True,uniform_full_native_unit_pair_locus=False)
 with pytest.raises(ValueError):diagnose(v)


def test_complete_diagnostic_archive_closure_and_fixed_path():
 root=Path(__file__).resolve().parents[1];sys.path[:0]=[str(root/'tools')]
 spec=importlib.util.spec_from_file_location('route_driver',root/'tools/diagnose_native705_route.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 loaded={str(Path(x.__file__).resolve().relative_to(root))for x in list(sys.modules.values())if getattr(x,'__file__',None)and Path(x.__file__).resolve().is_relative_to(root)and '/tests/'not in str(Path(x.__file__).resolve())}
 assert loaded<=set(m.SOURCES),sorted(loaded-set(m.SOURCES))
 assert all((root/n).is_file()for n in m.SOURCES)
 assert m.OUT.name=='run'and m.OUT.parent.name=='rotated001-diagnosis'
