import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fractions import Fraction as F
import pytest
from neural_shrimp import fresh_center_quarter as Q


def fixture():return [dict(index=r['index'],endpoints=r['endpoints'],passed=True,native_calls=126,dimension=705,complement_dimension=11)for r in Q.partition()]

def test_exact_full17_cell_union_without_duplicate_center():
 rows=Q.partition();assert len(rows)==16 and sum(F(r['endpoints'][1])-F(r['endpoints'][0])for r in rows)==F(15,1024)
 assert sum(F(r['halfwidth'])==F(1,4096)for r in rows)==2 and sum(F(r['halfwidth'])==F(1,2048)for r in rows)==14
 assert Q.complete_union(fixture(),dict(passed=True,endpoints=['-17/1024','-1/64']))['total_cells']==17

def test_no_partial_failed_shifted_or_duplicate_cover():
 prior=dict(passed=True,endpoints=['-17/1024','-1/64'])
 for kind in ['missing','false','shift','duplicate','dimension','calls']:
  r=fixture()
  if kind=='missing':r.pop()
  elif kind=='false':r[3]['passed']=False
  elif kind=='shift':r[3]['endpoints'][0]='0'
  elif kind=='duplicate':r[3]=r[2]
  elif kind=='dimension':r[3]['complement_dimension']=10
  else:r[3]['native_calls']=125
  with pytest.raises(ValueError):Q.complete_union(r,prior)

def test_measured_complete_unit_cost_floor_and_caps():
 r=Q.forecast(8,20*1024**2,36.4879,66721025,12,1131356);assert r['passed']and r['total_native_calls']==2160 and r['forecast_bytes']<3*1024**3
 assert not Q.forecast(50,20*1024**2,36.4879,66721025,12,1131356)['passed']
 assert not Q.forecast(8,20*1024**2,36.4879,120*1024**2,12,1131356)['passed']
