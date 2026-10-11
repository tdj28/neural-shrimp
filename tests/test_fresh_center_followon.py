import copy
from neural_shrimp import fresh_center_followon as H
import pytest


def outcome(c,passed=True):return dict(cell=c,passed=passed,completed=True,native_calls=126,dimension=705,complement_dimension=11)


def test_exact_two_domain_union_and_full_fine_partition():
 assert H.complete([outcome(H.cell(d))for d in(1,2)])['passed']
 fine=[outcome(H.cell(d,3,i))for d in(1,2)for i in range(8)]
 assert H.complete(fine)['accepted_cells']==16
 with pytest.raises(ValueError):H.complete(fine[:-1])
 with pytest.raises(ValueError):H.complete([fine[0]]+fine)


def test_only_complete_metric_failure_splits():
 c=H.cell(1);o=outcome(c,False);assert H.children(c,o)==[H.cell(1,1,0),H.cell(1,1,1)]
 for key,value in [('completed',False),('native_calls',125),('passed',True)]:
  bad=copy.deepcopy(o);bad[key]=value
  with pytest.raises(ValueError):H.children(c,bad)
 with pytest.raises(ValueError):H.children(H.cell(1,3),outcome(H.cell(1,3),False))


def test_worst_case_full_tree_count_and_measured_limits():
 assert sum(2**d for d in range(4))*2==30
 assert 144+2*90+30*126==4104
 assert H.forecast(16,20*1024**2,37,67*1024**2,20,1200000)['passed']
 assert not H.forecast(160,20*1024**2,370,67*1024**2,20,1200000)['passed']
