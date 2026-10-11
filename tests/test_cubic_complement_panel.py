from fractions import Fraction as F
import pytest
from neural_shrimp import cubic_complement_panel as H
from neural_shrimp import cubic_complement_geometry as G


def test_exact_full_chart_partitions_and_phase_budget():
 for bounds in [('205/2048','333/2048'),('83/512','115/512')]:
  rows=H.partition(bounds);assert len(rows)==8 and rows[0]['endpoints'][0]==bounds[0]and rows[-1]['endpoints'][1]==bounds[1]
  assert all(F(r['endpoints'][1])-F(r['endpoints'][0])==F(1,128)for r in rows)
 assert H.call_cap(16,96)==13680
 with pytest.raises(ValueError):H.partition(('0','1/127'))


def test_global_reservations_require_complete_predecessor():
 assert H.reserve(144,0,0,'domain',16,96)==90
 assert H.reserve(234,1,0,'cell',16,96)==126
 for args in [(235,1,0,'cell',16,96),(144+1440,16,0,'domain',16,96),(144+1440+12096,16,96,'cell',16,96),(144+90+12096,1,96,'domain',16,96)]:
  with pytest.raises(ValueError):H.reserve(*args)


def test_complete_chart_rejects_gap_duplicate_and_unfinished_cell():
 rows=H.partition(('0','1/16'));accepted=[dict(cell=G.cell(rows,i+1),passed=True,completed=True,native_calls=126,dimension=705,complement_dimension=11)for i in range(8)]
 assert H.chart_complete(rows,accepted)['accepted_cells']==8
 for bad in [accepted[:-1],accepted+[accepted[0]],[dict(accepted[0],passed=False)]+accepted[1:]]:
  with pytest.raises(ValueError):H.chart_complete(rows,bad)
