import pytest
from neural_shrimp.complement_partition import cell,children,next_action,coverage


def test_exact_nonuniform_cover_and_unique_cells():
    c,h='15/2048','1/128'
    cells=[cell(c,h,1,0),cell(c,h,2,2),cell(c,h,2,3)]
    assert coverage(c,h,cells)['complete']
    assert not coverage(c,h,cells[:2])['complete']
    with pytest.raises(ValueError):coverage(c,h,cells+[cells[0]])
    with pytest.raises(ValueError):coverage(c,h,[cell(c,h,0,0),cells[0]])


def test_shared_closed_midpoint_and_bounded_tree():
    x=cell('383/4096','1/64',0,0);kids=[cell('383/4096','1/64',d,k)for d,k in children(x)]
    assert kids[0]['endpoints']==['319/4096','383/4096']and kids[1]['endpoints']==['383/4096','447/4096']
    for args in [('0','0',0,0),('0','1',4,0),('0','1',1,2),('0','1',True,0)]:
        with pytest.raises(ValueError):cell(*args)


def test_only_full_negative_metric_can_split():
    c=cell('0','1/128',0,0);o=dict(native_calls=180,status='completed_correlated_bound_unresolved',passed=False,metric_passed=False,complement_dimension=11)
    assert next_action(o,c)=='split'
    for k,v in [('native_calls',179),('status','failed_or_interrupted'),('complement_dimension',10),('metric_passed',True)]:
        bad=dict(o,**{k:v});assert next_action(bad,c)=='stop'
    assert next_action(o,cell('0','1/128',3,0))=='stop'
    o.update(status='whole_chart_complement_attracting',passed=True,metric_passed=True);assert next_action(o,c)=='accept'
