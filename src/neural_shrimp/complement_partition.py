"""Exact per-chart subdivision bookkeeping; carries no dynamical proof by itself."""
from fractions import Fraction as F


def cell(center,halfwidth,depth,index):
    center,halfwidth=F(center),F(halfwidth)
    if halfwidth<=0 or type(depth)is not int or not 0<=depth<=3 or type(index)is not int or not 0<=index<2**depth:
        raise ValueError('positive fixed chart and bounded dyadic cell')
    h=halfwidth/2**depth;left=center-halfwidth+2*index*h;right=left+2*h
    return dict(depth=depth,index=index,center=str(left+h),halfwidth=str(h),shift=str(left+h-center),endpoints=[str(left),str(right)])


def children(c):
    d,k=c['depth'],c['index']
    if d>=3:raise ValueError('subdivision depth cap')
    return [(d+1,2*k),(d+1,2*k+1)]


def next_action(outcome,c):
    """Only complete negative complementary bounds permit splitting, never partial calls."""
    if outcome.get('native_calls')!=180:return 'stop'
    if outcome.get('status')=='whole_chart_complement_attracting'and outcome.get('passed')is True and outcome.get('metric_passed')is True and outcome.get('complement_dimension')==11:
        return 'accept'
    if outcome.get('status')=='completed_correlated_bound_unresolved'and outcome.get('passed')is False and outcome.get('metric_passed')is False and outcome.get('complement_dimension')==11 and c['depth']<3:
        return 'split'
    return 'stop'


def coverage(center,halfwidth,cells):
    """Validate a disjoint-interior union of accepted closed cells in one exact gauge."""
    original=cell(center,halfwidth,0,0);left,right=map(F,original['endpoints']);intervals=[];seen=set()
    for v in cells:
        expected=cell(center,halfwidth,v['depth'],v['index'])
        if v!=expected or(v['depth'],v['index'])in seen:raise ValueError('exact unique canonical cell')
        seen.add((v['depth'],v['index']));intervals.append(tuple(map(F,v['endpoints'])))
    intervals.sort();cursor=left;gap=False
    for a,b in intervals:
        if a<cursor:raise ValueError('duplicate overlapping cell interiors')
        if a!=cursor:gap=True
        cursor=b
    return dict(complete=bool(intervals)and not gap and cursor==right,original=[str(left),str(right)],accepted=[list(map(str,v))for v in intervals],scope='Exact scalar coverage in this authenticated chart gauge only; each cell still requires its own full11-mode proof and independent replay.')
