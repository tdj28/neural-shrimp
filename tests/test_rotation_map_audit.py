import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.rotation_map_audit import (DEFAULTS, analyze, geometric_edges,
    provisional_roots, segment_distance_envelope, validate_map)


def row(o,h=.5,angle=.125,kind='neimark_sacker'):
    return dict(log10_output=o,log10_hidden_boundary=h,theta_over_2pi=angle,type=kind)


def grid():
    h=o=np.linspace(0,1,101)
    shape=(101,101)
    return dict(log10_hidden=h,log10_output=o,rho=np.full(shape,.125),
                status=np.full(shape,4),period=np.zeros(shape,dtype=int),
                ftle=np.zeros(shape),planar_share=np.ones(shape))


def test_gaps_and_jumps_never_join():
    rows=[row(0),row(.1,kind='start_not_period_2'),row(.2),row(.3,h=.7),row(.4,h=.7,angle=.2),row(.5,h=.701,angle=.201)]
    edges=geometric_edges(rows,DEFAULTS)
    assert [e['reason'] for e in edges]==['non_ns_endpoint','non_ns_endpoint','hidden_jump','angle_jump','geometrically_admissible']
    assert len(provisional_roots(grid(),rows,edges,DEFAULTS,6000))==0
    assert geometric_edges([row(0),row(.1),row(.2),row(.8)],DEFAULTS)[-1]['reason']=='output_gap'


def test_families_retained_with_exclusions_and_no_best_angle_selection():
    m=grid(); m['status'][51,50]=3
    families={'a':dict(args=dict(start=1.1),rows=[row(.5),row(.6)]),
              'b':dict(args=dict(start=.95),rows=[row(.5,angle=.2),row(.6,angle=.2)])}
    cfg={**DEFAULTS,'offsets':[.01]}
    r=analyze(m,families,cfg,6000)
    assert set(r['families'])=={'a','b'}
    a,b=[r['families'][k]['comparisons'][0] for k in ['a','b']]
    assert a['exclusions']=={'chaotic_screen':1,'eligible':1}
    assert a['median_abs_difference']==0
    assert b['median_abs_difference']==pytest.approx(.075)
    assert a['ns_rows']==2 and len(a['pairs'])==2


def test_eligibility_details():
    m=grid(); rows=[row(.1),row(.2),row(.3),row(.4),row(.5),row(.6),row(.7),row(.8)]
    for j,status in [(10,0),(20,1),(30,2),(40,5)]: m['status'][51,j]=status
    m['period'][51,30]=2; m['rho'][51,50]=np.nan
    m['ftle'][51,60]=.006; m['planar_share'][51,70]=.5
    f={'a':dict(args=dict(start=1.1),rows=rows)}
    r=analyze(m,f,{**DEFAULTS,'offsets':[.01]},6000)
    reasons=[p['reason'] for p in r['families']['a']['comparisons'][0]['pairs']]
    assert reasons==['divergent_screen','fixed_screen','period_at_most_two_screen','drift_screen','nonfinite_diagnostic','ftle_outside_band','low_planarity','eligible']
    json.dumps(r,allow_nan=False)


def test_missing_period_candidates_retained_and_geometric_distance_envelope():
    m=grid(); rows=[row(.4,angle=.124),row(.5,angle=.126)]
    cfg={**DEFAULTS,'min_den':8,'max_den':8}
    edges=geometric_edges(rows,cfg)
    roots=provisional_roots(m,rows,edges,cfg,6000)
    assert len(roots)==1 and roots[0]['candidate_cells']==0
    assert roots[0]['interpolated_distance_grid_cells'] is None
    m['status'][51,45]=2; m['period'][51,45]=16
    root=provisional_roots(m,rows,edges,cfg,6000)[0]
    assert root['interpolated_distance_grid_cells']==pytest.approx(1)
    assert root['segment_distance_envelope_grid_cells']==pytest.approx([1,np.sqrt(26)])
    a,b,c=np.array([0.,0.]),np.array([2.,0.]),np.array([[.8,1.],[1.2,-.5]])
    lower,upper=segment_distance_envelope(a,b,c)
    ds=[np.linalg.norm(c-(a+t*(b-a)),axis=1).min() for t in np.linspace(0,1,100)]
    assert lower<=min(ds) and upper>=max(ds)


def test_bad_axes_rejected():
    m=grid(); m['log10_hidden'][1]=m['log10_hidden'][0]
    with pytest.raises(ValueError,match='uniformly increasing'): validate_map(m)
    with pytest.raises(ValueError,match='increasing output'): geometric_edges([row(.2),row(.1)],DEFAULTS)


def test_verifier_rejects_incomplete_inventory(tmp_path):
    script=Path(__file__).resolve().parents[1]/'tools/analyze_s1r_rotation_map.py'
    spec=importlib.util.spec_from_file_location('s1r_driver',script)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(schema='s1r-rotation-audit-v1',sources={})))
    with pytest.raises(ValueError,match='incomplete source'): module.verify(receipt)
    receipt.write_text(json.dumps(dict(schema='s1r-rotation-audit-v1',sources={p:'corrupt' for p in module.SOURCE_FILES})))
    with pytest.raises(ValueError,match='source hash'): module.verify(receipt)
