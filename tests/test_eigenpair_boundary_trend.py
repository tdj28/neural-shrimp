import numpy as np
import pytest
from neural_shrimp.eigenpair_boundary_trend import assignment,future_linear_prediction,unpack,summarize


def test_projective_matching_preserves_permuted_and_phased_modes():
    v=np.array([.4,.40001,.8]);q=np.eye(3,dtype=complex);order=np.array([2,0,1]);phase=np.array([1j,-1,1j])
    chosen,meta=assignment(v,q,v[order],q[:,order]*phase)
    assert np.array_equal(order[chosen],np.arange(3)) and meta['minimum_matched_overlap']==1


def test_fixed_fit_predicts_known_crossing_and_rejects_past():
    s=np.arange(20)/10;y=.5+.1*s
    assert abs(future_linear_prediction(s,y,8,1.)['future_gauge']-5)<1e-12
    assert future_linear_prediction(s,1.1+.1*s,8,1.)['future_gauge']is None
    with pytest.raises(ValueError):future_linear_prediction(s[::-1],y,8,1.)


def spectral(v):
    vals=np.r_[1.,v,np.linspace(.1,.5,11)];vec=np.eye(13)
    return {'eigenvalues':[[float(x),0.]for x in vals],'eigenvectors':np.stack([vec,np.zeros_like(vec)],axis=-1).tolist(),'critical_nearest_index':0,'relative_residuals':[0.]*13,'mode_residual':0.}


def test_all_modes_retained_and_fixed_complete_chain_tracked():
    rows=[{'gauge':str(i/10),'rates':[3.,.8],'spectrum':spectral(.8+i*.005),'fold':{'second_derivative_contraction':-4.,'border_frobenius_condition_proxy':2.}}for i in range(20)]
    r=summarize(rows);assert r['nearest_plus_mode_real_throughout'] and r['all_complements_numerically_inside_unit_circle']
    assert all(len(v['values'])==12 for v in r['rows']) and all(abs(v['future_gauge']-4.)<1e-12 for v in r['plus_one_linear_proposals'])
    rows[-1]['spectrum']['relative_residuals'][2]=.1
    with pytest.raises(ValueError):summarize(rows)


def test_no_positive_sign_change_inferred_from_fit_or_complex_mode():
    s=np.arange(20);r=future_linear_prediction(s,np.ones(20),8,1.)
    assert 'confidence' not in r and r['scope'].startswith('Linear continuation proposal')
    s= spectral(.9);s['eigenvalues'][1][1]=.01;unpack(s)


def driver():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).parents[1]/'tools/diagnose_eigenpair_boundary_trend.py';s=importlib.util.spec_from_file_location('boundary_trend_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_fixed_source_inventory_and_rational_gauges():
    from pathlib import Path
    d=driver();assert set(d.N.SOURCES)<=set(d.SOURCES)
    assert all((Path(__file__).parents[1]/n).is_file()for n in d.SOURCES)
    rows=[{'gauge':str(i)+'/10','rates':[3.,.8],'spectrum':spectral(.8+i*.005),'fold':{'second_derivative_contraction':-4.,'border_frobenius_condition_proxy':2.}}for i in range(20)]
    assert len(summarize(rows)['rows'])==20


def test_selected_file_authentication_detects_tampering(tmp_path):
    import json
    d=driver();root=tmp_path/'evidence';(root/'run').mkdir(parents=True);p=root/'run/one.json';p.write_text('{}');p.chmod(0o444);identity=d.entry(p)
    manifest=root/'run/manifest.json';manifest.write_text(json.dumps({'files':{'one.json':identity}}));outer=root/'manifest.json';outer.write_text(json.dumps({'files':{'run/one.json':identity,'run/manifest.json':d.entry(manifest)}}));mp=d.entry(manifest)['sha256'];op=d.entry(outer)['sha256']
    assert d.bound_selected(root,op,mp,'one.json',identity)==p
    p.chmod(0o644);p.write_text('{"changed":true}');p.chmod(0o444)
    with pytest.raises(ValueError):d.bound_selected(root,op,mp,'one.json',identity)
