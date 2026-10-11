from pathlib import Path
import copy
import importlib.util
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'tools/plot_two_write_saved.py';spec=importlib.util.spec_from_file_location('plot_two_write_saved',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture():
    balances=[];scores=[];contexts={}
    for setting in m.SETTINGS:
        for phase in range(24):
            cid=f'{setting}--p{phase:02d}';ok=phase!=9;split='development' if phase<8 else 'heldout'
            contexts[cid]=dict(qualified=ok,complete=ok)
            for arm in m.ARMS:
                for window in m.WINDOWS:
                    balances.append(dict(context_id=cid,setting=setting,phase_index=phase,split=split,arm=arm,window=window,
                        complete=ok,qualified=ok,with_second=.6 if ok else None,without_second=.7 if ok else None,interference_loss=.1 if ok else None))
                for first in (-1,0,1):
                    for second in (-1,0,1):
                        for method in (('rk4-coarse','rk4-fine','dop853-independent') if phase in (0,8) else ('rk4-coarse','rk4-fine')):
                            task=dict(setting=setting,phase_index=phase,split=split,arm=arm,first_sign=first,second_sign=second,method=method)
                            scores.append(dict(task=task,task_id=f'{cid}-{arm}-{first}-{second}-{method}',complete=ok,qualified=ok,
                                windows=dict(k4=dict(primary=dict(signed_margin=float(first*second),projected_mean=float(second),probability_correct=.6)))))
    return dict(analysis=dict(counts=dict(histories=1296,trace_tasks=2700,comparisons=1404),balanced_accuracies=balances,scores=scores,contexts=contexts))


def test_census_and_missing_columns():
    balances,fine,contexts=m.collect(fixture())
    assert len(balances)==576 and len(fine)==1296 and len(contexts)==48
    a=m.matrix(balances,'core',[('identity','k4','with_second')])
    assert a.shape==(1,24) and np.isnan(a[0,9]) and np.sum(np.isfinite(a))==23
    z=m.margin_matrix(fine,contexts,'core')
    assert z.shape==(6,24) and np.all(np.isnan(z[:,9]))
    assert np.all(z[::2,0]==-1) and np.all(z[1::2,0]==1)


@pytest.mark.parametrize('edit',[
    lambda s:s['analysis']['balanced_accuracies'].pop(),
    lambda s:s['analysis']['balanced_accuracies'].__setitem__(1,s['analysis']['balanced_accuracies'][0]),
    lambda s:s['analysis']['scores'].pop(),
    lambda s:s['analysis']['contexts'].pop('core--p00'),
    lambda s:s['analysis']['balanced_accuracies'][0].update(with_second=np.nan),
    lambda s:s['analysis']['balanced_accuracies'][0].update(qualified=False),
    lambda s:s['analysis']['balanced_accuracies'][9*12].update(with_second=.6),
    lambda s:s['analysis']['counts'].update(comparisons=1080),
])
def test_refuses_dropped_or_invented_cells(edit):
    s=fixture();edit(s)
    with pytest.raises(ValueError):m.collect(s)


def test_all_unavailable_is_retained():
    s=fixture()
    for row in s['analysis']['balanced_accuracies']:row.update(complete=False,qualified=False,with_second=None,without_second=None,interference_loss=None)
    for row in s['analysis']['contexts'].values():row.update(complete=False,qualified=False)
    for row in s['analysis']['scores']:row.update(complete=False,qualified=False,windows={})
    balances,fine,contexts=m.collect(s)
    assert np.isnan(m.matrix(balances,'core',[('identity','k4','with_second')])).all()
    assert np.isnan(m.margin_matrix(fine,contexts,'core')).all()


def test_synthetic_render_preserves_csv_denominators(tmp_path):
    result=m.render(fixture(),tmp_path)
    assert result['phase_rows']==result['signed_margin_rows']==576
    assert len((tmp_path/'phase-readouts.csv').read_text().splitlines())==577
    assert len((tmp_path/'signed-margins-at-4T.csv').read_text().splitlines())==577
    assert all(Path(p).stat().st_size>10000 for p in result['images'])
