"""Ensure mobile layout preserves the audited center-zoom evidence exactly."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
import plot_baby_shrimp_contraction_minima as base
import plot_baby_shrimp_contraction_minima_mobile as mobile

spec = importlib.util.spec_from_file_location("bsc5_plot_fixture", ROOT/"tests/test_plot_baby_shrimp_contraction_minima.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


@pytest.mark.parametrize("partial,no_center", [(False, False), (True, False), (True, True)])
def test_stacking_keeps_all_panel_artists_limits_and_labels(partial, no_center):
    receipt, plan = fixture.fixture_receipt(partial=partial, no_center=no_center)
    data = base.validate_plot_data(receipt, plan, allow_partial=partial)
    original = base.make_center_zoom(receipt, data, synthetic=True)
    stacked = mobile.stacked_center_zoom(receipt, data, synthetic=True)
    assert len(original.axes) == len(stacked.axes) == 2
    for before, after in zip(original.axes, stacked.axes, strict=True):
        assert before.get_xlim() == after.get_xlim()
        assert before.get_ylim() == after.get_ylim()
        assert before.get_aspect() == after.get_aspect() == 1.
        assert before.get_xlabel() == after.get_xlabel()
        assert before.get_ylabel() == after.get_ylabel()
        assert [t.get_text() for t in before.texts] == [t.get_text() for t in after.texts]
        for kind in ("lines", "collections"):
            left, right = getattr(before, kind), getattr(after, kind)
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                assert a.get_gid() == b.get_gid()
                getter = "get_xydata" if kind == "lines" else "get_offsets"
                assert np.array_equal(getattr(a, getter)(), getattr(b, getter)())
    normalize = lambda t: " ".join(t.replace("— SYNTHETIC QA", "SYNTHETIC QA").split())
    assert sorted(normalize(t.get_text()) for t in original.texts) == sorted(normalize(t.get_text()) for t in stacked.texts)
    assert [t.get_text() for t in original.legends[0].get_texts()] == [t.get_text() for t in stacked.legends[0].get_texts()]
    assert stacked.axes[0].get_position().y0 > stacked.axes[1].get_position().y1
    import matplotlib.pyplot as plt
    plt.close(original); plt.close(stacked)


def test_unexpected_base_label_cannot_be_silently_lost(monkeypatch):
    receipt, plan = fixture.fixture_receipt()
    data = base.validate_plot_data(receipt, plan)
    original = base.make_center_zoom
    def extra_label(*args, **kwargs):
        fig = original(*args, **kwargs)
        fig.text(.1, .1, "New evidence label")
        return fig
    monkeypatch.setattr(base, "make_center_zoom", extra_label)
    with pytest.raises(ValueError, match="refuse to omit"):
        mobile.stacked_center_zoom(receipt, data)
