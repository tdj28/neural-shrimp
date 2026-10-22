"""Publication layout preserves diagnostic evidence; synthetic inputs only."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import plot_baby_shrimp_contraction_minima as diagnostic
import plot_baby_shrimp_contraction_minima_publication as publication
from test_plot_baby_shrimp_contraction_minima import fixture_receipt


def artists(fig):
    """Data geometry, marker paths, styles, axes and section statuses."""
    result = []
    for ax in fig.axes:
        result.append({"limits": (ax.get_xlim(), ax.get_ylim()), "aspect": ax.get_aspect(),
            "lines": [(line.get_gid(), line.get_xydata().copy(), line.get_linestyle(), line.get_color(), line.get_linewidth()) for line in ax.lines],
            "collections": [(c.get_gid(), c.get_offsets().copy(), [p.vertices.copy() for p in c.get_paths()], c.get_sizes().copy()) for c in ax.collections],
            "annotations": [(t.get_text(), t.xy) for t in ax.texts if hasattr(t, "xy")]})
    return result


def assert_same_artists(before, after):
    assert len(before) == len(after) == 2
    for a, b in zip(before, after):
        assert a["limits"] == b["limits"] and a["aspect"] == b["aspect"] == 1
        assert a["annotations"] == b["annotations"]
        for kind in ("lines", "collections"):
            assert len(a[kind]) == len(b[kind])
            for old, new in zip(a[kind], b[kind]):
                assert old[0] == new[0]
                np.testing.assert_array_equal(old[1], new[1])
                if kind == "lines":
                    assert old[2:] == new[2:]
                else:
                    assert len(old[2]) == len(new[2])
                    for op, np_ in zip(old[2], new[2]):
                        np.testing.assert_array_equal(op, np_)
                    np.testing.assert_array_equal(old[3], new[3])


@pytest.mark.parametrize("case", ["complete", "partial", "no_center", "failed"])
def test_layout_preserves_every_artist_status_support_and_bracket(case):
    receipt, plan = fixture_receipt(partial=case == "partial", no_center=case == "no_center")
    data = diagnostic.validate_plot_data(receipt, plan, allow_partial=case != "complete")
    if case == "failed":
        selected = next(r for r in data["sections"] if r["section"]["longitudinal"] == min(plan["contraction_minima"]["longitudinal_steps"]))
        s, observation = next(iter(selected["observed"].items()))
        selected["observed"] = {s: observation}; selected["admitted"] = {}
        selected["section"]["search"] = {"status": "interrupted before search returned", "minima": []}
        selected["section"]["callback_attempts"] = [{"s": s, "status": "started"}]
        receipt["complete"] = False
    old = diagnostic.make_center_zoom(receipt, data, synthetic=True)
    new = publication.make_publication(receipt, data, synthetic=True)
    try:
        assert_same_artists(artists(old), artists(new))
        cards = lambda fig: [t.get_text() for t in fig.texts if t.get_text().startswith("Arm ")]
        assert cards(old) == cards(new)
        np.testing.assert_array_equal(new.get_size_inches(), [7, 5])
        assert "SYNTHETIC QA" in " ".join(t.get_text() for t in new.texts)
        assert ("PARTIAL / UNRESOLVED" in " ".join(t.get_text() for t in new.texts)) is (not receipt["complete"])
        for ax in new.axes:
            assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 9
        assert len(new.axes[1].lines) == sum(len(r["measurement"]["section"]["search"]["minima"]) for r in diagnostic.center_zoom_data(data)["rows"] if r["measurement"])
    finally:
        import matplotlib.pyplot as plt
        plt.close(old); plt.close(new)


def test_caption_retains_moved_metric_and_scientific_limitations():
    receipt, plan = fixture_receipt(); data = diagnostic.validate_plot_data(receipt, plan)
    caption = publication.caption_semantics(receipt, data)
    assert caption["coordinate_scales"] == plan["coordinate_scales"]
    assert caption["planned_longitudinal_distance"] == min(plan["contraction_minima"]["longitudinal_steps"])
    assert caption["display_multiplier"] == 1e4
    assert "Equal axes" in caption["metric"]
    assert "not measured valley" in caption["frame"]
    assert "not joined" in caption["measurements"]
    assert "not confidence intervals" in caption["brackets"]
    assert "sampled mesh" in caption["status"]
    assert "do not establish" in caption["limits"]
    assert len(caption["diagnostic_labels_replaced_or_moved"]) == 3


@pytest.mark.parametrize("change", ["extra warning", "changed footer", "duplicate status"])
def test_new_or_changed_diagnostic_labels_cannot_be_silently_hidden(monkeypatch, change):
    receipt, plan = fixture_receipt(); data = diagnostic.validate_plot_data(receipt, plan)
    original = diagnostic.make_center_zoom
    figures = []
    def modified(*args, **kwargs):
        fig = original(*args, **kwargs); figures.append(fig)
        if change == "extra warning":
            fig.text(.5, .5, "NEW SCIENTIFIC LIMITATION")
        elif change == "changed footer":
            next(t for t in fig.texts if t.get_text().startswith("Scaled coordinates")).set_text("Changed scientific limits")
        else:
            next(t for t in fig.texts if t.get_text().startswith("Arm 4")).set_text("Arm 3: wrong status")
        return fig
    monkeypatch.setattr(diagnostic, "make_center_zoom", modified)
    try:
        with pytest.raises(ValueError, match="unrecognized"):
            publication.make_publication(receipt, data)
    finally:
        import matplotlib.pyplot as plt
        for fig in figures:
            plt.close(fig)


@pytest.mark.parametrize("changed", publication.SOURCES)
def test_scientific_export_requires_both_sources_committed(monkeypatch, tmp_path, changed):
    monkeypatch.setattr(publication, "ROOT", tmp_path)
    for name in publication.SOURCES:
        path = tmp_path/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b"committed")
    (tmp_path/changed).write_bytes(b"uncommitted change")
    monkeypatch.setattr(diagnostic, "git_bytes", lambda commit, path: b"committed")
    with pytest.raises(ValueError, match="commit publication and diagnostic"):
        publication.checked_sources("a"*40)


@pytest.mark.parametrize("partial,no_center", [(False, False), (True, False), (True, True)])
def test_static_formats_and_artist_bounds_are_inside_page(tmp_path, partial, no_center):
    receipt, plan = fixture_receipt(partial=partial, no_center=no_center)
    data = diagnostic.validate_plot_data(receipt, plan, allow_partial=partial)
    fig = publication.make_publication(receipt, data, synthetic=True)
    try:
        fig.canvas.draw(); renderer = fig.canvas.get_renderer(); page = fig.bbox
        texts = list(fig.texts)
        for ax in fig.axes:
            texts += [ax.xaxis.label, ax.yaxis.label, ax._left_title] + ax.get_xticklabels()+ax.get_yticklabels()+ax.texts
        texts += [t for legend in fig.legends for t in legend.get_texts()]
        for text in texts:
            box = text.get_window_extent(renderer)
            assert box.x0 >= page.x0-1 and box.y0 >= page.y0-1, text.get_text()
            assert box.x1 <= page.x1+1 and box.y1 <= page.y1+1, text.get_text()
        legend = fig.legends[0].get_window_extent(renderer)
        for ax in fig.axes:
            assert not legend.overlaps(ax.xaxis.label.get_window_extent(renderer))
        for text in fig.texts:
            if text.get_text().startswith("Arm "):
                assert not legend.overlaps(text.get_window_extent(renderer))
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    publication.render(receipt, data, tmp_path, synthetic=True)
    for ext in ("png", "pdf"):
        assert (tmp_path/f"section-minima-center-zoom-publication.{ext}").stat().st_size > 5000
