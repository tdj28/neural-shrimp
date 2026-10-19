import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import period_colors as pc  # noqa: E402

PERIODS = [1, 2, 3, 5, 8, 13, 21, 40]


def sample_map(rng, periods=PERIODS, shape=(40, 50)):
    """A map with every period of `periods` plus the special codes 0 (no period), -1 and NaN (not computed)."""
    P = rng.choice(np.array(list(periods) + [0, -1], dtype=float), size=shape)
    P[0, : len(periods)] = periods                       # every period occurs at least once
    P[1, :3] = np.nan
    return P


def rendered(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(float) / 255


def test_one_colour_per_period_and_special_codes_untouched():
    P = sample_map(np.random.default_rng(0))
    cats = pc.categorize(P)
    assert cats.periods == tuple(PERIODS) and cats.other == ()
    assert cats.labels == [str(p) for p in PERIODS]
    assert len({tuple(c) for c in cats.colours}) == cats.n            # distinct colours
    assert not any(np.allclose(c[0], c[1:3]) for c in cats.colours)  # no greys among period colours
    idx = cats.index(P)
    assert np.isnan(idx[~(P > 0)]).all()                             # 0, -1, NaN: no category
    for k, p in enumerate(PERIODS):                                 # each period exactly one category
        assert set(np.unique(idx[P == p])) == {k}
        assert np.allclose(cats.cmap(cats.norm(k)), cats.colours[k])
    rgba = cats.rgba(P)
    assert np.allclose(rgba[P == 8], cats.colours[PERIODS.index(8)])
    assert (rgba[~(P > 0)] == 0).all()


def test_colorbar_blocks_and_labels_align_on_the_rendered_figure():
    P = sample_map(np.random.default_rng(1))
    cats = pc.categorize(P)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    im = ax.imshow(cats.index(P), cmap=cats.cmap, norm=cats.norm, interpolation="nearest")
    cb = cats.colorbar(im, ax=ax, label="period")
    img = rendered(fig)
    h = img.shape[0]
    ticks = cb.ax.get_yticks()
    assert np.array_equal(ticks, np.arange(cats.n))
    assert [t.get_text() for t in cb.ax.get_yticklabels()] == cats.labels
    b = cats.norm.boundaries
    assert np.allclose((b[:-1] + b[1:]) / 2, np.arange(cats.n))      # every tick at its block's centre
    x_mid = cb.ax.transAxes.transform((0.5, 0.5))[0]
    for k, label in enumerate(cb.ax.get_yticklabels()):
        y = cb.ax.transData.transform((0.5, k))[1]                    # tick (and label anchor) position
        lo, hi = (cb.ax.transData.transform((0.5, v))[1] for v in (k - 0.5, k + 0.5))
        assert lo < y < hi and abs(y - (lo + hi) / 2) < 1e-6           # centred on block k
        pixel = img[int(round(h - y)), int(round(x_mid))]
        assert np.allclose(pixel, cats.colours[k][:3], atol=2 / 255), (k, pixel)
        anchor = label.get_transform().transform(label.get_position())[1]
        assert abs(anchor - y) < 1.0                                   # label drawn at the tick
        assert label.get_text() == str(PERIODS[k])
    plt.close(fig)


def test_horizontal_colorbar_and_rgba_image_without_mappable():
    P = sample_map(np.random.default_rng(2))
    cats = pc.categorize(P)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    ax.imshow(cats.rgba(P), interpolation="nearest")
    cb = cats.colorbar(ax=ax, orientation="horizontal")
    img = rendered(fig)
    h = img.shape[0]
    y_mid = cb.ax.transAxes.transform((0.5, 0.5))[1]
    assert [t.get_text() for t in cb.ax.get_xticklabels()] == cats.labels
    for k in range(cats.n):
        x = cb.ax.transData.transform((k, 0.5))[0]
        assert np.allclose(img[int(round(h - y_mid)), int(round(x))], cats.colours[k][:3], atol=2 / 255)
    plt.close(fig)


def test_shared_categories_across_panels():
    a = np.array([[1.0, 2.0], [0.0, 7.0]])
    b = np.array([[7.0, 9.0], [np.nan, 2.0]])
    cats = pc.categorize([a, b])
    assert cats.periods == (1, 2, 7, 9)
    assert cats.index(a)[1, 1] == cats.index(b)[0, 0] == 2           # period 7: same category in both panels


def test_more_periods_than_colours_keeps_the_most_frequent():
    P = np.array([5] * 9 + [3] * 7 + [11] * 4 + [2] * 3 + [8] * 3 + [13] * 1, dtype=float)
    cats = pc.categorize(P, max_blocks=4)                              # three colours and "other"
    assert cats.periods == (3, 5, 11)                                  # most frequent, listed by period
    assert cats.other == (2, 8, 13)                                    # 2 and 8 tie with 3 cells: not kept
    assert cats.labels == ["3", "5", "11", "other"]
    assert np.allclose(cats.colours[-1], pc.OTHER_COLOUR)
    idx = cats.index(P)
    assert set(idx[np.isin(P, [2, 8, 13])]) == {3}
    cats = pc.categorize(P, max_blocks=5)                              # tie at 3 cells: the shorter period wins
    assert cats.periods == (2, 3, 5, 11) and cats.other == (8, 13)
    assert pc.categorize(P, max_blocks=6).other == ()                  # six periods fit six blocks
    full = pc.categorize(np.arange(1, 61, dtype=float))                # more periods than the palette
    assert len(full.periods) == len(pc.palette()) and full.labels[-1] == "other"
    fig, ax = plt.subplots(figsize=(4, 3))
    n = pc.blocks_that_fit(ax)
    length = ax.get_position().height * 3 * 72
    assert n == int(0.9 * length / 6.5)
    plt.close(fig)


def test_frequent_periods_get_the_first_colours_and_the_palette_has_no_greys():
    pal = pc.palette()
    assert len(pal) == 38 and len({tuple(c) for c in pal}) == 38
    assert not any(np.allclose(c[0], c[1:3]) for c in pal)
    P = np.array([14] * 50 + [7] * 30 + [2] * 30 + [28] * 5, dtype=float)
    cats = pc.categorize(P)
    assert cats.labels == ["2", "7", "14", "28"]                       # listed by period
    colour = dict(zip(cats.periods, map(tuple, cats.colours)))
    assert colour[14] == tuple(pal[0])                                 # most cells: first colour
    assert colour[2] == tuple(pal[1]) and colour[7] == tuple(pal[2])   # tie: the shorter period first
    assert colour[28] == tuple(pal[3])


def test_valid_mask_and_failures():
    P = np.array([[2.0, 3.0], [4.0, 0.0]])
    cats = pc.categorize(P)
    assert np.isnan(cats.index(P, valid=[[True, False], [True, True]])[0, 1])
    with pytest.raises(ValueError, match="not given"):
        cats.index(np.array([2.0, 6.0]))                               # unknown period: refuse, no fallback
    with pytest.raises(ValueError, match="no positive periods"):
        pc.categorize(np.array([0.0, -1.0, np.nan]))
    with pytest.raises(ValueError, match="whole numbers"):
        pc.categorize(np.array([2.0, 2.5]))
    with pytest.raises(ValueError, match="max_blocks"):
        pc.categorize(P, max_blocks=1)


def test_legend_handles_match_categories():
    cats = pc.categorize(np.array([4.0, 6.0, 6.0, 9.0]))
    hs = cats.handles()
    assert [h.get_label() for h in hs] == ["4", "6", "9"]
    assert all(np.allclose(h.get_facecolor(), c) for h, c in zip(hs, cats.colours))


# --- the soft-loop map renderers (tools/explore_llm_soft_loop.py) on small synthetic scans

def fake_scan(root, period, lyapunov, family=None, families=()):
    """A minimal finished scan under root/explore/s: scan.npz and scan.json."""
    import json

    d = root / "explore" / "s"
    d.mkdir(parents=True)
    rows, cols = period.shape
    np.savez(d / "scan.npz", log10_temperature=np.linspace(-0.1, 0.1, rows), alpha=np.linspace(0.0, 0.1, cols),
             lyapunov=lyapunov, period=period, entropy=np.zeros(period.shape),
             family=np.full(period.shape, -1) if family is None else family)
    args = {"model": "qwen0.5b", "window": 1, "layer": 12, "concept": "philosophy", "vectors": "v"}
    (d / "scan.json").write_text(json.dumps({"args": args, "families": list(families)}))


def captured(monkeypatch, tmp_path):
    import explore_llm_soft_loop as esl
    from matplotlib.figure import Figure

    figs = []
    monkeypatch.setattr(esl, "DATA", tmp_path)
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **k: figs.append(self))
    return esl, figs


def render_args(tmp_path, **kw):
    import argparse

    a = dict(scan="s", max_period=None, chaos_level=0.005, curves=None, name=str(tmp_path / "out.png"), title=None,
             window=None, style="period", families=0, box=None)                # the CLI's defaults
    a.update(kw)
    return argparse.Namespace(**a)


def legend_and_image(fig):
    ax = fig.axes[0]
    leg = ax.get_legend()
    colours = {t.get_text(): np.asarray(h.get_facecolor()) for t, h in zip(leg.get_texts(), leg.legend_handles)}
    return colours, np.asarray(ax.get_images()[0].get_array())


SCAN_P = np.array([[2, 3, 17, 33], [0, 2, 40, 33], [3, 0, 17, 2]], dtype=float)
# two cells with a positive exponent: the period-40 cell (a detected period wins: drawn by its period) and a cell
# without a period (chaotic by the scans' criterion: exponent above the level and no period)
SCAN_L = np.where(np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]) == 1, 0.2, -0.3)


def test_render_gives_every_drawn_period_its_own_entry(monkeypatch, tmp_path):
    esl, figs = captured(monkeypatch, tmp_path)
    fake_scan(tmp_path, SCAN_P, SCAN_L)
    esl.render(render_args(tmp_path))
    colours, img = legend_and_image(figs[0])
    assert [k for k in colours if k.isdigit()] == ["2", "3", "17", "33", "40"]
    for r, c in np.ndindex(SCAN_P.shape):
        p = int(SCAN_P[r, c])
        chaotic = SCAN_L[r, c] > 0.005 and p <= 0
        chaos_key = next(k for k in colours if k.startswith("chaotic (exponent > 0.005"))   # the legend's chaos entry
        want = colours[chaos_key] if chaotic else colours[str(p)] if p > 0 \
            else colours["no period, not chaotic"]
        assert np.allclose(img[r, c], want), (r, c, p)


def test_render_groups_long_periods_only_when_asked(monkeypatch, tmp_path):
    esl, figs = captured(monkeypatch, tmp_path)
    fake_scan(tmp_path, SCAN_P, SCAN_L)
    esl.render(render_args(tmp_path, max_period=16))
    colours, img = legend_and_image(figs[0])
    assert [k for k in colours if k.isdigit()] == ["2", "3"] and "other (> 16)" in colours
    assert np.allclose(img[0, 2], pc.OTHER_COLOUR) and np.allclose(img[0, 3], pc.OTHER_COLOUR)
    assert not np.allclose(img[0, 2], colours["no period, not chaotic"])


def test_render_families_label_the_other_attractors(monkeypatch, tmp_path):
    esl, figs = captured(monkeypatch, tmp_path)
    P = np.array([[2, 2, 3, 0], [2, 3, 3, 4]], dtype=float)
    fam = np.array([[0, 0, 1, -1], [0, 1, 1, 2]])
    fams = [{"family": i, "period": q, "cells": 1, "tokens": [t]} for i, q, t in ((0, 2, "a"), (1, 3, "b"), (2, 4, "c"))]
    fake_scan(tmp_path, P, np.full(P.shape, -0.3), fam, fams)
    esl.render(render_args(tmp_path, families=2))
    colours, img = legend_and_image(figs[0])
    assert "other periodic attractors" in colours
    assert np.allclose(img[1, 3], colours["other periodic attractors"])      # family 2 is not among the two shown
    assert np.allclose(img[0, 3], colours["no period, not chaotic"])         # legend grey = map grey


def test_replot_scan_figure_has_one_block_per_period(monkeypatch, tmp_path):
    esl, figs = captured(monkeypatch, tmp_path)
    P = np.array([[1, 2, 25, 0], [2, 2, 25, 1]], dtype=float)
    fake_scan(tmp_path, P, np.full(P.shape, -0.3))
    import argparse

    esl.replot(argparse.Namespace(scan="s", out=str(tmp_path / "scan.png")))
    bar = [ax for ax in figs[0].axes if ax.get_label() == "<colorbar>"][1]   # Lyapunov, period, entropy bars
    assert [t.get_text() for t in bar.get_yticklabels()] == ["1", "2", "25"]
    assert np.array_equal(bar.get_yticks(), [0, 1, 2])


def test_cvd_palette_hands_out_okabe_ito_colours():
    from matplotlib.colors import to_rgba

    cats = pc.categorize([np.array([6, 6, 6, 12, 12, 24])], kind="cvd")
    assert cats.periods == (6, 12, 24)
    assert np.allclose(cats.colours, [to_rgba(c) for c in pc.OKABE_ITO[:3]])   # most frequent period first
    with pytest.raises(ValueError):
        pc.palette("rainbow")
