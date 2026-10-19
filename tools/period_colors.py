"""Categorical colours for maps coloured by the period of the attractor.

A discrete colormap on a continuous norm (for example tab20 with vmin 0.5, vmax 40.5) gives each colour block two
periods and puts the colour-bar ticks between blocks; a norm that ends at 20.5 also gives every longer period the
top colour. Here each period that occurs gets its own colour and its own colour-bar block (or legend entry), with
the period written at the centre of that block. When more periods occur than there are distinct colours, the most
frequent periods keep their own colours and the rest share one block labelled "other".

Special codes (no period found, chaotic, not computed) stay with the caller: their cells get no category (NaN
index), so they are drawn in the colormap's "bad" colour (white by default) or overlaid exactly as before.

Typical use:

    import period_colors as pc
    cats = pc.categorize([P1, P2])            # the periods of all panels: one colour per period in every panel
    im = ax.imshow(cats.index(P1), cmap=cats.cmap, norm=cats.norm, interpolation="nearest", ...)
    cats.colorbar(im, ax=ax, label="period (white: none)")
"""
from dataclasses import dataclass

import numpy as np

OTHER_COLOUR = (0.55, 0.55, 0.62, 1.0)      # bluish grey: periods (or attractors) without a colour of their own
# Okabe and Ito's colours other than black, told apart under the common colour-vision deficiencies
OKABE_ITO = ("#0072B2", "#E69F00", "#56B4E9", "#CC79A7", "#009E73", "#D55E00", "#F0E442")


def palette(kind="tab"):
    """Qualitative colours without greys (grey marks special codes), in the order they are handed out. kind "tab": the
    nine saturated tab20 colours, their nine light partners, then tab20b; kind "cvd": the seven colours of OKABE_ITO,
    safe for readers with colour-vision deficiencies (with fewer colours, more periods share "other")."""
    import matplotlib

    if kind == "cvd":
        from matplotlib.colors import to_rgba

        return np.array([to_rgba(c) for c in OKABE_ITO])
    if kind != "tab":
        raise ValueError(f"unknown palette {kind!r}")
    tab20 = matplotlib.colormaps["tab20"](np.arange(20))
    tab20b = matplotlib.colormaps["tab20b"](np.arange(20))
    dark, light = np.delete(tab20[0::2], 7, axis=0), np.delete(tab20[1::2], 7, axis=0)   # index 7: the greys
    return np.concatenate([dark, light, tab20b])


def _positive_periods(arrays):
    """Positive finite values of the arrays, checked to be whole numbers."""
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float).ravel()
        vals.append(a[np.isfinite(a) & (a > 0)])
    vals = np.concatenate(vals) if vals else np.zeros(0)
    if vals.size and not np.array_equal(vals, np.round(vals)):
        raise ValueError(f"periods must be whole numbers, got e.g. {vals[vals != np.round(vals)][:3]}")
    return vals.astype(np.int64)


@dataclass
class PeriodCategories:
    """One category per period in `periods` (ascending), then "other" for the periods in `other`, if any."""
    periods: tuple
    other: tuple
    colours: np.ndarray
    bad: object = "white"

    @property
    def labels(self):
        return [str(p) for p in self.periods] + (["other"] if self.other else [])

    @property
    def n(self):
        return len(self.colours)

    @property
    def cmap(self):
        from matplotlib.colors import ListedColormap

        return ListedColormap(self.colours, name="periods").with_extremes(bad=self.bad)

    @property
    def norm(self):
        """Category k fills [k - 0.5, k + 0.5]: integer category indices sit at block centres."""
        from matplotlib.colors import BoundaryNorm

        return BoundaryNorm(np.arange(self.n + 1) - 0.5, self.n)

    def index(self, period, valid=None):
        """Category index of every cell (float, NaN for non-positive, non-finite or not `valid` cells). A positive
        period that was not given to categorize() is an error, not a silent fallback."""
        p = np.asarray(period, dtype=float)
        out = np.full(p.shape, np.nan)
        ok = np.isfinite(p) & (p > 0)
        if valid is not None:
            ok &= np.asarray(valid, dtype=bool)
        vals = p[ok]
        idx = np.full(vals.shape, -1, dtype=np.int64)
        for k, q in enumerate(self.periods):
            idx[vals == q] = k
        if self.other:
            idx[np.isin(vals, self.other)] = len(self.periods)
        if (idx < 0).any():
            missing = np.unique(vals[idx < 0])
            raise ValueError(f"periods {missing[:10].tolist()} were not given to categorize()")
        out[ok] = idx
        return out

    def rgba(self, period, valid=None, fill=(0.0, 0.0, 0.0, 0.0)):
        """RGBA image of the categories; cells without a category get `fill`."""
        idx = self.index(period, valid)
        img = np.empty(idx.shape + (4,))
        img[...] = fill
        has = ~np.isnan(idx)
        img[has] = self.colours[idx[has].astype(np.int64)]
        return img

    def colorbar(self, mappable=None, ax=None, cax=None, label=None, fontsize=None, **kwargs):
        """Colour bar with one block per category and its label at the block centre. Without a mappable, one is
        made from cmap and norm (for RGBA images). fontsize=None shrinks the labels to fit the bar's length."""
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        if mappable is None:
            mappable = plt.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
            mappable.set_array(np.arange(self.n, dtype=float))
        if ax is not None:
            fig = np.ravel(ax)[0].figure
        elif cax is not None:
            fig = cax.figure
        else:
            fig = mappable.axes.figure
        cb = fig.colorbar(mappable, ax=ax, cax=cax, ticks=np.arange(self.n), **kwargs)
        cb.ax.minorticks_off()
        cb.set_ticks(np.arange(self.n), labels=self.labels)
        if label is not None:
            cb.set_label(label)
        if fontsize is None:
            box = cb.ax.get_position()
            vertical = cb.orientation == "vertical"
            length = (box.height * fig.get_figheight() if vertical else box.width * fig.get_figwidth()) * 72
            default = FontProperties(size=matplotlib.rcParams["ytick.labelsize"]).get_size_in_points()
            # vertical bars: one line of digits per block; horizontal bars: about two digits per block
            per_block = length / self.n if vertical else length / (1.3 * self.n)
            fontsize = float(np.clip(0.9 * per_block, 4.0, default))
            if fontsize < 6.0:
                import warnings

                warnings.warn(f"{self.n} colour-bar labels fit only at {fontsize:.1f} pt; pass max_blocks to "
                              "categorize() (see blocks_that_fit)", stacklevel=2)
        cb.ax.tick_params(labelsize=fontsize, length=2)
        return cb

    def handles(self, alpha=None):
        """Legend patches, one per category, in colour-bar order."""
        from matplotlib.patches import Patch

        return [Patch(color=c, alpha=alpha, label=lab) for c, lab in zip(self.colours, self.labels)]


def blocks_that_fit(ax, min_fontsize=6.5):
    """How many colour-bar blocks fit beside `ax` (a vertical colour bar as long as the axes, before layout) with
    labels of at least `min_fontsize` points; pass it to categorize(max_blocks=...)."""
    length = ax.get_position().height * ax.figure.get_figheight() * 72
    return max(2, int(0.9 * length / min_fontsize))


def categorize(arrays, max_blocks=None, other_colour=OTHER_COLOUR, bad="white", kind="tab"):
    """Categories for the positive periods found in `arrays` (one array or a list of arrays: the panels sharing one
    colour scale). Non-positive and non-finite values are special codes and get no category.

    Colours are handed out in palette order by frequency (most cells first; ties: the shorter period), so the large
    regions get the saturated colours; categories, colour-bar blocks and legend entries are listed by period. With
    more periods than distinct colours (the palette) or than `max_blocks` (colour-bar blocks, "other" included),
    only the most frequent periods get colours and the rest are grouped as "other". `kind` selects the palette."""
    if isinstance(arrays, np.ndarray) or np.isscalar(arrays):
        arrays = [arrays]
    vals = _positive_periods(arrays)
    if vals.size == 0:
        raise ValueError("no positive periods: nothing to colour by period")
    colours = palette(kind)
    if max_blocks is not None and int(max_blocks) < 2:
        raise ValueError(f"max_blocks must be at least 2, got {max_blocks}")
    u, counts = np.unique(vals, return_counts=True)
    limit = len(colours)
    if max_blocks is not None and len(u) > int(max_blocks):
        limit = min(limit, int(max_blocks) - 1)               # one block is left for "other"
    order = np.lexsort((u, -counts))                      # most frequent first, shorter period first on ties
    ranked, other = u[order[:limit]], np.sort(u[order[limit:]])
    rank = {int(q): r for r, q in enumerate(ranked)}
    kept = np.sort(ranked)
    cols = [colours[rank[int(q)]] for q in kept]
    if other.size:
        cols.append(other_colour)
    return PeriodCategories(tuple(int(q) for q in kept), tuple(int(q) for q in other), np.asarray(cols, dtype=float),
                            bad)
