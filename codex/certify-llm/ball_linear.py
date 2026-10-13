"""Rigorous bounded-memory affine maps for exact stored binary weights.

``affine(W, X, bias, bias_columns=(0,))`` handles an Arb matrix whose columns
are a primal and tangent together. Bias is added only to column zero in that
example; the default adds it to every column. A sequence of Arb scalars instead
returns a list. ``W.T`` works as a noncontiguous view without a full conversion.

Each float16/32/64 weight is widened EXACTLY to a Python binary64 value, then
passed to python-flint's exact float constructor (not a decimal string). Arb
matrix products and sums enclose all rounding. There is no NumPy/BLAS product
or inferred floating-point error bound. The mathematical constants are the
stored binary values, not their decimal renderings or intended real values.

Conversion is limited to chunk_rows*chunk_cols scalars at a time. An optional
LRU bounds retained Arb coefficients by max_elements, NOT an estimated byte
count. Temporary storage also includes the current conversion block, small
input slices and the full output (n_outputs*n_columns). A cache key compares the
copied block's exact bytes, shape and precision; changing weights cannot return
an old block (this does not rely on cryptographic collision resistance). The caller must not mutate weights concurrently with an affine
call. Context precision and thread count are owned by the caller; this module
never changes either. Reuse a cache only within one thread/context.

This is an arithmetic component, not a certificate of a network or orbit.
Natural interval dependency and nonlinear wrapping need separate analysis.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from numbers import Integral
from typing import Sequence

import numpy as np
from flint import arb, arb_mat, ctx


def exact_dyadic(value: float) -> arb:
    """Return the exact stored finite binary16/32/64 value as an Arb point.

    python-flint's float constructor is exact even when ctx.prec is below 53.
    Explicitly reject wider or non-floating values to prevent silent rounding.
    Integer constants can be constructed directly with ``arb(integer)``.
    """
    if not isinstance(value, (float, np.floating)):
        raise TypeError("an IEEE binary16/32/64 floating value is required")
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise TypeError("wider-than-binary64 values cannot be silently rounded")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("dyadic constant must be finite")
    return arb(scalar)


def _positive_integer(value: int, name: str, *, zero_ok: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < (0 if zero_ok else 1):
        raise ValueError(f"{name} is out of range")
    return int(value)


class DyadicWeightCache:
    """Content-addressed, bounded LRU of exact Arb weight blocks.

    ``elements``/``peak_elements`` count retained scalar entries. They do not
    claim to bound all Python or FLINT allocation bytes. Exact cache keys also
    retain eight bytes per scalar, plus object overhead. Oversized blocks are
    computed but never cached. ``clear`` releases all retained matrices.
    """

    def __init__(self, max_elements: int = 262144):
        self.max_elements = _positive_integer(max_elements, "max_elements", zero_ok=True)
        self._blocks: OrderedDict[tuple, arb_mat] = OrderedDict()
        self.elements = 0
        self.peak_elements = 0
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._blocks.clear()
        self.elements = 0

    def _matrix(self, block: np.ndarray) -> arb_mat:
        key = (block.shape, int(ctx.prec), block.tobytes(order="C"))
        cached = self._blocks.get(key)
        if cached is not None:
            self.hits += 1
            self._blocks.move_to_end(key)
            return cached
        self.misses += 1
        # tolist returns Python floats. arb_mat's native float conversion is
        # exact; avoid first forming decimal strings or doing rounded sums.
        result = arb_mat(block.shape[0], block.shape[1], block.ravel().tolist())
        size = int(block.size)
        if size <= self.max_elements:
            while self.elements + size > self.max_elements:
                _, oldest = self._blocks.popitem(last=False)
                self.elements -= oldest.nrows() * oldest.ncols()
            self._blocks[key] = result
            self.elements += size
            self.peak_elements = max(self.peak_elements, self.elements)
        return result


def _floating_array(value: np.ndarray, ndim: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        raise TypeError(f"{name} must be a {ndim}-dimensional numpy ndarray")
    if value.dtype.kind != "f" or value.dtype.itemsize not in (2, 4, 8):
        raise TypeError(f"{name} must have binary16/32/64 floating dtype")
    return value


def _input_columns(x: Sequence[arb] | arb_mat, n: int) -> tuple[arb_mat, bool]:
    if isinstance(x, arb_mat):
        matrix, vector = x, False
        if matrix.nrows() != n or matrix.ncols() < 1:
            raise ValueError("input matrix must have n_inputs rows and at least one column")
    else:
        if not isinstance(x, (list, tuple)) or len(x) != n:
            raise ValueError("input vector must be a list/tuple of n_inputs Arb values")
        if any(not isinstance(v, arb) for v in x):
            raise TypeError("input vector values must already be Arb balls")
        matrix, vector = arb_mat(n, 1, list(x)), True
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            if not matrix[i, j].is_finite():
                raise ValueError("input balls must be finite")
    return matrix, vector


def affine(
    weights: np.ndarray,
    x: Sequence[arb] | arb_mat,
    bias: np.ndarray | None = None,
    *,
    chunk_rows: int = 32,
    chunk_cols: int = 256,
    bias_columns: Sequence[int] | None = None,
    cache: DyadicWeightCache | None = None,
) -> list[arb] | arb_mat:
    """Enclose ``weights @ x + bias`` without floating-point dot products.

    A matrix input shares each converted weight block over all columns. For a
    primal/tangent matrix explicitly use ``bias_columns=(0,)``. ``bias=None``
    skips bias. ``bias_columns=()`` skips it for every column. Weight and bias
    arrays are stored binary floating constants; x contains actual Arb balls.
    Nonfinite data and invalid shapes fail loudly. No inputs are modified.
    """
    weights = _floating_array(weights, 2, "weights")
    rows, inner = weights.shape
    if min(rows, inner) < 1:
        raise ValueError("weight dimensions must be positive")
    chunk_rows = _positive_integer(chunk_rows, "chunk_rows")
    chunk_cols = _positive_integer(chunk_cols, "chunk_cols")
    if cache is not None and not isinstance(cache, DyadicWeightCache):
        raise TypeError("cache must be DyadicWeightCache or None")
    matrix, vector = _input_columns(x, inner)
    columns = matrix.ncols()
    if bias_columns is None:
        selected = tuple(range(columns))
    else:
        selected = tuple(_positive_integer(i, "bias column", zero_ok=True) for i in bias_columns)
        if len(set(selected)) != len(selected) or any(i >= columns for i in selected):
            raise ValueError("bias columns must be distinct valid column indices")
    if bias is not None:
        bias = _floating_array(bias, 1, "bias")
        if bias.shape != (rows,):
            raise ValueError("bias must contain n_outputs entries")
        if not np.isfinite(bias).all():
            raise ValueError("bias constants must be finite")

    # These copies contain only O(n_inputs*n_columns) balls, never weights.
    input_blocks = []
    for start in range(0, inner, chunk_cols):
        stop = min(start + chunk_cols, inner)
        input_blocks.append((start, stop, arb_mat(stop - start, columns,
            [matrix[i, j] for i in range(start, stop) for j in range(columns)])))
    output = arb_mat(rows, columns)
    for row0 in range(0, rows, chunk_rows):
        row1 = min(rows, row0 + chunk_rows)
        accumulated = arb_mat(row1 - row0, columns)
        for col0, col1, input_block in input_blocks:
            # Always copy this bounded chunk first, making its key and exact
            # conversion describe the same bytes. Widening is exact.
            block = np.array(weights[row0:row1, col0:col1], dtype=np.float64, order="C", copy=True)
            if not np.isfinite(block).all():
                raise ValueError("weight constants must be finite")
            if cache is None:
                weight_block = arb_mat(block.shape[0], block.shape[1], block.ravel().tolist())
            else:
                weight_block = cache._matrix(block)
            accumulated = accumulated + weight_block * input_block
        for i in range(row0, row1):
            bias_ball = exact_dyadic(bias[i]) if bias is not None else None
            for j in range(columns):
                value = accumulated[i - row0, j]
                if bias_ball is not None and j in selected:
                    value = value + bias_ball
                if not value.is_finite():
                    raise ArithmeticError("nonfinite affine enclosure")
                output[i, j] = value
    return [output[i, 0] for i in range(rows)] if vector else output
