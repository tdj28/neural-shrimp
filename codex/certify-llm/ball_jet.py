"""Signed multi-direction first derivatives for the inherited exact-real map.

``BallQwenJet(..., direction_count=8).forward(x, directions)`` returns a list
of primal Arb balls and an Arb matrix with one JVP per column. Directions are
an Arb matrix of shape (state_dimension, K), or a finite binary16/32/64 NumPy
array of shape (K, state_dimension). Scalar/zero directions must be supplied
explicitly as that shape; omitted directions mean K exact zero columns.

Each scalar carries K signed derivative coefficients. Affine operations pack
one primal plus all K tangent columns per token in a single native Arb product.
This preserves signed linear combinations for each direction; it does not
replace all input directions by an interval cube. All decoder, attention,
rotary, steering and cache control flow is inherited from BallQwen through its
six arithmetic hooks. There is no copied transformer architecture here.

This is first-order directional arithmetic, NOT a Taylor remainder kernel,
full Jacobian certificate, low-rank tail bound, or orbit proof. If state inputs
are balls, each JVP still encloses that specified directional action on them.
Eight columns do not certify the other 888 directions of the target model.

Like BallQwen, inputs and precision must remain immutable during cache use.
A custom progress callback receives Jet objects: .tangent is a tuple of K Arb
scalars, not a scalar as it is for the single-direction Dual implementation.
No checkpoint access or target computation occurs on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
from flint import arb, arb_mat

from ball_linear import affine
from ball_qwen import BallQwen, exact


@dataclass(frozen=True)
class Jet:
    value: arb
    tangent: tuple[arb, ...]

    def __post_init__(self):
        if not isinstance(self.value, arb):
            raise TypeError("Jet value must be an Arb ball")
        if not isinstance(self.tangent, tuple) or not self.tangent or any(not isinstance(v, arb) for v in self.tangent):
            raise TypeError("Jet tangent must be a nonempty tuple of Arb balls")

    @staticmethod
    def make(value, tangent: Sequence):
        return Jet(exact(value), tuple(exact(v) for v in tangent))

    @property
    def dimension(self):
        return len(self.tangent)

    def constant(self, value):
        if isinstance(value, Jet):
            if value.dimension != self.dimension:
                raise ValueError("mixed Jet direction counts")
            return value
        return Jet(exact(value), (arb(0),) * self.dimension)

    def __add__(self, other):
        other = self.constant(other)
        return Jet(self.value + other.value, tuple(a+b for a,b in zip(self.tangent, other.tangent)))
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, tuple(-v for v in self.tangent))

    def __sub__(self, other):
        return self + -self.constant(other)

    def __rsub__(self, other):
        return self.constant(other) + -self

    def __mul__(self, other):
        other = self.constant(other)
        return Jet(self.value*other.value,
                   tuple(a*other.value + self.value*b for a,b in zip(self.tangent, other.tangent)))
    __rmul__ = __mul__

    def reciprocal(self):
        if self.value.contains(0):
            raise ValueError("denominator enclosure contains zero")
        inverse = 1/self.value
        factor = -inverse*inverse
        return Jet(inverse, tuple(factor*v for v in self.tangent))

    def __truediv__(self, other):
        return self*self.constant(other).reciprocal()

    def __rtruediv__(self, other):
        return self.constant(other)*self.reciprocal()

    def exp(self):
        value = self.value.exp()
        return Jet(value, tuple(value*v for v in self.tangent))

    def rsqrt(self):
        if not self.value > 0:
            raise ValueError("positive RMS denominator not established")
        inverse = self.value.rsqrt()
        factor = -inverse*inverse*inverse/2
        return Jet(inverse, tuple(factor*v for v in self.tangent))


def finite_jet(value: Jet) -> bool:
    return value.value.is_finite() and all(v.is_finite() for v in value.tangent)


def _dimension(values: Sequence[Jet]) -> int:
    if not values or any(not isinstance(q, Jet) for q in values):
        raise TypeError("nonempty sequence of Jets required")
    dimension = values[0].dimension
    if any(q.dimension != dimension for q in values):
        raise ValueError("mixed Jet direction counts")
    return dimension


def softmax(values: Sequence[Jet]) -> list[Jet]:
    dimension = _dimension(values)
    if any(not finite_jet(q) for q in values):
        raise ValueError("nonfinite softmax input")
    shift = max((q.value.mid() for q in values), key=float)
    exps = [(q.value-shift).exp() for q in values]
    denominator = sum(exps, arb(0))
    if not denominator > 0:
        raise ValueError("positive softmax denominator not established")
    probabilities = [v/denominator for v in exps]
    average = [sum((p*q.tangent[k] for p,q in zip(probabilities,values)), arb(0))
               for k in range(dimension)]
    return [Jet(p, tuple(p*(q.tangent[k]-average[k]) for k in range(dimension)))
            for p,q in zip(probabilities,values)]


def rmsnorm(values: Sequence[Jet], weights, epsilon) -> list[Jet]:
    dimension = _dimension(values)
    if len(weights) != len(values):
        raise ValueError("RMSNorm weight dimension mismatch")
    zero = Jet(arb(0), (arb(0),)*dimension)
    variance = sum((q*q for q in values), zero)/len(values)
    inverse = (variance+exact(epsilon)).rsqrt()
    return [q*inverse*exact(w) for q,w in zip(values,weights)]


def linear_sequence(weight, sequence, bias=None, **options) -> list[list[Jet]]:
    if not sequence:
        raise ValueError("nonempty sequence required")
    dimension = _dimension(sequence[0])
    n, count = len(sequence[0]), len(sequence)
    if any(len(state) != n or _dimension(state) != dimension for state in sequence):
        raise ValueError("ragged sequence or inconsistent direction count")
    stride = dimension+1
    matrix = arb_mat(n, stride*count)
    for j,state in enumerate(sequence):
        for i,q in enumerate(state):
            matrix[i,stride*j] = q.value
            for k,v in enumerate(q.tangent):
                matrix[i,stride*j+k+1] = v
    output = affine(weight, matrix, bias,
                    bias_columns=tuple(range(0,stride*count,stride)), **options)
    return [[Jet(output[i,stride*j], tuple(output[i,stride*j+k+1] for k in range(dimension)))
             for i in range(output.nrows())] for j in range(count)]


class BallQwenJet(BallQwen):
    """Unchanged inherited real map with K explicitly signed JVP columns."""

    normalize = staticmethod(rmsnorm)
    probabilities = staticmethod(softmax)
    affine_sequence = staticmethod(linear_sequence)
    finite_number = staticmethod(finite_jet)

    def __init__(self, *args, direction_count=8, **kwargs):
        if isinstance(direction_count, bool) or not isinstance(direction_count, Integral) or direction_count < 1:
            raise ValueError("positive integer direction_count required")
        self._direction_count = int(direction_count)
        super().__init__(*args, **kwargs)

    @property
    def direction_count(self):
        return self._direction_count

    def scalar(self, value, tangent=0):
        if isinstance(tangent, (int, np.integer, float, np.floating, arb)):
            if exact(tangent) != 0:
                raise ValueError("nonzero tangent requires K explicit coefficients")
            directions = (arb(0),) * self.direction_count
        elif isinstance(tangent, (list, tuple)) and len(tangent) == self.direction_count:
            directions = tuple(exact(v) for v in tangent)
        else:
            raise ValueError("one explicit tangent coefficient per direction required")
        return Jet(exact(value), directions)

    def coerce(self, value):
        if isinstance(value, Jet):
            if value.dimension != self.direction_count:
                raise ValueError("mixed Jet direction counts")
            return value
        return self.scalar(value)

    def forward(self, state, directions=None):
        """Return (primal[d], JVP[d,K]); no numerical products in admission.

        ``directions`` is Arb[d,K] or finite NumPy[K,d] of stored binary16/32/64
        values. Each direction is independent. An Arb column may itself be an
        interval direction, but that does not imply bounds for omitted columns.
        """
        if directions is None:
            rows = [(arb(0),)*self.direction_count for _ in range(self.d)]
        elif isinstance(directions, arb_mat):
            if (directions.nrows(), directions.ncols()) != (self.d, self.direction_count):
                raise ValueError("Arb directions must have shape(d,K)")
            rows = [tuple(exact(directions[i,k]) for k in range(self.direction_count)) for i in range(self.d)]
        elif isinstance(directions, np.ndarray):
            if (directions.shape != (self.direction_count,self.d) or directions.dtype.kind != "f"
                    or directions.dtype.itemsize not in (2,4,8)):
                raise ValueError("NumPy directions must be binary16/32/64 of shape(K,d)")
            if not np.isfinite(directions).all():
                raise ValueError("directions must be finite")
            rows = [tuple(exact(directions[k,i]) for k in range(self.direction_count)) for i in range(self.d)]
        else:
            raise TypeError("directions must be an Arb matrix or a floating NumPy array")
        values, derivatives = super().forward(state, rows)
        return values, arb_mat(self.d, self.direction_count, [v for row in derivatives for v in row])
