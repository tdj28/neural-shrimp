"""Symbolic words of periodic orbits and unimodal (MSS) kneading words.

For a smooth map with Jacobian ``J`` whose eigenvalues are real (gradient
descent: ``J = I - P H`` is similar to ``I - P^1/2 H P^1/2``), an orbit point is
labeled ``R`` when the smallest eigenvalue of ``J`` is negative (the step
reverses orientation along its stiffest direction: curvature beyond the
critical value ``1/eta``) and ``L`` otherwise. The point closest to the
critical set (smallest ``|lambda_min|``) plays the role of the critical point
``C``; the word is read from the next point, giving ``p - 1`` symbols as in the
Metropolis-Stein-Stein (MSS) convention for superstable unimodal orbits.

``mss_words(n)`` returns the admissible words of superstable period-``n``
orbits of the logistic map in order of increasing parameter, computed
numerically, so window words of a reduced unimodal dynamics can be compared.
"""

import numpy as np
from scipy.optimize import brentq


def orbit_word(orbit, jacobian):
    """Return (word, lambda_min per point, negative-eigenvalue counts, C index)."""
    orbit = [np.asarray(point, dtype=np.float64) for point in orbit]
    lam_min, negatives = [], []
    for point in orbit:
        eigenvalues = np.linalg.eigvals(np.asarray(jacobian(point), dtype=np.float64))
        real = eigenvalues.real
        lam_min.append(float(real.min()))
        negatives.append(int(np.sum(real < 0)))
    lam_min = np.array(lam_min)
    c_index = int(np.argmin(np.abs(lam_min)))
    p = len(orbit)
    order = [(c_index + k) % p for k in range(1, p)]
    word = "".join("R" if lam_min[k] < 0 else "L" for k in order)
    return word, lam_min.tolist(), negatives, c_index


def _logistic_orbit_word(r, n):
    x = 0.5
    symbols = []
    for _ in range(n - 1):
        x = r * x * (1.0 - x)
        symbols.append("L" if x < 0.5 else "R")
    return "".join(symbols)


def mss_words(n, grid=200000):
    """Admissible MSS words of superstable period-``n`` logistic orbits, by parameter."""
    if n < 2:
        raise ValueError("period must be at least 2")

    def g(r):
        x = 0.5
        for _ in range(n):
            x = r * x * (1.0 - x)
        return x - 0.5

    rs = np.linspace(1.0 + 1e-9, 4.0, grid)
    values = np.array([g(r) for r in rs])
    words = []
    for a, b, fa, fb in zip(rs[:-1], rs[1:], values[:-1], values[1:]):
        if fa == 0.0 or fa * fb < 0:
            r = a if fa == 0.0 else brentq(g, a, b, xtol=1e-14)
            # keep only primitive period-n superstable orbits
            x, primitive = 0.5, True
            for k in range(1, n):
                x = r * x * (1.0 - x)
                if abs(x - 0.5) < 1e-9:
                    primitive = False
                    break
            if primitive:
                word = _logistic_orbit_word(r, n)
                if not words or words[-1][1] != word:
                    words.append((float(r), word))
    return words


_RANK = {"L": 0, "C": 1, "R": 2}


def _unimodal_compare(a, b):
    """Compare two equal-length symbol strings in the unimodal (parity) order.

    Symbols are ordered L < C < R; after an odd number of R symbols in the
    common prefix, the order of the next symbol is reversed. Returns -1, 0, 1.
    """
    flips = 0
    for x, y in zip(a, b):
        if x != y:
            sign = -1 if _RANK[x] < _RANK[y] else 1
            return -sign if flips % 2 else sign
        if x == "R":
            flips += 1
    return 0


def is_mss_admissible(word):
    """True if ``word`` (p - 1 symbols L/R) is a superstable unimodal kneading word.

    MSS admissibility: the periodic sequence (word + 'C') is shift-maximal in
    the unimodal order and has primitive period ``len(word) + 1``.
    """
    sequence = word + "C"
    n = len(sequence)
    repeated = sequence * 3
    reference = repeated[:2 * n]
    for k in range(1, n):
        shifted = repeated[k:k + 2 * n]
        if shifted == reference:
            return False
        if _unimodal_compare(shifted, reference) > 0:
            return False
    return True
