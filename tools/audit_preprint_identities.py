#!/usr/bin/env python3
"""Exact independent-spin audit, not a reproduction of neural target results.

The comparison identifies consequences of printed arXiv:2609.07341v1 SM
Eqs. S40--S41. It cannot determine which formulas the authors implemented.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json


def lag_table(
    delta: Fraction = Fraction(1, 10),
    beta: Fraction = Fraction(1),
    max_lag: int = 5,
) -> list[dict[str, object]]:
    """Derive rational moments for an unbiased independent refreshed spin.

    With probability delta the spin is resampled from its zero-field law;
    otherwise it persists. A field impulse acts at the first update only.
    Exact values are strings so JSON serialization preserves rational results.
    """
    delta, beta = Fraction(delta), Fraction(beta)
    if not 0 < delta <= 1:
        raise ValueError("delta must lie in (0, 1]")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if isinstance(max_lag, bool) or not isinstance(max_lag, int) or max_lag < 1:
        raise ValueError("max_lag must be a positive integer")
    rows: list[dict[str, object]] = []
    for lag in range(1, max_lag + 1):
        persistence = (1 - delta) ** (lag - 1)
        values = {
            "q_update_law": (1 - delta) * persistence,
            "q_printed_S40": Fraction(0),
            "chi_update_law": beta * delta * persistence,
            "chi_printed_S41": beta * persistence,
        }
        rows.append({
            "lag": lag,
            "exact": {name: str(value) for name, value in values.items()},
            "numeric": {name: float(value) for name, value in values.items()},
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=Fraction, default=Fraction(1, 10))
    parser.add_argument("--beta", type=Fraction, default=Fraction(1))
    parser.add_argument("--max-lag", type=int, default=5)
    args = parser.parse_args()
    try:
        rows = lag_table(args.delta, args.beta, args.max_lag)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({
        "scope": "synthetic independent-spin algebra; no neural target outcomes",
        "source_version": "arXiv:2609.07341v1",
        "delta": str(args.delta),
        "beta": str(args.beta),
        "rows": rows,
        "interpretation": (
            "Printed-identity comparison only; implementation impact unknown. "
            "A finite strictly causal response matrix is nilpotent, so its "
            "spectral radius is zero; infinite-time stability needs a separate "
            "operator definition."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
