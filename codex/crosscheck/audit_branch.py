"""Independent saved-data arithmetic audit; no model or repository-tool imports.

This uses 60-digit Decimal arithmetic and a separately written normal-equation
solver. It checks the historical branch fit, not the underlying map, saved
eigenvalues, or asymptotic extrapolation. It is not exact or interval arithmetic.
Run from the standard library alone; --output is optional (otherwise stdout).
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


DATA = Path('/data2/neural-shrimp-data/explore/llm-soft-loop')
ATTEMPTS = Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck')


def read(path):
    return json.loads(path.read_text(), parse_float=Decimal)


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def fit(x, y, degree):
    """Unscaled normal equations, partial-pivot Gaussian elimination at Decimal60."""
    size = degree + 1
    system = [[sum(z ** (i+j) for z in x) for j in range(size)] +
              [sum(z ** i * value for z, value in zip(x, y))]
              for i in range(size)]
    for i in range(size):
        pivot_row = max(range(i, size), key=lambda k: abs(system[k][i]))
        system[i], system[pivot_row] = system[pivot_row], system[i]
        pivot = system[i][i]
        if not pivot:
            raise ValueError('singular normal equations')
        system[i] = [value / pivot for value in system[i]]
        for k in range(size):
            if k != i:
                multiplier = system[k][i]
                system[k] = [a - multiplier*b for a, b in zip(system[k], system[i])]
    coefficients = [row[-1] for row in system]
    residuals = [value - sum(coefficients[i]*z**i for i in range(size))
                 for z, value in zip(x, y)]
    return dict(coefficients_ascending=coefficients,
                rms=(sum(value*value for value in residuals)/len(y)).sqrt(),
                maximum_absolute_residual=max(abs(value) for value in residuals))


def audit(data, attempts):
    paths = dict(branch=data/'branch64-p2-t-0.115-alpha-up/branch.json',
                 normal_form=data/'normal-form-p2-t-0.115/normal-form.json',
                 strict_summary=attempts/'branch-001/run/branch-summary.json',
                 amended_summary=attempts/'branch-002/run/branch-summary.json',
                 amended_manifest=attempts/'branch-002/manifest.json',
                 amended_receipt=attempts/'branch-002/receipt.json')
    branch, normal_form = read(paths['branch']), read(paths['normal_form'])
    strict, saved = read(paths['strict_summary']), read(paths['amended_summary'])
    with localcontext() as context:
        context.prec = 60
        alpha_c = normal_form['source']['alpha']
        low, high = Decimal('0.00005'), Decimal('0.0004')
        rows = [row for row in branch['points'] if low <= alpha_c-row['lambda'] <= high]
        if len(rows) != 18:
            raise ValueError('expected 18 saved rows in fixed distance range')
        if [row['index'] for row in rows] != [row['saved_index'] for row in saved['selected_rows']]:
            raise ValueError('saved selection mismatch')
        if rows != [row['input_record'] for row in saved['selected_rows']]:
            raise ValueError('saved input records differ from original branch')
        for row in rows:
            multipliers = row['multipliers']
            largest = max(multipliers, key=lambda value: value[0]**2+value[1]**2)
            if largest is not multipliers[0] or largest[1] != 0 or not 0 < largest[0] < 1:
                raise ValueError('expected positive real first dominant multiplier')
            if row['multipliers_converged'] is not True or row['spectral_radius'] != largest[0]:
                raise ValueError('saved spectral metadata mismatch')
        delta = [alpha_c-row['lambda'] for row in rows]
        x = [value.sqrt() for value in delta]
        y = [(1-row['multipliers'][0][0])**2/(4*distance) for row, distance in zip(rows, delta)]
        fits = {}
        for degree, name in ((2, 'quadratic'), (1, 'linear')):
            result = fit(x, y, degree)
            result['intercept_minus_saved'] = result['coefficients_ascending'][0]-saved[name]['intercept']
            result['rms_minus_saved'] = result['rms']-saved[name]['residual_root_mean_square']
            result['intercept_minus_reference_ab'] = result['coefficients_ascending'][0]-normal_form['ab']
            fits[name] = result
        strict_failed = [row['index'] for row in rows if row['residual'] >= Decimal('1e-11')]
        producer_failed = [row['index'] for row in rows if row['residual'] >= branch['args']['tol']]
        gates = dict(original_records_match=True, strict_failed_row3=strict_failed == [3],
                     producer_tolerance_matches_1e_minus_10=branch['args']['tol'] == Decimal('1e-10'),
                     producer_all_rows_qualify=not producer_failed,
                     failed_strict_attempt_retained=strict['passed'] is False and
                         strict['tolerances']['closure'] == Decimal('1e-11') and
                         [r['saved_index'] for r in strict['selected_rows'] if not r['passed']] == [3],
                     amended_tolerance_disclosed=saved['passed'] is True and
                         saved['tolerances']['closure'] == branch['args']['tol'] and bool(saved.get('criterion_note')),
                     fits_agree_within_1e_minus_9=all(abs(result['intercept_minus_saved']) < Decimal('1e-9') and
                         abs(result['rms_minus_saved']) < Decimal('1e-9') for result in fits.values()))
        return dict(scope='Independent saved-data arithmetic reconstruction; no model or eigenpair reevaluation; no exact or interval claim.',
                    method='Decimal precision 60, unscaled normal equations and partial-pivot Gaussian elimination',
                    source_sha256=identity(Path(__file__).resolve())['sha256'],
                    inputs={name: identity(path) for name, path in paths.items()},
                    fit_source_commit=read(paths['amended_manifest'])['source_commit'],
                    alpha_c=alpha_c, selected_indices=[row['index'] for row in rows],
                    y_range=[min(y), max(y)], row3_saved_residual=rows[3]['residual'],
                    producer_saved_tolerance=branch['args']['tol'], strict_1e11_failed_rows=strict_failed,
                    producer_tolerance_failed_rows=producer_failed, fits=fits, gates=gates,
                    passed=all(gates.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--attempts', type=Path, default=ATTEMPTS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.data, args.attempts)
    text = json.dumps(result, indent=2, default=str, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
