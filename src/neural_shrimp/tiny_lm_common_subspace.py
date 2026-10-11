"""Saved-action common-subspace experiment; no model or derivative acquisition.

All claims are conditional on the fixed reduced-model fidelity gates. The
commuting sequence preserves matched per-time eigenvalues while freezing their
basis. Its mean generally differs from the mean of the original sequence.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

GATES = dict(rank_relative=1e-8, condition=1e8, orthogonality=1e-12,
             input_closure=1e-6, action_closure=1e-6, symmetry=1e-6,
             subspace_leakage=.01, direction_closure=1e-10, fidelity=.01,
             eigenvalue_gap=1e-6, matched_overlap_squared=.5,
             matching_margin_squared=.1)
HORIZONS = (1, 4, 16)


def need(ok, text):
    if not ok:
        raise ValueError(text)


def relative(a, b):
    return float(np.linalg.norm(a-b)) / max(float(np.linalg.norm(b)), 1e-12)


def match_eigenvalues(operators):
    """Fixed consecutive overlap assignment, with gap/uniqueness exclusions."""
    n = operators.shape[1]
    values, vectors, diagnostics = [], [], []
    previous = None
    for t, matrix in enumerate(operators):
        lam, vec = np.linalg.eigh(matrix)
        scale = max(1., float(np.linalg.norm(matrix, 2)))
        gap = float(np.min(np.diff(lam))) / scale
        row = dict(time=t, relative_gap=gap, gap_pass=gap > GATES['eigenvalue_gap'])
        if previous is not None:
            overlap = (previous.T @ vec)**2
            rr, cc = linear_sum_assignment(-overlap)
            need(np.array_equal(rr, np.arange(n)), 'complete eigenvalue assignment')
            selected = overlap[rr, cc]
            other = overlap.copy(); other[rr, cc] = -np.inf
            margin = selected - np.max(other, axis=1)
            row.update(minimum_overlap_squared=float(selected.min()),
                       minimum_margin_squared=float(margin.min()),
                       matching_pass=bool(np.all(selected > GATES['matched_overlap_squared']) and
                                          np.all(margin > GATES['matching_margin_squared'])))
            lam, vec = lam[cc], vec[:, cc]
            vec *= np.where(np.sum(previous*vec, axis=0) < 0, -1., 1.)
        else:
            row['matching_pass'] = True
        values.append(lam); vectors.append(vec); diagnostics.append(row); previous = vec
    return np.asarray(values), np.asarray(vectors), diagnostics


def product_response(operators, u):
    rows = [u.copy()]
    for matrix in operators:
        rows.append(matrix @ rows[-1])
    return np.asarray(rows)


def comparisons(operators, u, eigenvalues, eigenvectors):
    """Three-term vector identity avoids attributing mean failure to ordering."""
    basis = eigenvectors[0]
    commuting = np.asarray([(basis*lam) @ basis.T for lam in eigenvalues])
    average = operators.mean(axis=0)
    commuting_average = commuting.mean(axis=0)
    ordered = product_response(operators, u)
    mean = product_response(np.repeat(average[None], 16, axis=0), u)
    scalar = product_response(commuting, u)
    scalar_mean = product_response(np.repeat(commuting_average[None], 16, axis=0), u)
    reverse = [u.copy()]
    for h in range(1, 17):
        reverse.append(product_response(operators[:h][::-1], u)[-1])
    reverse = np.asarray(reverse)
    alignment = ordered-scalar
    temporal = scalar-scalar_mean
    changed_mean = scalar_mean-mean
    residual = np.linalg.norm(alignment+temporal+changed_mean-(ordered-mean))
    scale = max(1., sum(float(np.linalg.norm(v)) for v in (ordered, scalar, scalar_mean, mean)))
    need(residual <= 128*np.finfo(float).eps*scale, 'complete three-term vector decomposition')
    return dict(ordered=ordered, mean=mean, reversed=reverse, commuting=scalar,
                commuting_mean=scalar_mean, basis_alignment_term=alignment,
                temporal_gain_term=temporal, changed_mean_term=changed_mean,
                commuting_operators=commuting)


def analyze(mean_powers, actions, ordered_full, eta, save=lambda *_: None):
    """actions[h,t,:] is saved H(theta_t) mean_powers[h], h,t=0..15.

    Caller authenticates exact original objects, directions, time ordering and
    all acquired operations. This helper never expands the span or acquires data.
    save() retains completed arrays before any subsequent interpretation gate.
    """
    powers = np.asarray(mean_powers, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    full = np.asarray(ordered_full, dtype=np.float64)
    need(powers.ndim == 2 and powers.shape[0] == 17 and powers.shape[1] >= 2,
         'complete mean powers')
    N = powers.shape[1]
    need(actions.shape == (16, 16, N) and full.shape == powers.shape and
         all(np.isfinite(v).all() for v in (powers, actions, full)) and
         np.isfinite(eta) and eta > 0, 'finite fixed common-action data')
    need(np.array_equal(powers[0], full[0]), 'same literal initial direction')
    report = dict(passed=False, status='abstained', gates=dict(GATES), reasons=[], dimension=N)
    def gate(ok, reason):
        if not ok:
            report['reasons'].append(reason)
    norms = np.linalg.norm(powers[:16], axis=1)
    gate(bool(np.all(norms > 0) and np.isfinite(norms).all()), 'nonzero finite Krylov columns')
    if report['reasons']: return report
    K = powers[:16].T/norms
    left, singular, vt = np.linalg.svd(K, full_matrices=False)
    save('column_norms', norms); save('singular_values', singular)
    rank = int(np.sum(singular >= GATES['rank_relative']*singular[0]))
    report.update(rank=rank, retained_condition=float(singular[0]/singular[rank-1]))
    gate(rank >= 2, 'at least two resolved common directions')
    gate(report['retained_condition'] <= GATES['condition'], 'fixed retained conditioning')
    if report['reasons']: return report
    E = left[:, :rank]; W = vt[:rank].T/singular[:rank]
    save('basis', E); save('coefficient_map', W)
    coordinates = E.T @ K
    ortho = float(np.max(np.abs(E.T @ E-np.eye(rank))))
    input_error = relative(E @ coordinates, K)
    u = E.T @ powers[0]
    direction_error = relative(E @ u, powers[0])
    report.update(orthogonality=ortho, input_closure=input_error, direction_closure=direction_error)
    gate(ortho <= GATES['orthogonality'], 'orthogonal retained basis')
    gate(input_error <= GATES['input_closure'], 'normalized input closure')
    gate(direction_error <= GATES['direction_closure'], 'literal initial direction closure')
    operators, diagnostics = [], []
    for t in range(16):
        B = actions[:, t, :].T/norms
        HE = B @ W
        applied = E-eta*HE
        raw = E.T @ applied
        skew = relative(raw, raw.T)
        action_error = relative(HE @ coordinates, B)
        leakage = relative(E @ raw, applied)
        operators.append((raw+raw.T)/2)
        diagnostics.append(dict(time=t, action_closure=action_error, symmetry=skew,
                                subspace_leakage=leakage))
        save(f'raw_operator_{t:02d}', raw)
        gate(action_error <= GATES['action_closure'], f'action closure t={t}')
        gate(skew <= GATES['symmetry'], f'projected symmetry t={t}')
        gate(leakage <= GATES['subspace_leakage'], f'operator subspace leakage t={t}')
    operators = np.asarray(operators)
    save('symmetric_operators', operators); report['operator_diagnostics'] = diagnostics
    if report['reasons']: return report
    ordered = product_response(operators, u)
    mean = product_response(np.repeat(operators.mean(axis=0)[None], 16, axis=0), u)
    save('projected_ordered', ordered); save('projected_mean', mean)
    fidelity = []
    for h in HORIZONS:
        oe, me = relative(E @ ordered[h], full[h]), relative(E @ mean[h], powers[h])
        fidelity.append(dict(horizon=h, ordered_error=oe, mean_error=me))
        gate(oe <= GATES['fidelity'], f'full ordered fidelity h={h}')
        gate(me <= GATES['fidelity'], f'full mean fidelity h={h}')
    report['fidelity'] = fidelity
    if report['reasons']: return report
    values, vectors, matching = match_eigenvalues(operators)
    save('matched_eigenvalues', values); save('matched_eigenvectors', vectors)
    report['matching'] = matching
    gate(all(x['gap_pass'] for x in matching), 'no near-degenerate reduced eigenvalues')
    gate(all(x['matching_pass'] for x in matching), 'unambiguous consecutive eigenvalue matching')
    if report['reasons']: return report
    result = comparisons(operators, u, values, vectors)
    for key, value in result.items(): save(key, value)
    rows = []
    for h in HORIZONS:
        denominator = max(float(np.linalg.norm(full[h])), 1e-12)
        errors = {k: relative(E @ v[h], full[h]) for k, v in result.items()
                  if k in ('ordered', 'mean', 'reversed', 'commuting', 'commuting_mean')}
        terms = {k: float(np.linalg.norm(v[h]))/denominator for k, v in result.items()
                 if k.endswith('_term')}
        rows.append(dict(horizon=h, full_ordered_relative_errors=errors,
                         decomposition_term_norms=terms))
    report.update(passed=True, status='qualified_reduced_model', rows=rows,
                  scope='Retrospective reduced-model sufficiency; no full-operator causality, noncommutativity isolation, forecast, geometry or utility claim.')
    return report
