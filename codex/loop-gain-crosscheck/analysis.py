"""Independent tied-readout factorization and finite dense loop diagnostics.

Only the already reviewed ManualQwen tensor architecture is reused. This module
never imports a producer analysis or model implementation.
"""
import math
import time


def normalized_hidden(model, state, alpha, cache):
    """Final normalized current-position hidden state, including prefix steering."""
    import torch
    alpha = model._scalar(alpha)
    cache = model._external_cache(cache, alpha)
    hidden = model._state(state)[:, None, :]
    positions = torch.tensor([len(model.prefix_ids)])
    for index in range(model.layers):
        hidden, _ = model._block(hidden, index, alpha, positions, cache.layers[index])
    hidden = model._norm(hidden[:, -1, :], model.weights['model.norm.weight'])
    return hidden[0] if state.ndim == 1 else hidden


def joint_map(model, state, alpha, log_temperature, cache):
    import torch
    hidden = normalized_hidden(model, state, alpha, cache)
    temperature = 10. ** model._scalar(log_temperature)
    probabilities = torch.softmax((hidden @ model.embedding.T) / temperature, dim=-1)
    return hidden, probabilities @ model.embedding


def covariance(embedding, probabilities, *, chunk=4096, check=lambda: None):
    """All-token centered second moment, accumulated by fixed vocabulary chunks.

    This avoids a vocabulary-square matrix and the subtraction of two large
    uncentered moments. No token threshold, truncation or renormalization.
    """
    import torch
    if embedding.ndim != 2 or probabilities.shape != (embedding.shape[0],):
        raise ValueError('full vocabulary probabilities/embedding rows required')
    if embedding.dtype != torch.float64 or probabilities.dtype != torch.float64:
        raise ValueError('binary64 covariance operands required')
    if not bool(torch.isfinite(embedding).all()) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError('finite covariance operands required')
    if bool((probabilities < 0).any()) or abs(float(probabilities.sum()) - 1.) > 1e-12:
        raise ValueError('nonnegative normalized probabilities required')
    if type(chunk) is not int or chunk < 1:
        raise ValueError('positive chunk required')
    mean = probabilities @ embedding
    answer = torch.zeros((embedding.shape[1],) * 2, dtype=torch.float64)
    for start in range(0, len(probabilities), chunk):
        check()
        centered = embedding[start:start+chunk] - mean
        weighted = centered * torch.sqrt(probabilities[start:start+chunk])[:, None]
        answer += weighted.T @ weighted
    return answer, mean


def relative(left, right):
    import torch
    error = float(torch.linalg.vector_norm(left - right))
    scale = float(torch.linalg.vector_norm(right))
    if not math.isfinite(error) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('finite positive reference derivative norm required')
    return error / scale


def chronological_product(matrices):
    """Input J_i = DF(x_i); return J_(q-1) ... J_1 J_0."""
    import numpy as np
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != values.shape[2] or not np.isfinite(values).all():
        raise ValueError('finite nonempty square step matrices required')
    product = np.eye(values.shape[1], dtype=np.float64)
    for matrix in values:
        product = matrix @ product
    return product


def loop_spectra(matrices):
    import numpy as np
    matrices = np.asarray(matrices, dtype=np.float64)
    product = chronological_product(matrices)
    step_singular = np.stack([np.linalg.svd(m, compute_uv=False) for m in matrices])
    multipliers = np.linalg.eigvals(product)
    multipliers = multipliers[np.argsort(-np.abs(multipliers), kind='stable')]
    singular = np.linalg.svd(product, compute_uv=False)
    return dict(product=product, step_singular_values=step_singular,
                monodromy_eigenvalues=multipliers, monodromy_singular_values=singular,
                product_step_norms=float(np.prod(step_singular[:, 0])))


def orbit(model, alpha, log_temperature, *, transient, period, cache, emit, check):
    import torch
    state = model.initial_state()
    emit('fixed-start', dict(state=state, prefix_ids=model.prefix_ids))
    started = time.monotonic()
    with torch.no_grad():
        for i in range(transient):
            check()
            state = model.forward(state, alpha, log_temperature, cache=cache)
            if not bool(torch.isfinite(state).all()):
                emit('nonfinite-transient', dict(index=i+1, state=state))
                raise ValueError('nonfinite transient')
            if (i+1) % 100 == 0:
                emit('transient-progress', dict(steps=i+1, state=state, seconds=time.monotonic()-started))
        states = [state.clone()]
        for _ in range(period):
            check()
            state = model.forward(state, alpha, log_temperature, cache=cache)
            states.append(state.clone())
    states = torch.stack(states)
    emit('orbit', dict(states=states, transient=transient, period=period,
                       seconds=time.monotonic()-started))
    if not bool(torch.isfinite(states).all()):
        raise ValueError('finite full orbit required')
    scale = max(float(torch.linalg.vector_norm(states[:-1], dim=1).mean()), 1e-30)
    closure = float(torch.linalg.vector_norm(states[-1]-states[0])) / scale
    separation = min(float(torch.linalg.vector_norm(states[i]-states[j])) / scale
                     for i in range(period) for j in range(i))
    gates = dict(relative_closure=closure, minimum_relative_phase_separation=separation,
                 closure_pass=closure <= 1e-10, distinct_phases_pass=separation > 1e-8)
    emit('orbit-gates', gates)
    if not gates['closure_pass'] or not gates['distinct_phases_pass']:
        raise ValueError('independent orbit closure/primitive separation gate failed')
    return states, gates


def phase(model, state, alpha, log_temperature, *, cache, batch, seed, emit, check):
    """Full h/F dense derivatives plus separate original-map random JVP controls."""
    import numpy as np
    import torch
    dimension = model.d
    if state.shape != (dimension,) or type(batch) is not int or batch < 1:
        raise ValueError('single full state and positive batch required')
    timings = {}
    started = time.monotonic()
    check()
    with torch.no_grad():
        hidden, image = joint_map(model, state, alpha, log_temperature, cache)
        original_image = model.forward(state, alpha, log_temperature, cache=cache)
        logits = hidden @ model.embedding.T
        temperature = 10. ** model._scalar(log_temperature)
        probabilities = torch.softmax(logits / temperature, dim=-1)
    emit('readout', dict(state=state, hidden=hidden, image=image,
                        original_image=original_image, probabilities=probabilities,
                        temperature=temperature, alpha=alpha, log10_temperature=log_temperature))
    forward_relative = relative(image, original_image)
    if forward_relative > 1e-12:
        raise ValueError('new hidden readout disagrees with original ManualQwen map')
    timings['readout'] = time.monotonic()-started
    started = time.monotonic()
    sigma, mean = covariance(model.embedding, probabilities, check=check)
    emit('covariance', dict(matrix=sigma, mean=mean, seconds=time.monotonic()-started))
    timings['covariance'] = time.monotonic()-started
    eye = torch.eye(dimension, dtype=torch.float64)
    hidden_columns, map_columns = [], []
    started = time.monotonic()
    function = lambda x: joint_map(model, x, alpha, log_temperature, cache)
    for first in range(0, dimension, batch):
        check()
        directions = eye[first:first+batch]
        points = state[None].expand(len(directions), -1).clone()
        began = time.monotonic()
        (h, f), (dh, df) = torch.func.jvp(function, (points,), (directions,))
        emit('basis', dict(first_column=first, state=state, directions=directions,
                           hidden_primal=h[0], image_primal=f[0], hidden_tangents=dh,
                           image_tangents=df, seconds=time.monotonic()-began))
        if not all(bool(torch.isfinite(x).all()) for x in (h, f, dh, df)):
            raise ValueError('nonfinite basis return')
        hidden_columns.append(dh.detach()); map_columns.append(df.detach())
    jh = torch.cat(hidden_columns).T.contiguous()
    jf = torch.cat(map_columns).T.contiguous()
    timings['joint_dense'] = time.monotonic()-started
    emit('dense', dict(hidden_jacobian=jh, map_jacobian=jf, columns=dimension,
                       seconds=timings['joint_dense']))
    factored = sigma @ jh / temperature
    dense_relative = relative(factored, jf)
    emit('factorized-dense', dict(matrix=factored, relative_error=dense_relative))
    rng = np.random.Generator(np.random.PCG64(seed))
    directions = torch.tensor(rng.normal(size=(8, dimension)), dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=1)[:, None]
    rows = []
    started = time.monotonic()
    for index, direction in enumerate(directions):
        check()
        _, direct = torch.func.jvp(lambda x:model.forward(x, alpha, log_temperature, cache=cache),
                                  (state,), (direction,))
        _, dh = torch.func.jvp(lambda x:normalized_hidden(model, x, alpha, cache),
                              (state,), (direction,))
        predicted = sigma @ dh / temperature
        row = dict(index=index, direction=direction, original_map_jvp=direct,
                   hidden_jvp=dh, predicted_jvp=predicted,
                   factor_relative=relative(predicted, direct),
                   dense_relative=relative(jf @ direction, direct),
                   hidden_dense_relative=relative(jh @ direction, dh))
        emit('direction', row)
        rows.append({key:row[key] for key in ('index','factor_relative','dense_relative','hidden_dense_relative')})
    timings['eight_random_controls'] = time.monotonic()-started
    started = time.monotonic()
    sigma_np = sigma.detach().numpy()
    symmetry_relative = float(np.linalg.norm(sigma_np-sigma_np.T)/max(np.linalg.norm(sigma_np),1e-30))
    sigma_values = np.linalg.eigvalsh(sigma_np)[::-1].copy()
    singular_values = np.linalg.svd(jf.detach().numpy(), compute_uv=False)
    timings['dense_eigen_svd'] = time.monotonic()-started
    emit('spectra', dict(covariance_eigenvalues=sigma_values, map_singular_values=singular_values,
                        seconds=timings['dense_eigen_svd']))
    checks = dict(forward_relative=forward_relative, factor_dense_relative=dense_relative,
                  covariance_symmetry_relative=symmetry_relative,
                  covariance_min_eigenvalue=float(sigma_values[-1]),
                  factorization_pass=dense_relative <= 1e-9 and all(r['factor_relative'] <= 1e-9 for r in rows),
                  dense_direction_pass=all(max(r['dense_relative'],r['hidden_dense_relative']) <= 1e-9 for r in rows),
                  covariance_pass=symmetry_relative <= 1e-12 and sigma_values[-1] >= -1e-12*max(1.,sigma_values[0]))
    summary = dict(dimension=dimension, vocabulary=model.vocab, batch=batch, seed=seed,
                   timings=timings, checks=checks, directions=rows,
                   first_two_step_singular_values=singular_values[:2].tolist(),
                   step_second_first_ratio=float(singular_values[1]/singular_values[0]),
                   first_two_covariance_eigenvalues=sigma_values[:2].tolist(),
                   covariance_second_first_ratio=float(sigma_values[1]/sigma_values[0]))
    emit('phase-summary', summary)
    if not all(checks[k] for k in ('factorization_pass','dense_direction_pass','covariance_pass')):
        raise ValueError('independent phase precision/factorization gate failed')
    return summary, jf.detach().numpy().copy()


def composed_controls(model, state, alpha, log_temperature, *, period, cache, matrix,
                      expected_image, seed, emit, check):
    """Direct original composed-map AD against the chronological dense product."""
    import numpy as np
    import torch
    matrix = torch.as_tensor(matrix, dtype=torch.float64)
    if matrix.shape != (model.d, model.d): raise ValueError('full composed product required')
    rng = np.random.Generator(np.random.PCG64(seed))
    directions = torch.tensor(rng.normal(size=(3, model.d)), dtype=torch.float64)
    directions /= torch.linalg.vector_norm(directions, dim=1)[:, None]
    rows = []
    for index, direction in enumerate(directions):
        check(); began = time.monotonic()
        primal, direct = torch.func.jvp(
            lambda x:model.period_map(x, alpha, log_temperature, period, cache=cache),
            (state,), (direction,))
        predicted = matrix @ direction
        row = dict(index=index, seed=seed, state=state, direction=direction, period=period,
                   original_composed_primal=primal, expected_image=expected_image,
                   original_composed_jvp=direct, dense_product_jvp=predicted,
                   derivative_relative=relative(predicted, direct),
                   primal_relative=relative(primal, expected_image),
                   seconds=time.monotonic()-began)
        emit('composed-direction', row)
        rows.append({k:row[k] for k in ('index','seed','period','derivative_relative','primal_relative','seconds')})
    passed = all(r['derivative_relative'] <= 1e-8 and r['primal_relative'] <= 1e-12 for r in rows)
    emit('composed-controls-summary', dict(passed=passed, rows=rows, derivative_tolerance=1e-8, primal_tolerance=1e-12))
    if not passed: raise ValueError('composed original-map/product precision gate failed')
    return dict(passed=passed, rows=rows, derivative_tolerance=1e-8, primal_tolerance=1e-12)
