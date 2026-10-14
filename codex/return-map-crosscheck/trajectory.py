"""Fixed-start finite twin experiment; arbitrary step callable enables toy tests.

Each record row stores F(input) and the readout of that SAME input. The readout
is not recomputed on the output. Epsilons are absolute Euclidean offsets.
"""
import math
import numpy as np


def checked(value, shape):
    a = np.asarray(value)
    if a.dtype != np.float64 or a.shape != shape or not np.isfinite(a).all():
        raise ValueError('finite full binary64 state required')
    return a


def advance(step, initial, sink, transient=1000, record=2000, offsets=(1e-6, 1e-9)):
    """All completed returns, even nonfinite ones, reach sink before refusal."""
    x = checked(initial, (len(initial),)).copy(); dimension = len(x)
    for i in range(transient):
        sink('input', 'transient', i, {'state': x[None].copy()})
        result = step(x[None])
        sink('return', 'transient', i, result)
        x = checked(result['state'], (1,dimension))[0].copy()
        for key in ('pmax', 'entropy'):
            if not np.isfinite(result[key]).all():
                raise ValueError('nonfinite completed transient readout')
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(len(offsets), dimension))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    state = np.concatenate([x[None], x[None]+directions*np.array(offsets)[:,None]])
    sink('initial_twins', 'record', -1, dict(state=state.copy(), directions=directions, offsets=list(offsets)))
    growth = []; clamps = []; distance = []
    for i in range(record):
        sink('input', 'record', i, {'state': state.copy()})
        result = step(state)
        sink('return', 'record', i, result)
        output = checked(result['state'], state.shape)
        for key in ('pmax', 'entropy'):
            values = np.asarray(result[key])
            if values.shape != (len(offsets)+1,) or not np.isfinite(values).all():
                raise ValueError('nonfinite or incomplete completed readout')
        if np.asarray(result['lead']).shape != (len(offsets)+1,) or not np.issubdtype(np.asarray(result['lead']).dtype, np.integer):
            raise ValueError('complete integer leading-token readout required')
        if np.any((result['pmax'] <= 0) | (result['pmax'] > 1)):
            raise ValueError('probability outside (0,1]')
        differences = output[1:]-output[0]
        raw = np.linalg.norm(differences, axis=1)
        floors = np.array(offsets)*1e-30
        distances = np.maximum(raw, floors)
        logs = np.log(distances/offsets)
        if not np.isfinite(logs).all():
            raise ValueError('nonfinite twin growth')
        state = np.concatenate([output[:1], output[0]+differences*(np.array(offsets)/distances)[:,None]])
        sink('growth', 'record', i, dict(raw_distance=raw, used_distance=distances,
            log_growth=logs, clamped=(raw < floors), renormalized_state=state.copy()))
        growth.append(logs); clamps.append(raw < floors); distance.append(raw)
    growth = np.asarray(growth)
    return dict(transient_steps=transient, recorded_steps=record, native_step_calls=transient+record,
        row_forwards=transient+(len(offsets)+1)*record, offsets=list(offsets), seed=0,
        lyapunov=growth.mean(0).tolist(), floors_used=np.asarray(clamps).sum(0).tolist(),
        disjoint_windows=[dict(start=int(ids[0]), stop=int(ids[-1])+1,
                              means=growth[ids].mean(0).tolist()) for ids in np.array_split(np.arange(record), 4)],
        running_means=[dict(stop=i, means=growth[:i].mean(0).tolist()) for i in range(100,record+1,100)],
        exact_orbit_or_asymptotic_chaos_proved=False)


def state_diagnostics(states, tolerance=1e-8, maximum=64):
    s = np.asarray(states)
    if s.ndim != 2 or len(s) <= maximum or not np.isfinite(s).all():
        raise ValueError('complete finite state record required')
    scale = max(1., float(np.linalg.norm(s[-1])))
    rows = [dict(lag=q, relative_distance=float(np.linalg.norm(s[-q-1]-s[-1])/scale)) for q in range(1,maximum+1)]
    period = next((r['lag'] for r in rows if r['relative_distance'] < tolerance), 0)
    tail = s[-400:]; radius = np.linalg.norm(tail-tail.mean(0), axis=1).max()
    return dict(period=period, all_lags=rows, tolerance=tolerance, scale=scale,
                diameter_relative=float(2*radius/scale), detector='last-state finite-lag return only')
