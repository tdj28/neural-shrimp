"""Two observed derivative snapshots; later states are evaluation data only."""
import numpy as np
from neural_shrimp.tiny_lm_response import finite, relative

def forecast(initial_response, eta, theta0, theta1, hvp, save, check):
    first = finite(initial_response).copy()
    trajectories = {}
    for label in ('alternating', 'latest'):
        values = [first.copy()]
        try:
            for step in range(1,16):
                check()
                theta = theta1 if label == 'latest' or step % 2 else theta0
                action=np.asarray(hvp(theta,values[-1]),dtype=np.float64)
                save(f'{label}-hvp-{step:02d}',action)
                values.append(finite(values[-1]-eta*finite(action)))
        finally:
            save(label,np.asarray(values))
        trajectories[label]=values
    return trajectories

def scores(predictions, responses, readout, slopes):
    out=[]
    for h in (4,16):
        target=finite(responses[h]);row=dict(horizon=h)
        for name,values in predictions.items():
            v=finite(values[h-1])
            row[name+'_vector_error']=relative(v,target)
            row[name+'_gain']=float(np.linalg.norm(v))
            row[name+'_ce_slope']=float(np.dot(readout[h],v))
            row[name+'_ce_slope_relative_error']=abs(row[name+'_ce_slope']-slopes[h])/max(abs(slopes[h]),1e-8)
        out.append(row)
    return out

def candidate_gate(rows):
    chosen=[r for r in rows if r['arm']==256 and r['direction']=='gradient' and r['horizon']==16]
    if len(chosen)!=2 or {r['seed']for r in chosen}!={28161,28162}:raise ValueError('two fixed cases')
    return all(r['mechanism_qualified'] and r['alternating_vector_error']<.2
               and 10*r['alternating_vector_error']<r['latest_vector_error']
               and 10*r['alternating_vector_error']<r['persistence_vector_error'] for r in chosen)
