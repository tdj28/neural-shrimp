"""Fixed quadratic-path merit comparison; the forcing gate applies to its tangent."""
import numpy as np
from .cifar_current_correction import direction_gate
from .cifar_reorthogonalized import vector
SETTINGS=dict(alphas=[1/32,1/64],acceleration_alpha=1/64,correction_ratio_limit=.2,trust_relative=.05,armijo=1e-4,action_relative=1e-8)
def evaluate(x,g,origin,p,predicted_hp,hp,report,a,predicted_ha,ha,gradient,*,event=lambda r:None):
    x=vector(x);n=len(x)
    g,origin,p,predicted_hp,hp,a,predicted_ha,ha=[vector(v,n)for v in(g,origin,p,predicted_hp,hp,a,predicted_ha,ha)]
    gate=direction_gate(g,predicted_hp,hp,report)
    scale=max(1.,float(np.linalg.norm(origin)));ng=float(np.linalg.norm(g));pn=float(np.linalg.norm(p))
    if not gate['passed']or pn<=0 or float(np.linalg.norm(x-origin))/scale>SETTINGS['trust_relative']:raise ValueError('original tangent and original global trust required')
    err=float(np.linalg.norm(ha-predicted_ha))/max(ng,float(np.linalg.norm(ha)),float(np.linalg.norm(predicted_ha)),1e-12)
    action_ok=np.isfinite(err)and err<=SETTINGS['action_relative']
    rows=[];vectors=[];phi=.5*float(np.dot(g,g))
    for alpha in SETTINGS['alphas']:
        y=x+alpha*p+alpha*alpha*a
        distance=float(np.linalg.norm(y-origin))/scale;ratio=alpha*float(np.linalg.norm(a))/pn
        admitted=bool(action_ok and distance<=SETTINGS['trust_relative']and ratio<=SETTINGS['correction_ratio_limit'])
        row=dict(alpha=alpha,relative_origin_distance=distance,correction_to_primary_step=ratio,admitted=admitted,accepted=False)
        if admitted:
            gy=vector(gradient(y.copy()),n);merit=.5*float(np.dot(gy,gy));bound=phi+SETTINGS['armijo']*alpha*gate['slope']
            row.update(merit=merit,merit_ratio=merit/phi,armijo_bound=bound,accepted=bool(merit<=bound),gradient_norm=float(np.linalg.norm(gy)),relative_gradient=float(np.linalg.norm(gy))/max(1.,float(np.linalg.norm(y))))
            vectors.append((y,gy))
        else:vectors.append(None)
        rows.append(row);event(dict(kind='curved_path_trial',**row))
    accepted=next((j for j,r in enumerate(rows)if r['accepted']),None)
    return dict(tangent_gate=gate,acceleration_action_relative_error=err,acceleration_action_qualified=bool(action_ok),trials=rows,accepted_index=accepted,scope='Forcing qualifies initial tangent p only. Both fixed quadratic-path points evaluated; first accepted alpha selected. Composite p+alpha*a has no asserted forcing qualification.'),vectors
