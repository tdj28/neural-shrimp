"""Fixed early-information warning rules and separate retrospective scoring."""
import math
import numpy as np
ORIGINS=(0,4,8,12)
HORIZONS=(1,2,3,4)
THRESHOLD=.2
QUALIFICATION=.01


def vector(v):
    v=np.asarray(v,dtype=np.float64)
    if v.ndim!=1 or not v.size or not np.isfinite(v).all():raise ValueError('finite nonempty vector')
    return v


def relative(a,b):return float(np.linalg.norm(a-b))/max(float(np.linalg.norm(b)),1e-12)


def first_crossing(scores):
    if len(scores)!=4 or not all(math.isfinite(x)and x>=0 for x in scores):raise ValueError('four finite nonnegative fixed scores')
    return next((h for h,x in zip(HORIZONS,scores)if x>THRESHOLD),None)


def predict(u,v1,a2,f2,g0,g1,loss0,loss1):
    """Arguments contain no future states, derivatives, responses or labels."""
    u,v1,a2,f2,g0,g1=[vector(v)for v in(u,v1,a2,f2,g0,g1)]
    if len({v.shape for v in(u,v1,a2,f2,g0,g1)})!=1 or not all(math.isfinite(v)for v in(loss0,loss1)):
        raise ValueError('same dimension and finite early losses')
    if abs(np.linalg.norm(u)-1)>1e-10:raise ValueError('normalized current tangent')
    defect=relative(f2,a2);gradient=relative(g1,g0)
    loss=abs(loss1-loss0)/max(abs(loss0),1e-12)
    curvature=relative(u,v1)
    scores=dict(drift=[h*(h-1)*defect/2 for h in HORIZONS],
                persistent_defect=[defect]*4,
                gradient_change=[h*gradient for h in HORIZONS],
                loss_change=[h*loss for h in HORIZONS],
                static_curvature=[h*curvature for h in HORIZONS])
    scores['baseline_composite']=[max(scores[k][i]for k in('persistent_defect','gradient_change','loss_change','static_curvature'))for i in range(4)]
    return dict(scores=scores,predicted_crossing={k:first_crossing(v)for k,v in scores.items()},
                early_defect=defect,gradient_change=gradient,relative_loss_change=loss,
                directional_curvature=curvature,threshold=THRESHOLD,
                scope='Triangular drift is a fixed growth heuristic of early defect, not an error bound or additional information.')


def evaluate(prediction,u,frozen,ordered,response_large,response_small):
    """Evaluation arrays enter only after every predictor record is sealed."""
    u=vector(u);groups=[np.asarray(x,dtype=np.float64)for x in(frozen,ordered,response_large,response_small)]
    if any(a.shape!=(5,len(u))or not np.isfinite(a).all()for a in groups):raise ValueError('full local0..4 evaluation arrays')
    f,t,big,small=groups;rows=[]
    # Local origin qualification matters because these are continued perturbations.
    origin_qualified=relative(big[0],small[0])<=QUALIFICATION and relative(t[0],small[0])<=QUALIFICATION
    for h in HORIZONS:
        agreement=relative(big[h],small[h]);transport=relative(t[h],small[h])
        gain=float(np.linalg.norm(small[h]));scalar=float(np.linalg.norm(f[1]))**h
        rows.append(dict(horizon=h,finite_difference_disagreement=agreement,
            transported_error=transport,qualified=bool(origin_qualified and agreement<=QUALIFICATION and transport<=QUALIFICATION),
            frozen_error=relative(f[h],small[h]),persistence_error=relative(u,small[h]),
            actual_gain=gain,scalar_gain=scalar,scalar_gain_error=abs(scalar-gain)/max(gain,1e-12)))
    resolved=all(r['qualified']for r in rows)
    errors=[r['frozen_error']for r in rows]
    crossing=first_crossing(errors)if resolved else None
    future_eligible=bool(resolved and all(r['frozen_error']<=THRESHOLD for r in rows[:2]))
    return dict(rows=rows,origin_qualified=bool(origin_qualified),resolved=resolved,
                observed_crossing=crossing,right_censored=bool(resolved and crossing is None),
                future_warning_eligible=future_eligible,
                warnings={k:bool(any(s>THRESHOLD for s in v[2:]))for k,v in prediction['scores'].items()},
                future_failure=bool(future_eligible and any(x>THRESHOLD for x in errors[2:])))


def summarize(rows):
    if len(rows)!=32:raise ValueError('all32 fixed correlated development rows')
    methods=('drift','persistent_defect','gradient_change','loss_change','static_curvature','baseline_composite')
    eligible=[r for r in rows if r['evaluation']['future_warning_eligible']]
    counts={}
    for method in methods:
        c=dict(true_warning=0,false_warning=0,missed_failure=0,correct_quiet=0)
        for r in eligible:
            e=r['evaluation'];warning=e['warnings'][method];failure=e['future_failure']
            key=('true_warning'if failure else'false_warning')if warning else('missed_failure'if failure else'correct_quiet');c[key]+=1
        counts[method]=c
    return dict(total_rows=len(rows),resolved_rows=sum(r['evaluation']['resolved']for r in rows),
                future_eligible_rows=len(eligible),already_failing_rows=sum(r['evaluation']['resolved']and not r['evaluation']['future_warning_eligible']for r in rows),
                future_warning_counts=counts,
                drift_only_caught_failures=sum(r['evaluation']['future_failure']and r['evaluation']['warnings']['drift']and not r['evaluation']['warnings']['baseline_composite']for r in eligible),
                scope='Correlated consumed development prefixes; no population inference, fitting, intervention or claimed generalization.')
