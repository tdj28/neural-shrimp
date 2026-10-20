#!/usr/bin/env python3
"""Task40 supplemental Sec V saved-data audit. No model, producer imports or old-audit replay.
Fresh: small saved passage/branch reductions; reused: exact closed Task2/5/5B receipts.
Run only after Root checkpoint/admission. All files read before decoding; new exclusive result.
"""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time

ROOT=Path('/data2/neural-shrimp')
DATA=Path('/data2/neural-shrimp-data')
SOFT=DATA/'explore/llm-soft-loop'
OUT=ROOT/'codex/arxiv-gonogo2-task40/fold-evidence.json'
MANIFEST={}

def read(path, expected=None):
    p=Path(path)
    if not (p.is_relative_to(SOFT) or p.is_relative_to(ROOT/'codex/crosscheck') or
            p.is_relative_to(ROOT/'codex/confirm-fold') or p.is_relative_to(ROOT/'codex/confirm-fold-2') or
            p in (ROOT/'docs/wp-window-predictions.json',ROOT/'docs/wp-line-cells.json') or
            p in (DATA/'codex/task5-fresh-prediction/passage-audit-v1.json',
                  DATA/'codex/task5b-fresh-prediction/audit-01/audit.json')):
        raise ValueError('outside fixed saved-evidence roots')
    before=p.stat()
    if before.st_size>64*1024*1024: raise ValueError('input cap')
    raw=p.read_bytes(); after=p.stat()
    identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    if identity(before)!=identity(after) or len(raw)!=before.st_size: raise ValueError('input changed')
    sha=hashlib.sha256(raw).hexdigest()
    if expected is not None and sha!=expected: raise ValueError('historical audit hash differs')
    MANIFEST[str(p)]={'bytes':len(raw),'sha256':sha}
    return raw

def obj(p, expected=None):
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d: raise ValueError('duplicate key')
            d[k]=v
        return d
    return json.loads(read(p,expected),object_pairs_hook=pairs,
                      parse_constant=lambda t: 'NONFINITE:'+t)

def array(p):
    x=np.load(io.BytesIO(read(p)),allow_pickle=False)
    if isinstance(x,np.ndarray):
        if not np.isfinite(x).all(): raise ValueError('nonfinite saved array')
        return x
    d={k:x[k] for k in x.files};x.close()
    if any(np.issubdtype(v.dtype,np.number) and not np.isfinite(v).all() for v in d.values()):
        raise ValueError('nonfinite saved archive')
    return d

def count(v,a,b,e,U):
    # Strict first exit, no default index if absent.
    for n in range(100000):
        if v>U:return n
        v+=a*e+b*v*v
        if not math.isfinite(v):raise ValueError('nonfinite scalar recurrence')
    return None

def complete_runs(mask):
    # Maximal true runs; censor both observation endpoints, with no phase stitching.
    out=[];i=0
    while i<len(mask):
        if not mask[i]:i+=1;continue
        start=i
        while i<len(mask) and mask[i]:i+=1
        if start>0 and i<len(mask):out.append((start,i))
    return out

def all_runs(labels, target, reference_index=None):
    """Keep every run; select the declared seed's run or a unique longest run."""
    runs=[];i=0
    while i<len(labels):
        if labels[i]!=target:i+=1;continue
        start=i
        while i<len(labels) and labels[i]==target:i+=1
        runs.append((start,i))
    if not runs:raise ValueError('period absent from saved grid')
    best=max(e-s for s,e in runs)
    winners=([r for r in runs if r[0]<=reference_index<r[1]] if reference_index is not None
             else [r for r in runs if r[1]-r[0]==best])
    if len(winners)!=1:raise ValueError('ambiguous widest saved window')
    return runs,winners[0]

def main():
    global np
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(k)!='1':raise ValueError('one-thread environment required')
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('offline CPU environment required')
    import numpy as np
    tic=time.monotonic()
    nf=obj(SOFT/'normal-form-p2-t-0.115/normal-form.json')
    a,b=abs(nf['a']),abs(nf['b']);ab=a*b;ac=nf['source']['alpha']
    vv=array(SOFT/'normal-form-p2-t-0.115/vectors.npz')
    normal={'record':nf,'critical_error':abs(complex(*nf['multiplier'])-1),
            'norm_r':float(np.linalg.norm(vv['r'])),'left_right':float(vv['l']@vv['r']),
            'ab_recomputed':ab,'full_prefactor':math.pi**2/ab,'half_prefactor':math.pi**2/(4*ab)}
    br=obj(SOFT/'branch-p2-t-0.115-alpha-up/branch.json')
    states=array(SOFT/'branch-p2-t-0.115-alpha-up/states.npy')
    if len(states)!=len(br['points']):raise ValueError('branch row mismatch')
    projection=(states.reshape(len(states),-1)-vv['x_c'].ravel())@vv['l'].ravel()/float(vv['l']@vv['r'])
    branch={'args':br['args'],'rows':[{'index':p['index'],'alpha':p['lambda'],
      'residual':p['residual'],'newton':p['newton'],'turned':p['turned_since_previous'],
      'multipliers_converged':p['multipliers_converged'],'leading':p['multipliers'][0],
      'u':float(projection[i]),'saved_residual_pass':p['residual']<=1e-10}
      for i,p in enumerate(br['points'])],
      'limit':'Projection and saved gates only; original mixed-precision spectra not re-evaluated.'}
    rec=obj(SOFT/'passage-p2-t-0.115/passage.json');u=array(SOFT/'passage-p2-t-0.115/u.npy')
    if u.shape[1]!=len(rec['rows']):raise ValueError('controlled shape mismatch')
    controlled=[]
    for i,r in enumerate(rec['rows']):
        crossings=np.flatnonzero(u[:,i]>r['u_out'])
        n=int(crossings[0])+1 if len(crossings) else None
        pred=count(r['u_in'],a,b,r['eps'],r['u_out'])
        controlled.append({**r,'recomputed_measured_G':n,'recomputed_discrete_G':pred,
          'saved_matches':n==r['measured_G'] and pred==r['predicted_discrete_G'],
          'within_one':n is not None and pred is not None and abs(n-pred)<=1})
    nat=obj(SOFT/'passage-natural-p2-t-0.115/passage.json')
    un=array(SOFT/'passage-natural-p2-t-0.115/u.npy')
    tr=array(SOFT/'passage-natural-p2-t-0.115/transverse.npy')
    if un.shape!=tr.shape or un.shape[1]!=len(nat['rows']):raise ValueError('natural shape mismatch')
    U=nat['U'];natural={}
    definitions={'article_figure_U_only':(U,None),'original_3U_transverse005':(3*U,.05)}
    for name,(bound,transbound) in definitions.items():
        groups={}
        for col,r in enumerate(nat['rows']):
            e=r['eps'];mask=np.abs(un[:,col])<bound
            if transbound is not None:mask &= tr[:,col]<transbound
            rows=groups.setdefault(str(e),{'eps':e,'trajectories':0,'phases':[]})
            rows['trajectories']+=1
            for start,end in complete_runs(mask):
                pred=count(float(un[start,col]),a,b,e,bound)
                n=end-start
                rows['phases'].append({'trajectory':r['entrance'],'start':start,'end':end,
                  'observed':n,'u_entry':float(un[start,col]),'predicted':pred,
                  'relative_error_prediction_denominator':None if pred is None else (n-pred)/pred,
                  'relative_error_observed_denominator':None if pred is None else (n-pred)/n,
                  'exit_by_u':bool(abs(un[end,col])>=bound),
                  'exit_by_transverse':bool(transbound is not None and tr[end,col]>=transbound)})
        for rows in groups.values():
            phases=rows['phases']; n=[p['observed'] for p in phases]
            errors=[p['relative_error_prediction_denominator'] for p in phases if p['predicted'] is not None]
            full=count(-bound,a,b,rows['eps'],bound)
            rows.update({'phase_count':len(n),'full_discrete_passage':full,
               'observed_range':[min(n),max(n)] if n else None,
               'mean_observed':float(np.mean(n)) if n else None,
               'mean_fraction':float(np.mean(n))/full if n and full else None,
               'median_relative_error':float(np.median(errors)) if errors else None,
               'max_absolute_relative_error':max(map(abs,errors)) if errors else None,
               'prediction_missing':len(phases)-len(errors)})
        natural[name]={'bound':bound,'transverse_bound':transbound,'groups':groups}
    # Figure panel d selects natural-3 at eps1e-5, first600 displayed; record complete phases.
    ci=next(i for i,r in enumerate(nat['rows']) if r['eps']==1e-5 and r['entrance']=='natural-3')
    shown=[(s,e,e-s) for s,e in complete_runs(np.abs(un[:600,ci])<U)]
    # Author's center coordinates are also saved in the original orbit-verification records.
    orb=obj(SOFT/'table1-cut-t-0.115/orbits.json')
    centers=[]
    for o in orb['orbits']:
        p=o['period'];N=(p-6)//2
        centers.append({'period':p,'N':N,'alpha':o['alpha'],'d':o['alpha']-ac,
          'product':(o['alpha']-ac)*N*N,'converged':o['converged'],
          'detected_period':o['detected_period'],'residuals':o['newton_residuals'],
          'divisor_returns':o['divisor_return_distances'],'multipliers':o['multipliers']})
    selected=[r for r in centers if 4<=r['N']<=7]
    ns=np.array([r['N'] for r in selected],dtype=float);ds=np.array([r['d'] for r in selected])
    windows={'rows':centers,'selected_N4to7':selected,
      'inverse_sqrt_fit':np.polyfit(ns,1/np.sqrt(ds),1).tolist(),
      'loglog_fit':np.polyfit(np.log(ns),np.log(ds),1).tolist(),
      'fractions':(np.sqrt(ds*ns**2*ab)/math.pi).tolist(),
      'full_slope':math.sqrt(ab)/math.pi,
      'limit':'Center-coordinate products independently reduced, but original float32 cut midpoint extraction not repeated.'}
    # Reuse completed independent audits unchanged; do not read/replay their full trajectories.
    old={}
    for name,path,sha in [
      ('task5',DATA/'codex/task5-fresh-prediction/passage-audit-v1.json','76217ba867aca5e43ef801dce45263a200a5c0854c2e3dd2b84f0d4e9d12eab6'),
      ('task5b',DATA/'codex/task5b-fresh-prediction/audit-01/audit.json','e906239572e15002e9811b217b1c87aa2e0018755a6ed29741666bb7a6a15e7d')]:
        old[name]=obj(path,sha)
    # Article TableV enumerates every measurement; compare all48, preserving seven primary failures.
    expected={'A':[[118,59,117],[56,28,56],[26,13,26],[10,5,10]],
              'B':[[26,13,25],[10,5,10],[4,2,4],[2,1,2]],
              'C':[[23,12,23],[45,23,45],[86,43,86],[155,78,155]],
              'D':[[22,11,22],[42,22,42],[81,43,82],[149,78,150]]}
    prospective=[]
    for study in old.values():
        for t in study['targets']:
            lab=t['label']
            if len(t['rows'])!=12:raise ValueError('prospective missing rows')
            for j,r in enumerate(t['rows']):
                expected_n=expected[lab][j//3][j%3]
                prospective.append({'target':lab,**r,'article_measured':expected_n,
                                    'article_matches':r['measured_G']==expected_n})
    reused={p:obj(ROOT/p) for p in ['codex/crosscheck/audits/fold-001.json',
      'codex/crosscheck/audits/branch-002.json','codex/confirm-fold/results-receipt.json',
      'codex/confirm-fold-2/results-receipt.json']}
    wp=obj(ROOT/'docs/wp-window-predictions.json');cells=obj(ROOT/'docs/wp-line-cells.json')
    # Pre-run amendment: original finite grids and original three extrapolated fold edges.
    cutmeta=obj(SOFT/'cut-t-0.115/scan.json');cut=array(SOFT/'cut-t-0.115/scan.npz')
    ca=cut['alpha'].ravel();cp=cut['period'].ravel()
    if len(ca)!=len(cp):raise ValueError('cut axes mismatch')
    cutcenters=[]
    for p in range(6,21,2):
        table_alpha=next(x['alpha'] for x in centers if x['period']==p)
        runs,(s,e)=all_runs(cp,p,int(np.argmin(abs(ca-table_alpha))))
        center=float((ca[s]+ca[e-1])/2)
        cutcenters.append({'period':p,'runs':runs,'selected':[s,e],'count':e-s,
          'alpha_first':float(ca[s]),'alpha_last':float(ca[e-1]),'center':center,
          'center_table':table_alpha,'selection':'contiguous run containing saved table coordinate'})
    rawwp=obj(SOFT/'pod64/hidden/wp-line/hidden-chaos.json')
    wc=rawwp['cells']
    if len(wc)!=len(cells['cells']):raise ValueError('WP input row count')
    for r,c in zip(wc,cells['cells']):
        if [r['log10_temperature'],r['alpha']]!=c:raise ValueError('WP input coordinate mismatch')
        if not all(math.isfinite(v) for v in r['lyapunov']):raise ValueError('WP nonfinite exponent')
    wa=np.array([r['alpha'] for r in wc]);wpers=[r['period'] for r in wc]
    if not np.all(np.diff(wa)>0):raise ValueError('WP unordered cells')
    gridrows=[]
    for p in range(22,33,2):
        runs,(s,e)=all_runs(wpers,p);d=float(wa[s]-wp['alpha_c'])
        pred=next((r for r in wp['predictions'] if r['period']==p),None)
        gridrows.append({'period':p,'N':(p-6)//2,'runs':runs,'selected':[s,e],
          'first_alpha':float(wa[s]),'last_alpha':float(wa[e-1]),'d':d,
          'label_before':wpers[s-1] if s else None,
          'edge_bracket':[float(wa[s-1]),float(wa[s])] if s else None,
          'label_after':wpers[e] if e<len(wc) else None,'prediction':pred,
          'law_relative_error_prediction_denominator':(d-pred['eps_law'])/pred['eps_law'] if pred else None,
          'inside_R_band':min(pred['eps_model'].values())<=d<=max(pred['eps_model'].values()) if pred else None,
          'full_state_closure_replayed':False})
    edges=obj(SOFT/'pod64/edges/stripes-t-0.115/edges.json');foldedges=[]
    for w in edges['windows']:
        if w['period'] not in (16,18,20):continue
        folds=[x for x in w['edges'] if x['kind']=='fold']
        if len(folds)!=1:raise ValueError('missing/ambiguous fold edge')
        edge=folds[0];xy=np.array(edge['fit_points'])
        coeff=np.polyfit(xy[:,0],(1-xy[:,1])**2,1)
        foldedges.append({'period':w['period'],'N':(w['period']-6)//2,**edge,
          'linear_root_recomputed':float(-coeff[1]/coeff[0]),
          'limit':'Extrapolated from four saved stable multipliers; no solved unit-circle crossing record.'})
    en=np.array([r['N'] for r in foldedges],float)
    ed=np.array([r['lambda']-wp['alpha_c'] for r in foldedges])
    slope,intercept=np.polyfit(en,1/np.sqrt(ed),1)
    combined=[{'N':r['N'],'alpha':r['lambda']} for r in foldedges]+[
        {'N':r['N'],'alpha':r['first_alpha']} for r in gridrows]
    cn=np.array([r['N'] for r in combined],float)
    cd=np.array([r['alpha']-.0627983 for r in combined])
    wp_result={'args':rawwp['args'],'all_input_coordinates_match':True,'raw_grid_cells':len(wc),
       'grid_rows':gridrows,'fold_edges':foldedges,'edge_slope':float(slope),
       'edge_intercept':float(intercept),'edge_C':float(1/slope**2),'edge_N0':float(intercept/slope),
       'edge_effective_fraction':float(math.sqrt(ab)/math.pi/slope),
       'edge_slope_full_ratio':float(slope/(math.sqrt(ab)/math.pi)),
       'combined_edges':combined,'combined_alpha_c_convention':.0627983,
       'combined_loglog_fit':np.polyfit(np.log(cn),np.log(cd),1).tolist(),
       'combined_width_over_U':(np.sqrt(a*cd/b)/U).tolist(),
       'limit':'Saved finite-grid labels and extrapolated edge estimates; no formal edge error bounds, new full-state solves, or free-exponent nonlinear refit.'}
    result={'scope':'Fresh saved small reductions plus reuse of closed original audits, not model/old-audit replay',
      'article_commit':'bb42a82189da91c37f50804c077832df9e2237ae','normal_form':normal,
      'branch_projection':branch,'controlled':controlled,'natural':natural,'figure_d_complete_runs':shown,
      'window_centers':windows,'prospective_48':prospective,'reused_closed_audits':old,
      'reused_receipts':reused,'wp_prediction_metadata':wp,'wp_line_input_metadata':{'keys':list(cells)},
      'wp_reduction':wp_result,'original_float32_cut':{'args':cutmeta['args'],'centers':cutcenters},
      'manifest':MANIFEST,'elapsed_seconds':time.monotonic()-tic,'model_calls':0,'old_audit_reruns':0}
    with OUT.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'output':str(OUT),'inputs':len(MANIFEST),'elapsed':result['elapsed_seconds']}))

if __name__=='__main__':main()
