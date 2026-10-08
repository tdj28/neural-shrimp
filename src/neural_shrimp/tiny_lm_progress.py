"""Fixed development-data identity and causal continuous-progress comparisons."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .tiny_lm_prediction import FAMILIES, PENALTIES, family_columns

CORPUS_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
CORPUS_BYTES = 1115394
TRAIN_STARTS = tuple(65536+256*j for j in range(256))
PROBE_STARTS = tuple(196608+256*j for j in range(32))
TARGET_SEEDS = (28161, 28162)
PROFILE_SEED = 28160
DIRECTION_SEED = 28165
HORIZON = 16
SPLITS = {'fit': (32,383), 'tune': (400,583), 'test': (600,783)}
ORDINARY = ('persistence', 'AR1', 'AR2', 'B0', 'B1')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_hash(value):
    return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def token_hash(array):
    return digest(np.asarray(array,dtype='<i8').tobytes(order='C'))


def validate_intervals():
    selected=[(s,s+33) for s in TRAIN_STARTS+PROBE_STARTS]
    old=[(4096*j,4096*j+33) for j in range(8)]
    old += [(262144+4096*j,262144+4096*j+33) for j in range(8)]
    old += [(base+4096*j,base+4096*j+33) for base in (524288,655360) for j in range(16)]
    old += [(1003854,CORPUS_BYTES)]
    for i,(a,b) in enumerate(selected):
        if any(not (b<=c or d<=a) for c,d in selected[i+1:]+old):
            raise ValueError('selected window overlaps another role or protected interval')
    return selected


def select_windows(raw, vocabulary, starts):
    """Select only stated byte intervals; ASCII makes character offsets identical."""
    mapping={c:i for i,c in enumerate(vocabulary)}
    windows=[raw[s:s+33] for s in starts]
    if any(len(w)!=33 for w in windows):
        raise ValueError('incomplete selected byte window')
    rows=np.asarray([[mapping[c] for c in w.decode('ascii')] for w in windows],dtype='<i8')
    identity={'starts':list(starts),'byte_character_intervals':[[s,s+33] for s in starts],
              'input_intervals':[[s,s+32] for s in starts],
              'target_intervals':[[s+1,s+33] for s in starts],
              'selected_bytes_sha256':digest(b''.join(windows)),
              'window_tokens_sha256':token_hash(rows),'window_token_ids':rows.tolist(),
              'shape':list(rows.shape),'targets_per_evaluation':len(starts)*32}
    return rows[:,:-1].copy(),rows[:,1:].copy(),identity


def load_fixed_data(folder):
    folder=Path(folder);raw=(folder/'input.txt').read_bytes()
    source=json.loads((folder/'receipt.json').read_text())
    if digest(raw)!=CORPUS_SHA256 or len(raw)!=CORPUS_BYTES or source['corpus_sha256']!=CORPUS_SHA256:
        raise ValueError('fixed corpus identity mismatch')
    validate_intervals()
    prefix=raw[:1003854].decode('ascii')
    vocabulary=sorted(set(prefix))+['<UNK>']
    if len(vocabulary)!=66:raise ValueError('changed vocabulary')
    arrays={};roles={}
    for name,starts in [('expanded',TRAIN_STARTS),('control',TRAIN_STARTS[:8]),('probe',PROBE_STARTS)]:
        x,y,identity=select_windows(raw,vocabulary,starts)
        arrays[name]=(x,y);roles[name]=identity
    if not all(np.array_equal(a[:8],b) for a,b in zip(arrays['expanded'],arrays['control'])):
        raise ValueError('control is not the exact expanded-data prefix')
    data={'source':source,'corpus_sha256':CORPUS_SHA256,'corpus_bytes':len(raw),
          'vocabulary':vocabulary,'vocabulary_sha256':canonical_hash(vocabulary),
          'roles':roles,'target_rule':'X=chars[s:s+32]; Y=chars[s+1:s+33], next character at every position',
          'scope':'One fixed nested development text pool; seed replication is initialization only.',
          'sealed':'Old assessment/validation and final-10-percent outcomes unused.'}
    data['data_identity_sha256']=canonical_hash({k:data[k] for k in ('corpus_sha256','corpus_bytes','vocabulary','roles','target_rule')})
    return arrays,data


def progress_targets(ce):
    ce=np.asarray(ce,float);y=np.full(len(ce),np.nan)
    for t in range(max(0,len(ce)-HORIZON)):
        if np.isfinite(ce[t:t+HORIZON+1]).all():y[t]=ce[t+HORIZON]-ce[t]
    return y


def split_indices(length,split):
    lo,hi=SPLITS[split]
    return np.arange(lo,min(length,hi+1),dtype=int)


def ar_progress(history,order):
    """Strict full-rank 32-observation OLS, then 16 recursive steps; no fallback."""
    h=np.asarray(history,float)[-32:]
    if len(h)!=32:return np.nan,'insufficient_history'
    if not np.isfinite(h).all():return np.nan,'nonfinite_history'
    design=np.column_stack([np.ones(32-order)]+[h[order-j-1:31-j] for j in range(order)])
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            beta,_,rank,_=np.linalg.lstsq(design,h[order:],rcond=None)
            if rank!=order+1:return np.nan,'rank_deficient'
            if not np.isfinite(beta).all():return np.nan,'nonfinite_fit'
            future=list(h)
            for _ in range(HORIZON):
                value=float(beta@np.r_[1.,future[-order:][::-1]])
                if not np.isfinite(value):return np.nan,'nonfinite_recursion'
                future.append(value)
            return float(future[-1]-h[-1]),'ok'
    except (FloatingPointError,OverflowError,np.linalg.LinAlgError):
        return np.nan,'numerical_failure'


def baseline_series(ce,order):
    pairs=[ar_progress(ce[:t+1],order) for t in range(len(ce))]
    return np.asarray([p[0] for p in pairs]),np.asarray([p[1] for p in pairs])


def fit_scale(record):
    y=record['y'][split_indices(len(record['y']),'fit')]
    if len(y)!=352 or not np.isfinite(y).all():raise ValueError('incomplete fitting labels')
    return max(.01,float(np.sqrt(np.mean(y*y))))


def ridge_fit(X,y,penalty):
    X,y=np.asarray(X,float),np.asarray(y,float)
    if not len(X) or not np.isfinite(y).all():raise ValueError('nonempty finite fit labels required')
    if np.isinf(X).any():raise ValueError('infinite feature; structural NaN alone is imputable')
    count=np.isfinite(X).sum(0)
    mean=np.divide(np.where(np.isfinite(X),X,0.).sum(0),count,out=np.zeros(X.shape[1]),where=count>0)
    filled=np.where(np.isfinite(X),X,mean);scale=np.maximum(filled.std(0),1e-8)
    Z=np.c_[np.ones(len(X)),(filled-mean)/scale]
    regularizer=np.diag(np.r_[0.,np.repeat(penalty,X.shape[1])])
    weights=np.linalg.solve(Z.T@Z/len(X)+regularizer,Z.T@y/len(X))
    if not np.isfinite(weights).all():raise ValueError('nonfinite ridge coefficients')
    return {'mean':mean.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),
            'penalty':float(penalty),'training_rows':len(y)}


def ridge_predict(model,X):
    X=np.asarray(X,float)
    if np.isinf(X).any():raise ValueError('infinite prediction feature')
    mean=np.asarray(model['mean']);scale=np.asarray(model['scale'])
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        Z=np.c_[np.ones(len(X)),(np.where(np.isfinite(X),X,mean)-mean)/scale]
        return Z@np.asarray(model['weights'])


def score(y,p,scale,expected_rows):
    y,p=np.asarray(y,float),np.asarray(p,float)
    bad=np.flatnonzero(~(np.isfinite(y)&np.isfinite(p)))
    out={'rows':len(y),'expected_rows':expected_rows,'invalid_row_indices':bad.tolist(),
         'complete':len(y)==expected_rows and not len(bad),'normalization':float(scale)}
    if not out['complete']:
        return {**out,'normalized_mse':None,'mse':None,'mae':None,'signed_bias':None}
    try:
        with np.errstate(over='raise',invalid='raise'):
            error=p-y
            metrics={'normalized_mse':float(np.mean((error/scale)**2)),
                     'mse':float(np.mean(error*error)),'mae':float(np.mean(abs(error))),
                     'signed_bias':float(np.mean(error))}
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError()
        return {**out,**metrics}
    except FloatingPointError:
        return {**out,'complete':False,'failure':'nonfinite_score','normalized_mse':None,'mse':None,'mae':None,'signed_bias':None}


def select_progress_model(records,family):
    columns=family_columns(records[0]['groups'],family)
    X=np.concatenate([r['X'][split_indices(len(r['y']),'fit')][:,columns] for r in records])
    y=np.concatenate([r['y'][split_indices(len(r['y']),'fit')] for r in records])
    candidates=[]
    for penalty in PENALTIES:
        candidate={'penalty':penalty}
        try:
            model=ridge_fit(X,y,penalty);scores=[]
            for r in records:
                ix=split_indices(len(r['y']),'tune')
                scores.append(score(r['y'][ix],ridge_predict(model,r['X'][ix][:,columns]),fit_scale(r),184))
            if not all(s['complete'] for s in scores):raise ValueError('incomplete or invalid tuning forecasts')
            candidate.update(model=model,tune_scores=scores,mean_tune_normalized_mse=float(np.mean([s['normalized_mse'] for s in scores])),status='ok')
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            candidate.update(status='failed',error=repr(error))
        candidates.append(candidate)
    good=[c for c in candidates if c['status']=='ok']
    selected=min(good,key=lambda c:(c['mean_tune_normalized_mse'],-c['penalty'])) if good else None
    return {'columns':columns.tolist(),'candidates':candidates,'selected':selected,
            'all_candidates_valid':len(good)==len(candidates)}


def verify_inventory(folder,manifest):
    folder=Path(folder);names=[v['path'] for v in manifest['files']]
    if len(names)!=len(set(names)):raise ValueError('duplicate manifest path')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'manifest.json'}
    if actual!=set(names):raise ValueError('manifest must match exact input file inventory')
    for v in manifest['files']:
        name=Path(v['path']);p=folder/name
        if name.is_absolute() or '..' in name.parts or not p.resolve().is_relative_to(folder.resolve()) or p.is_symlink():
            raise ValueError('unsafe manifest path')
        if p.stat().st_size!=v['bytes'] or digest(p.read_bytes())!=v['sha256']:raise ValueError('manifest digest/size mismatch')
