"""Exact all44 immutable input authority; no model or producer imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from definitions import ALL,L6,L10,TRAIN,TEST,SECOND,SECOND_REVISION,FIRST,FIRST_REVISION,replicas,filenames

RAW=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/msm')
HERE=Path(__file__).resolve().parent
PRODUCER_MANIFEST_SHA='c90331389be2b2a6496c1b02925d99766252d945ea5a7b1c3698a5547af044ad'
PRODUCER_SOURCE='feac2a796e4aa673e2d7f44654a9dc06b2070416'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(1024*1024): h.update(b)
    return h.hexdigest()


def authority(manifest):
    """Run before array loading, and again before success; no fallback discovery."""
    if manifest.get('status')!='complete_immutable' or not manifest.get('writer_closed_reference'):
        raise ValueError('all44 writer-complete immutable authority is required')
    if manifest.get('arms')!=list(ALL): raise ValueError('literal all44 arm order differs')
    expected=[str(RAW/a/f) for a in ALL for f in filenames(a)]
    if [r['path'] for r in manifest.get('files',[])]!=expected:
        raise ValueError('missing, duplicate or changed input identity')
    if set(manifest.get('sample_args',{}))!=set(ALL): raise ValueError('all44 sample args required')
    validate_producer(manifest)
    for row in manifest['files']:
        p=Path(row['path'])
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:
            raise ValueError('input changed: '+str(p))
    return manifest


def validate_producer(manifest):
    p=HERE/'producer-score-manifest.json'
    if digest(p)!=PRODUCER_MANIFEST_SHA:raise ValueError('producer closure manifest changed')
    owner=json.loads(p.read_text())
    if manifest.get('producer_manifest_sha256')!=PRODUCER_MANIFEST_SHA or manifest.get('producer_source_commit')!=PRODUCER_SOURCE:
        raise ValueError('producer authority not pinned')
    if owner['classifier']!=SECOND or owner['revision']!=SECOND_REVISION or not owner['writer_status'].startswith('completed: all 44 runs scored, exit status 0'):
        raise ValueError('producer classifier or completion differs')
    expected={str(Path('/data2/neural-shrimp-data')/r['path']):(r['bytes'],r['sha256']) for r in owner['files']}
    actual={r['path']:(r['bytes'],r['sha256']) for r in manifest['files'] if Path(r['path']).name=='hostility-second.npz'}
    if len(owner['files'])!=44 or len(expected)!=44 or actual!=expected:raise ValueError('second scores differ from closed producer44')


def validate_scores(first,labels,second,model,revision,n):
    if list(labels)!=['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
        raise ValueError('first classifier column semantics')
    if model!=SECOND or revision!=SECOND_REVISION: raise ValueError('second classifier identity')
    if first.shape!=(n,40,6) or second.shape!=(n,40): raise ValueError('score shape')
    for a in (first,second):
        if a.dtype!=np.float32 or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
            raise ValueError('scores must be finite stored float32 probabilities')


def load_arm(name,manifest,features=False):
    n=replicas(name);p=RAW/name
    meta=json.loads((p/'sample.json').read_text());args=meta['args']
    if args!=manifest['sample_args'][name] or args['replicas']!=n or args['turns']!=40 or args['model']!='qwen0.5b':
        raise ValueError('sample metadata differs')
    if len(meta['texts'])!=n or any(len(x)!=40 for x in meta['texts']): raise ValueError('text identity shape')
    old=json.loads((p/'hostility.json').read_text())
    if old['classifier']!=FIRST or old['revision']!=FIRST_REVISION: raise ValueError('first classifier identity')
    with np.load(p/'hostility.npz',allow_pickle=False) as z:
        first=z['classifier'];labels=z['labels'].tolist()
    with np.load(p/'hostility-second.npz',allow_pickle=False) as z:
        if set(z.files)!={'toxic','classifier','revision'}: raise ValueError('second archive schema')
        second=z['toxic'];model=z['classifier'].item();revision=z['revision'].item()
    validate_scores(first,labels,second,model,revision,n)
    out={'name':name,'first':first[:,:,0].astype(float),'second':second.astype(float),'args':args}
    if name in L6+L10:
        with np.load(p/'turns.npz',allow_pickle=False) as z:
            out['counter']=z['counter'].astype(float)
            if name in L10:
                if 'regenerated' not in z.files: raise ValueError('L10 regeneration identity absent')
                out['regenerated']=z['regenerated']
            else: out['regenerated']=np.zeros((n,40),bool)
        out['texts']=meta['texts']
        if out['counter'].shape!=(n,40) or not np.isfinite(out['counter']).all(): raise ValueError('counter schema')
        if out['regenerated'].shape!=(n,40) or not np.isin(out['regenerated'],[0,1]).all(): raise ValueError('regeneration schema')
    if features:
        if name not in TRAIN+TEST: raise ValueError('features only in frozen detection arms')
        out['dose']=.04 if name.endswith('040') else .045
        if args['alphas']!=[out['dose']]: raise ValueError('dose differs')
        with np.load(p/'in-context.npz',allow_pickle=False) as z:
            out['pre']=z['pre_reply_18'];unit=z['unit']
        acts=np.load(p/'activations.npy',allow_pickle=False,mmap_mode='r')
        if out['pre'].shape!=(n,40,896) or acts.shape!=(n,40,896) or unit.shape!=(896,): raise ValueError('state shape')
        out['axis']=acts@unit
        if not np.isfinite(out['pre']).all() or not np.isfinite(out['axis']).all(): raise ValueError('nonfinite state/baseline')
        out['feature_set']='baseline_state'
    return out
