"""One fixed-pool development report: no policy, assessment or follow-on launch."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.tiny_lm_prediction import FAMILIES
from neural_shrimp.tiny_lm_progress import (CORPUS_SHA256,CORPUS_BYTES,TRAIN_STARTS,PROBE_STARTS,ORDINARY,TARGET_SEEDS,baseline_series,
    canonical_hash,digest,fit_scale,progress_targets,ridge_predict,score,select_progress_model,
    split_indices,token_hash,verify_inventory)
from explore_tiny_lm_prediction import save_json
from explore_tiny_lm_broader_data import committed_identity

ANALYSIS_SOURCES=['tools/analyze_tiny_lm_broader_data.py','src/neural_shrimp/tiny_lm_progress.py',
                  'src/neural_shrimp/tiny_lm_prediction.py','tools/explore_tiny_lm_prediction.py',
                  'tools/explore_tiny_lm.py','tools/explore_tiny_lm_broader_data.py',
                  'src/neural_shrimp/tiny_lm.py','docs/plans/explore-tiny-lm-broader-data.md']


def verify_data(folder):
    d=json.loads((folder/'data.json').read_text())
    expected=canonical_hash({k:d[k] for k in ('corpus_sha256','corpus_bytes','vocabulary','roles','target_rule')})
    if d['corpus_sha256']!=CORPUS_SHA256 or d['corpus_bytes']!=CORPUS_BYTES or d['data_identity_sha256']!=expected:
        raise ValueError('dataset identity mismatch')
    with np.load(folder/'data-tokens.npz') as arrays:
        for role,starts in [('control',TRAIN_STARTS[:8]),('expanded',TRAIN_STARTS),('probe',PROBE_STARTS)]:
            if d['roles'][role]['starts']!=list(starts) or d['roles'][role]['byte_character_intervals']!=[[v,v+33] for v in starts]:
                raise ValueError('declared data window mismatch')
            rows=np.asarray(d['roles'][role]['window_token_ids'],dtype='<i8')
            if token_hash(rows)!=d['roles'][role]['window_tokens_sha256']:
                raise ValueError('token identity mismatch')
            np.testing.assert_equal(rows[:,:-1],arrays[role+'_X'])
            np.testing.assert_equal(rows[:,1:],arrays[role+'_Y'])
        np.testing.assert_equal(arrays['expanded_X'][:8],arrays['control_X'])
        np.testing.assert_equal(arrays['expanded_Y'][:8],arrays['control_Y'])
    return d


def record_scores(r,p,status):
    try:normalization=fit_scale(r)
    except ValueError:normalization=None
    out={}
    for split,count in (('fit',352),('tune',184),('test',184)):
        ix=split_indices(len(r['y']),split)
        if normalization is None:
            value={'complete':False,'rows':len(ix),'expected_rows':count,'normalized_mse':None,
                   'mse':None,'mae':None,'signed_bias':None,'failure':'invalid_fit_normalization'}
        else:value=score(r['y'][ix],p[ix],normalization,count)
        value.update(steps=ix.tolist(),prediction=p[ix].tolist(),status=status[ix].tolist())
        bad=[int(t) for t in ix if status[t]!='ok']
        value['failed_forecast_steps']=bad
        if bad:value['complete']=False
        out[split]=value
    return out


def probe_value(r,step):
    matches=[p for p in r.get('probe',[]) if p['step']==step]
    if len(matches)!=1 or matches[0]['status']!='ok':return None
    value=matches[0]['ce']
    return float(value) if value is not None and np.isfinite(value) else None


def analyze(records):
    if sorted((r['seed'],r['arm']) for r in records)!=[(s,a) for s in TARGET_SEEDS for a in (8,256)]:
        raise ValueError('exactly four declared seed/arm cases required')
    if any(r['role']!='collect' for r in records):raise ValueError('profile/assessment data rejected')
    if len({r['data_identity_sha256'] for r in records})!=1:raise ValueError('unmatched data identities')
    for r in records:
        np.testing.assert_equal(r['y'],progress_targets(r['ce']))
        if r['X'].ndim!=2 or len(r['X'])!=len(r['ce']) or r['X'].shape[1]!=len(r['groups']):
            raise ValueError('feature/target shape mismatch')
        if r['groups']!=records[0]['groups']:raise ValueError('feature-family mismatch')
    report={'seeds':{},'learning':{},'scope':'Exploratory fixed text pool; two initialization seeds only. Time-held forecasts, no independent-window uncertainty or data-pool generalization.'}
    for seed in TARGET_SEEDS:
        pair=sorted([r for r in records if r['seed']==seed],key=lambda r:r['arm'])
        if pair[0]['eta']!=pair[1]['eta'] or pair[0]['initial_state_sha256']!=pair[1]['initial_state_sha256']:
            raise ValueError('arms do not share exact rate and initialization')
        family_output={};models={};all_valid=all(r['outcome']=='completed' and r['completed_updates']==800 and
            r.get('final_train_ce') is not None and np.isfinite(r['final_train_ce']) for r in pair)
        for family in ('persistence','AR1','AR2',*FAMILIES):
            selected=None
            if family in FAMILIES:
                selected=select_progress_model(pair,family);models[family]=selected
                all_valid=all_valid and selected['all_candidates_valid']
            arms={}
            for r in pair:
                if family=='persistence':p=np.zeros(len(r['ce']));status=np.repeat('ok',len(p))
                elif family.startswith('AR'):p,status=baseline_series(r['ce'],int(family[-1]))
                else:
                    p=np.full(len(r['ce']),np.nan);status=np.repeat('model_unavailable',len(p))
                    if selected['selected'] is not None:
                        # Row-local evaluation retains valid forecasts when another
                        # row fails, without dropping that failed comparator row.
                        for split in ('fit','tune','test'):
                            for t in split_indices(len(r['y']),split):
                                try:
                                    p[t]=ridge_predict(selected['selected']['model'],r['X'][t:t+1,selected['columns']])[0]
                                    status[t]='ok' if np.isfinite(p[t]) else 'nonfinite_value'
                                except (ValueError,FloatingPointError) as error:
                                    status[t]='numerical_failure'
                                    models[family].setdefault('prediction_failures',[]).append({'arm':r['arm'],'step':int(t),'error':repr(error)})
                arms[str(r['arm'])]=record_scores(r,p,status)
                all_valid=all_valid and all(v['complete'] for v in arms[str(r['arm'])].values())
            family_output[family]=arms
        ordinary_scores=[]
        for family in ORDINARY:
            values=[family_output[family][str(arm)]['tune'] for arm in (8,256)]
            ordinary_scores.append({'family':family,'valid':all(v['complete'] for v in values),
                'mean_tune_normalized_mse':float(np.mean([v['normalized_mse'] for v in values])) if all(v['complete'] for v in values) else None})
        ordinary=None
        if all(v['valid'] for v in ordinary_scores):
            ordinary=min(ordinary_scores,key=lambda v:(v['mean_tune_normalized_mse'],ORDINARY.index(v['family'])))['family']
        b2=family_output['B2']['256']['test'];b3=family_output['B3']['256']['test']
        ordinary_test=family_output[ordinary]['256']['test'] if ordinary else None
        enough=(ordinary_test is not None and all(v['complete'] for v in (b2,b3,ordinary_test)) and
                b2['normalized_mse']>=1e-6 and ordinary_test['normalized_mse']>=1e-6)
        positive=(all_valid and enough and all(b3['normalized_mse']<=.95*v['normalized_mse'] and
                  b3['mae']<=v['mae'] for v in (b2,ordinary_test)))
        report['seeds'][str(seed)]={'models':models,'scores':family_output,'ordinary_candidates':ordinary_scores,
            'selected_ordinary':ordinary,'all_comparators_and_eligible_rows_valid':bool(all_valid),
            'comparison_above_error_floor':bool(enough),'positive_forecast_interpretation':bool(positive)}
        small,large=pair
        s800,l800,l25=probe_value(small,800),probe_value(large,800),probe_value(large,25)
        report['learning'][str(seed)]={'probe_CE_8_at800':s800,'probe_CE_256_at800':l800,'probe_CE_256_at25':l25,
            'equal_updates_difference':None if s800 is None or l800 is None else l800-s800,
            'equal_training_tokens_difference':None if s800 is None or l25 is None else l25-s800,
            'final_train_CE_8':small.get('final_train_ce'),'final_train_CE_256':large.get('final_train_ce'),
            'train_probe_gap_8':None if s800 is None or small.get('final_train_ce') is None else s800-small['final_train_ce'],
            'train_probe_gap_256':None if l800 is None or large.get('final_train_ce') is None else l800-large['final_train_ce'],
            'gradient_calls':{str(r['arm']):r['completed_updates'] for r in pair},
            'processed_training_targets':{str(r['arm']):r['completed_updates']*r['arm']*32 for r in pair},
            'elapsed_seconds':{str(r['arm']):r.get('elapsed_seconds') for r in pair},
            'all_scheduled_probe_observations':{str(r['arm']):r.get('probe',[]) for r in pair},
            'scope':'Paired initialization, one nested text pool; differing token/runtime cost at equal updates.'}
    report['learning_aggregate']={}
    for name in ('equal_updates_difference','equal_training_tokens_difference'):
        values=[r[name] for r in report['learning'].values()]
        report['learning_aggregate'][name]={'seed_values':values,
            'mean':float(np.mean(values)) if all(v is not None for v in values) else None,
            'range':[min(values),max(values)] if all(v is not None for v in values) else None,
            'scope':'Two initialization seeds on one fixed pool, descriptive only; no window-level uncertainty.'}
    report['positive_forecast_interpretation']=all(v['positive_forecast_interpretation'] for v in report['seeds'].values())
    report['next_action']='Stop after this report regardless of outcome; no policy/assessment/inference stage authorized.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('collection',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.collection.resolve()):raise ValueError('output must be outside immutable inputs')
    source=committed_identity(ANALYSIS_SOURCES)
    manifest_path=args.collection/'manifest.json';manifest=json.loads(manifest_path.read_text())
    verify_inventory(args.collection,manifest);data=verify_data(args.collection)
    records=[]
    for path in sorted(args.collection.glob('seed-*/result.json')):
        r=json.loads(path.read_text())
        if r['data_identity_sha256']!=data['data_identity_sha256']:raise ValueError('case/data identity mismatch')
        with np.load(path.with_name('features.npz')) as z:
            records.append({**r,'X':z['X'],'ce':z['ce'],'y':z['progress'],'groups':r['feature_groups']})
    try:result=analyze(records)
    except Exception as error:result={'status':'failed','error':repr(error),'positive_forecast_interpretation':False}
    result.update(analysis_identity={**source,'numpy_version':np.__version__},
                  input_manifest_sha256=digest(manifest_path.read_bytes()),data_identity_sha256=data['data_identity_sha256'])
    if args.output.exists():raise FileExistsError(args.output)
    save_json(args.output,result)


if __name__=='__main__':main()
