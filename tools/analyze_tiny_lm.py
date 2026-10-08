"""Offline hash and numerical receipt check for the bounded text-transformer pilot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(root, relative):
    directory=root/relative
    path=directory/'manifest.json'
    manifest=json.loads(path.read_text())
    for entry in manifest['files']:
        raw=directory/entry['path']
        if raw.stat().st_size!=entry['bytes'] or sha(raw)!=entry['sha256']:
            raise ValueError(f'raw evidence changed: {raw}')
    return {'relative_path':relative,'manifest_sha256':sha(path),'manifest_bytes':path.stat().st_size,
            'source_commit':manifest['metadata']['source_commit'],'file_count':len(manifest['files'])}


def derive(root):
    corpus=bind(root,'datasets/tinyshakespeare/attempt-001')
    attempts=[bind(root,'explore/tiny-lm/'+name) for name in ('attempt-001-unit','attempt-002-rates','attempt-003-high-rates')]
    rows=[]
    for attempt in attempts:
        directory=root/attempt['relative_path']
        saved=json.loads((directory/'results.json').read_text())
        attempt['parameters']=saved['plan']['parameters']
        attempt['plan_sha256']=sha(directory/'plan.json')
        attempt['peak_cuda_allocated_bytes']=saved['peak_cuda_allocated_bytes']
        attempt['validation']=json.loads((directory/'validation.json').read_text())
        for index,case in enumerate(saved['results']):
            raw=np.load(directory/f'case-{index:02d}/trajectory.npz')
            trace,tail=raw['trace'],raw['tail_states']
            row={key:case[key] for key in ('factor','eta','completed_steps','outcome','initial_cross_entropy',
                 'final_recorded_cross_entropy','seconds_per_update')}
            row.update(attempt=Path(attempt['relative_path']).name,case=index,
                       loss_tail_std=float(np.std(trace[-128:,0])),final_gradient_norm=float(trace[-1,2]),
                       full_state_label=case.get('finite_horizon_label','stopped_before_tail'))
            if len(tail)>=32:
                row['tail_max_relative_recurrence']=case['tail_max_relative_recurrence']
                row['tail_parameter_displacement']=float(np.linalg.norm(tail[-1]-tail[0])/max(np.linalg.norm(tail[0]),1))
                row['tail_parameter_norm_relative_change']=float(np.linalg.norm(tail[-1])/np.linalg.norm(tail[0])-1)
            ar=case['loss_ar']
            if ar['outcome']=='fit':
                row['ar2_over_ar1_time_held_mse']=ar['ar2']['normalized_test_mse']/max(ar['ar1']['normalized_test_mse'],1e-300)
                row['ar2_roots']=ar['ar2']['roots']
                row['ar2_root_angles_cycles_per_update']=[float(abs(np.angle(complex(*z)))/(2*np.pi)) for z in ar['ar2']['roots']]
                row['relative_loss_trend_over_ar_window']=ar['relative_trend_over_window']
            if 'final_hessian_ritz' in case:
                row['final_max_hessian_ritz']=case['final_hessian_ritz']['maximum']
                row['final_min_hessian_ritz']=case['final_hessian_ritz']['minimum']
                row['eta_times_final_max_ritz']=case['eta']*case['final_hessian_ritz']['maximum']['value']
            rows.append(row)
    main=[row for row in rows if row['attempt']!='attempt-001-unit']
    data=json.loads((root/attempts[0]['relative_path']/'data.json').read_text())
    return {'label':'exploratory','evidence_rung':0,'generator_sha256':sha(Path(__file__)),
            'data_root':str(root),'corpus':corpus,'attempts':attempts,
            'corpus_sha256':data['source']['corpus_sha256'],'corpus_source':data['source']['source'],
            'license_note':data['source']['license_note'],'training_offsets':data['training_character_offsets'],
            'target_convention':data['target_rule'],'validation':data['validation'],
            'parameters':attempts[0]['parameters'],'tokens_per_update':data['tokens_per_update'],
            'main_case_counts':dict(Counter(row['outcome'] for row in main)),
            'candidate_counts':dict(Counter(row['full_state_label'] for row in main)),
            'rows':rows,'scope':'One seed and one fixed batch; no validation outcomes, stationary orbit, Floquet, chaos, invariant-circle, or training-utility claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/data2/neural-shrimp-data'))
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    result=derive(args.data_root)
    if args.write:
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        args.receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    elif json.loads(args.receipt.read_text())!=result:
        raise ValueError('receipt differs from retained raw evidence')
    print(json.dumps({'parameters':result['parameters'],'main_case_counts':result['main_case_counts'],
                      'candidate_counts':result['candidate_counts'],'receipt_matches_raw_evidence':True}))


if __name__=='__main__':
    main()
