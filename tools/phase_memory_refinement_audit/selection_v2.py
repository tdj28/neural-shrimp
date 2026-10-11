"""Prospective signed-complete refinement; preserves the earlier 61-case proposal."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import selection as previous


def select(summary):
    old=previous.select(summary)
    failed=[r for r in old['cases'] if r['selection_category']=='failed-large-pulse']
    affected={(r['setting'],r['phase_index'],r['direction']) for r in failed}
    contexts={(r['setting'],r['phase_index']) for r in failed}
    if len(affected)!=13 or len(contexts)!=7:raise ValueError('Exact 13 binary/7 phase contexts required')
    failed_ids={r['id'] for r in failed};selected=[]
    for row in previous.all_cases():
        if row['kind']=='sham':
            if (row['setting'],row['phase_index']) not in contexts:continue
            category='affected-context-sham'
        else:
            if (row['setting'],row['phase_index'],row['direction']) not in affected:continue
            category='failed-large-pulse' if row['id'] in failed_ids else ('matched-large-opposite-sign' if row['dose_index']==2 else 'matched-lower-dose-both-signs')
        selected.append(dict(row,selection_category=category))
    if len(selected)!=85:raise AssertionError('Internal signed-complete census')
    return dict(selection='Both signs in every binary setting/phase/direction context containing an original failed large pulse, at all three doses; one sham per affected setting/phase. This prospectively supersedes the 61-case proposal before new trajectories.',cases=selected,counts=dict(physical_cases=85,failed_large_pulses=18,additional_large_opposite_signs=8,matched_lower_doses=52,shams=7,binary_contexts=13,phase_contexts=7,new_methods=5,new_traces=425,old_traces=255,combined_methods=8,pairs_per_case=28,all_method_pairs=2380,same_method_ba_cells=3744),by_setting=dict(Counter(r['setting'] for r in selected)),binary_contexts=sorted(affected),contexts=sorted(contexts))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.summary.read_bytes();result=select(json.loads(raw))
    result['source_summary']=dict(path=str(a.summary.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    result['selection_sources']=[dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest()) for q in (Path(__file__),Path(previous.__file__))]
    result['case_census_sha256']=hashlib.sha256(previous.canonical(result['cases'])).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','selection')}))


if __name__=='__main__':main()
