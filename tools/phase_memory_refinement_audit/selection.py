"""Independent metadata-only reconstruction of the post-EXP014 refinement panel."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

METHODS=('rk4-0.0025','rk4-0.00125','dop853')
SETTINGS=('core','failed-near-edge')
DIRECTIONS=('phase_write','phase_null','neuron_one')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def all_cases():
    rows=[]
    for setting in SETTINGS:
        for phase in range(8):
            for dose,epsilon in enumerate((1e-4,1e-3,1e-2)):
                for direction in DIRECTIONS:
                    for sign in (-1,1):
                        rows.append(dict(id=f'{setting}--p{phase}--d{dose}--{direction}--s{sign:+d}',setting=setting,phase_index=phase,kind='pulse',dose_index=dose,epsilon=epsilon,direction=direction,sign=sign))
            rows.append(dict(id=f'{setting}--p{phase}--sham',setting=setting,phase_index=phase,kind='sham',dose_index=None,epsilon=None,direction=None,sign=None))
    return rows


def select(summary):
    science=summary['scientific']
    if summary['complete'] is not True or science['complete'] is not True:
        raise ValueError('Complete original diagnostic required')
    cases=all_cases(); by_id={r['id']:r for r in cases}
    expected={(r['id']+'--'+a,r['id']+'--'+b) for r in cases for a,b in zip(METHODS[:-1],METHODS[1:])}
    comparisons=science['trace_comparisons']
    if len(comparisons)!=608 or {tuple(r['ids']) for r in comparisons}!=expected:
        raise ValueError('Exact 608 original comparison census required')
    failed=set()
    for row in comparisons:
        if type(row['qualified']) is not bool or row['missing'] is not False:
            raise ValueError('Complete typed numerical comparison required')
        case_id=row['ids'][0].rsplit('--',1)[0]
        if row['setting']!=by_id[case_id]['setting']:
            raise ValueError('Comparison setting changed')
        if not row['qualified']:failed.add(case_id)
    if len(failed)!=18 or any(by_id[k]['kind']!='pulse' or by_id[k]['dose_index']!=2 for k in failed):
        raise ValueError('The declared 18 failed large-dose signed physical cases are required')
    lower={case_id.replace('--d2--',f'--d{dose}--') for case_id in failed for dose in (0,1)}
    sham={f"{by_id[k]['setting']}--p{by_id[k]['phase_index']}--sham" for k in failed}
    if (len(lower),len(sham))!=(36,7):raise ValueError('The declared lower-dose/context census differs')
    category={**{k:'failed-large-pulse' for k in failed},**{k:'matched-lower-dose' for k in lower},**{k:'affected-context-sham' for k in sham}}
    selected=[dict(row,selection_category=category[row['id']]) for row in cases if row['id'] in category]
    if len(selected)!=61:raise AssertionError('Internal selection census')
    return dict(selection='All failed original method comparisons collapse to physical signed cases; add both smaller doses with same sign/direction and one sham for every affected setting/phase. Original canonical order retained.',cases=selected,counts=dict(physical_cases=61,failed_large_pulses=18,matched_lower_doses=36,shams=7,new_methods=5,new_traces=305,old_traces=183,combined_methods=8,pairs_per_case=28,all_method_pairs=1708),by_setting=dict(Counter(r['setting'] for r in selected)),contexts=sorted({(r['setting'],r['phase_index']) for r in selected}))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=a.summary.read_bytes();result=select(json.loads(raw))
    result['source_summary']=dict(path=str(a.summary.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    result['selection_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['case_census_sha256']=hashlib.sha256(canonical(result['cases'])).hexdigest()
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','selection')}))


if __name__=='__main__':main()
