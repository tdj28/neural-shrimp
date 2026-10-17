"""One proposed manufactured C12 workload/serialization measurement; no target IO."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import time
import traceback

OUTPUT='/data2/neural-shrimp-data/codex/task27-ready-review-20261017/run-001/c12-profile'


def own(name):
    p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location('profile_own_'+name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    card=json.loads(Path(a.card).read_text());out=Path(a.out)
    if card.get('status')!='admitted' or str(out)!=OUTPUT or card['output']!=OUTPUT:
        raise ValueError('prepared/wrong manufactured card/output')
    for name,pin in card['source_files'].items():
        b=Path(__file__).with_name(name).read_bytes()
        if len(b)!=pin['bytes'] or hashlib.sha256(b).hexdigest()!=pin['sha256']:raise ValueError('source mismatch '+name)
    D=own('run')
    if card['limits']!=D.LIMITS:raise ValueError('wrong manufactured resource contract')
    out.mkdir(parents=False,exist_ok=False);started=time.monotonic();W=D.Writer(out)
    try:
        C=own('controls');R=own('reducers');controls=C.run_controls();W.write('controls.json',controls)
        if controls['failed'] or controls['passed']!=15:raise ValueError('manufactured controls failed')
        payload=C.full_shape_payload()
        # Retain all synthetic scientific rows once, as in the actual adapter.
        c3=payload['c12_3'];rows3=c3['48'].pop('row_ledger');c3['24'].pop('row_ledger')
        c3['24'].pop('pair_steps',None)
        selection=payload['c12_4c_selection'];rows4=[{'run':k,**r} for k,s in zip(('a','b'),selection) for r in s.pop('row_ledger')]
        W.write('c12-3.json',c3);W.lines('c12-3-rows.jsonl',rows3)
        W.write('c12-4c.json',{'result':payload['c12_4c'],'selection':selection,'checker':payload['checker'],'fixed_coverage':payload['fixed_coverage']})
        W.lines('c12-4c-rows.jsonl',rows4)
        count=0
        def compared():
            nonlocal count
            projections=[('c3-analysis',{s:v['summary'] for s,v in c3.items()}),
                         ('c3-publication',{s:v['summary'] for s,v in c3.items()}),
                         ('c4-confirm',payload['c12_4c']),('c4-publication',payload['c12_4c']),
                         ('c4-check',payload['checker'])]
            for name,v in projections:
                for row in R.compare_leaves(v,v):count+=1;yield {'document':name,**row}
        W.lines('comparisons.jsonl',compared())
        report={'schema':'c12-manufactured-resource-v1','computational_completed':True,'resource_profile_only':True,
            'scientific_claims_accepted':False,'groups':15,'source_rows':[8192,2304,1152],
            'comparison_leaves':count,'scientific_output_bytes_before_report':W.used,
            'elapsed_through_scientific_serialization':time.monotonic()-started,
            'rss_kib_observed_before_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'scope':'Manufactured source-shaped rows and fixed bootstrap streams only; no target paths opened. Includes two C3 readouts, full C4 summaries and checker. Comparisons intentionally include extra own witness leaves. Does not measure target hash IO or guarantee adversarial/worst-case cost.'}
        W.write('report.json',report);W.write('manifest.json',{'files':list(W.files)})
    except Exception as exc:
        W.write('failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        raise


if __name__=='__main__':main()
