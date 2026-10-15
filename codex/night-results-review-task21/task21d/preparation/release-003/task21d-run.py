"""Prepared bounded own saved-line rescore; no producer imports or model execution."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

SCHEMA='codex-task21d-saved-rescore-v1'
LIMITS={'cpu_seconds':30,'wall_seconds':60,'as_bytes':1024**3,'input_bytes':4*1024**2,
        'output_bytes':16*1024**2,'startup_available_bytes':20*1024**3,'runtime_available_bytes':16*1024**3}


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError('duplicate JSON key: '+k)
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique)


def safe(obj):
    if isinstance(obj,float) and not math.isfinite(obj):return {'nonfinite_saved_or_derived':repr(obj)}
    if isinstance(obj,dict):return {k:safe(v) for k,v in obj.items()}
    if isinstance(obj,list):return [safe(v) for v in obj]
    return obj


def import_own(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();cardraw=Path(args.card).read_bytes();card=read_json(cardraw)
    if card.get('schema')!=SCHEMA or card.get('admission')!='reviewed' or card.get('limits')!=LIMITS:
        raise SystemExit('prepared or invalid card: refusing execution')
    out=Path(args.out)
    if not out.is_absolute() or str(out)!=card['output'] or out.exists():raise SystemExit('exact fresh output required')
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10:raise SystemExit('priority stronger than exact nice10')
    os.nice(10-nice)
    hard=resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard!=resource.RLIM_INFINITY and hard<LIMITS['as_bytes']:raise SystemExit('insufficient inherited AS hard limit')
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['as_bytes'],LIMITS['as_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_seconds']))
    for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        if os.environ.get(var)!='1':raise SystemExit('exact CPU1 environment required: '+var)
    here=Path(__file__).resolve().parent
    if set(card['source_sha256'])!={'run.py','score.py','inputs.py'}:raise SystemExit('exact own source inventory required')
    for name,pin in card['source_sha256'].items():
        if sha((here/name).read_bytes())!=pin:raise SystemExit('own source mismatch: '+name)
    score=import_own('task21d_score',here/'score.py');helpers=import_own('task21d_inputs',here/'inputs.py')
    if score.np.__version__!=card['numpy_version']:raise SystemExit('NumPy version mismatch')
    admitted=[];total=0;seen=set()
    def load(binding,role):
        nonlocal total
        path=Path(binding['path'])
        if not path.is_absolute() or str(path) in seen:raise ValueError('duplicate/nonabsolute input')
        raw=path.read_bytes();total+=len(raw);seen.add(str(path))
        if total>LIMITS['input_bytes'] or len(raw)!=binding['bytes'] or sha(raw)!=binding['sha256']:
            raise ValueError('input size/hash mismatch: '+str(path))
        admitted.append({'path':str(path),'bytes':len(raw),'sha256':binding['sha256'],'role':role})
        return read_json(raw)
    out.mkdir(parents=False,exist_ok=False)
    records=[]
    try:
        manifest=load(card['inputs_manifest'],'input_manifest')
        if manifest['schema']!='task21d-saved-inputs-v1' or [(m['id'],m['base_period'],m['expected_cells']) for m in manifest['models']]!=[('FL15',4,1700),('FL7',2,1000)]:
            raise ValueError('exact two-model/period/coverage contract required')
        authority=load(manifest['authority'],'inherited_original_authority')
        sourcemap=load(manifest['source_map'],'inherited_source_association')
        prefix_authority=load(card['prefix_authority'],'inherited_resolved_prefix_authority')
        tic=time.perf_counter()
        for spec in manifest['models']:
            row={'id':spec['id'],'completed':False};records.append(row)
            line=load(spec['line'],'line_'+spec['id']);nf=load(spec['normal_form'],'normal_form_'+spec['id'])
            curve=load(spec['normal_form_source_curve'],'curve_'+spec['id'])
            definition=load(spec['cell_definition'],'definition_'+spec['id'])
            row['admission']=helpers.admit_line(line,definition,nf,curve,spec)
            row['normal_form']=helpers.normal_form(nf,{'source':{'period':spec['base_period']},
                'args':{'param':'temperature','index':0,'nearest_fold':False},'coordinate':'log10_temperature'})
            row['independent']=score.rescore(line['cells'],nf,spec['base_period'])
            row['producer_comparisons']={}
            for old,mode in [('original','original_stop'),('amended','amended_skip')]:
                producer=load(spec['producer_scores'][old],old+'_score_'+spec['id'])
                row['producer_comparisons'][old]=score.compare_leaves(producer,row['independent']['variants'][mode])
            row['amendment_comparison']=score.compare_leaves(row['independent']['variants']['original_stop'],row['independent']['variants']['amended_skip'])
            row['completed']=True
        for b in admitted:
            raw=Path(b['path']).read_bytes()
            if len(raw)!=b['bytes'] or sha(raw)!=b['sha256']:raise ValueError('input changed during rescore')
        result={'schema':SCHEMA,'computational_completed':True,'scientific_claims_accepted':False,'models':records,
                'input_pins':admitted,'card_sha256':sha(cardraw),'source_sha256':card['source_sha256'],
                'numpy_version':score.np.__version__,'original_authority_inherited':authority,
                'source_association_inherited':sourcemap,'prefix_authority_inherited':prefix_authority,'target_cell_count':sum(m['expected_cells'] for m in manifest['models']),
                'calculation_input_io_and_posthash_seconds':time.perf_counter()-tic,
                'observed_peak_rss_kib_before_serialization':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'scope':'Independent unblinded saved-line arithmetic; original native model/fold/period/precision premises inherited.'}
        payload=json.dumps(safe(result),indent=2,allow_nan=False).encode()+b'\n'
        inv=json.dumps({'files':[{'path':'report.json','bytes':len(payload),'sha256':sha(payload)}]},indent=2).encode()+b'\n'
        if len(payload)+len(inv)>LIMITS['output_bytes']:raise ValueError('output byte ceiling exceeded')
        with (out/'report.json').open('xb') as f:f.write(payload)
        with (out/'manifest.json').open('xb') as f:f.write(inv)
    except Exception as e:
        failure={'schema':SCHEMA,'computational_completed':False,'error_type':type(e).__name__,'error':str(e),'models':records,'admitted':admitted}
        payload=json.dumps(safe(failure),indent=2,allow_nan=False).encode()+b'\n'
        if len(payload)<=LIMITS['output_bytes']:
            with (out/'failure.json').open('xb') as f:f.write(payload)
        raise
    return 0


if __name__=='__main__':sys.exit(main())
