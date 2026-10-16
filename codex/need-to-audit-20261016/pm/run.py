"""One admitted PM/PM2 saved-row audit, manufactured controls first.

Independent source, no producer imports/model/native work. Exact source/receipt
history is inherited from pinned finite metadata, not a recreated historical job.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

SCHEMA='independent-pm-pm2-v1'
OUTPUT='/data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/pm'
PM_ORDER=['top15-s64','replay','greedy','top1-s64','full-s64','top15-s16','full-s16']
CELL_SPECS=[('A-gsm8k','PM2-A','r1qwen7b','gsm8k','primary','NVIDIA H100 NVL'),
 ('A-math','PM2-A','r1qwen7b','math500','primary','NVIDIA H100 NVL'),
 ('B-gsm8k','PM2-BCD','r1llama8b','gsm8k','primary','NVIDIA H100 NVL'),
 ('B-math','PM2-BCD','r1llama8b','math500','primary','NVIDIA H100 NVL'),
 ('C-gsm8k','PM2-BCD','qwq32b','gsm8k','primary','NVIDIA H200'),
 ('D-gsm8k','PM2-BCD','r1qwen7b','gsm8k','replication','NVIDIA A100-SXM4-80GB')]
LIMIT=32*1024**2


def pin(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def verify(p):
    actual=pin(p['path'])
    if any(actual[k]!=p[k] for k in ['bytes','sha256']):raise ValueError('pin mismatch: '+p['path'])
    return actual


def own(name):
    spec=importlib.util.spec_from_file_location('own_pm_'+name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def save(out,name,value):
    body=(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n').encode()
    used=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if used+len(body)>LIMIT:raise ValueError('own output cap exceeded; existing failure artifacts retained')
    with (out/name).open('xb') as f:f.write(body)


def load_inventory(card):
    verify(card['inputs']);inv=json.loads(Path(card['inputs']['path']).read_text())
    if inv['schema']!='pm-input-inventory-v1' or len(inv['arms'])!=19:raise ValueError('input schema/count')
    if inv['pm_order']!=PM_ORDER or inv['pm2_order']!=[x[0] for x in CELL_SPECS] or inv['cross']!=[['A-gsm8k','D-gsm8k']]:raise ValueError('frozen order changed')
    expected=[dict(zip(['name','group','model','task','role','gpu'],x)) for x in CELL_SPECS]
    if inv['cells']!=expected:raise ValueError('cell role/model/hardware mapping changed')
    paths=[p['path'] for p in inv['pins']]
    if len(paths)!=len(set(paths)):raise ValueError('duplicate pin path')
    if sum(p['bytes'] for p in inv['pins'])>48*1024**2:raise ValueError('input ceiling')
    before=[verify(p) for p in inv['pins']]
    declared={p['path']:p for p in before}
    lookup={}
    for x in inv['arms']:
        key=x['group']+'/'+x['arm']
        if key in lookup:raise ValueError('duplicate arm identity')
        for field in ['records','run']:
            p=x[field]
            if p['path'] not in declared or any(p[k]!=declared[p['path']][k] for k in ['bytes','sha256']):raise ValueError('unpinned arm member')
        if json.loads(Path(x['run']['path']).read_text())!=x['config']:raise ValueError('config metadata mismatch')
        cfg=x['config'];a=cfg['args']
        if a['seed']!=20261016 or a['dtype']!='bfloat16' or a['device']!='cuda' or a['split']!='train':raise ValueError('source protocol mismatch')
        if x['group']=='PM':
            if x['arm'] not in PM_ORDER or a['model']!='qwen7b' or a['layouts']!=[8,16] or a['limit']!=256 or a['offset']!=0 or cfg['gpu']!='NVIDIA H100 NVL':raise ValueError('PM association')
        else:
            matched=[v for v in CELL_SPECS if x['group']==v[1] and a['model']==v[2] and a['task']==v[3]]
            if len(matched)!=1 or cfg['gpu']!=matched[0][5] or a['arm'] not in ['soft','plain'] or x['arm']!=a['model']+'-'+a['task']+'-'+a['arm']:raise ValueError('PM2 association')
            math_task=a['task']=='math500'
            if a['layouts']!=([1,8] if math_task else [8,16]) or a['limit']!=(64 if math_task else 256) or a['offset']!=256:raise ValueError('PM2 layout/task selection')
            is_a='/pi/pm2-A/' in x['records']['path']
            if is_a!=(x['group']=='PM2-A'):raise ValueError('A versus ordinary BCD path isolation')
        lookup[key]=x
    expected_keys={'PM/'+n for n in PM_ORDER}|{g+'/'+model+'-'+task+'-'+mode for _,g,model,task,_,_ in CELL_SPECS for mode in ['soft','plain']}
    if set(lookup)!=expected_keys:raise ValueError('missing/unexpected arm identity')
    if len({x['records']['path'] for x in lookup.values()})!=19:raise ValueError('shared raw path')
    return inv,lookup,before


def chronology(inv):
    designs=[{'commit':d['show'][0],'commit_time':d['show'][2],
              'epoch':datetime.fromisoformat(d['show'][2]).timestamp(),'subject':d['show'][3]} for d in inv['design_chronology']]
    result=[]
    for name in inv['original_logs']:
        rows=[]
        for lineno,line in enumerate(Path(name).read_text().splitlines(),1):
            try:v=json.loads(line)
            except json.JSONDecodeError:v=None
            row={'line':lineno,'text':line,'parsed':v}
            if isinstance(v,dict) and 'started' in v:
                row['started_utc']=datetime.fromtimestamp(v['started'],timezone.utc).isoformat()
                relevant=designs[:2] if Path(name).name=='PM-orchestrator.log' else designs[2:]
                row['design_chronology']=[{'commit':d['commit'],'seconds_before_recorded_start':v['started']-d['epoch'],
                 'precedes_recorded_start':d['epoch']<v['started']} for d in relevant]
            rows.append(row)
        result.append({'path':name,'rows':rows})
    return {'designs':designs,'logs':result,'metadata_attempt_rows':inv['attempt_metadata'],
      'scope':'Commit and original orchestrator timestamps/receipts only. Preserve every startup/price/capacity refusal, failed/incomplete receipt and termination statement. No new remote verification, independent historical clock attestation, or reconstructed job. Distinguish pre-run startup refusals from data-producing reruns.'}


def ranges(p1,p2):
    entries=[]
    for n,a in p1['arms'].items():
        for L,v in a['accuracy'].items():entries.append({'study':'PM','cell':n,'mode':a['mode'],'layout':L,'accuracy':v})
    for n,c in p2['cells'].items():
        for mode in ['soft','plain']:
            for L,v in c[mode]['accuracy'].items():entries.append({'study':'PM2','cell':n,'mode':mode,'layout':L,'accuracy':v})
    groups={'PM_all_arms':lambda x:x['study']=='PM','PM_non_greedy':lambda x:x['study']=='PM' and x['cell']!='greedy',
      'PM2_Qwen_QwQ_H100_H200_GSM_soft':lambda x:x['study']=='PM2' and x['cell'] in ['A-gsm8k','C-gsm8k'] and x['mode']=='soft',
      'PM2_Qwen_QwQ_H100_H200_GSM_both_modes':lambda x:x['study']=='PM2' and x['cell'] in ['A-gsm8k','C-gsm8k'],
      'PM2_including_A100_Qwen_QwQ_GSM_soft':lambda x:x['study']=='PM2' and x['cell'] in ['A-gsm8k','C-gsm8k','D-gsm8k'] and x['mode']=='soft',
      'PM2_all_cells_both_modes':lambda x:x['study']=='PM2'}
    out={}
    for n,f in groups.items():
        chosen=[x for x in entries if f(x)];finite=[x for x in chosen if x['accuracy'] is not None]
        lo=min((x['accuracy'] for x in finite),default=None);hi=max((x['accuracy'] for x in finite),default=None)
        out[n]={'members':chosen,'minimum':lo,'maximum':hi,'minimizers':[x for x in finite if x['accuracy']==lo],
                'maximizers':[x for x in finite if x['accuracy']==hi],'undefined':[x for x in chosen if x['accuracy'] is None]}
    return out


def diagnostic_counts(d):
    r=d['record_checks'];p=d['paired_rows']
    return {'record_count':len(r),'paired_count':len(p),'unpaired_count':len(d['unpaired']),
      'duplicate_count':len(d['duplicates_last_write_retained']),'negative_rows':len(d['excluded_negative_index_rows']),
      'unexpected_layout_rows':len(d['unexpected_layout_rows']),
      'stored_label_discrepancies':sum(not x['stored_label_consistent'] for x in r),
      'suffix_extraction_discrepancies':sum(not x['suffix_extraction_matches'] for x in r),
      'possibly_truncated_suffixes':sum(x['may_be_truncated_suffix'] for x in r),
      'marker_present_records':sum(any(x['markers'].values()) for x in r),
      'marker_vs_stored_answer_changes':sum(x['marker_normalized_answer']!=x['stored_answer'] for x in r),
      'stored_flip_count':sum(x['flip'] for x in p),'suffix_flip_count':sum(x['suffix_flip'] for x in p),
      'marker_flip_count':sum(x['marker_normalized_flip'] for x in p),
      'marker_correct_counts':{k:sum(x['marker_normalized_correct_'+k] for x in p) for k in ['a','b']},
      'paired_gold_mismatch':sum(not x['gold_match'] for x in p),'paired_prompt_length_mismatch':sum(not x['prompt_length_match'] for x in p),
      'scope':'All detailed identifiers/values retained in row-diagnostics.json; normalized suffix counts are a separate diagnostic, not frozen-endpoint replacements.'}


def main(card_path,out):
    card=json.loads(Path(card_path).read_text())
    if card['admission']!='reviewed':raise ValueError('prepared card refuses execution')
    if str(out)!=OUTPUT or card['output']!=OUTPUT:raise ValueError('fixed output identity')
    if card['schema']!=SCHEMA or card['controls_groups']!=12:raise ValueError('card contract')
    for name in ['run.py','core.py','controls.py']:
        p=Path(__file__).with_name(name)
        if pin(p)['sha256']!=card['source_sha256'][name]:raise ValueError('source pin: '+name)
    out.mkdir(exist_ok=False);start=time.perf_counter()
    try:
        controls=own('controls');controls.main(out/'controls')
        controls_result=json.loads((out/'controls/report.json').read_text())
        if controls_result['groups']!=12 or controls_result['passed']!=12:raise ValueError('manufactured control refusal')
        inv,lookup,before=load_inventory(card);core=own('core')
        diagnostic={};loaded={}
        for key,x in lookup.items():
            kind='pm' if x['group']=='PM' else 'pm2'
            rows,layouts,extra,detail=core.load_rows(x['records']['path'],x['config']['args'],kind)
            if detail['record_count']!=x['record_count']:raise ValueError('record count mismatch: '+key)
            loaded[key]={'rows':rows,'layouts':layouts,'extra':extra,'cfg':x['config']['args']};diagnostic[key]=detail
        arms={n:loaded['PM/'+n] for n in PM_ORDER};cells={}
        for n,g,model,task,role,_ in CELL_SPECS:
            cells[n]={'role':role,**{mode:loaded[g+'/'+model+'-'+task+'-'+mode] for mode in ['soft','plain']}}
        p1,d1=core.analyze_pm(arms);p2,d2=core.analyze_pm2(cells,inv['cross'])
        published={name:json.loads(Path(p['path']).read_text()) for name,p in inv['published'].items()}
        if list(published['pm-summary']['arms'])!=PM_ORDER or list(published['pm2-summary']['cells'])!=[v[0] for v in CELL_SPECS]:raise ValueError('published order proxy changed')
        comparisons={'PM':core.compare_leaves(p1,published['pm-summary']),'PM2':core.compare_leaves(p2,published['pm2-summary'])}
        save(out,'row-diagnostics.json',{'arms':diagnostic,'PM_contrast_pairing':d1,'PM2_contrast_pairing':d2})
        save(out,'comparisons.json',comparisons);save(out,'chronology.json',chronology(inv))
        after=[verify(p) for p in inv['pins']]
        if before!=after:raise ValueError('input mutation')
        report={'schema':SCHEMA,'computational_completed':True,'scientific_claims_accepted':False,
          'PM':p1,'PM2':p2,'accuracy_ranges':ranges(p1,p2),
          'four_secondary_bonferroni_diagnostic':{n:{'p':p1['contrasts']['top15-s64 - '+n]['mcnemar_p'],
              'p_times_four_capped':min(1,4*p1['contrasts']['top15-s64 - '+n]['mcnemar_p'])} for n in ['replay','top15-s16','greedy','top1-s64']},
          'row_inventory':{k:diagnostic_counts(v) for k,v in diagnostic.items()},
          'comparison_counts':{k:dict(Counter(x['status'] for x in v)) for k,v in comparisons.items()},
          'input_binding':{'inventory':card['inputs'],'before_after_identical':True,'pins':before,
             'models_hardware_configs':[{k:x[k] for k in ['group','arm','records','run','config']} for x in inv['arms']]},
          'rng':{'seed':core.SEED,'draws_per_bootstrap':core.DRAWS,'order_basis':inv['order_basis'],'PM_order':PM_ORDER,'PM2_order':inv['pm2_order'],
                 'primary_cluster_order':'first encounter in JSONL; per-arm and pairwise cluster order uses sorted problem IDs'},
          'controls':controls_result,'limits':['Unblinded independent saved-row arithmetic, no model replay.',
            'Original source/recorded start and termination authority inherited from pinned metadata; no historical loaded-weight or independent timestamp reconstruction.',
            'Saved suffix extraction cannot recover discarded answer prefixes; byte-marker normalization is an explicit alternative diagnostic.',
            'Stored answer/correct fields remain the original endpoints even where independent diagnostics differ.',
            'No equivalence conclusion from nonsignificant secondary or cross-GPU tests.'],
          'seconds_through_input_posthash_before_final_report':time.perf_counter()-start,
          'rss_KiB_observed_before_final_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'numpy_version':core.np.__version__,'python':sys.version}
        save(out,'report.json',report)
        members=[{**pin(p),'path':str(p.relative_to(out))} for p in sorted(out.rglob('*')) if p.is_file()]
        save(out,'manifest.json',{'schema':SCHEMA,'files':members,'file_count':len(members),'computational_completed':True})
    except BaseException as e:
        if not (out/'failure.json').exists():
            (out/'failure.json').write_text(json.dumps({'computational_completed':False,'error_type':type(e).__name__,'error':str(e)},indent=2)+'\n')
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();main(a.card,Path(a.out))
