"""Bounded own saved-data adapter, --card/--out. Prepared cards cannot execute."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(pin):
    p = Path(pin['path'])
    if p.stat().st_size != pin['bytes'] or sha(p) != pin['sha256']:
        raise ValueError('input pin mismatch: ' + str(p))


def read(pin):
    verify(pin)
    return json.loads(Path(pin['path']).read_text())


def clean(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return 'NaN' if math.isnan(x) else ('Infinity' if x > 0 else '-Infinity')
    if isinstance(x, dict):
        return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)):
        return [clean(v) for v in x]
    return x


def write(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=1, allow_nan=False) + '\n')


def compare(own, producer, path='', inherited=False):
    """Every union leaf retained; no numerical tolerance or selected endpoint search."""
    if isinstance(own, dict) and isinstance(producer, dict):
        rows = []
        for k in list(own) + [k for k in producer if k not in own]:
            p = path + '/' + str(k).replace('~','~0').replace('/','~1')
            if k not in own or k not in producer:
                rows.append({'path':p,'status':'inherited' if k.startswith('text_') else ('producer_only' if k not in own else 'own_only'),
                             'own':own.get(k),'producer':producer.get(k)})
            else:
                rows.extend(compare(own[k], producer[k], p, inherited or k.startswith('text_') or k=='check'))
        return rows
    if isinstance(own, list) and isinstance(producer, list):
        rows = []
        for i in range(max(len(own),len(producer))):
            if i >= len(own) or i >= len(producer):
                rows.append({'path':path+'/'+str(i),'status':'producer_only' if i >= len(own) else 'own_only',
                             'own':own[i] if i<len(own) else None,'producer':producer[i] if i<len(producer) else None})
            else:
                rows.extend(compare(own[i],producer[i],path+'/'+str(i),inherited))
        return rows
    equal = type(own) is type(producer) and own == producer
    numeric = isinstance(own,(float,int)) and not isinstance(own,bool) and isinstance(producer,(float,int)) and not isinstance(producer,bool)
    if numeric:
        equal = own == producer
    row = {'path':path,'status':'inherited' if inherited else ('exact' if equal else 'different'),
           'own':own,'producer':producer}
    if numeric and math.isfinite(own) and math.isfinite(producer):
        row.update(signed_difference=own-producer,absolute_difference=abs(own-producer))
    return [row]


def cohort_gates(run, groups, expected):
    """Full-cohort admission is separate from core common-row estimates."""
    cfg = run['args']; rows = run['records']; layouts = sorted(cfg['layouts'])
    expected_keys = {(i,l) for i in expected for l in layouts}
    observed = [(r['index'],r['layout']) for r in rows if r['index'] >= 0]
    batch_map = {}
    for l in layouts:
        b = 0
        for g in sorted(groups,key=lambda g:g['length']):
            for start in range(0,len(g['indices']),l):
                for pos,i in enumerate(g['indices'][start:start+l]):
                    batch_map[i,l] = (b,pos,g['length'])
                b += 1
    issues = []
    for r in rows:
        key = (r['index'],r['layout'])
        if key not in expected_keys:
            issues.append({'key':key,'issue':'unexpected/filler'}); continue
        if (r['batch'],r['row'],r['prompt_len']) != batch_map[key]:
            issues.append({'key':key,'issue':'batch/row/prompt differs from frozen length group'})
        n = cfg['soft_steps'] if cfg['mode']=='soft' else None
        step_lengths = []
        for field in ['top1','pmax','entropy','gap'] + (['gain'] if n is not None else []) + (['temp'] if cfg.get('gain_cap') else []):
            if field not in r or not isinstance(r[field],list):
                raise ValueError('required step field absent: '+field)
            arr = np.asarray(r[field],dtype=float)
            if arr.ndim != 1: raise ValueError('step field is not a vector: '+field)
            step_lengths.append(len(arr))
            bad = int((~np.isfinite(arr)).sum())
            if bad or n is not None and len(arr)!=n:
                issues.append({'key':key,'field':field,'nonfinite':bad,'length':len(arr),'expected':n})
        if len(set(step_lengths))!=1 or n is None and not 0 < step_lengths[0] <= cfg['max_new']:
            issues.append({'key':key,'issue':'step lengths disagree or exceed hard budget','lengths':step_lengths})
        if cfg.get('perturb'):
            realized=r.get('perturb_realized')
            if realized is None or not math.isfinite(realized) or realized < 0:
                issues.append({'key':key,'issue':'missing/nonfinite/negative realized perturbation','value':realized})
        if r['correct'] != (r['answer'] is not None and r['answer']==r['gold']):
            issues.append({'key':key,'issue':'saved correct flag differs'})
    return {'run':run['id'],'expected':len(expected_keys),'records':len(rows),
            'missing':sorted(expected_keys-set(observed)),'duplicates':len(observed)-len(set(observed)),
            'issues':issues,'qualified':set(observed)==expected_keys and len(observed)==len(expected_keys) and not issues}


def calculate(e, s, runs, membership, association, sync, published_lambda):
    expected = membership['indices']
    gates = [cohort_gates(r,membership['groups'],expected) for r in runs.values()]
    gold_by_index = {}
    for r in runs.values():
        for row in r['records']:
            if row['index'] in expected: gold_by_index.setdefault(row['index'],set()).add(row['gold'])
    gold_conflicts = {i:sorted(v) for i,v in gold_by_index.items() if len(v)!=1}
    arm_specs = association['e2']['arms']
    ordered = [dict(runs[a['run']],id=a['name']) for a in arm_specs]
    first = e.e2(ordered,'uncapped',expected)
    second = e.e2(ordered,'t0.5',expected)
    rr = association['e3']['metadata']['runs']
    third = e.e3(runs[rr['soft0']],[runs[p] for p in rr['soft']],runs[rr['greedy']],
                 [runs[p] for p in rr['sample']],expected)
    extra = []
    for r in runs.values():
        rows=r['records']; cfg=r['args']
        extra.append({'run':r['id'],'missing_answers':sum(x['answer'] is None for x in rows),
                      'at_hard_token_budget':sum(len(x.get('tokens',[]))>=cfg['max_new'] for x in rows) if cfg['mode']=='hard' else None,
                      'missing_perturb_realized':sum(x.get('perturb_realized') is None for x in rows) if cfg.get('perturb') else None,
                      'correct':sum(x['answer'] is not None and x['answer']==x['gold'] for x in rows), 'count':len(rows)})
    unc = {x['index']:x for x in ordered[0]['records'] if x['layout']==16}
    soft0 = {x['index']:x for x in runs[rr['soft0']]['records']}
    cross = compare({str(i):unc[i] for i in sorted(unc)}, {str(i):soft0[i] for i in sorted(soft0)},'/soft0_vs_uncapped_layout16')
    tj=sync['transverse.json']; tz=sync['transverse.npz']
    transverse=s.transverse(tj['cells'],tj['pairs'],tz['jvp_logs'],tz['logs'],published_lambda)
    cells=transverse['cells']
    for c in cells.values(): c['sweeps']={}
    sweep_gates=[]
    for name in ['sweep-near','sweep-far','sweep-refine']:
        meta=sync[name+'.json']; arrays=sync[name+'.npz']
        scored=s.sweep(meta,arrays['dist'],arrays['tok'],final=arrays['final'])
        for ci,value in scored.items():
            cells[ci]['sweeps'][name]=value
            sweep_gates.append({'sweep':name,'cell':ci,'full_record_and_tail':value['full_record_and_tail'],
                                'all_distances_qualified':value['all_distances_qualified']})
    periodic={name:s.periodic(sync[name+'.json']) for name in ['periodic','periodic-synced']}
    for rows in periodic.values():
        for row in rows:
            raw=row['saved']; sm=sync[raw['sweep']+'.json']
            candidates=[k for k,p in enumerate(sm['pairs']) if p['cell']==raw['cell'] and abs(p['c']-raw['c'])<1e-12]
            row['source_pair_candidates']=candidates
            row['source_steps_match']=raw['steps_before']==sm['steps']
            if len(candidates)!=1: raise ValueError('periodic source pair not unique')
            k=candidates[0]; row['source_c_difference']=sm['pairs'][k]['c']-raw['c']
            row['source_final_distance_minus_pair_now']=float(sync[raw['sweep']+'.npz']['dist'][k,-1])-raw['pair_distance_now']
    checks=s.check_scalars(sync['check.json'])
    check_projection=[{'index':r['saved']['index'],'bitwise':r['saved']['bitwise'],
                       'max_abs_diff_rel':r['saved']['max_abs_diff_rel'],
                       'fd_rel_err':dict(zip(map(str,r['saved']['fd_h']),r['saved']['fd_rel_err']))} for r in checks]
    summary={'cells':cells,'check':check_projection,'sync_tol':1e-8,
             'sync_rule':'max distance over the last 500 steps below 1e-08'}
    if tj.get('commit',{}).get('tools_dirty'): summary['warnings']=['transverse run had uncommitted tool changes']
    e2projection={'arms':first['arms'],'contrasts':first['contrasts'],
                  'contrasts_vs_t0.5':{k:second['contrasts'][k] for k in association['e2']['contrast_key_order']['contrasts_vs_t0.5']}}
    selection=e.selection(membership)
    return {'e2':{'first':first,'second':second,'projection':e2projection},'e3':third,
            'synchronization':{'projection':summary,'transverse':transverse,'periodic':periodic,'precision':checks},
            'gates':{'cohorts':gates,'selection':selection,'sweeps':sweep_gates,
                     'e3_diagnostics':extra,'soft0_vs_uncapped':cross,
                     'gold_conflicts':gold_conflicts,
                     'all_cohorts_qualified':selection['all_structural_checks'] and all(g['qualified'] for g in gates) and not gold_conflicts}}


def execute(card,out):
    started=time.perf_counter()
    if card['status']!='reviewed-for-execution': raise ValueError('unreviewed card')
    for name,expected in card['source_sha256'].items():
        if sha(HERE/(name+'.py'))!=expected: raise ValueError('source mismatch '+name)
    for p in card['pins']: verify(p)
    index={p['path']:p for p in card['pins']}
    if len(index)!=len(card['pins']): raise ValueError('duplicate input path')
    def admitted(path):
        if path not in index: raise ValueError('undeclared input '+path)
        return read(index[path])
    def arrays(path,expected_shapes):
        if path not in index: raise ValueError('undeclared NPZ')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!=set(expected_shapes): raise ValueError('NPZ field mismatch')
            a={k:z[k] for k in z.files}
        for k,(shape,dtype) in expected_shapes.items():
            if list(a[k].shape)!=shape or a[k].dtype.str!=dtype: raise ValueError('NPZ header mismatch '+k)
        return a
    roles=card['roles']; meta=admitted(roles['metadata']); assoc=admitted(roles['association'])
    membership=admitted(roles['membership']); published=admitted(roles['published_lambda'])
    profile=admitted(roles['manufactured_report'])
    if profile['schema']!='task27-e2e3-sync-profile-v1' or profile['controls']['passed']!=12 or profile['controls']['failed']!=0:
        raise ValueError('missing successful manufactured evidence')
    runs={}
    for kind in ['e2','e3']:
        for r in meta['e2e3'][kind]:
            cfg=admitted(r['config']['path'])
            if cfg['args']!=r['config']['metadata']['args']: raise ValueError('config differs from admitted metadata')
            if r['records']['path'] not in index: raise ValueError('unadmitted records')
            records=[json.loads(line) for line in Path(r['records']['path']).read_text().splitlines() if line.strip()]
            if [{'index':x['index'],'layout':x['layout']} for x in records]!=r['records']['ordered_identity']:
                raise ValueError('row order differs from metadata')
            path=cfg['args']['out']
            if path in runs: raise ValueError('duplicate run')
            runs[path]={'id':path,'args':cfg['args'],'records':records}
    if len(meta['e2e3']['e2'])!=5 or len(meta['e2e3']['e3'])!=18: raise ValueError('declared E2/E3 run counts')
    arms=assoc['e2']['arms']; rr=assoc['e3']['metadata']['runs']
    if [a['name'] for a in arms]!=['uncapped','cap0.37','cap0.10','cap0.015','t0.5']:
        raise ValueError('unreviewed E2 replay order')
    e2paths=[a['run'] for a in arms]
    e3paths=[rr['soft0']]+rr['soft']+[rr['greedy']]+rr['sample']
    if len(e3paths)!=18 or len(set(e2paths+e3paths))!=23 or set(e2paths+e3paths)!=set(runs):
        raise ValueError('run-role coverage/duplicates')
    sync={}
    for key,r in meta['sync'].items():
        name=key.split('/')[0]; ext=Path(key).suffix; dest=(name if name not in ['check','transverse'] else name)+ext
        if ext=='.json':
            sync[dest]=admitted(r['path'])
            for field,value in r.get('metadata',{}).items():
                if sync[dest].get(field)!=value: raise ValueError('sync metadata mismatch '+key+'/'+field)
        else: sync[dest]=arrays(r['path'],card['array_shapes'][key])
    for name in ['sweep-near','sweep-far','sweep-refine']:
        if sync[name+'.json']['sweep_npz_sha256']!=index[meta['sync'][name+'/sweep.npz']['path']]['sha256']:
            raise ValueError('saved sweep digest mismatch')
    if sync['transverse.json']['transverse_npz_sha256']!=index[meta['sync']['transverse/transverse.npz']['path']]['sha256']:
        raise ValueError('saved transverse digest mismatch')
    cells=admitted(roles['cells'])['cells']
    expected_cells=[{'index':i,'log10_temperature':float(c[0]),'alpha':float(c[1])} for i,c in enumerate(cells)]
    if sync['transverse.json']['cells']!=expected_cells: raise ValueError('cell coordinate association')
    if len(expected_cells)!=3 or {r['index'] for r in sync['check.json']['cells']}!={0,1,2} or len(sync['check.json']['cells'])!=3:
        raise ValueError('check/transverse cell coverage')
    for name in ['transverse','sweep-near','sweep-far','sweep-refine']:
        sm=sync[name+'.json']; a=sm['args']; per=json.loads(a['c_per_cell']) if a.get('c_per_cell') else None
        cs=a['c'] if a['c'] is not None else np.round(np.arange(a['c_min'],a['c_max']+1e-12,a['c_step']),6).tolist()
        pairs=[{'cell':ci,'c':float(c)} for ci in [0,1,2] for c in (per[str(ci)] if per else cs)]
        if a['cells']!=[0,1,2] or sm['pairs']!=pairs: raise ValueError('native pair grid/order mismatch '+name)
    for name,count in [('periodic',14),('periodic-synced',4)]:
        sm=sync[name+'.json']
        if sm['args']['settle']!=0 or sm['args']['max_period']!=40 or sm['args']['tol']!=1e-9:
            raise ValueError('periodic contract differs')
        wanted=[(n,int(i),float(c)) for n,i,c in (x.split(':') for x in sm['args']['pairs'])]
        got=[(r['sweep'],r['cell'],r['c']) for r in sm['pairs']]
        if len(got)!=count or got!=wanted: raise ValueError('periodic selection/order mismatch')
    pcells=published['cells']
    if len(pcells)!=3 or {c['index'] for c in pcells}!={0,1,2} or published.get('complete') is not True:
        raise ValueError('published spectrum cell coverage/completion')
    for c in pcells:
        expected=expected_cells[c['index']]
        if any(c[k]!=expected[k] for k in ['log10_temperature','alpha']) or not c['exponents']:
            raise ValueError('published spectrum coordinate/column association')
    pub={c['index']:c['exponents'][0] for c in pcells}
    e,s=module('e2e3'),module('synchronization')
    result=calculate(e,s,runs,membership,assoc,sync,pub)
    comparisons=[]
    for key,own in [('e2',result['e2']['projection']),('e3',result['e3']['projection']),('sync',result['synchronization']['projection'])]:
        p=admitted(assoc[key]['path'])
        selected={k:p[k] for k in own if k in p}
        comparisons.extend(compare(own,selected,'/'+key))
        comparisons.extend({'path':'/'+key+'/'+k,'status':'inherited_publication_wrapper',
                            'producer':v,'own':None} for k,v in p.items() if k not in own)
    write(out/'e2.json',result['e2']);write(out/'e3.json',result['e3']);write(out/'synchronization.json',result['synchronization'])
    write(out/'gates.json',result['gates']);write(out/'comparisons.json',comparisons)
    elapsed=time.perf_counter()-started; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for p in card['pins']: verify(p)
    files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.glob('*.json'))]
    report={'schema':'task27-e2e3-sync-independent-v1','computational_status':'complete',
            'source_sha256':card['source_sha256'],'input_pins':card['pins'],'outputs':files,
            'comparison_status_counts':{k:sum(r['status']==k for r in comparisons) for k in sorted({r['status'] for r in comparisons})},
            'all_cohorts_qualified':result['gates']['all_cohorts_qualified'],
            'runtime':{'seconds_before_final_rehash_report':elapsed,'observed_peak_kib_before_final_rehash_report':rss},
            'scope':'independent saved arithmetic; no native replay; original order/precision/periodic premises remain scoped',
            'bootstrap_schedule':card['bootstrap_schedule']}
    write(out/'report.json',report)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);created=False
    try:
        card=json.loads(Path(args.card).read_text());out.mkdir(parents=True,exist_ok=False);created=True
        execute(card,out)
    except Exception as exc:
        if created: write(out/'failure.json',{'computational_status':'failed','error':repr(exc)})
        raise


if __name__=='__main__': main()
