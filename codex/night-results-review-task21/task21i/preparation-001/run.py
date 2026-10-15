"""Independent Task21i BZ/BP adapter. Prepared; no producer imports or model code."""
import argparse
import hashlib
import importlib.util
import json
import math
import resource
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('task21i_statistics', HERE/'statistics.py')
s = importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
TASKS = ('arithmetic', 'nq', 'mbpp')
TEMPERATURES = (.25, .32, .4)
SEED = 20261015


def unique_object(pairs):
    out = {}
    for k, v in pairs:
        s.require(k not in out, f'duplicate JSON key: {k}')
        out[k] = v
    return out


def decode(text):
    return json.loads(text, object_pairs_hook=unique_object)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def pin(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}


def small_pin(row):
    return {k: row[k] for k in ('path', 'bytes', 'sha256')}


def verify(row):
    s.require(pin(row['path']) == small_pin(row), 'changed pinned input: '+row['path'])


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return str(x)
    return x


def write_json(path, x):
    with open(path, 'x') as f: json.dump(clean(x), f, indent=1, allow_nan=False); f.write('\n')


def compare(path, own, saved, scope):
    if isinstance(own, dict) and isinstance(saved, dict):
        result = []
        for k in sorted(set(own) | set(saved)):
            if k not in own or k not in saved:
                result.append({'path':path+'/'+str(k), 'own_present':k in own, 'saved_present':k in saved, 'equal':False, 'scope':scope})
            else: result += compare(path+'/'+str(k), own[k], saved[k], scope)
        return result
    if isinstance(own, list) and isinstance(saved, list):
        if len(own) != len(saved): return [{'path':path, 'own_length':len(own), 'saved_length':len(saved), 'equal':False, 'scope':scope}]
        return [r for i,(a,b) in enumerate(zip(own,saved)) for r in compare(path+'/'+str(i),a,b,scope)]
    equal = own == saved
    numeric = type(own) in (int,float) and type(saved) in (int,float)
    if numeric and not (math.isfinite(own) and math.isfinite(saved)):
        equal = clean(own) == clean(saved)
    return [{'path':path,'own':own,'saved':saved,'equal':bool(equal),
             'difference':float(own-saved) if numeric and math.isfinite(own) and math.isfinite(saved) else None,'scope':scope}]


def response(row, task, item, cap, identity):
    s.require(type(row['correct']) is bool and isinstance(row['reply'],str), 'invalid saved response')
    out = dict(identity) | {'saved_correct':row['correct'], 'reply':row['reply']}
    out['text_score'] = s.text_score_record(row['reply'],row['correct'],task,item,cap)
    if 'entropy' in row:
        s.require(type(row['entropy']) in (int,float) and type(row['length']) is int, 'invalid saved entropy/length type')
        out.update(entropy=row['entropy'],length=row['length'],entropy_finite=math.isfinite(row['entropy']),length_nonnegative=row['length']>=0)
    return out


def bz_reduce(file_records, pairs, tasks, order, emit, draws=10000):
    s.require(order['tasks']==list(TASKS) and order['metrics']==['d_acc','d_entropy','d_length'], 'BZ order differs')
    cells_def = {c['cell']:c for c in pairs['cells']}
    s.require(len(cells_def)==len(pairs['cells'])==90 and set(cells_def)==set(range(90)), 'BZ declared cell allocation differs')
    s.require([p['pair'] for p in pairs['pairs']]==list(range(30)), 'BZ declared pair order differs')
    s.require([len(tasks[t]) for t in TASKS]==[50,50,30], 'BZ task allocation differs')
    sources = sorted(file_records, key=lambda r:(order['runs'].index('llm-soft-loop/pod64/bz/'+Path(r[0]['path']).parent.name),Path(r[0]['path']).name))
    cells, comparisons, count = {}, [], 0
    for entry, rec in sources:
        meta=entry['metadata']; c=rec['cell']; ident=c['cell']
        s.require(c==meta['cell']==cells_def[ident] and ident not in cells, 'BZ cell identity/duplicate differs')
        s.require(list(rec['groups'])==meta['task_order']==list(TASKS), 'BZ task order differs')
        groups={}
        for task in TASKS:
            group=rec['groups'][task]; rows=group['rows']; m=meta['groups'][task]
            keys=[(r['item'],r['sample']) for r in rows]
            k=2 if task=='mbpp' else 3
            expected=[(i,j) for i in range(len(tasks[task])) for j in range(k)]
            s.require(keys==expected==[tuple(v) for v in m['item_sample_identities']] and len(rows)==m['rows'], 'BZ item/sample coverage differs')
            s.require(all(list(r)==m['row_key_order'] for r in rows), 'BZ response schema differs')
            for r in rows:
                emit(response(r,task,tasks[task][r['item']],400,{'study':'BZ','cell':ident,'pair':c['pair'],'label':c['label'],'task':task,'item':r['item'],'sample':r['sample']}));count+=1
            own={'accuracy':float(np.mean([r['correct'] for r in rows])),
                 'mean_entropy':float(np.mean([r['entropy'] for r in rows])),
                 'mean_length':float(np.mean([r['length'] for r in rows]))}
            for key,value in own.items(): comparisons+=compare(f'BZ/cell/{ident}/{task}/{key}',value,group[key],'raw-response arithmetic')
            groups[task]=own|{'responses':len(rows),'nonfinite_entropy':sum(not math.isfinite(r['entropy']) for r in rows),'negative_length':sum(r['length']<0 for r in rows)}
        cells[ident]={'cell':c,'groups':groups,'source_file':small_pin(entry),'attempt':entry['attempt']}
    s.require(set(cells)==set(cells_def) and count==32400, 'BZ completed coverage differs')
    grouped={}
    for cell in cells.values():
        c=cell['cell'];s.require(c['label'] not in grouped.setdefault(c['pair'],{}),'duplicate BZ pair label')
        grouped[c['pair']][c['label']]=cell
    rng=np.random.default_rng(SEED); literal={'cells':len(cells),'tasks':{}}; diagnostics={}
    for task in TASKS:
        rows=[]
        for pair, group in sorted(grouped.items()):
            s.require(set(group)=={'chaotic','periodic','unsteered'},'BZ missing pair arm')
            c,p,u=[group[g] for g in ('chaotic','periodic','unsteered')]
            s.require(c['cell']['log10_temperature']==p['cell']['log10_temperature']==u['cell']['log10_temperature'],'BZ pair temperature mismatch')
            cg,pg,ug=[v['groups'][task] for v in (c,p,u)]
            rows.append({'pair':pair,'log10_T':c['cell']['log10_temperature'],'acc_chaotic':cg['accuracy'],'acc_periodic':pg['accuracy'],
                         'acc_unsteered':ug['accuracy'],'d_acc':cg['accuracy']-pg['accuracy'],'d_entropy':cg['mean_entropy']-pg['mean_entropy'],'d_length':cg['mean_length']-pg['mean_length']})
        summary={'pairs':len(rows),'rows':rows}; diagnostics[task]={}
        for metric in order['metrics']:
            result=s.paired_bootstrap([r[metric] for r in rows],rng,draws)
            summary[metric]={k:result[k] for k in ('mean','positive','nonzero','sign_test_p','bootstrap_95')}
            diagnostics[task][metric]={k:v for k,v in result.items() if k not in summary[metric]}
        literal['tasks'][task]=summary
    selection=[]
    for p in pairs['pairs']:
        c,q=p['chaotic'],p['periodic'];gap=abs(c['alpha']-q['alpha'])
        selection.append({'pair':p['pair'],'map':p['map'],'log10_temperature':p['log10_temperature'],'alpha_gap':gap,
                          'declared_chaotic_exponent':c['exponent'],'declared_periodic_exponent':q['exponent'],'declared_period':q['period'],
                          'saved_selection_conditions':gap<=.001 and c['exponent']>.05 and q['exponent']<-.05 and q['period']>0,
                          'scope':'Only saved pair-definition predicates checked; source maps/nearest-row/minimum-gap selection not replayed.'})
    return {'literal_comparable':literal,'diagnostics':diagnostics,'cells':list(cells.values()),'selection':selection,
            'cell_summary_comparisons':comparisons,'response_rows':count,
            'scope':'Pair statistics use own raw response means; source uses stored cell means. Every such difference is retained. Sorted pair order fixes BZ resampling despite unavailable historical run CLI.'}


def label_diagnostic(cell):
    ex=cell['lyapunov']; p=cell['period']; cls=cell['class']
    s.require(isinstance(ex,list) and len(ex)==2 and all(type(v) in (int,float) for v in ex), 'BP twin field differs')
    s.require(type(p) in (int,float) and isinstance(cls,str),'BP period/class type differs')
    mapped='chaotic' if 'chaos' in cls and 'irregular' not in cls else 'periodic' if cls=='periodic' else 'other'
    union='periodic' if p>0 else 'chaotic' if all(v>.001 for v in ex) else 'other'
    valid=math.isfinite(p) and p==int(p) and 0<=p<=64 and all(math.isfinite(v) for v in ex)
    return {'literal_analysis_label':mapped,'reconstructed_union':union,'union_equal':mapped==union,'finite_period_qualified':valid,
            'subclass_scope':'Lead-period absent; hidden versus visible subclass is inherited, not reconstructed.'}


def bp_reduce(label_records, generation_entry, generation, tasks, model, order, emit, draws=10000):
    items=tasks['items']; s.require([v['prompt'] for v in items]==list(range(200)), 'BP task allocation differs')
    taskmap={v['prompt']:v for v in items};s.require([v['task'] for v in items]==['arithmetic']*100+['nq']*100,'BP task order differs')
    labels={}; origin={}
    source_order=order['label_runs'][model]
    s.require(len(label_records)==len(source_order),'BP shard count differs')
    for entry,records in sorted(label_records,key=lambda r:source_order.index(Path(r[0]['path']).parent.name)):
        m=entry['metadata'];s.require(len(records)==m['rows'],'BP shard length differs')
        expected=m['identifiers_in_file_order']
        actual=[{'prompt':r['prompt'],'task':r['task'],'temperatures':[c['log10_temperature'] for c in r['cells']]} for r in records]
        s.require(actual==expected,'BP label identifier/order differs')
        for r in records:
            p=r['prompt'];s.require(p not in labels and p in taskmap and r['task']==taskmap[p]['task'],'BP duplicate/unknown label')
            s.require([c['log10_temperature'] for c in r['cells']]==list(TEMPERATURES),'BP temperature coverage differs')
            labels[p]=r;origin[p]=small_pin(entry)
    s.require(set(labels)==set(range(200)),'BP prompt coverage differs')
    gm=generation_entry['metadata'];greedy=generation['greedy'];sampled=generation['sampled']
    s.require(list(greedy)==gm['greedy_prompt_order']==list(map(str,range(200))),'BP greedy allocation/order differs')
    s.require(list(sampled)==gm['sampled_temperature_order']==list(map(str,TEMPERATURES)),'BP sampled temperature order differs')
    s.require(order['temperature_order']==list(TEMPERATURES) and order['subset_order']==['all','greedy_correct'],'BP declared analyze order differs')
    responses=0
    for pstr,r in greedy.items():
        p=int(pstr);emit(response(r,taskmap[p]['task'],taskmap[p],200,{'study':'BP','model':model,'prompt':p,'task':taskmap[p]['task'],'arm':'greedy'}));responses+=1
    canonical_rows=[];literal={};permutation_details={};task_tables={};rng=np.random.default_rng(SEED)
    for lt in TEMPERATURES:
        key=str(lt); sm=gm['sampled'][key]
        s.require(list(sampled[key])==sm['prompt_order']==list(map(str,range(200))),'BP sampled prompt order differs')
        rows=[]
        for p,record in labels.items():
            item=taskmap[p];samples=sampled[key][str(p)]
            s.require(len(samples)==sm['sample_counts'][str(p)]==10,'BP repeated response allocation differs')
            for j,r in enumerate(samples):
                emit(response(r,item['task'],item,200,{'study':'BP','model':model,'prompt':p,'task':item['task'],'arm':'sampled','log10_temperature':lt,'sample':j}));responses+=1
            cell=next(c for c in record['cells'] if c['log10_temperature']==lt)
            diag=label_diagnostic(cell)
            row={'prompt':p,'task':item['task'],'label':diag['literal_analysis_label'],'greedy':greedy[str(p)]['correct'],
                 'acc':float(np.mean([v['correct'] for v in samples])),
                 'generation_entropy':float(np.mean([v['entropy'] for v in samples])),
                 'generation_length':float(np.mean([v['length'] for v in samples])),
                 'saved_cell':cell,'label_diagnostic':diag,'label_origin':origin[p],
                 'prefix_tokens':record['prefix_tokens'],'sample_count':len(samples),'model':model,'log10_temperature':lt}
            rows.append(row)
        res={};permutation_details[key]={};task_tables[key]={}
        for subset in order['subset_order']:
            keep=[r for r in rows if (subset=='all' or r['greedy']) and r['label'] in ('chaotic','periodic')]
            result=s.prompt_permutation([r['acc'] for r in keep if r['label']=='chaotic'],[r['acc'] for r in keep if r['label']=='periodic'],rng,draws)
            permutation_details[key][subset]=result
            if result['defined']:res[subset]={k:result[k] for k in ('chaotic','periodic','acc_chaotic','acc_periodic','difference','permutation_p')}
            task_tables[key][subset]=s.task_composition(keep)
        res['label_counts']={label:sum(r['label']==label for r in rows) for label in ('chaotic','periodic','other')}
        literal[key]=res;canonical_rows+=rows
    s.require(responses==6200 and len(canonical_rows)==600,'BP complete row count differs')
    return {'literal_comparable':literal,'prompt_rows':canonical_rows,'permutation_details':permutation_details,
            'task_composition':task_tables,'response_rows':responses,'canonical_prompt_order':list(labels),
            'scope':'Canonical supplied-shard then file-line order, not historically attested CLI order. Prompt-unit permutation fractions may differ because original label insertion order is unavailable. No adjusted-regression replay.'}


def runtime_pins(card, inputs, source_map):
    rows=[card['inputs']]+[inputs[k] for k in ('authority','source_map','analysis_order')]+inputs['files']+inputs['published']+source_map['source_files']+source_map['claims']
    unique={}
    for r in rows:
        p=small_pin(r);s.require(p['path'] not in unique or unique[p['path']]==p,'conflicting duplicate pin')
        unique[p['path']]=p
    return list(unique.values())


def execute(card_path, output):
    started=time.monotonic();card=decode(card_path.read_text())
    s.require(card['admission']=='reviewed' and output.resolve()==Path(card['output']).resolve(),'prepared card/wrong output')
    for name,digest in card['source_sha256'].items():s.require(sha(HERE/name)==digest,'captured own source mismatch')
    verify(card['inputs']);inputs=decode(Path(card['inputs']['path']).read_text())
    verify(inputs['source_map']);source_map=decode(Path(inputs['source_map']['path']).read_text())
    pins=runtime_pins(card,inputs,source_map)
    s.require(inputs['target_files']==len(inputs['files'])==100 and inputs['target_bytes']==10231893,'fixed target inventory differs')
    for p in pins:verify(p)
    order=decode(Path(inputs['analysis_order']['path']).read_text())
    def source(repo,commit):
        rows=[p for p in source_map['source_files'] if p.get('repository_path')==repo and p.get('source_commit','').startswith(commit)]
        s.require(len(rows)==1,'ambiguous task source');return decode(Path(rows[0]['path']).read_text())
    pairs=source('docs/bz-pairs.json','9113b64e');bz_tasks=source('docs/bz-tasks.json','9113b64e');bp_tasks=source('docs/bp-tasks.json','6a7c7ac1')
    output.mkdir(parents=False,exist_ok=False)
    try:
        files=inputs['files']; bz_entries=[r for r in files if r['kind']=='bz-cell'];labels=[r for r in files if r['kind']=='bp-label'];gens=[r for r in files if r['kind']=='bp-generation']
        s.require((len(bz_entries),len(labels),len(gens))==(90,8,2),'input kinds differ')
        with open(output/'responses.jsonl','x') as f:
            def emit(row):f.write(json.dumps(clean(row),allow_nan=False)+'\n')
            bz=bz_reduce([(r,decode(Path(r['path']).read_text())) for r in bz_entries],pairs,bz_tasks,order['bz'],emit)
            bp={}
            for model in ('q05','q7'):
                lab=[(r,[decode(line) for line in Path(r['path']).read_text().splitlines()]) for r in labels if Path(r['path']).parent.name.startswith(model+'-')]
                g=[r for r in gens if Path(r['path']).parent.name==model+'-gen'];s.require(len(g)==1,'BP generation identity differs')
                bp[model]=bp_reduce(lab,g[0],decode(Path(g[0]['path']).read_text()),bp_tasks,model,order['bp'],emit)
        comparisons=[]
        for row in inputs['published']:
            name=Path(row['path']).name
            own=bz['literal_comparable'] if name=='bz-analysis.json' else bp[{'bp-q05.json':'q05','bp-q7.json':'q7'}[name]]['literal_comparable']
            comparisons+=compare(name,own,decode(Path(row['path']).read_text()),'Saved-field arithmetic with declared canonical source RNG order; missing historical analyze invocation not invented.')
        report={'schema':'task21i-independent-saved-behavior-v1','computational_status':'complete','source_sha256':card['source_sha256'],'input_pins':pins,
                'authority':inputs['authority'],'analysis_order':order,'bz':bz,'bp':bp,'published_comparisons':comparisons,
                'published_comparison_count':len(comparisons),'published_nonexact_count':sum(not r['equal'] for r in comparisons),
                'coverage':{'target_files':100,'BZ_cells':90,'BZ_pairs':30,'BZ_responses':bz['response_rows'],'BP_models':2,'BP_label_prompts':400,'BP_prompt_temperature_rows':1200,'BP_responses':sum(r['response_rows'] for r in bp.values()),'all_response_rows':44800},
                'historical_adjustment':{'disposition':'cannot check exact estimator','basis':order['ad_hoc_adjustment'],'no_inferred_historical_formula':True},
                'scope':'Original BZ/BP only; BPC/BPCS exposed context, not analyzed. No model, native label recomputation, generated-code execution or causal adjustment. Prior summaries disclosed.',
                'runtime':{'seconds_before_posthash_report_write':time.monotonic()-started,'observed_peak_kib_before_posthash_report_write':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           'scope':'Includes response serialization; excludes final pin rehash/report/manifest serialization; supervisor observes full worker wall, not full peak RSS.'}}
        for p in pins:verify(p)
        write_json(output/'report.json',report)
        write_json(output/'manifest.json',{'computational_status':'complete','files':[pin(output/n) for n in ('responses.jsonl','report.json')]})
    except BaseException as exc:
        write_json(output/'failure.json',{'type':type(exc).__name__,'message':str(exc)});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--card',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(a.card,a.out)
