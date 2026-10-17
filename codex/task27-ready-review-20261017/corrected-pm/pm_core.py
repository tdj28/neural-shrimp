"""Independent saved-row PM/PM2 arithmetic; no producer imports or model work.

Literal endpoints use the producer's stored answer/correct fields, after retaining
every extraction/label diagnostic. Suffix/byte-marker views are explicit sensitivity
counts, never replacements for frozen endpoints. RNG order uses the frozen published-summary insertion order as a disclosed proxy;
no historical analyzer CLI is attested and no favorable-order search is allowed.
"""
import json
import math
import re
from collections import Counter, OrderedDict

import numpy as np

SEED = 20261016
DRAWS = 10000
NUMBER = re.compile(r'-?\d[\d,]*\.?\d*')


def number_string(s):
    s = s.replace(',', '').rstrip('.')
    try:
        f = float(s)
    except ValueError:
        return None
    if not math.isfinite(f):
        return None
    return str(int(f)) if f.is_integer() else str(f)


def answer(text, kind):
    if kind == 'pm':
        at = text.lower().rfind('answer is')
        found = NUMBER.findall(text[at + 9:] if at >= 0 else text)
        return number_string(found[0] if at >= 0 else found[-1]) if found else None
    if kind != 'pm2':
        raise ValueError('unknown extraction kind')
    at = text.rfind('\\boxed{')
    if at >= 0:
        start = at + 7
        depth = 1
        stop = None
        for i in range(start, len(text)):
            depth += int(text[i] == '{') - int(text[i] == '}')
            if depth == 0:
                stop = i
                break
        if stop is not None:
            s = re.sub(r'\\text\{[^}]*\}|\\!|\\,|\$|\s|,', '', text[start:stop])
            s = s.replace('\\%', '').replace('%', '')
            return str(int(float(s))) if re.fullmatch(r'-?\d+(\.0+)?', s) else s
    found = NUMBER.findall(text)
    return number_string(found[-1]) if found else None


def byte_marker_text(text):
    """Minimal disclosed diagnostic only: ByteLevel space/newline/tab markers.

    This does not claim full tokenizer reconstruction or recover a missing prefix.
    Marker-present source and transformed text remain distinguishable.
    """
    return text.translate(str.maketrans({'Ġ': ' ', 'Ċ': '\n', 'ĉ': '\t'}))


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959964
    p = successes / total
    den = 1 + z*z/total
    center = (p + z*z/(2*total)) / den
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return [center-half, center+half]


def group_counts(rows, key, order='sorted'):
    groups = OrderedDict()
    indices = sorted(rows) if order == 'sorted' else rows
    for idx in indices:
        r = rows[idx]
        c = r['cluster']
        if c not in groups:
            groups[c] = [0, 0]
        groups[c][0] += r[key]
        groups[c][1] += 1
    return groups


def boot(groups, rng):
    if not groups:
        return None
    values = np.asarray(list(groups.values()), dtype=float)
    draw = rng.integers(len(values), size=(DRAWS, len(values)))
    means = values[draw, 0].sum(axis=1) / values[draw, 1].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def pairwise(x, y, key, rng):
    ids = sorted(x.keys() & y.keys())
    groups = OrderedDict()
    b = c = total = 0
    for i in ids:
        u, v = bool(x[i][key]), bool(y[i][key])
        b += int(u and not v)
        c += int(v and not u)
        delta = int(u) - int(v)
        total += delta
        cluster = x[i]['cluster']
        if cluster not in groups:
            groups[cluster] = [0, 0]
        groups[cluster][0] += delta
        groups[cluster][1] += 1
    return {'problems': len(ids), 'difference': total/len(ids) if ids else None,
            'cluster95': boot(groups, rng), 'discordant': [b, c],
            'mcnemar_p': exact_mcnemar(b, c)}, {
                'paired_ids': ids, 'left_only': sorted(x.keys()-y.keys()),
                'right_only': sorted(y.keys()-x.keys()),
                'left_cluster_sizes': [{'cluster': k, 'n': v[1]} for k,v in groups.items()],
                'gold_mismatch_ids': [i for i in ids if x[i]['gold'] != y[i]['gold']],
                'cluster_id_mismatch_ids': [i for i in ids if x[i]['cluster'] != y[i]['cluster']]}


def arm_stats(rows, layouts, rng):
    n = len(rows)
    flips = sum(r['flip'] for r in rows.values())
    groups = group_counts(rows, 'flip')
    return {'problems': n, 'clusters': len(groups), 'answer_flips': flips,
            'flip_rate': flips/n if n else None, 'flip_wilson95': wilson(flips, n),
            'flip_cluster95': boot(groups, rng),
            'accuracy': {str(layouts[j]): sum(r['correct_'+suffix] for r in rows.values())/n if n else None
                         for j,suffix in enumerate(['a','b'])},
            'exactly_one_correct': sum(r['correct_a'] != r['correct_b'] for r in rows.values())/n if n else None,
            'missing_answers': {str(layouts[j]): sum(r['missing_'+suffix] for r in rows.values())
                                for j,suffix in enumerate(['a','b'])}}


def load_rows(records_path, cfg, kind):
    """Keep source last-write semantics, plus every duplicate and incomplete index."""
    layouts = tuple(sorted(cfg['layouts'])[:2])
    if len(layouts) != 2 or layouts[0] == layouts[1]:
        raise ValueError('exactly two distinct layout values required')
    data = OrderedDict()
    duplicates, record_checks, excluded, unexpected = [], [], [], []
    for line_number,line in enumerate(open(records_path), 1):
        r = json.loads(line)
        for field in ['index','layout','batch','row','prompt_len']:
            if type(r[field]) is not int:
                raise ValueError(f'{line_number}: noninteger {field}')
        if type(r['correct']) is not bool or not isinstance(r['text'],str):
            raise ValueError(f'{line_number}: invalid label/text schema')
        if r['answer'] is not None and not isinstance(r['answer'], str):
            raise ValueError(f'{line_number}: answer must be string or null')
        if not isinstance(r['gold'], str):
            raise ValueError(f'{line_number}: gold must be string')
        identity = {'line':line_number,'index':r['index'],'layout':r['layout']}
        raw = answer(r['text'], kind)
        normalized = answer(byte_marker_text(r['text']), kind)
        label = r['answer'] is not None and r['answer'] == r['gold']
        record_checks.append({**identity,'stored_answer':r['answer'],'gold':r['gold'],
            'stored_correct':r['correct'],'label_from_stored_answer':label,'stored_label_consistent':label==r['correct'],
            'suffix_extracted_answer':raw,'suffix_extraction_matches':raw==r['answer'],
            'marker_normalized_answer':normalized,'marker_normalized_correct':normalized is not None and normalized==r['gold'],
            'markers':{c:r['text'].count(c) for c in ['Ġ','Ċ','ĉ']},
            'saved_text_chars':len(r['text']),
            'may_be_truncated_suffix':len(r['text']) >= (400 if kind=='pm' else 600)})
        if r['index'] < 0:
            excluded.append(identity)
            continue
        if r['layout'] not in layouts:
            unexpected.append(identity)
        versions = data.setdefault(r['index'], OrderedDict())
        if r['layout'] in versions:
            old = versions[r['layout']]
            duplicates.append({**identity,'previous_line':old['_line'],
                'equal_saved_fields':{k:old.get(k)==r.get(k) for k in r},
                'old_answer':old['answer'],'new_answer':r['answer'],
                'old_correct':old['correct'],'new_correct':r['correct']})
        versions[r['layout']] = {**r,'_line':line_number,'_suffix_answer':raw,'_marker_answer':normalized}
    rows = OrderedDict(); row_ledger = []; unpaired = []
    for i,v in data.items():
        if any(L not in v for L in layouts):
            unpaired.append({'index':i,'present_layouts':list(v)})
            continue
        a,b = (v[L] for L in layouts)
        row = {'flip':a['answer'] != b['answer'],'correct_a':a['correct'],'correct_b':b['correct'],
               'missing_a':a['answer'] is None,'missing_b':b['answer'] is None,
               'cluster':b['batch'],'gold':a['gold']}
        rows[i]=row
        ta,tb = a.get('top1',[]),b.get('top1',[])
        first = next((j+1 for j,(u,w) in enumerate(zip(ta,tb)) if u!=w),None)
        row_ledger.append({'index':i,**row,'lines':[a['_line'],b['_line']],
            'answer_a':a['answer'],'answer_b':b['answer'],
            'gold_match':a['gold']==b['gold'],'prompt_length_match':a['prompt_len']==b['prompt_len'],
            'suffix_flip':a['_suffix_answer']!=b['_suffix_answer'],
            'marker_normalized_flip':a['_marker_answer']!=b['_marker_answer'],
            'marker_normalized_correct_a':a['_marker_answer'] is not None and a['_marker_answer']==a['gold'],
            'marker_normalized_correct_b':b['_marker_answer'] is not None and b['_marker_answer']==b['gold'],
            'top1_common_prefix_first_different_step_1based':first,
            'top1_lengths':[len(ta),len(tb)],
            'top1_clock_scope':'saved per-step top1 readout only; not native state distance or necessarily answer phase'})
    big_records = [v[layouts[1]] for v in data.values() if layouts[1] in v]
    cluster_ledger=[]
    for c in dict.fromkeys(r['cluster'] for r in rows.values()):
        ids=[i for i,r in rows.items() if r['cluster']==c]
        lengths=[data[i][layouts[1]]['prompt_len'] for i in ids]
        positions=[data[i][layouts[1]]['row'] for i in ids]
        cluster_ledger.append({'cluster':c,'indices':ids,'n':len(ids),
            'prompt_lengths':lengths,'one_prompt_length':len(set(lengths))==1,
            'row_positions':positions,'row_positions_unique':len(set(positions))==len(positions)})
    extra={'forced_switch':sum(bool(r.get('forced_switch',False)) for r in big_records),
           'switch_steps':[r['switch_step'] for r in big_records if r.get('switch_step') is not None],
           'larger_layout_record_ids_including_unpaired':[r['index'] for r in big_records]}
    return rows,layouts,extra,{'record_checks':record_checks,'paired_rows':row_ledger,
        'duplicates_last_write_retained':duplicates,'unpaired':unpaired,'excluded_negative_index_rows':excluded,
        'unexpected_layout_rows':unexpected,'clusters':cluster_ledger,
        'record_count':len(record_checks),'source_index_encounter_order':list(data),
        'scope':'Stored answer/correct endpoints retained. Extraction of a saved suffix cannot recover discarded full text; byte-marker mapping is a separate diagnostic.'}


def analyze_pm(arms):
    rng=np.random.default_rng(SEED); projection={'arms':{},'contrasts':{},'primary':None}; diagnostics={}
    for name,x in arms.items():
        a=arm_stats(x['rows'],x['layouts'],rng);cfg=x['cfg']
        projection['arms'][name]={'run':cfg.get('out'),'mode':cfg.get('mode',cfg.get('arm')),
            'top_n':cfg.get('top_n'),'soft_steps':cfg.get('soft_steps'),'temperature':cfg['temperature'],
            'layouts':list(x['layouts']),**a}
    if 'top15-s64' in arms:
        a=projection['arms']['top15-s64'];ci=a['flip_cluster95']
        projection['primary']={'arm':'top15-s64','flip_rate':a['flip_rate'],'cluster95':ci,
            'confirmed':bool(a['flip_rate'] is not None and ci is not None and a['flip_rate']>.10 and ci[0]>.05)}
        for name,x in arms.items():
            if name=='top15-s64':continue
            result,detail=pairwise(arms['top15-s64']['rows'],x['rows'],'flip',rng)
            if not result['problems']:continue
            result['flip_rate_difference']=result.pop('difference')
            projection['contrasts']['top15-s64 - '+name]=result
            diagnostics[name]=detail
    return projection,diagnostics


def analyze_pm2(cells,cross):
    rng=np.random.default_rng(SEED); projection={'cells':{},'primary':None,'cross_gpu':{}};diagnostics={}
    for name,c in cells.items():
        s,p=c['soft'],c['plain'];a=arm_stats(s['rows'],s['layouts'],rng);b=arm_stats(p['rows'],p['layouts'],rng)
        a.update(forced_switch=s['extra']['forced_switch'],median_switch_step=float(np.median(s['extra']['switch_steps'])) if s['extra']['switch_steps'] else None)
        b['forced_switch']=p['extra']['forced_switch']
        fl,fd=pairwise(s['rows'],p['rows'],'flip',rng);acc,ad=pairwise(s['rows'],p['rows'],'correct_b',rng)
        projection['cells'][name]={'role':c['role'],'model':s['cfg']['model'],'task':s['cfg']['task'],
            'layouts':list(s['layouts']),'soft':a,'plain':b,'soft_minus_plain_flips':fl,
            'soft_minus_plain_accuracy_larger_layout':acc}
        diagnostics[name]={'flip_pairing':fd,'accuracy_pairing':ad}
    primary=[(n,c) for n,c in cells.items() if c['role']=='primary']
    if primary:
        groups=[np.asarray(list(group_counts(c['soft']['rows'],'flip','encounter').values()),dtype=float) for _,c in primary]
        if any(len(v)==0 for v in groups):
            projection['primary']={'status':'undefined_empty_primary_cell','cells':[n for n,_ in primary]}
        else:
            total=sum(float(v[:,1].sum()) for v in groups);flips=sum(float(v[:,0].sum()) for v in groups)
            samples=[]
            for _ in range(DRAWS):
                numerator=denominator=0.0
                for v in groups:
                    take=rng.integers(len(v),size=len(v))
                    numerator+=sum(v[j,0] for j in take);denominator+=sum(v[j,1] for j in take)
                samples.append(numerator/denominator)
            ci=[float(v) for v in np.quantile(samples,[.025,.975])];rate=flips/total
            projection['primary']={'cells':[n for n,_ in primary],'problems':int(total),'answer_flips':int(flips),
                'flip_rate':rate,'cluster95':ci,'confirmed':bool(rate>.05 and ci[0]>.02)}
    for a,b in cross:
        key=a+','+b;projection['cross_gpu'][key]={};diagnostics[key]={}
        for mode in ['soft','plain']:
            result,detail=pairwise(cells[a][mode]['rows'],cells[b][mode]['rows'],'flip',rng)
            projection['cross_gpu'][key][mode]=result;diagnostics[key][mode]=detail
    return projection,diagnostics


def compare_leaves(own,published,path=''):
    if isinstance(own,dict) and isinstance(published,dict):
        return [q for k in sorted(own.keys()|published.keys()) for q in
                compare_leaves(own[k],published[k],path+'/'+str(k))] if own.keys()==published.keys() else [
            q for k in sorted(own.keys()|published.keys()) for q in
            (compare_leaves(own[k],published[k],path+'/'+str(k)) if k in own and k in published else
             [{'path':path+'/'+str(k),'status':'own_only' if k in own else 'published_only','own':own.get(k),'published':published.get(k)}])]
    if isinstance(own,list) and isinstance(published,list):
        return [q for i in range(max(len(own),len(published))) for q in
            (compare_leaves(own[i],published[i],path+'/'+str(i)) if i<len(own) and i<len(published) else
             [{'path':path+'/'+str(i),'status':'length_mismatch','own':own[i] if i<len(own) else None,'published':published[i] if i<len(published) else None}])]
    out={'path':path,'own':own,'published':published,'status':'exact' if own==published else 'different'}
    if type(own) in (int,float) and type(published) in (int,float):out['own_minus_published']=own-published
    return [out]
