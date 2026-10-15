"""One bounded literal Task21i presentation; no scientific recomputation."""
import argparse, collections, hashlib, json, os, resource, time, traceback
from pathlib import Path

BASE = Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21i/run-001')
OUTPUT = Path('/tmp/task21i-saved-read-001')
CAP = 128 * 1024**2
EXPECTED = {'target_files':100,'BZ_cells':90,'BZ_pairs':30,'BZ_responses':32400,
            'BP_models':2,'BP_label_prompts':400,'BP_prompt_temperature_rows':1200,
            'BP_responses':12400,'all_response_rows':44800}


def key(value):
    return json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)


def main():
    ap = argparse.ArgumentParser()
    for name in ('report','responses','manifest','authority'):
        ap.add_argument('--'+name+'-sha', required=True)
    args = ap.parse_args()
    pins = {n:getattr(args,n+'_sha') for n in ('report','responses','manifest','authority')}
    if any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in pins.values()):
        raise ValueError('exact completed-output and authority SHA256 arguments required')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(50,50))
    n = os.getpriority(os.PRIO_PROCESS,0)
    if n > 10: raise RuntimeError('exact nice10 unavailable')
    if n < 10: os.nice(10-n)
    OUTPUT.mkdir(exist_ok=False)
    start = time.perf_counter(); written = 0

    def write(stream, data):
        nonlocal written
        if written + len(data) > CAP: raise RuntimeError('literal-presentation output ceiling')
        stream.write(data); written += len(data)

    def line(stream, obj):
        write(stream, (json.dumps(obj, allow_nan=False, separators=(',',':'))+'\n').encode())

    def emit(name, obj):
        with (OUTPUT/name).open('xb') as stream: line(stream,obj)

    def bound(path, digest):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest: raise ValueError('released saved-file hash: '+str(path))
        return data

    def flag(counter, name, value):
        counter.setdefault(name,collections.Counter())[key(value)] += 1

    try:
        # Only the coordinator-released saved output/authority, not original target inputs.
        ab = bound(BASE.parent/'run-001-authority/report.json',pins['authority'])
        rb = bound(BASE/'report.json',pins['report'])
        mb = bound(BASE/'manifest.json',pins['manifest'])
        report = json.loads(rb); oldmanifest = json.loads(mb)
        if report['schema']!='task21i-independent-saved-behavior-v1' or report['computational_status']!='complete':
            raise ValueError('completed source schema required')
        if report['coverage']!=EXPECTED: raise ValueError('released coverage contract')
        if oldmanifest['computational_status']!='complete' or len(oldmanifest['files'])!=2:
            raise ValueError('original two-file manifest contract')
        indexed = {Path(p['path']).name:p for p in oldmanifest['files']}
        if set(indexed)!={'report.json','responses.jsonl'}: raise ValueError('original output identities')
        for name,data in [('literal-report.json',rb),('literal-worker-manifest.json',mb),('literal-authority.json',ab)]:
            with (OUTPUT/name).open('xb') as stream: write(stream,data)

        groups = {}; seen = set(); count = 0; size = 0; digest = hashlib.sha256(); exceptions = 0
        with (BASE/'responses.jsonl').open('rb') as src, (OUTPUT/'literal-responses.jsonl').open('xb') as dst, (OUTPUT/'response-flag-exceptions.jsonl').open('xb') as exc:
            for raw in src:
                digest.update(raw); size += len(raw); write(dst,raw); row = json.loads(raw); count += 1
                if row['study']=='BZ':
                    identity = ('BZ',row['cell'],row['task'],row['item'],row['sample'])
                    group = ('BZ',row['task'],row['label'])
                elif row['study']=='BP':
                    identity = ('BP',row['model'],row['arm'],row.get('log10_temperature'),row['prompt'],row.get('sample'))
                    group = ('BP',row['model'],row['arm'],row.get('log10_temperature'),row['task'])
                else: raise ValueError('unknown saved response study')
                if identity in seen: raise ValueError('duplicate saved presentation identity')
                seen.add(identity)
                item = groups.setdefault(key(group),{'records':0,'saved_fields':{}}); item['records'] += 1
                score = row['text_score']
                flag(item['saved_fields'],'saved_correct',row['saved_correct'])
                for name in ('own_score','equal','full_text_available_from_length','comparison'):
                    flag(item['saved_fields'],'text_score/'+name,score[name] if name in score else {'absent':True})
                for name in ('entropy_finite','length_nonnegative'):
                    flag(item['saved_fields'],name,row[name] if name in row else {'absent':True})
                unusual = (score.get('equal') is False or score.get('own_score') is None or
                           score.get('full_text_available_from_length') is False or
                           row.get('entropy_finite') is False or row.get('length_nonnegative') is False)
                if unusual:
                    line(exc,{'identity':identity,'record':row}); exceptions += 1
        if count!=44800 or digest.hexdigest()!=pins['responses']:
            raise ValueError('released response inventory/hash')
        actual = {'report.json':{'bytes':len(rb),'sha256':pins['report']},
                  'responses.jsonl':{'bytes':size,'sha256':digest.hexdigest()}}
        for name,p in indexed.items():
            if p['path']!=str(BASE/name) or p['bytes']!=actual[name]['bytes'] or p['sha256']!=actual[name]['sha256']:
                raise ValueError('saved output-manifest association '+name)

        bz = report['bz']
        if len(bz['cells'])!=90 or len(bz['selection'])!=30 or set(bz['literal_comparable']['tasks'])!={'arithmetic','nq','mbpp'}:
            raise ValueError('saved BZ summary coverage')
        for task,summary in bz['literal_comparable']['tasks'].items():
            if len(summary['rows'])!=30 or set(bz['diagnostics'][task])!={'d_acc','d_entropy','d_length'}:
                raise ValueError('saved BZ pair/metric coverage')
        selection_flags = collections.Counter(key(r['saved_selection_conditions']) for r in bz['selection'])
        cell_leaves = bz['cell_summary_comparisons']; published = report['published_comparisons']

        bp = {}; label_inventory = {}; prompt_count = 0
        if set(report['bp'])!={'q05','q7'}: raise ValueError('saved BP model identities')
        with (OUTPUT/'bp-prompt-fields.jsonl').open('xb') as stream:
            for model,entry in report['bp'].items():
                if len(entry['prompt_rows'])!=600: raise ValueError('saved BP prompt coverage')
                bp[model] = {k:v for k,v in entry.items() if k!='prompt_rows'}
                seen_prompts = set()
                for row in entry['prompt_rows']:
                    identity = (row['prompt'],row['log10_temperature'])
                    if identity in seen_prompts: raise ValueError('duplicate saved BP presentation identity')
                    seen_prompts.add(identity); prompt_count += 1; line(stream,row)
                    group = key((model,row['log10_temperature'],row['task']))
                    inv = label_inventory.setdefault(group,{'records':0,'saved_fields':{}}); inv['records'] += 1
                    for name,value in row['label_diagnostic'].items(): flag(inv['saved_fields'],name,value)
                    flag(inv['saved_fields'],'literal_label',row['label'])
        if prompt_count!=1200: raise ValueError('saved BP total coverage')

        adj = report['historical_adjustment']; adjusted = {}; fit_count = 0
        if set(adj['temperatures'])!={'0.25','0.32','0.4'} or adj['total_draws']!=15000:
            raise ValueError('saved adjustment coverage')
        with (OUTPUT/'ols-fits.jsonl').open('xb') as stream:
            for temperature,item in adj['temperatures'].items():
                draws = item['permutations']
                if len(draws)!=5000 or [r['draw'] for r in draws]!=list(range(5000)):
                    raise ValueError('saved OLS draw inventory')
                inv = {}; bad = []
                for row in draws:
                    line(stream,{'temperature':temperature,'record':row}); fit_count += 1
                    for name in ('defined','rank','full_column_rank','reason'): flag(inv,name,row[name])
                    if row['defined'] is not True or row['full_column_rank'] is not True: bad.append(row)
                adjusted[temperature] = {k:v for k,v in item.items() if k!='permutations'}
                adjusted[temperature]['saved_fit_flag_inventory'] = inv
                adjusted[temperature]['all_undefined_or_rank_deficient_fits'] = bad
        printed = adj['printed_comparisons']
        if fit_count!=15000 or len(printed['fields'])!=120 or printed['count']!=120:
            raise ValueError('saved OLS/printed-field coverage')
        emit('review-summary.json',{
            'scope':'Literal completed-output copies, saved-field occurrence inventories and selections only. No response scoring, label recomputation, means, regression, bootstrap, permutation, quantile or original-input audit.',
            'released_pins':pins,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'coverage':report['coverage'],'response_inventory':{'records':count,'groups':groups,'exception_records':exceptions,
                'exception_scope':'Saved disagreement, unavailable MBPP score, possibly truncated prefix, nonfinite entropy or negative-length flag; not all are scientific failures.'},
            'all_BZ':bz,'BZ_saved_selection_flags':selection_flags,
            'BZ_cell_comparison_exceptions':[r for r in cell_leaves if r['equal'] is not True or r.get('difference') not in (None,0)],
            'all_BP_summaries':bp,'BP_prompt_label_inventory':label_inventory,
            'all_adjustment_summaries':adjusted,'adjustment_other_metadata':{k:v for k,v in adj.items() if k not in ('temperatures','printed_comparisons')},
            'all_printed_comparisons':printed,'printed_comparison_exceptions':[r for r in printed['fields'] if r['equal_at_printed_precision'] is not True],
            'all_published_comparisons':published,
            'published_comparison_exceptions':[r for r in published if r['equal'] is not True or r.get('difference') not in (None,0)],
            'reported_published_comparison_count':report['published_comparison_count'],
            'reported_published_nonexact_count':report['published_nonexact_count'],
            'original_runtime':report['runtime'],'historical_order_amendment':report['historical_order_amendment'],
            'elapsed_before_summary_write':time.perf_counter()-start,
            'rss_kib_observed_before_summary_write':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        entries = []
        for path in sorted(OUTPUT.iterdir()):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for data in iter(lambda:stream.read(1024**2),b''): h.update(data)
            entries.append({'path':path.name,'bytes':path.stat().st_size,'sha256':h.hexdigest()})
        emit('manifest.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':entries,
                             'scope':'Presentation files only; original scientific result and authority unchanged.'})
        print(json.dumps({'completed':True,'response_records':count,'BP_prompt_records':prompt_count,
                          'OLS_fit_records':fit_count,'output_bytes':written,'elapsed_before_print':time.perf_counter()-start}))
    except BaseException:
        with (OUTPUT/'failure.json').open('x') as stream:
            json.dump({'error':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start},stream)
        raise


if __name__=='__main__':main()
