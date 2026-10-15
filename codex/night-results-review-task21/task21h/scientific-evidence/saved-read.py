"""One bounded literal Task21h output presentation; no scientific recomputation."""
import argparse, collections, hashlib, json, os, resource, time, traceback
from pathlib import Path
BASE = Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21h/run-001')
OUTPUT = Path('/tmp/task21h-saved-read-001')
CAP = 128 * 1024**2
CELL_FLAGS = ('literal_chaotic','reconstructed_chaotic','finite_valid','valid_period')
JOIN_FLAGS = ('plane_chaotic','a1_chaotic','a2_chaotic','finite_valid_triplet','near_a1_chaotic','near_a2_chaotic',
              'exact_controls_nonchaotic','exact_controls_periodic','near_axis_nonchaotic_all_plane',
              'near_axis_periodic_all_plane','near_axis_nonchaotic_excluding_nearest_row_column')


def main():
    ap = argparse.ArgumentParser()
    for name in ('report','cells','joins','manifest','authority'):
        ap.add_argument('--'+name+'-sha', required=True)
    a = ap.parse_args()
    pins = {n:getattr(a,n+'_sha') for n in ('report','cells','joins','manifest','authority')}
    if any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in pins.values()):
        raise ValueError('exact completed-output and authority SHA256 arguments required')
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(50,50))
    n=os.getpriority(os.PRIO_PROCESS,0)
    if n>10: raise RuntimeError('exact nice10 unavailable')
    if n<10: os.nice(10-n)
    OUTPUT.mkdir(exist_ok=False)
    start=time.perf_counter(); written=0
    def write(f,b):
        nonlocal written
        if written+len(b)>CAP: raise RuntimeError('literal-presentation output ceiling')
        f.write(b); written+=len(b)
    def emit(name,obj):
        with (OUTPUT/name).open('xb') as f:
            write(f,(json.dumps(obj,allow_nan=False,separators=(',',':'))+'\n').encode())
    def read_bound(path,digest):
        b=path.read_bytes()
        if hashlib.sha256(b).hexdigest()!=digest: raise ValueError('released saved-file hash: '+str(path))
        return b
    try:
        # This binds coordinator-released output authority; no original input/archive audit.
        ab=read_bound(BASE.parent/'run-001-authority/report.json',pins['authority'])
        rb=read_bound(BASE/'report.json',pins['report'])
        mb=read_bound(BASE/'manifest.json',pins['manifest'])
        report=json.loads(rb); oldmanifest=json.loads(mb)
        if report['schema']!='task21h-independent-saved-grid-v1' or report['computational_status']!='complete':
            raise ValueError('completed source schema required')
        expected={'original_runs':15,'source_grids':21,'source_cells':119552,'control_join_cells':36864}
        if report['coverage']!=expected: raise ValueError('released coverage contract')
        for name,b in [('literal-report.json',rb),('literal-worker-manifest.json',mb),('literal-authority.json',ab)]:
            with (OUTPUT/name).open('xb') as f: write(f,b)
        outputs=oldmanifest['files']
        if oldmanifest['computational_status']!='complete' or len(outputs)!=3:
            raise ValueError('original three-file manifest contract')
        indexed={Path(p['path']).name:p for p in outputs}
        if set(indexed)!={'report.json','cells.jsonl','control-joins.jsonl'}:
            raise ValueError('original output identities')
        actual={'report.json':{'bytes':len(rb),'sha256':pins['report']}}
        inventories={}
        for filename,key,flagfields,total,groups in [
            ('cells.jsonl','grid',CELL_FLAGS,119552,21),
            ('control-joins.jsonl','comparison',JOIN_FLAGS,36864,4)]:
            label='cells' if key=='grid' else 'joins'; h=hashlib.sha256(); size=0; count=0
            by={}; seen=set()
            with (BASE/filename).open('rb') as src,(OUTPUT/('literal-'+filename)).open('xb') as dst:
                for line in src:
                    h.update(line);size+=len(line);write(dst,line);z=json.loads(line);count+=1
                    ident=(z[key],z['row'],z['column'])
                    if ident in seen: raise ValueError('duplicate saved presentation identity')
                    seen.add(ident)
                    q=by.setdefault(z[key],{'records':0,'saved_flag_patterns':collections.Counter(),
                                            'saved_flags':{k:collections.Counter() for k in flagfields},
                                            'literal_nonfinite_tokens':collections.Counter()})
                    q['records']+=1
                    if any(type(z[k]) is not bool for k in flagfields): raise ValueError('saved flag type changed')
                    q['saved_flag_patterns'][''.join('1' if z[k] else '0' for k in flagfields)]+=1
                    for k in flagfields:q['saved_flags'][k][str(z[k])]+=1
                    for k in ('lyapunov','period') if key=='grid' else ('a1_period','a2_period'):
                        if isinstance(z[k],str):q['literal_nonfinite_tokens'][k+':'+z[k]]+=1
            if h.hexdigest()!=pins[label]: raise ValueError('released ledger hash '+filename)
            if count!=total or len(by)!=groups: raise ValueError('literal record/group inventory')
            actual[filename]={'bytes':size,'sha256':h.hexdigest()}
            inventories[key]={'records':count,'groups':by,'flag_pattern_order':flagfields}
        for name,p in indexed.items():
            if p['path']!=str(BASE/name) or p['bytes']!=actual[name]['bytes'] or p['sha256']!=actual[name]['sha256']:
                raise ValueError('coordinator output-manifest association '+name)
        for name,q in inventories['grid']['groups'].items():
            if q['records']!=report['grids'][name]['summary']['allocated_cells']:
                raise ValueError('saved grid-summary presentation association')
        if set(inventories['grid']['groups'])!=set(report['grids']) or set(inventories['comparison']['groups'])!=set(report['two_block_controls']):
            raise ValueError('saved summary identifier association')
        leaves=report['producer_comparisons']
        emit('review-summary.json',{
            'scope':'Literal completed-output copies, record/Boolean flag occurrence inventory and saved-field selection only; no LE, period, mask, coordinate, geometry, statistical or original-input recomputation.',
            'released_pins':pins,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inventories':inventories,'all21_grid_summaries':report['grids'],'all5_layer_summaries':report['layer_sweep'],
            'all4_control_summaries':report['two_block_controls'],'all_comparison_leaves':leaves,
            'comparison_exceptions':[x for x in leaves if x['equal'] is not True],
            'reported_comparison_count':report['comparison_count'],
            'reported_comparison_nonzero_or_unequal':report['comparison_nonzero_or_unequal'],
            'elapsed_before_summary_write':time.perf_counter()-start,
            'rss_kib_observed_before_summary_write':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        entries=[]
        for p in sorted(OUTPUT.iterdir()):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            entries.append({'path':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
        emit('manifest.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':entries,
                             'scope':'Presentation files only; original scientific result and authority unchanged.'})
        print(json.dumps({'completed':True,'cell_records':119552,'join_records':36864,'output_bytes':written,
                          'elapsed_before_print':time.perf_counter()-start}))
    except BaseException:
        # Only the freshly created presentation directory is owned. Original outputs stay untouched.
        with (OUTPUT/'failure.json').open('x') as f:
            json.dump({'error':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-start},f)
        raise


if __name__=='__main__':main()
