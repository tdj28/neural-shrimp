"""Own finite saved DSS reader. Admission and output only; no producer imports."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('own_dss_core',HERE/'core.py')
C=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
np=C.np
RUNS={'p7-blob':(31,31,807,'DSS1','7217f136aa447b932ad4db0b91f8ddf3c488e05d'),
      'p7-blob-lo':(16,31,477,'DSS1b','68e9d2c32c9a45f272485fd4bacb0967dda91ca3'),
      'p7-blob-hi':(16,31,483,'DSS1c','68e9d2c32c9a45f272485fd4bacb0967dda91ca3'),
      'p7-zoom-a':(12,33,396,'DSS2a','d89d5bc9d40f0790902bf7971bf90c64b6c7cf2f'),
      'p7-zoom-b':(12,33,396,'DSS2b','d89d5bc9d40f0790902bf7971bf90c64b6c7cf2f'),
      'p7-zoom-c':(13,33,429,'DSS2c','d89d5bc9d40f0790902bf7971bf90c64b6c7cf2f')}
INPUT_SHA='bff3fbc5937f345f35ca6312e87517c9927b7a9b709c48a4ade1ba0292100aff'
AXIS_SHA='69302df5649a263fb0089c92fd5e82aa1e766ad58e0211b36b73b58b4d3d9547'
CORE_SHA='b7b29685e9c1bf4a9211a9a76d9fa04e451de6a1a3de1ad63783965bbee9c60a'
OUTPUT='/data2/neural-shrimp-data/codex/night-results-review-task21/task21f/run-001'
LIMITS={'cpu_threads':1,'nice':10,'as_bytes':1073741824,'cpu_seconds':120,'wall_seconds':180,
        'max_output_bytes':268435456,'max_input_bytes':8388608,'startup_available_gib':20,'runtime_available_gib':16}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def authenticate(pin):
    p=Path(pin['path'])
    if p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:
        raise ValueError('input/source hash mismatch: '+str(p))
    return pin


def pins_in(obj):
    if isinstance(obj,dict):
        if {'path','sha256','bytes'}<=obj.keys():yield {k:obj[k] for k in ('path','sha256','bytes')}
        else:
            for value in obj.values():yield from pins_in(value)
    elif isinstance(obj,list):
        for value in obj:yield from pins_in(value)


class Sink:
    def __init__(self,out,limit):
        self.out=Path(out);self.limit=limit;self.bytes=0;self.entries={}
    def write(self,name,value,append=False):
        data=(json.dumps(C.plain(value),allow_nan=False,separators=(',',':'))+'\n').encode()
        if self.bytes+len(data)>self.limit:raise RuntimeError('declared output ceiling')
        if '/' in name or name.startswith('.'):raise ValueError('output basename')
        mode='ab' if append and name in self.entries else 'xb'
        with open(self.out/name,mode) as f:f.write(data)
        self.bytes+=len(data);self.entries[name]=self.entries.get(name,0)+1
    def manifest(self):
        return [{'path':name,'bytes':(self.out/name).stat().st_size,'sha256':sha(self.out/name),'records':count}
                for name,count in sorted(self.entries.items())]


def load_run(r):
    """All fields are exactly the admitted historical numeric NPZ fields."""
    spec={x['member'][:-4]:x for x in r['metadata']['npz_members']}
    expected_fields=set(C.SCALARS+C.FLAGS+('alpha','log10_temperature','factors','seed','seed_tokens'))
    if r['id'].startswith('p7-zoom-'):expected_fields.update(C.TRANSPORT)
    if len(spec)!=len(r['metadata']['npz_members']) or set(spec)!=expected_fields:raise ValueError('historical field inventory')
    path=r['files']['grid.npz']['path']
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!={k+'.npy' for k in spec}:
            raise ValueError('NPZ duplicate/missing/extra members')
    with np.load(path,allow_pickle=False) as z:
        g={}
        for k,s in spec.items():
            a=z[k]
            if a.dtype.hasobject or a.dtype.str!=s['dtype'] or list(a.shape)!=s['shape'] or a.nbytes!=s['array_nbytes']:
                raise ValueError('NPZ schema '+k)
            g[k]=a.copy()
    shape=C.validate_grid(g,7)
    expected=RUNS[r['id']][:2]
    if shape!=expected:raise ValueError('fixed grid shape')
    for k,v in r['metadata']['axis_and_seed_metadata'].items():
        if not np.array_equal(g[k],np.asarray(v,dtype=g[k].dtype)):raise ValueError('exact axis/seed '+k)
    if g['seed_tokens'].shape!=(7,) or g['seed_tokens'].dtype.kind not in 'iu':raise ValueError('token schema')
    transported=r['id'].startswith('p7-zoom-')
    if all(k in g for k in C.TRANSPORT)!=transported:raise ValueError('historical method association')
    progress=[json.loads(line) for line in Path(r['files']['progress.jsonl']['path']).read_text().splitlines()]
    indices=[[x['i'],x['j']] for x in progress if 'i' in x and 'j' in x]
    pm=r['metadata']['progress']
    if len(progress)!=pm['line_count'] or indices!=pm['indices'] or len(indices)!=RUNS[r['id']][2]:
        raise ValueError('original BFS coverage association')
    if len({tuple(x) for x in indices})!=len(indices):raise ValueError('duplicate BFS cell')
    for x in progress:
        if 'i' in x or 'j' in x:
            i,j=x['i'],x['j']
            if type(i)!=int or type(j)!=int or not 0<=i<shape[0] or not 0<=j<shape[1]:raise ValueError('BFS index')
            if x['alpha']!=g['alpha'][j] or x['log10_temperature']!=g['log10_temperature'][i]:raise ValueError('BFS exact coordinates')
            if type(x['converged'])!=bool:raise ValueError('BFS converged schema')
            required=set(C.SCALARS+C.FLAGS+('factors','split','seconds'))-{'ok'}
            required|=set(C.TRANSPORT) if transported else set()
            if x['converged'] and not required<=x.keys():raise ValueError('successful BFS fields missing')
    seed=[x for x in progress if 'i' not in x and 'j' not in x]
    if len(seed)!=1 or seed[0]['seed']!=g['seed'].tolist() or seed[0]['tokens']!=g['seed_tokens'].tolist():
        raise ValueError('seed source association')
    saved=None
    if r['files']['grid.json'] is not None:
        saved=json.loads(Path(r['files']['grid.json']['path']).read_text())
        if saved['args']!=r['declared_args'] or saved['seed_tokens']!=g['seed_tokens'].tolist():raise ValueError('final args/seed')
        if saved['args']['period']!=7:raise ValueError('declared period')
    elif r['id']!='p7-blob':raise ValueError('unexpected missing final JSON')
    return g,progress,saved


def view_masks(g):
    raw=C.masks(g)
    selected={'ok':g['ok'],'ok_same_tokens':g['ok']&g['same_tokens'],
              'ok_exact_real':g['ok']&raw['exact_real_finite'],
              'ok_clean_exact_real':g['ok']&g['clean']&raw['exact_real_finite']}
    selected.update({k:v for k,v in raw.items() if k.startswith(('ok_clean_','transport_'))})
    unique=[];names={}
    for name,m in selected.items():
        found=next((i for i,u in enumerate(unique) if np.array_equal(m,u)),None)
        if found is None:found=len(unique);unique.append(m)
        names[name]=found
    quads=[m[:-1,:-1]&m[:-1,1:]&m[1:,1:]&m[1:,:-1] for m in unique]
    return {'names':names,'cell_masks':unique,'quad_masks':quads,
            'reuse':'exact identical masks share an ID; distinct masks filter the same base-ok segments without repeated geometry'},raw


def geometry(g,field,views,sink,run):
    """Once per original field; one phase/pair at a time is serialized."""
    masks=views['quad_masks'];nq=(len(g['alpha'])-1)*(len(g['log10_temperature'])-1);nc=len(g['alpha'])-1
    if field not in g:return {'available':False}
    phases=[];summaries=[]
    for k in range(7):
        phase=C.contours(g[field][...,k],g['ok'],g['alpha'],g['log10_temperature'])
        sink.write('geometry.jsonl',{'run':run,'field':field,'phase':k,'value':phase},True)
        phases.append(phase)
        summaries.append({'phase':k,'segments':len(phase['segments']),
                          'quad_count':len(phase['quads']),
                          'by_mask':[{'mask_id':i,'eligible_quads':int(m.sum()),
                                      'segments':sum(len(q['segment_indices']) for q in phase['quads'] if m[q['i'],q['j']]),
                                      'checkerboards':sum(q['literal_checkerboard'] for q in phase['quads'] if m[q['i'],q['j']])}
                                     for i,m in enumerate(masks)]})
    pairs=[]
    for a,b in C.combinations(range(7),2):
        ga,gb=phases[a],phases[b]
        result=C.intersections(ga['segments'],gb['segments'])
        four=[];count=[]
        for qa,qb in zip(ga['quads'],gb['quads']):
            av,bv=qa['values'],qb['values']
            valid=bool(np.isfinite(av).all() and np.isfinite(bv).all() and (av!=0).all() and (bv!=0).all())
            number=len(set(zip(np.sign(av).tolist(),np.sign(bv).tolist()))) if valid else None
            count.append(number);four.append(valid and number==4)
        if len(four)!=nq:raise ValueError('phase quad identity')
        for label in ('intersections','collinear_overlaps','near_parallel_candidate_pairs','nonfinite_arithmetic_candidates'):
            for event in result[label]:
                sa,sb=event['pair'];qa=ga['segments'][sa]['quad'];qb=gb['segments'][sb]['quad']
                same=qa==qb
                event.update(quad_a=qa,quad_b=qb,same_quad=same,
                             mask_ids=[i for i,m in enumerate(masks) if m[tuple(qa)] and m[tuple(qb)]],
                             same_quad_four_sign_support=bool(four[qa[0]*nc+qa[1]]) if same else None)
        rows={'run':run,'field':field,'phases':[a,b],'crossings':result,
              'quad_order':'row-major','quad_finite_nonzero_sign_pair_counts':count,
              'quad_four_sign_support':four,'scope':'sampled corners/interpolated segments, not a native crossing; cross-quad four-sector support unresolved'}
        sink.write('geometry.jsonl',rows,True)
        pairs.append({'phases':[a,b],'all_ok_pairs':result['pairs'],'all_ok_intersections':len(result['intersections']),
                      'all_ok_collinear_overlaps':len(result['collinear_overlaps']),
                      'by_mask':[{'mask_id':i,'four_sign_quads':sum(bool(x) and bool(m.flat[k]) for k,x in enumerate(four)),
                                  'intersections':sum(i in x['mask_ids'] for x in result['intersections']),
                                  'same_quad_four_sign_intersections':sum(i in x['mask_ids'] and x['same_quad_four_sign_support'] is True for x in result['intersections']),
                                  'collinear_overlaps':sum(i in x['mask_ids'] for x in result['collinear_overlaps'])}
                                 for i,m in enumerate(masks)]})
    return {'available':True,'phases':summaries,'pairs':pairs,'method':result['method']}


def calculate(packet,sink):
    start=time.perf_counter();summaries=[];grids=[]
    for r in packet['runs']:
        name=r['id'];g,progress,saved=load_run(r);grids.append((name,g));views,rawm=view_masks(g)
        sink.write('masks.jsonl',{'run':name,**views},True)
        rows=C.cell_rows(g,7)
        for x in rows:sink.write('cells.jsonl',{'run':name,**x},True)
        coverage=C.progress_coverage(g,progress)
        for x in coverage['all_progress_records']:
            rec=x['record']
            if rec.get('converged') is True:
                x['saved_newton_below_original_tolerance']=rec['newton']<max(1e-12,7e-13)
                x['saved_primitive_split_gate']=rec['split']>1e-7
            sink.write('progress.jsonl',{'run':name,**x},True)
        for x in coverage['cells']:sink.write('coverage.jsonl',{'run':name,**x},True)
        sink.write('progress.jsonl',{'run':name,'noncell_records':coverage['noncell_records'],'no_snapshot_reconstruction':True},True)
        own={'cells':g['mu'].size,'solved':int(g['ok'].sum()),'clean':int((g['ok']&g['clean']).sum()),'same_tokens':int((g['ok']&g['same_tokens']).sum())}
        mu_geom=C.contours(g['mu'],g['ok'],g['alpha'],g['log10_temperature'])
        sink.write('geometry.jsonl',{'run':name,'field':'mu','mask':'ok','value':mu_geom},True)
        methods={field:geometry(g,field,views,sink,name) for field in ('factors','factors_transported')}
        diff_names=['fixed_product_vs_saved_product','fixed_product_vs_mu_real']
        if 'factors_transported' in g:diff_names+=['transport_product_vs_mu_real','phase_absolute_factor_vs_stretching','absolute_product_vs_stretching_product']
        diff_stats={key:C.stats(np.asarray([x[key]['signed'] for x in rows])) for key in diff_names}
        success=[x for x in progress if x.get('converged') is True]
        row={'id':name,'attempt':r['attempt'],'source_commit':r['source_commit'],'original_status_lines':r['original_tool_status_lines'],
             'declared_args':r['declared_args'],'source_counts':own,'final_json_available':saved is not None,
             'original_final_json':saved,'final_summary_deltas':{k:C.delta(v,saved[k]) for k,v in own.items()} if saved else None,
             'progress_cells':len(coverage['all_progress_records']),'progress_converged':len(success),
             'progress_refused':sum(x.get('converged') is False for x in progress),
             'no_progress_cells':sum(not x['progress_sequences'] for x in coverage['cells']),
             'initialized_snapshot_cells':sum(x['snapshot_failed_vs_unvisited_unresolved'] for x in coverage['cells']),
             'all_saved_scalar_stats':{k:C.stats(g[k]) for k in C.SCALARS+C.TRANSPORT if k in g},
             'leading_modulus_stats':C.stats(np.hypot(g['mu'],g['mu_imag'])),
             'mask_counts':{k:int(v.sum()) for k,v in rawm.items()},'mask_equivalence':views['names'],
             'leading_modulus_by_mask':{k:C.stats(np.hypot(g['mu'],g['mu_imag']),v) for k,v in rawm.items()},
             'scalar_difference_stats':diff_stats,'source_clean_mismatch_count':sum(x['source_clean_mismatch_on_finite'] for x in rows),
             'axes':{k:{'values':g[k],'hex':[float(v).hex() for v in g[k]],'spacing':C.stats(np.diff(g[k]))} for k in ('alpha','log10_temperature')},
             'literal_coarse':{'segments':len(mu_geom['segments']),'checkerboards':mu_geom['literal_checkerboards'],
                               'finite_nonzero_checkerboards':mu_geom['finite_nonzero_checkerboards']},
             'methods':methods,'nonfinite_and_failures_retained':True}
        summaries.append(row);sink.write('run-summaries.jsonl',row,True)
    fields=list(C.SCALARS+C.FLAGS+('factors',)+C.TRANSPORT)
    overlaps=C.exact_overlaps(grids,fields)
    for row in overlaps:sink.write('overlaps.jsonl',row,True)
    by_name=dict(grids);zoom=C.plot_zoom([(name,by_name[name]) for name in ('p7-zoom-a','p7-zoom-b','p7-zoom-c')])
    sink.write('zoom.json',zoom)
    low,high,original=(by_name[name] for name in ('p7-blob-lo','p7-blob-hi','p7-blob'))
    joined_t=np.concatenate([low['log10_temperature'],high['log10_temperature'][1:]])
    ordinal={'scope':'fixed lo/all + hi/drop-first scalar comparison only, not exact-coordinate matching or factor gauge alignment',
             'alpha':C.delta(low['alpha'],original['alpha']),'boundary_temperature':C.delta(high['log10_temperature'][0],low['log10_temperature'][-1]),
             'row_temperature_deltas':C.delta(joined_t,original['log10_temperature']),
             'rows':[{'i':i,'j':j,'original_temperature':original['log10_temperature'][i],
                      'selected_temperature':joined_t[i],'source':['p7-blob-lo',i] if i<16 else ['p7-blob-hi',i-15],
                      'fields':{k:C.delta((low[k][i,j] if i<16 else high[k][i-15,j]),original[k][i,j]) for k in fields if k in original}}
                     for i,j in np.ndindex(original['mu'].shape)]}
    sink.write('coarse-ordinal.json',ordinal)
    return {'schema':'codex-task21f-saved-dss-v1','computational_completed':True,'scientific_claims_accepted':False,
            'runs':summaries,'run_count':len(summaries),'allocated_cells':sum(x['source_counts']['cells'] for x in summaries),
            'exact_coordinate_overlap_pairs':len(overlaps),'zoom':{k:v for k,v in zoom.items() if k not in ('literal_contours','row_ledger','joins')},
            'zoom_literal_segments':len(zoom['literal_contours']['segments']),
            'native_parent_gauge_and_derivatives':'inherited, unavailable for independent scalar reconstruction; original Ritz attempts convergence but does not enforce final residual and saves no Ritz residual/history',
            'axis_refinement_ratio_ranges':{name:{k:{'all_spacing_ratio_min':float(np.diff(original[k]).min()/np.diff(g[k]).max()),'all_spacing_ratio_max':float(np.diff(original[k]).max()/np.diff(g[k]).min())} for k in ('alpha','log10_temperature')} for name,g in grids if name.startswith('p7-zoom-')},
            'gap_1_3e_minus3':'unverified: no exact original distance definition/analysis source supplied',
            'resource_observation':{'calculation_and_incremental_serialization_seconds':time.perf_counter()-start,
                                    'peak_rss_kib_observed_before_final_rehash_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--card',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();card=json.loads(Path(args.card).read_text());out=Path(args.out)
    if card.get('schema')!='codex-task21f-card-v1' or card.get('admission')!='reviewed' or card.get('limits')!=LIMITS or str(out)!=OUTPUT or card.get('output')!=OUTPUT:
        raise ValueError('prepared/unrecognized card or output refuses launch')
    if card['inputs']['sha256']!=INPUT_SHA or card['axis_metadata']['sha256']!=AXIS_SHA or sha(HERE/'core.py')!=CORE_SHA:
        raise ValueError('fixed source/input association')
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['as_bytes'],LIMITS['as_bytes']))
    resource.setrlimit(resource.RLIMIT_CPU,(LIMITS['cpu_seconds'],LIMITS['cpu_seconds']))
    nice=os.getpriority(os.PRIO_PROCESS,0)
    if nice>10:raise ValueError('nice exceeds exact10')
    if nice<10:os.nice(10-nice)
    out.mkdir(exist_ok=False)
    sink=Sink(out,LIMITS['max_output_bytes']);started=time.perf_counter()
    try:
        for key in ('inputs','axis_metadata'):authenticate(card[key])
        if len(card['sources'])!=2 or {p['name'] for p in card['sources']}!={'run.py','core.py'}:raise ValueError('own source inventory')
        for p in card['sources']:
            if sha(HERE/p['name'])!=p['sha256']:raise ValueError('own source pin')
        packet=json.loads(Path(card['inputs']['path']).read_text())
        if packet['run_count']!=6 or [r['id'] for r in packet['runs']]!=list(RUNS):raise ValueError('run inventory')
        for r in packet['runs']:
            fixed=RUNS[r['id']]
            if (r['attempt'],r['source_commit'])!=fixed[3:]:raise ValueError('attempt/source association')
        for k in ('authority','source_map'):authenticate(packet[k])
        source_map=json.loads(Path(packet['source_map']['path']).read_text())
        pins=list(pins_in(packet))+list(pins_in(source_map))+[card['inputs'],card['axis_metadata']]
        unique={}
        for p in pins:
            if p['path'] in unique and unique[p['path']]!=p:raise ValueError('conflicting pin')
            unique[p['path']]=p
        if sum(p['bytes'] for p in unique.values())>LIMITS['max_input_bytes']:raise ValueError('input byte cap')
        for p in unique.values():authenticate(p)
        report=calculate(packet,sink)
        for p in unique.values():authenticate(p)
        report.update(input_authentication=list(unique.values()),source_authority_scope='original receipt/archive/process binding inherited from accepted coordinator authority; all selected target and source bytes independently hashed before/after',
                      actual_source_pins=card['sources'],card_sha256=sha(args.card),axis_metadata_pin=card['axis_metadata'],
                      seconds_before_final_report_write=time.perf_counter()-started)
        sink.write('report.json',report)
        sink.write('manifest.json',{'schema':'codex-task21f-output-v1','computational_completed':True,'files':sink.manifest()})
    except BaseException:
        # Only this freshly created output is owned. Retain earlier outputs.
        with open(out/'failure.json','x') as f:json.dump({'computational_completed':False,'error':traceback.format_exc(),'source':sha(__file__)},f,indent=2)
        raise


if __name__=='__main__':main()
