"""One bounded literal saved-output traversal. No numerical/model recomputation."""
import argparse,collections,hashlib,json,math,os,resource,time,traceback
from pathlib import Path
FILES=('cells.jsonl','progress.jsonl','coverage.jsonl','masks.jsonl','geometry.jsonl','run-summaries.jsonl','overlaps.jsonl','zoom.json','coarse-ordinal.json','report.json','manifest.json')
REPORT_SHA='b7347ffd2ef4c61e69d3cab25be276cc515ce8fb57ead609ec0bff159a9e1c31'
AUTH_SHA='1acde00bd152d9fd6f57888af3c1da98d67268d14cf6b9bb077463a0a43c60d3'
BASE=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21f/run-001')


def bad_number(x):
 if isinstance(x,dict):return 'nonfinite' in x or any(bad_number(v) for v in x.values())
 if isinstance(x,list):return any(bad_number(v) for v in x)
 return isinstance(x,float) and not math.isfinite(x)


def nonzero(x):
 if isinstance(x,list):return any(nonzero(v) for v in x)
 return type(x) in (int,float) and math.isfinite(x) and x!=0


def all_true(x):
 if isinstance(x,list):return all(all_true(v) for v in x)
 return x is True


def differences(x,path=''):
 if isinstance(x,dict):
  if {'own','saved','signed','absolute','both_finite','exact_equal_including_nan'}<=x.keys():
   kind=[]
   if nonzero(x['signed']):kind.append('nonzero_saved_difference')
   if not all_true(x['both_finite']):kind.append('nonfinite_comparison')
   if not all_true(x['exact_equal_including_nan']):kind.append('not_exactly_equal')
   if kind:yield {'pointer':path,'kinds':kind,'literal_comparison':x}
  else:
   for k,v in x.items():yield from differences(v,path+'/'+str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):yield from differences(v,path+'/'+str(i))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(50,50))
 n=os.getpriority(os.PRIO_PROCESS,0)
 if n>10:raise RuntimeError('exact nice10 unavailable')
 if n<10:os.nice(10-n)
 out=Path(a.out);out.mkdir(exist_ok=False);start=time.perf_counter();budget=96*1024**2;written=0
 counters={};run_counts=collections.defaultdict(collections.Counter);phase_keys=[];pair_keys=[];exceptions=[]
 def dump(f,x):
  nonlocal written
  b=(json.dumps(x,allow_nan=False,separators=(',',':'))+'\n').encode();written+=len(b)
  if written>budget:raise RuntimeError('saved traversal output cap')
  f.write(b)
 try:
  # Only the already released report and authority identities are bound here.
  # This is not a repeat of the coordinator's source/input/process audit.
  report_bytes=(BASE/'report.json').read_bytes()
  if hashlib.sha256(report_bytes).hexdigest()!=REPORT_SHA:raise ValueError('released report hash')
  authority=BASE.parent/'run-001-authority/report.json'
  ab=authority.read_bytes()
  if hashlib.sha256(ab).hexdigest()!=AUTH_SHA:raise ValueError('released authority hash')
  with open(out/'literal-extraction.jsonl','xb') as full,open(out/'comparison-exceptions.jsonl','xb') as diff:
   for name in FILES:
    counts=collections.Counter();p=BASE/name
    if name.endswith('.jsonl'):
     stream=open(p,'r');records=(json.loads(line) for line in stream)
    else:
     stream=None;records=iter([json.loads(report_bytes) if name=='report.json' else json.loads(p.read_text())])
    try:
     for number,obj in enumerate(records,1):
      counts['records']+=1;run=obj.get('run') if isinstance(obj,dict) else None
      if run:run_counts[run][name]+=1
      dump(full,{'source_file':name,'source_record':number,'literal':obj})
      for row in differences(obj):
       counts['comparison_exceptions']+=1
       for kind in row['kinds']:counts[kind]+=1
       dump(diff,{'source_file':name,'source_record':number,'run':run,**row})
      if bad_number(obj):counts['records_containing_nonfinite']+=1
      if name=='cells.jsonl':
       if obj.get('source_clean_mismatch_on_finite'):exceptions.append({'file':name,'record':number,'run':run,'i':obj['i'],'j':obj['j'],'kind':'source_clean_mismatch'})
       for key in ('fixed_product_minus_complex_mu','transport_product_minus_complex_mu'):
        if key in obj and (bad_number(obj[key]) or nonzero(obj[key])):counts[key+'_nonzero_or_nonfinite']+=1
       if not obj['saved']['ok']:counts['saved_ok_false']+=1
      if name=='progress.jsonl' and 'record' in obj:
       if obj['record'].get('converged') is False:exceptions.append({'file':name,'record':number,'run':run,'kind':'original_newton_refusal','literal':obj['record']})
       for key in ('saved_newton_below_original_tolerance','saved_primitive_split_gate'):
        if obj.get(key) is False:exceptions.append({'file':name,'record':number,'run':run,'kind':key,'literal':obj['record']})
      if name=='coverage.jsonl':
       if obj['snapshot_failed_vs_unvisited_unresolved']:counts['initialized_snapshot_cells']+=1
       if not obj['progress_sequences']:counts['no_progress_cells']+=1
      if name=='geometry.jsonl':
       if 'phase' in obj:phase_keys.append([run,obj['field'],obj['phase']])
       if 'phases' in obj:pair_keys.append([run,obj['field'],obj['phases']])
       if 'value' in obj:
        for q in obj['value']['quads']:
         counts['quads_traversed']+=1
         if q['eligible']:counts['eligible_quads']+=1
         if len(q['cuts'])==4:counts['four_cut_quads']+=1
         if q.get('zero_corner_indices'):counts['zero_corner_quads']+=1
         if q.get('opposite_sign_underflow_edges'):counts['underflow_quads']+=1
       if 'crossings' in obj:
        c=obj['crossings']
        for key in ('intersections','collinear_overlaps','near_parallel_candidate_pairs','nonfinite_arithmetic_candidates'):
         counts[key+'_records']+=len(c[key])
        for event in c['intersections']:
         if event['same_quad_four_sign_support'] is True:counts['same_quad_corner_supported_intersections']+=1
         if not event['mask_ids']:counts['intersections_with_no_requested_mask']+=1
    finally:
     if stream:stream.close()
    counters[name]=dict(counts)
  index={'scope':'Literal traversal and saved-field selection only; no source inputs/NPZ, statistics, geometry, model, QR, AD or eigenanalysis replay.',
         'source_report':{'path':str(BASE/'report.json'),'sha256':REPORT_SHA},'coordinator_authority':{'path':str(authority),'sha256':AUTH_SHA},
         'all_expected_files':list(FILES),'per_file_occurrence_counts':counters,'per_run_record_counts':{k:dict(v) for k,v in run_counts.items()},
         'phase_records':phase_keys,'phase_pair_records':pair_keys,'retained_original_refusals_and_gate_exceptions':exceptions,
         'count_scope':'Comparison exceptions count saved occurrences, including duplicate summary transcriptions; they are not unique scientific discrepancies.',
         'elapsed_before_index_write':time.perf_counter()-start,'peak_rss_kib_observed':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         'bounded_read_limits':{'cpu_threads':1,'nice':10,'as_bytes':1024**3,'cpu_seconds':50,'wall_seconds':60,'output_bytes':budget}}
  with open(out/'read-index.json','xb') as f:dump(f,index)
  manifest={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':[]}
  for p in sorted(out.iterdir()):
   h=hashlib.sha256()
   with open(p,'rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   manifest['files'].append({'path':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
  with open(out/'manifest.json','xb') as f:dump(f,manifest)
  print(json.dumps({'completed':True,'output_bytes':written,'per_file_counts':counters,'elapsed_before_finish':time.perf_counter()-start}))
 except BaseException:
  with open(out/'failure.json','x') as f:json.dump({'error':traceback.format_exc(),'elapsed':time.perf_counter()-start},f,indent=2)
  raise


if __name__=='__main__':main()
