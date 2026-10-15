"""One bounded literal saved presentation pass; no scalar-analysis recomputation."""
import argparse,collections,hashlib,json,os,resource,time,traceback
from pathlib import Path
BASE=Path('/data2/neural-shrimp-data/codex/night-results-review-task21/task21g/run-001')
REPORT_SHA='83a09773d414e7efe735f768424d90e7a19c9f250b6df3a6281f8b3096b80d26'
AUTH_SHA='26da80d0cfa9519acfacc362cf351115d3fb93afc3227e9190bed8e5972e5daf'

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);args=a.parse_args()
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(50,50))
 n=os.getpriority(os.PRIO_PROCESS,0)
 if n>10:raise RuntimeError('nice10 unavailable')
 if n<10:os.nice(10-n)
 out=Path(args.out);out.mkdir(exist_ok=False);start=time.perf_counter();written=0
 def emit(f,x):
  nonlocal written
  b=(json.dumps(x,allow_nan=False,separators=(',',':'))+'\n').encode();written+=len(b)
  if written>128*1024**2:raise RuntimeError('presentation output ceiling')
  f.write(b)
 try:
  b=(BASE/'report.json').read_bytes()
  if hashlib.sha256(b).hexdigest()!=REPORT_SHA:raise ValueError('released report hash')
  auth=BASE.parent/'run-001-authority/report.json';ab=auth.read_bytes()
  if hashlib.sha256(ab).hexdigest()!=AUTH_SHA:raise ValueError('released authority hash')
  report=json.loads(b)
  with (out/'literal-report.json').open('xb') as f:f.write(b);written+=len(b)
  counts=collections.Counter();summary=[];dispositions=[]
  with (out/'literal-phases.jsonl').open('xb') as full,(out/'partition-dispositions.jsonl').open('xb') as parts,(out/'phase-summary.jsonl').open('xb') as phases:
   for i,line in enumerate((BASE/'phases.jsonl').open(),1):
    p=json.loads(line);counts['phase_records']+=1;emit(full,p)
    r=report['all_phase_summaries'][i-1]
    if (p['cell'],p['phase'])!=(r['cell'],r['phase']):raise ValueError('saved presentation association')
    emit(phases,r)
    brief={'cell':p['cell'],'phase':p['phase'],'coordinate':p['coordinate'],'groups':p['groups'],'producer_summary':p['producer_summary'],'comparisons':p['comparisons'],'scalar_scope':r['scalar_scope'],'partitions':[]}
    for q,c in zip(p['own']['partitions'],r['partitions']):
     if (q['mode'],q['bins'])!=(c['mode'],c['bins']):raise ValueError('partition association')
     counts['partition_records']+=1;counts['partition_status_'+q['status']]+=1
     # Every saved bin retains its x/y/parity distributions and temporal indices.
     emit(parts,{'cell':p['cell'],'phase':p['phase'],'coordinate':p['coordinate'],'saved_compact_summary':c,'saved_partition':q})
     brief['partitions'].append({'mode':q['mode'],'bins':q['bins'],'status':q['status'],'occupied_bins':c.get('occupied_bins'),'singleton_bins':c.get('singleton_bins'),'empty_bins':q.get('empty_bins'),'runs':[{'bin_ids':s['bin_ids'],'initial_interior':s['profile']['initial']['interior'],'events':s['profile']['events'],'endpoint_only':s['profile']['endpoint_only']} for s in q['runs']]})
    for name,c in p['comparisons'].items():
     counts['comparison_'+name+'_'+c['status']]+=1
     counts['comparison_'+name+'_exact_'+str(c.get('exact'))]+=1
     if c.get('exact') is not True or c['status']!='finite':dispositions.append({'cell':p['cell'],'phase':p['phase'],'name':name,'saved':c})
    summary.append(brief)
  with (out/'review-summary.json').open('xb') as f:emit(f,{'scope':'Literal saved-field reading and selection only; no PCA, spectra, model, geometry, extrema, binning, count, statistical or parity recomputation. List lengths used only as record inventories.', 'report_sha256':REPORT_SHA,'authority_sha256':AUTH_SHA,'counts':dict(counts),'comparison_exceptions':dispositions,'all_phase_summaries':summary,'elapsed_before_summary_write':time.perf_counter()-start,'rss_kib_observed':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
  entries=[]
  for p in sorted(out.iterdir()):
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
   entries.append({'path':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
  with (out/'manifest.json').open('xb') as f:emit(f,{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'files':entries})
  print(json.dumps({'completed':True,'counts':dict(counts),'bytes':written,'elapsed_before_print':time.perf_counter()-start}))
 except BaseException:
  with (out/'failure.json').open('x') as f:json.dump({'error':traceback.format_exc(),'elapsed':time.perf_counter()-start},f,indent=2)
  raise
if __name__=='__main__':main()
