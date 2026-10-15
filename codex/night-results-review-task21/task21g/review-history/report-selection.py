"""Select/tally stored result dispositions only; no original sequence analysis."""
import json,hashlib,collections
from pathlib import Path
base=Path('/tmp/task21g-saved-read-001');out=Path('/tmp/task21g-partition-table.jsonl');rows=[];ph=[]
with out.open('x') as f:
 for line in (base/'phase-summary.jsonl').open():
  s=json.loads(line);pp=[]
  for p in s['partitions']:
   both=one=neither=positive=negative=zero=undefined=0
   for q in p.get('parity_counts_by_bin',[]):
    c=q['parity_counts'];d=q.get('odd_minus_even_median_y')
    if len(c)==2 and all(x>0 for x in c):both+=1
    elif any(x>0 for x in c):one+=1
    else:neither+=1
    if d is None:undefined+=1
    elif d>0:positive+=1
    elif d<0:negative+=1
    else:zero+=1
   runrows=[]
   for q in p.get('runs',[]):
    z=q['profile'];runrows.append({'bin_ids':q['bin_ids'],'plateau_records':len(z['initial']['plateaus']),'initial_extremum_records':len(z['initial']['interior']),'initial_extrema':z['initial']['interior'],'cancellation_events':z['events'],'endpoint_only':z['endpoint_only']})
   r={'cell':s['cell'],'phase':s['phase'],'coordinate':s['coordinate'],'mode':p['mode'],'bins':p['bins'],'status':p['status'],'occupied_bins':p.get('occupied_bins'),'singleton_bins':p.get('singleton_bins'),'empty_bins':p.get('empty_bins'),'total_pairs':p.get('total_pairs'),'runs':runrows,'parity_disposition_record_counts':{'both_present':both,'one_only':one,'neither':neither,'positive_odd_minus_even_median_y':positive,'negative':negative,'zero':zero,'undefined':undefined},'scope':'List/flag occurrence counts from already saved output. Complete parity x/y distributions and original pairs remain in literal-phases.jsonl. No turning-count or binning recomputation.'}
   f.write(json.dumps(r,separators=(',',':'))+'\n');rows.append(r);pp.append(r)
  ph.append({'cell':s['cell'],'phase':s['phase'],'coordinate':s['coordinate'],'groups':s['groups'],'producer_summary':s['producer_summary'],'comparisons':s['comparisons'],'inherited_pca':s['inherited_pca'],'scalar_scope':s['scalar_scope'],'partitions':pp})
p=Path('/tmp/task21g-scientific-findings.json');p.write_text(json.dumps({'source_selector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Saved-field reporting selections and record inventories only. No unimodality threshold selected.','all49':ph},indent=2)+'\n')
for q in (out,p):print(q,q.stat().st_size,hashlib.sha256(q.read_bytes()).hexdigest())
for s in ph:
 print('PHASE',s['cell'],s['phase'],'coord',s['coordinate'],'saved',s['producer_summary'],'initial10',[[len(r['initial_extrema']) for r in p['runs']] for p in s['partitions']])
