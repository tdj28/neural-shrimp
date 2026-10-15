"""Reporting selections/tallies of saved fields from the completed literal extraction.
No products, spectra, geometry, contours, coordinates or statistics are recomputed.
"""
import collections,hashlib,json
from pathlib import Path
base=Path('/tmp/task21f-saved-read-001');runs=json.loads(Path('/tmp/task21f-saved-compact-runs.json').read_text());cells={};zoom=None;tail=[];ordinal=None
for line in (base/'literal-extraction.jsonl').open():
 d=json.loads(line);x=d['literal'];f=d['source_file']
 if f=='cells.jsonl':cells[x['run'],x['i'],x['j']]=x
 elif f=='zoom.json':zoom=x
 elif f=='coarse-ordinal.json':ordinal=x
 elif f=='progress.jsonl' and x['run']=='p7-blob' and 801<=d['source_record']<=807:tail.append(x)
joined_counts=collections.Counter();joined_cell_ids=[]
for row in zoom['row_ledger']:
 for j in range(33):
  key=(row['band'],row['source_row'],j);joined_cell_ids.append(list(key))
  for k,v in cells[key]['masks'].items():
   if v:joined_counts[k]+=1
dropped=[]
for name in ('p7-zoom-b','p7-zoom-c'):
 cc=collections.Counter()
 for j in range(33):
  for k,v in cells[name,0,j]['masks'].items():
   if v:cc[k]+=1
 dropped.append({'run':name,'source_row':0,'cell_count':33,'saved_boolean_counts':dict(cc)})
methods=[]
for r in runs:
 for field,m in r['methods'].items():
  if not m['available']:continue
  masks={}
  for phase in m['phases']:
   for row in phase['by_mask']:
    c=masks.setdefault(row['mask_id'],collections.Counter());c['phase_segments']+=row['segments'];c['phase_checkerboards']+=row['checkerboards']
  for pair in m['pairs']:
   for row in pair['by_mask']:
    c=masks.setdefault(row['mask_id'],collections.Counter())
    for k in ('four_sign_quads','intersections','same_quad_four_sign_intersections','collinear_overlaps'):c[k]+=row[k]
  methods.append({'run':r['id'],'field':field,'mask_names':r['mask_equivalence'],'tallied_existing_summary_counts':{k:dict(v) for k,v in masks.items()},'all_phases':m['phases'],'all_pairs':m['pairs'],'scope':'Counts tally saved phase/pair summaries; repeated/endpoint segment events are not deduplicated roots.'})
obj={'scope':'Read-only saved-field selection and integer occurrence tallies from the prior complete literal extraction. No numerical replay or new scientific acceptance gate.',
 'source_extraction':{'path':str(base/'literal-extraction.jsonl'),'sha256':'c0d383ec1493796d28c040b8562236457ebcae0ca7048a496a7077bea00d89dd'},
 'source_selector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'runs':runs,'joined_zoom':{'saved_row_ledger':[{'band':x['band'],'source_row':x['source_row'],'temperature':x['temperature']} for x in zoom['row_ledger']],'selected_cell_ids':joined_cell_ids,'selected_cell_count':len(joined_cell_ids),'saved_boolean_counts':dict(joined_counts),'dropped_boundary_rows':dropped,'saved_modulus_summary':zoom['finite_modulus_summary'],'saved_literal_checkerboards':zoom['literal_contours']['literal_checkerboards'],'saved_finite_nonzero_checkerboards':zoom['literal_contours']['finite_nonzero_checkerboards']},
 'minimum_modulus_cell':cells['p7-zoom-b',4,3],'methods':methods,'original_blob_post_snapshot_success_rows':tail,
 'coarse_ordinal_coordinate_deltas':ordinal['row_temperature_deltas'],
 'limitations':['No per-phase gauge or origin alignment inferred between independent bands.','All reported crossings are saved segment-pair events, not certified or unique zeros.','All 11 outputs and all original nonfinite/nonzero values are retained in literal-extraction.jsonl; this reporting selection is not an additional audit.']}
p=Path('/tmp/task21f-scientific-findings.json');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
print('JOINED',dict(joined_counts));print('DROPPED',json.dumps(dropped));print('METHOD_COUNTS')
for m in methods:print(m['run'],m['field'],m['tallied_existing_summary_counts'])
print('FINDINGS',p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
