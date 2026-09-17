#!/usr/bin/env python3
"""Metadata/source-only EXP015 preflight: gain/scoring/ODE APIs forbidden."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import hashlib,json,itertools,sys,subprocess,datetime
ROOT=Path('/data2/neural-shrimp');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import run_phase_transverse_gain as run
checks=[]
def bind(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path):return json.loads(Path(path).read_text())
def check(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise AssertionError(name)
def forbidden(*a,**k):raise AssertionError('Forbidden numerical outcome/solver call in metadata preflight')
run.gain.evaluate_cell=forbidden;run.gain.decompose=forbidden;run.gain.propagated_matrices=forbidden
run.gain.pm._hermite=forbidden;run.gain.pm.full_linear_response=forbidden;run.gain.hopfield_rhs=forbidden
planpath=ROOT/'experiments/hopfield/EXP-015-plan.json';plan=read(planpath);plan_before=bind(planpath)
source_before=[bind(ROOT/x['path']) for x in plan['source_files']]
expected=[dict(id=f'{s}--p{p}--{d}--k{k}',setting=s,phase_index=p,direction=d,delay_cycles=k) for s,p,d,k in itertools.product(('core','failed-near-edge'),range(8),('phase_write','phase_null','neuron_one'),(1,2,4,8))]
check('independent_exact192_ordered_cells',plan['cells_census']==expected and len({r['id'] for r in expected})==192)
check('denominators',plan['cells']==192 and plan['realized_pulse_rows']==1152 and plan['mathematical_contract']['unit_noise_rows']==576 and plan['mathematical_contract']['realized_noise_rows']==3456)
check('no_solve_fit_paid',plan['new_integrations']==plan['model_fits']==plan['child_workers']==0 and plan['paid_compute'] is False and plan['paid_review'] is False)
check('frozen_resource_limits',all(plan[k]==v for k,v in dict(threads=1,producer_seconds=300,per_cell_seconds=10,sealing_seconds=60,owned_rss_bytes=4*1024**3,launch_available_ram_bytes=8*1024**3,reserve_ram_bytes=4*1024**3,raw_budget_bytes=256*1024**2).items()))
check('frozen_math_gates',plan['gates']==dict(algebra_rtol=1e-12,algebra_atol=1e-14,normalization_atol=1e-7,potential_max=1e-6,activation_max=1e-7,standardized_readout=.01,primary_noise=1e-4))
check('original_failure_scope_unchanged',plan['original_setting_qualification']=={'core':False,'failed-near-edge':False} and plan['mathematical_contract']['setting_means'] is False and plan['mathematical_contract']['original_study_requalified'] is False)
old=Path(plan['original_directory']);receipt=read(old/'receipt.json');manifest=read(old/'manifest.json');index={r['path']:r for r in receipt['artifacts']}
check('original_receipt_and_source',bind(old/'receipt.json')['sha256']==plan['original_receipt_sha256']=='2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8' and plan['original_source']==manifest['source_commit']==receipt['source_commit']=='406b89553b956bac1b97ef05413c799e383a9103')
names={'receipt.json','manifest.json','summary.json','all-terminal-records.json','postflight.json','reference-seal.json','reference-seal.npz','reference-seal-replay.json','events.jsonl'}
ledger=read(old/'all-terminal-records.json')
for stage,ids in [('references',{'core','failed-near-edge'}),('propagators',{f'{s}--p{j}' for s,j in itertools.product(('core','failed-near-edge'),range(8))})]:
 rows=ledger[stage];check('old_prerequisite_census:'+stage,len(rows)==len(ids) and {r['id'] for r in rows}==ids)
 for r in rows:
  ident=r['id'];meta=f'{stage}/payloads/{ident}/packet.json';descriptor=read(old/meta)
  check('old_qualified:'+ident,r['status']=='complete' and r['qualified'] is True and r['packet_path']==f'payloads/{ident}/packet.json' and bind(old/meta)['sha256']==r['packet_sha256'])
  names.update((meta,f'{stage}/records/{ident}.json',str(Path(meta).parent/descriptor['arrays']['path'])))
check('exact63_declared_inputs',len(names)==63 and len(plan['input_files'])==63 and {r['path'] for r in plan['input_files']}==names)
check('no_pulse_trace_inputs',all(not p.startswith('traces/') for p in names))
for row in plan['input_files']:
 b=bind(old/row['path']);check('bound_old_input:'+row['path'],b['bytes']==row['bytes'] and b['sha256']==row['sha256'])
 if row['path']!='receipt.json':check('prior_receipt_member:'+row['path'],index[row['path']]==row)
seal=read(old/'reference-seal.json')['tree'];pulse_ids={f'{s}--p{j}--d{d}--{direction}--s{sign:+d}' for s,j,d,direction,sign in itertools.product(('core','failed-near-edge'),range(8),range(3),('phase_write','phase_null','neuron_one'),(-1,1))}
check('old_seal_context288pulse_16sham_census',len(seal['contexts'])==16 and set(seal['cases'])==pulse_ids|{f'{s}--p{j}--sham' for s,j in itertools.product(('core','failed-near-edge'),range(8))})
compatibility=[]
for cell in expected:
 ids=[f'{cell["setting"]}--p{cell["phase_index"]}--d{d}--{cell["direction"]}--s{sign:+d}' for d,sign in itertools.product(range(3),(-1,1))]
 check('six_original_identity_metadata:'+cell['id'],all(i in seal['cases'] and str(cell['delay_cycles']) in seal['cases'][i]['readouts'] for i in ids))
 compatibility.extend((cell['id'],i) for i in ids)
check('exact1152_unique_cell_pulse_identities',len(compatibility)==len(set(compatibility))==1152)
old_sources=manifest['plan']['source_files'];check('source_count_27plus7',len(old_sources)==27 and len(plan['source_files'])==34 and len({r['path'] for r in plan['source_files']})==34)
check('original27_bindings_preserved',all(r in plan['source_files'] for r in old_sources))
for row in plan['source_files']:
 b=bind(ROOT/row['path']);check('current_source:'+row['path'],b['bytes']==row['bytes'] and b['sha256']==row['sha256'])
for row in old_sources:
 data=subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+row['path']]);check('historical_source:'+row['path'],len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'])
check('metadata_only_exact_plan_validate',run.validate(plan)==dict(cells=192,realized_pulse_rows=1152,sources=34,inputs=63,new_integrations=0))
check('unchanged_through_review',plan_before==bind(planpath) and source_before==[bind(ROOT/r['path']) for r in plan['source_files']])
(OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
report=dict(schema_version='EXP015-independent-metadata-preflight-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS; metadata/source/census ready for exact reviewed freeze',checks=len(checks),failures=0,plan=plan_before,script=bind(__file__),details=bind(OUT/'checks.json'),source_files=plan['source_files'],input_files_count=63,counts=dict(cells=192,realized_compatibility_rows=1152,unit_noise_rows=576,realized_noise_rows=3456),scope='Source/input hash and symbolic identity checks only. Gain, Hermite interpolation, full-linear propagation and RHS APIs were replaced by exceptions during validation. No new mathematical target outputs, ODE or fits. Synthetic test and mathematical review records are supplied separately.',new_gain_cells_computed=0,new_integrations=0)
f=OUT/'EXP-015-independent-metadata-preflight.json';f.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','checks','failures','plan','counts')},indent=2));print(json.dumps(bind(f),indent=2))
