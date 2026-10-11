#!/usr/bin/env python3
"""Review final integrated plan and old saved-input compatibility; never integrate."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from collections import Counter
import json,hashlib,sys,itertools,time,subprocess
ROOT=Path('/data2/neural-shrimp');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import run_phase_memory_refinement as r
start=time.monotonic();checks=[]
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise AssertionError(name)
planpath=ROOT/'experiments/hopfield/EXP-014-refinement-plan.json';p=read(planpath);planbinding=bind(planpath)
sourcesbefore=[bind(ROOT/x['path']) for x in p['source_files']]
testrecord=read(OUT/'synthetic-tests.json')
check('reviewed_tests_passed',testrecord['exit_code']==0 and testrecord['sources']==[bind(ROOT/x) for x in r.NEW_SOURCE_PATHS if x.endswith('.py')])
old=Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002');receipt=read(old/'receipt.json');manifest=read(old/'manifest.json');summary=read(old/'summary.json')
check('old_receipt_pinned',bind(old/'receipt.json')['sha256']=='2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8')
check('old_source_pinned',receipt['source_commit']==manifest['source_commit']==p['original_source']=='406b89553b956bac1b97ef05413c799e383a9103')
oldindex={x['path']:x for x in receipt['artifacts']}
check('old_inventory_unique',len(oldindex)==len(receipt['artifacts']))
oldtasks=manifest['plan']['tasks'];failed=[x for x in summary['scientific']['trace_comparisons'] if x['qualified'] is not True]
failedsigned={i.rsplit('--',1)[0] for x in failed for i in x['ids']}
check('thirty_pairs_eighteen_signed_failures',len(failed)==30 and len(failedsigned)==18)
byid={x['id']:x for x in oldtasks};selectedtasks=[byid[i] for x in failed for i in x['ids']]
check('all_failures_largest_dose',all(x['kind']=='pulse' and x['dose_index']==2 and x['epsilon']==.01 for x in selectedtasks))
binary={(x['setting'],x['phase_index'],x['direction']) for x in selectedtasks};phases={(s,j) for s,j,d in binary}
expected=[]
for task in oldtasks:
 if task['method']!='rk4-0.0025':continue
 if not ((task['kind']=='sham' and (task['setting'],task['phase_index']) in phases) or (task['kind']=='pulse' and (task['setting'],task['phase_index'],task['direction']) in binary)):continue
 x={k:v for k,v in task.items() if k!='method'};x['id']=task['id'].rsplit('--',1)[0];expected.append(x)
check('independent_exact_ordered85census',expected==p['selection']['cases'] and len(expected)==85)
check('signed_complete_context_counts',len(binary)==13 and len(phases)==7)
check('asymmetric_setting_counts',Counter(x['setting'] for x in expected)=={'core':26,'failed-near-edge':59})
check('category_counts',Counter(p['selection']['selection_categories'].values())=={'original-failed-strong-sign':18,'matched-strong-opposite-sign':8,'matched-lower-dose-both-signs':52,'matched-sham':7})
methods=[dict(id=f'rk4-{h:g}',kind='RK4',dt=h) for h in (.000625,.0003125,.00015625)]+[dict(id=f'dop853-1e-13-h{h:g}',kind='DOP853',rtol=1e-13,atol=1e-15,max_step=h) for h in (.0125,.00625)]
check('exact5_new_solver_methods',p['new_methods']==methods)
expectedtasks=[dict(id=c['id']+'--'+m['id'],kind='refinement_trace',case=c,method=m) for c in expected for m in methods]
check('exact425_target_identities',r.tasks(p)==expectedtasks and len(expectedtasks)==425)
allmethods=['rk4-0.0025','rk4-0.00125','dop853']+[m['id'] for m in methods]
specs=r.evaluate.comparison_specs(expected)
check('all2380_ordered_pairs',[(x['case_id'],x['pair']) for x in specs]==[(c['id'],list(pair)) for c in expected for pair in itertools.combinations(allmethods,2)] and len(specs)==2380)
check('pair_groups',Counter(x['group'] for x in specs)=={'old-old':255,'old-new':1275,'new-new':850})
check('ba_census',13*3*8*4*3==p['paired_ba_cells']==3744)
check('exact_duration_sample_readout_noise',all(p[k]==v for k,v in dict(duration=320.,sample_dt=.025,samples=12801,readout_delays=[1,2,4,8],noise_levels=[1e-5,1e-4,1e-3],primary_noise=1e-4,late_interval=[240.,320.]).items()))
check('unchanged_thresholds',all(p[k]==v for k,v in dict(potential_max_threshold=1e-6,activation_max_threshold=1e-7,primary_standardized_readout_threshold=.01).items()))
check('bounded_local_resources',all(p[k]==v for k,v in dict(workers=4,thread_limit=1,producer_seconds=900,per_trace_seconds=60,sealing_seconds=120,launch_available_ram_bytes=16*1024**3,reserve_ram_bytes=8*1024**3,owned_rss_bytes=12*1024**3,raw_budget_bytes=2*1024**3,paid_compute=False,paid_review=False).items()))
check('old27_plus6_source_count',len(p['source_files'])==33 and len(manifest['plan']['source_files'])==27)
check('all27_prior_source_bindings_unchanged',all(x in p['source_files'] for x in manifest['plan']['source_files']))
for x in p['source_files']:
 actual=bind(ROOT/x['path']);check('current_source:'+x['path'],actual['bytes']==x['bytes'] and actual['sha256']==x['sha256'])
for x in manifest['plan']['source_files']:
 data=subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+x['path']]);check('frozen_old_source:'+x['path'],len(data)==x['bytes'] and hashlib.sha256(data).hexdigest()==x['sha256'])
check('input_census803',len(p['input_files'])==803 and len({x['path'] for x in p['input_files']})==803)
for x in p['input_files']:
 actual=bind(old/x['path']);check('old_input:'+x['path'],actual['bytes']==x['bytes'] and actual['sha256']==x['sha256'])
 if x['path']!='receipt.json':check('old_receipt_membership:'+x['path'],oldindex.get(x['path'])==x)
check('exact_original_selection_status',p['original_setting_qualification']=={'core':False,'failed-near-edge':False})
check('exact_plan_reconstruction',r.validate(p,reconstruct=True)==dict(physical_cases=85,new_traces=425,old_traces=255,comparisons=2380,paired_ba_cells=3744))
replaystart=time.monotonic();prepared=r.codec.load_packet(old/'reference-seal.json');contexts=r.make_contexts(p,prepared);packets,scores=r.original_packets(p,contexts,replay=True)
check('all255_saved_packets_1020_readouts_replayed',len(contexts)==85 and len(packets)==len(scores)==255 and sum(len(s['readouts']) for s in scores.values())==1020)
replayseconds=time.monotonic()-replaystart
check('sources_unchanged_during_review',sourcesbefore==[bind(ROOT/x['path']) for x in p['source_files']] and planbinding==bind(planpath))
(OUT/'metadata-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
report=dict(schema_version='EXP014-refinement-independent-preflight-v1',status='PASS; ready for exact reviewed private freeze and guarded run',checks=len(checks),failures=0,
 script=bind(__file__),plan=planbinding,source_files=p['source_files'],input_files_count=803,check_details=bind(OUT/'metadata-checks.json'),
 synthetic_tests=bind(OUT/'synthetic-tests.json'),
 old_saved_input_replay=dict(contexts=85,packets=255,readouts=1020,rtol=1e-12,atol=1e-14,seconds=replayseconds,new_integrations=0),
 census=dict(physical_cases=85,new_traces=425,old_traces=255,all_method_rows=680,comparison_pairs=2380,same_method_paired_BA_cells=3744),
 scientific_conclusion='Exact outcome-selected signed-complete85-case diagnosis, unchanged original references/clock/orientation/time grid and scientific thresholds. The same-method BA probabilities use assumed Gaussian measurement noise and retain unsaturated signedq. First crossings refer to saved grid samples or four declared readout windows, not continuous-time proofs. No original setting averages, solver selection, gate waiver or topology/utility inference.',
 closed_review_findings=['Analysis completeness participates in producer/verifier completion, with failures retained.','No-dispatch interruption after persisted input seal is accepted only as validated failed partial evidence; dispatched traces require full seal chronology.','Original immutable evidence paths are protected from output placement.','True solver method/ID metadata bound.','All full and partial score packets, including absolute shamreadouts, retained.'],
 limitations=['No new refinement outcomes were acquired or examined in this preflight.','Saved old readout replay uses the supplied unchanged mathematical scoring primitive, not independent alternative-interpolant arithmetic.','Prior EXP014 remains unqualified; its independent alternative-arithmetic strict replay failures remain preserved.','Uncatchable kills/native buffers have the inherited explicitly documented partial-evidence limits.','Root must commit/private-push exact reviewed code,plan and local review and satisfy live resource/source gates before acquisition.'],
 elapsed_seconds=time.monotonic()-start)
(OUT/'EXP-014-refinement-independent-preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','checks','failures','plan','old_saved_input_replay','elapsed_seconds')},indent=2))
