#!/usr/bin/env python3
"""Reconstruct every saved diagnostic decision; no trajectory replay/ODE/fit."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,math,itertools,datetime
ROOT=Path('/data2/neural-shrimp');RAW=Path('/data2/neural-shrimp-data/raw/EXP-014-refinement/attempt-001');OUT=Path(__file__).resolve().parent
NEW=('rk4-0.000625','rk4-0.0003125','rk4-0.00015625','dop853-1e-13-h0.0125','dop853-1e-13-h0.00625');OLD=('rk4-0.0025','rk4-0.00125','dop853');METHODS=OLD+NEW
checks=[]
def read(p):return json.loads(Path(p).read_text())
def bind(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def check(name,value):
 checks.append(dict(name=name,passed=bool(value)))
 if not value:raise AssertionError(name)
receipt=read(RAW/'receipt.json');summary=read(RAW/'summary.json');manifest=read(RAW/'manifest.json');a=summary['analysis'];cases=manifest['plan']['selection']['cases'];original_failed=set(manifest['plan']['selection']['failed_signed_ids'])
check('receipt_pinned',bind(RAW/'receipt.json')['sha256']=='23ade00539cb1a138dbac3afc64e183b306f34d108beb7e6f58873b4c062243e')
check('source_and_plan_pinned',receipt['source_commit']==manifest['source_commit']=='67be43cc5ed821543bca384eaf574dbff26d73f3' and receipt['plan_sha256']==manifest['plan_sha256']=='150e18cf4b528ea8740e498a2cc610454e7f313dcce482a6a4735e0016104bfe')
index={x['path']:x for x in receipt['artifacts']}
for filename in ('summary.json','manifest.json','postflight.json','all-terminal-records.json'):
 b=bind(RAW/filename);check('receipt_bound:'+filename,index[filename]==dict(path=filename,bytes=b['bytes'],sha256=b['sha256']))
check('complete_valid_source',summary['complete'] is True and summary['source_valid'] is True and summary['error'] is None and summary['postflight_error'] is None and receipt['status']=='complete')
check('no_original_requalification',a['original_study_requalified'] is False and a['setting_means'] is None and summary['original_unchanged'] is True)
check('425_new_complete',summary['status_counts']=={'complete':425} and len(read(RAW/'all-terminal-records.json'))==425)
check('680_complete_scores',len(a['terminal_rows'])==680 and all(r['status']=='complete' for r in a['terminal_rows']))
expected=[(c['id'],list(pair)) for c in cases for pair in itertools.combinations(METHODS,2)]
check('2380_exact_ordered_pairs',[(r['spec']['case_id'],r['spec']['pair']) for r in a['comparisons']]==expected)
for r in a['comparisons']:
 check(r['spec']['id']+':complete',r['complete'] is True)
 state=r['state_difference']['maximum_absolute']<=1e-6;activation=r['activation_difference']['maximum_absolute']<=1e-7
 q=[v['standardized_mean_difference']<=.01 for v in r['readouts'].values()]
 check(r['spec']['id']+':decisions',state==r['state_pass'] and activation==r['activation_pass'] and q==[v['passed'] for v in r['readouts'].values()] and r['qualified']==(state and activation and all(q)))
summary_groups={};case_details=[]
for group in ('old-old','old-new','new-new'):
 rows=[r for r in a['comparisons'] if r['spec']['group']==group]
 stats=dict(required=len(rows),complete=len(rows),passed=sum(r['qualified'] for r in rows))
 check('saved_group_counts:'+group,stats==a['comparison_groups'][group])
 stats.update(failed=len(rows)-stats['passed'],failing_cases=len({r['spec']['case_id'] for r in rows if not r['qualified']}),state_failures=sum(not r['state_pass'] for r in rows),activation_failures=sum(not r['activation_pass'] for r in rows),readout_failures=sum(any(not v['passed'] for v in r['readouts'].values()) for r in rows),failing_readout_cells=sum(not v['passed'] for r in rows for v in r['readouts'].values()))
 for key,field in [('maximum_state','state_difference'),('maximum_activation','activation_difference')]:
  row=max(rows,key=lambda r:r[field]['maximum_absolute']);stats[key]=dict(value=row[field]['maximum_absolute'],pair_id=row['spec']['id'],saved_time=row[field]['maximum_time'],component=row[field]['maximum_component'])
 row,delay=max(((r,d) for r in rows for d in r['readouts']),key=lambda item:item[0]['readouts'][item[1]]['standardized_mean_difference']);stats['maximum_standardized_readout_difference']=dict(value=row['readouts'][delay]['standardized_mean_difference'],pair_id=row['spec']['id'],delay_cycles=int(delay))
 stats['failure_count_by_method_pair']=[dict(methods=list(pair),failed=sum(r['spec']['pair']==list(pair) and not r['qualified'] for r in rows)) for pair in itertools.combinations(METHODS,2) if any(r['spec']['pair']==list(pair) for r in rows)]
 stats['failure_count_by_setting']=dict(Counter(r['spec']['setting'] for r in rows if not r['qualified']))
 summary_groups[group]=stats
for c in cases:
 detail=dict(case=c,original_failed_sign=c['id'] in original_failed,groups={})
 for group in ('old-old','old-new','new-new'):
  rows=[r for r in a['comparisons'] if r['spec']['case_id']==c['id'] and r['spec']['group']==group];bad=[r for r in rows if not r['qualified']]
  detail['groups'][group]=dict(required=len(rows),passed=len(rows)-len(bad),failed_pairs=[r['spec']['pair'] for r in bad],earliest_state_sample=min((r['state_difference']['first_threshold_crossing_time'] for r in rows if r['state_difference']['first_threshold_crossing_time'] is not None),default=None),earliest_activation_sample=min((r['activation_difference']['first_threshold_crossing_time'] for r in rows if r['activation_difference']['first_threshold_crossing_time'] is not None),default=None),first_failed_declared_readout=min((int(d) for r in rows for d,v in r['readouts'].items() if not v['passed']),default=None))
 case_details.append(detail)
newfailed=[c for c in case_details if c['groups']['new-new']['passed']<10];allpass=[c for c in case_details if all(x['passed']==x['required'] for x in c['groups'].values())]
check('all_new_failure_original_strong',all(c['original_failed_sign'] and c['case']['epsilon']==.01 for c in newfailed))
# Recompute same-method BA from saved q only. This does not independently reconstruct q from raw arrays.
keys=('setting','phase_index','direction','dose_index','delay_cycles','sigma');bagroups=defaultdict(list)
check('BA_count_complete',len(a['paired_readouts'])==3744 and all(b['status']=='complete' for b in a['paired_readouts']))
maxbadiff=0.
for i,b in enumerate(a['paired_readouts']):
 z=[b['orientation']*b['q_plus']/(b['sigma']*b['noise_gain']),-b['orientation']*b['q_minus']/(b['sigma']*b['noise_gain'])];ba=sum(.5*(1+math.erf(v/math.sqrt(2))) for v in z)/2;diff=abs(ba-b['balanced_accuracy']);maxbadiff=max(maxbadiff,diff)
 check(f'BA{i}:same_method_saved_q_replay',diff<=2e-15 and all(abs(x-y)<=2e-12*max(1,abs(y)) for x,y in zip(z,b['standardized_correct_means'])))
 bagroups[tuple(b[k] for k in keys)].append(b)
ba_ranges=[]
for key,rows in bagroups.items():
 check('all8_BA_methods:'+str(key),len(rows)==8 and {r['method'] for r in rows}==set(METHODS))
 for label,methods in [('all8',METHODS),('new5',NEW)]:
  members=[r for r in rows if r['method'] in methods];values=[r['balanced_accuracy'] for r in members]
  ba_ranges.append(dict(zip(keys,key),methods=label,range=max(values)-min(values),minimum=min(values),maximum=max(values)))
newbinary={(c['case']['setting'],c['case']['phase_index'],c['case']['direction']) for c in newfailed}
report=dict(schema_version='EXP014-refinement-saved-interpretation-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Saved summary arithmetic and scientific interpretation, not independent raw trajectory replay. No new integration or fitting.',script=bind(__file__),source_commit=receipt['source_commit'],plan_sha256=receipt['plan_sha256'],raw_receipt=bind(RAW/'receipt.json'),raw_summary=bind(RAW/'summary.json'),artifact_count=len(receipt['artifacts']),artifact_bytes=receipt['artifact_bytes'],checks=len(checks),failures=0,
 counts=dict(new_traces=425,old_traces=255,physical_cases=85,score_rows=680,comparisons=2380,paired_BA_cells=3744,all_28_pairs_agree_cases=len(allpass),all_10_new_pairs_agree_cases=85-len(newfailed),new_method_disagreement_cases=len(newfailed),original_failed_signs_with_all_new_pairs_agree=sum(c['original_failed_sign'] and c['groups']['new-new']['passed']==10 for c in case_details),strong_binary_contexts_with_both_signs_new_consistent=13-len(newbinary),strong_binary_contexts_total=13,lower_dose_binary_contexts_with_both_signs_new_consistent=26,lower_dose_binary_contexts_total=26,shams_all_pairs_agree=sum(c['case']['kind']=='sham' for c in allpass)),
 groups=summary_groups,new_method_disagreement_cases=newfailed,
 BA_diagnostics=dict(saved_q_BA_max_arithmetic_difference=maxbadiff,all_sigma_all8_max_range=max((b for b in ba_ranges if b['methods']=='all8'),key=lambda b:b['range']),primary_sigma_all8_max_range=max((b for b in ba_ranges if b['methods']=='all8' and b['sigma']==1e-4),key=lambda b:b['range']),all_sigma_new5_max_range=max((b for b in ba_ranges if b['methods']=='new5'),key=lambda b:b['range']),primary_sigma_new5_max_range=max((b for b in ba_ranges if b['methods']=='new5' and b['sigma']==1e-4),key=lambda b:b['range']),exact_probability_counts={str(v):sum(b['balanced_accuracy']==v for b in a['paired_readouts']) for v in (0.,.5,1.)}),
 timing=dict(stage_wall_seconds=summary['stage_wall_seconds'],producer_and_analysis_seconds=summary['producer_and_analysis_seconds']),
 conclusion='Acquisition complete; large-dose finite-time numerical sensitivity persists in eight selected signed cases. Seventy-seven cases agree across all five new methods, but both original setting results remain unqualified. Seven cases fail every new-method pair, including both tighter DOP853 solves; near-edge p3 neuron-one positive fails only four pairs involving RK4 .000625. This is not a license to select a favorable solver. Small BA ranges coexist with severe state, activation and unsaturated-q differences.',
 next_step='Proceed to a separately frozen saved-reference phase/transverse observable-gain decomposition over all16 original phase contexts, three directions and four windows (192 cells), without new ODE or fitted decoder. Keep large-dose unresolved cases explicit and compare only conditionally numerically supported finite-pulse cases; no original setting means or posthoc horizon change. Do not infer chaos, new attractors, topology utility or trained-network generalization from disagreement.',
 distinctions=['The48 old-old failures comprise the30 original comparisons plus18 additional coarse-RK4 versus original-DOP comparisons; the original gate did not change.','The13 selected binary contexts were chosen from old failures and settings have unequal selected counts; no between-setting incidence or population estimate is justified.','The425 new solves add diagnostic numerical methods, not independent scientific replicates.','BA near0 means the frozen decoder orientation is opposite the signed response; no silent label inversion or inference of absent information.','No first saved-grid crossing is a continuous-time onset proof.','All3,744 diagnostic BA cells remain available, but none requalifies the original study.'])
ind=Path('/data2/neural-shrimp-data/audits/EXP-014-refinement/attempt-001-independent/report.json');ind_data=read(ind)
report['independent_raw_audit']=dict(binding=bind(ind),evidence_passed=ind_data['evidence_and_scientific_decisions_passed'],evidence_checks=ind_data['evidence_checks'],evidence_failures=len(ind_data['evidence_failures']),strict_arithmetic_passed=ind_data['strict_arithmetic_replay_passed'],strict_arithmetic_checks=ind_data['strict_arithmetic_checks'],strict_arithmetic_failures=len(ind_data['strict_arithmetic_failures']),maximum_failed_scalar_difference=max(x['details']['maximum_absolute_difference'] for x in ind_data['strict_arithmetic_failures']),maximum_scalar_difference_including_tolerance_passes=ind_data['maximum_scalar_arithmetic_difference'],arithmetic_tolerances=ind_data['arithmetic_tolerances'],note='Evidence and scientific decisions pass; strict alternate arithmetic replay fails. No tolerance changed or audit PASS claim. The maximum across all scalar checks can exceed the maximum failed check because relative tolerances also apply.')
(OUT/'case-and-BA-diagnostics.json').write_text(json.dumps(dict(cases=case_details,BA_ranges=ba_ranges),indent=2)+'\n');report['case_diagnostics']=bind(OUT/'case-and-BA-diagnostics.json')
(OUT/'saved-summary-checks.json').write_text(json.dumps(checks,indent=2)+'\n');report['checks_binding']=bind(OUT/'saved-summary-checks.json')
path=OUT/'EXP-014-refinement-scientific-interpretation.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('checks','counts','BA_diagnostics','artifact_count','artifact_bytes')},indent=2));print(json.dumps(bind(path),indent=2))
