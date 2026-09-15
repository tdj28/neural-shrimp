"""Independent EXP-010 terminal-value ranks and qualification audit, no integrations."""
import collections,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
P=Path('/data2/neural-shrimp-data/raw/EXP-010/attempt-001')
plan=json.loads((P/'manifest.json').read_text()); records=json.loads((P/'terminal-records.json').read_text()); summary=json.loads((P/'summary.json').read_text())
checks=0

def check(actual,expected,label):
 global checks
 checks+=1
 assert actual==expected,(label,actual,expected)

def dist(values):
 a=np.asarray(values,dtype=float)
 return {'n':len(a),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'maximum':float(np.max(a))} if len(a) else {'n':0,'median':None,'p95':None,'maximum':None}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

native=['rk4-0.0025','rk4-0.00125']; independent=['dop853-1e-11','dop853-1e-12']; levels=['old_0.005']+native
r={(x['case_id'],x['method']):x for x in records};cases={x['id']:x for x in plan['census']};details={x['id']:x for x in summary['details']}
check(len(r),len(records),'duplicate terminal keys');check(len(cases),3456,'cases');check(len(details),3456,'saved detail census')
expected={(c['id'],m) for c in cases.values() for m in native+(independent if c['independent_labels'] else [])}
check(set(r),expected,'exact terminal method census');check(len(expected),6944,'expected tasks');check(set(details),set(cases),'detail IDs')
check(collections.Counter(x['status'] for x in records),{'complete':6944},'all planned methods complete')
for x in records:check(math.isfinite(x['rmse']),True,'finite terminal RMSE')
check(sum(bool(c['independent_labels']) for c in cases.values()),16,'sentinel count')
# Reconstruct preselection from original scalar values within all 8 role/radius strata.
for role,rad in itertools.product(['development','test'],[.00025,.001,.004,.016]):
 group=[c for c in cases.values() if c['key'][0]==role and c['key'][3]==rad]
 check(len(group),432,'sentinel stratum population')
 ordered=sorted(group,key=lambda c:(c['old_rmse'],tuple(c['key'])))
 median=ordered[215];maximum=min(group,key=lambda c:(-c['old_rmse'],tuple(c['key'])))
 planned=next(s for s in plan['independent_strata'] if s['role']==role and s['radius']==rad)
 check(median['key'],planned['lower_median_key'],'lower median sentinel');check(maximum['key'],planned['maximum_key'],'max sentinel')
 check({c['id'] for c in group if c['independent_labels']},{median['id'],maximum['id']},'sentinel membership')

calc={}; contexts=collections.defaultdict(list); pair_metrics=collections.defaultdict(list)
for ident,c in cases.items():
 d=details[ident]; expectedmethods=native+(independent if c['independent_labels'] else [])
 v={m:r[ident,m]['rmse'] for m in expectedmethods}; old=c['old_rmse'];d1=abs(v[native[0]]-old);d2=abs(v[native[1]]-v[native[0]])
 fields=dict(d1=d1,d2=d2,threshold_gate=d2<=.0001,nonincreasing_gate=d2<=d1,rmse_stable_at_this_resolution=d2<=.0001 and d2<=d1,finest_minus_old=v[native[1]]-old,all_required_methods_complete=all(r[ident,m]['status']=='complete' for m in expectedmethods))
 for k,x in fields.items():check(d[k],x,ident+':'+k)
 check(d['new_scores'],{m:{'status':r[ident,m]['status'],'rmse':v[m]} for m in expectedmethods},ident+':all method values')
 ind=True
 if c['independent_labels']:
  ind=all(abs(v[a]-v[b])<=.0001 for a,b in itertools.combinations([native[1]]+independent,2));check(d['independent_rmse_agreement'],ind,ident+':independent scalar gate')
 full=fields['rmse_stable_at_this_resolution'] and ind
 check(d['full_case_scalar_qualification'],full,ident+':full scalar gate')
 waves=[]
 for pname,w in d['waveform_comparisons'].items():
  a,b=pname.split('_vs_');check(w['rmse_difference'],abs(v[a]-v[b]),ident+':pair scalar')
  check(w['activation_waveform_agreement'],w['activation_rms']<=.001 and w['activation_max']<=.01,ident+':wave threshold')
  pair_metrics[pname].append(dict(case_id=ident,**w));waves.append(w['activation_waveform_agreement'])
 calc[ident]=dict(**fields,full_scalar_qualification=full,independent_scalar_gate=ind,all_required_waveform_flags=all(waves),old=old,scores=v)
 contexts[tuple(c['key'][:4])].append(c)
check(len(contexts),192,'contexts')
# Exact assignment-to-case multiset preservation; no selected-only replacement denominator.
assignments=collections.defaultdict(list)
for a in plan['assignments']:assignments[tuple(a['key'])].append(a['policy'])
check(len(plan['assignments']),384,'assignments');check(len(assignments),220,'unique selected')
for c in cases.values():check(sorted(c['original_selected_policies']),sorted(assignments[tuple(c['key'])]),c['id']+':selected policy membership')

selected=[c for c in cases.values() if c['original_selected_policies']]
check(summary['original_selected_unique'],dict(required=220,native_complete=220,native_all_required_complete=True,all_required_methods_complete=True,full_scalar_qualification=all(calc[c['id']]['full_scalar_qualification'] for c in selected),scalar_stable=sum(calc[c['id']]['rmse_stable_at_this_resolution'] for c in selected)),'selected summary')
check(summary['qualification'],dict(native_all_required_complete=True,all_required_methods_complete=True,full_scalar_qualification=all(v['full_scalar_qualification'] for v in calc.values()),all_native_scalar_flags_pass=all(v['rmse_stable_at_this_resolution'] for v in calc.values()),all_native_waveform_flags_pass=all(v['activation_waveform_agreement'] for v in pair_metrics[native[0]+'_vs_'+native[1]])),'global full qualification')
for field,expectedvalue in dict(required=6944,observed=6944,cases=3456,scalar_stable=3456,scalar_unresolved=0,scalar_undefined=0,all_complete=True,status_counts={'complete':6944}).items():check(summary[field],expectedvalue,'summary '+field)
strata={}
for role,rad,subset in itertools.product(['development','test'],[.00025,.001,.004,.016],['all','original_selected_unique']):
 members=[c for c in cases.values() if c['key'][0]==role and c['key'][3]==rad and (subset=='all' or c['original_selected_policies'])]
 if not members:continue
 n=len(members);actual=dict(required=n,native_complete=n,native_all_required_complete=True,all_required_methods_complete=True,full_group_scalar_qualification=all(calc[c['id']]['full_scalar_qualification'] for c in members),scalar_stable=sum(calc[c['id']]['rmse_stable_at_this_resolution'] for c in members),scalar_unresolved=sum(not calc[c['id']]['rmse_stable_at_this_resolution'] for c in members),waveform_agree=sum(details[c['id']]['waveform_comparisons'][native[0]+'_vs_'+native[1]]['activation_waveform_agreement'] for c in members),d2_completed_only=dist([calc[c['id']]['d2'] for c in members]),absolute_old_to_finest_completed_only=dist([abs(calc[c['id']]['finest_minus_old']) for c in members]))
 key=f'{role}-r{rad:g}-{subset}';check(summary['role_radius_groups'][key],actual,'stratum '+key);strata[key]=actual
check(set(summary['role_radius_groups']),set(strata),'all strata')

savedranks={tuple(x['context']):x for x in summary['context_rankings']};check(set(savedranks),set(contexts),'ranking contexts')
comparisons=collections.defaultdict(list); near_envelope=[];rank_payload=[]
for key,group in sorted(contexts.items()):
 group.sort(key=lambda c:c['key'][4]);check([c['key'][4] for c in group],list(range(18)),'18 entries, duplicate directions retained')
 scores={level:[c['old_rmse'] if level==levels[0] else r[c['id'],level]['rmse'] for c in group] for level in levels};rankings={}
 for level,values in scores.items():
  order=np.lexsort((np.arange(18),values)).tolist();axisIDs=np.array([0,4,8,12]);ax=int(axisIDs[np.lexsort((axisIDs,np.array(values)[axisIDs]))[0]])
  rankings[level]=dict(order=order,argmin=order[0],top_four=order[:4],best_second_gap=values[order[1]]-values[order[0]],fourth_fifth_gap=values[order[4]]-values[order[3]],signed_axes_argmin=ax,signed_axes_minimum=values[ax])
 check(savedranks[key]['scores'],scores,'all 54 context values');check(savedranks[key]['rankings'],rankings,'all rank fields');check(savedranks[key]['candidate_ids'],list(range(18)),'saved IDs')
 rank_payload.append(dict(context=key,scores=scores,rankings=rankings))
 for a,b in itertools.combinations(levels,2):
  va,vb=scores[a],scores[b];ra,rb=rankings[a],rankings[b];delta=[y-x for x,y in zip(va,vb)];env=max(map(abs,delta));mc=min(vb)-min(va);ac=rb['signed_axes_minimum']-ra['signed_axes_minimum']
  result=dict(argmin_changed=ra['argmin']!=rb['argmin'],top_four_membership_changed=set(ra['top_four'])!=set(rb['top_four']),ranking_changed=ra['order']!=rb['order'],signed_axes_argmin_changed=ra['signed_axes_argmin']!=rb['signed_axes_argmin'],signed_candidate_deltas=delta,empirical_max_absolute_score_change=env,old_best_second_gap_exceeds_twice_change=ra['best_second_gap']>2*env,old_fourth_fifth_gap_exceeds_twice_change=ra['fourth_fifth_gap']>2*env,signed_minimum_error_change=mc,minimum_error_change_above_numerical_tolerance=abs(mc)>.0001,signed_axes_minimum_error_change=ac,absolute_axes_minimum_error_change=abs(ac))
  pname=a+'_vs_'+b
  for k,v in result.items():check(savedranks[key]['comparisons'][pname][k],v,pname+':'+k)
  comparisons[pname].append(dict(context=key,**result))
  if not result['old_best_second_gap_exceeds_twice_change'] or not result['old_fourth_fifth_gap_exceeds_twice_change']:
   near_envelope.append(dict(context=key,comparison=pname,earlier_ranking=ra,later_ranking=rb,earlier_scores=va,later_scores=vb,**result))

aggregates={}
for pair,rows in comparisons.items():
 aggregates[pair]={'contexts':len(rows),'counts':{field:sum(x[field] for x in rows) for field in ['argmin_changed','top_four_membership_changed','ranking_changed','signed_axes_argmin_changed','old_best_second_gap_exceeds_twice_change','old_fourth_fifth_gap_exceeds_twice_change','minimum_error_change_above_numerical_tolerance']},'axes_minimum_changes_above_numerical_tolerance':sum(x['absolute_axes_minimum_error_change']>.0001 for x in rows),'candidate_score_change_envelope':dist([x['empirical_max_absolute_score_change'] for x in rows]),'absolute_minimum_error_change':dist([abs(x['signed_minimum_error_change']) for x in rows]),'absolute_axes_minimum_error_change':dist([x['absolute_axes_minimum_error_change'] for x in rows]),'maximum_candidate_change_context':max(rows,key=lambda x:x['empirical_max_absolute_score_change'])['context'],'maximum_minimum_change_context':max(rows,key=lambda x:abs(x['signed_minimum_error_change']))['context']}
wave_aggregates={}
for pair,rows in pair_metrics.items():
 wave_aggregates[pair]={'required_pairs':len(rows),'passed':sum(x['activation_waveform_agreement'] for x in rows),**{k:dist([x[k] for x in rows]) for k in ['rmse_difference','activation_rms','activation_max','state_rms','state_max']},'maximum_rmse_difference_case':max(rows,key=lambda x:x['rmse_difference'])['case_id']}
sentinels=[]
for c in cases.values():
 if c['independent_labels']:sentinels.append(dict(case_id=c['id'],key=c['key'],selection_labels=c['independent_labels'],old_rmse=c['old_rmse'],new_rmse=calc[c['id']]['scores'],d1=calc[c['id']]['d1'],d2=calc[c['id']]['d2'],scalar_qualification=calc[c['id']]['full_scalar_qualification'],all_waveform_flags=calc[c['id']]['all_required_waveform_flags'],independent_pair_differences={a+'_vs_'+b:abs(r[c['id'],a]['rmse']-r[c['id'],b]['rmse']) for a,b in itertools.combinations([native[1]]+independent,2)}))
report=dict(audit='EXP-010 independent terminal-value rank and qualification reconstruction',producer_sources_imported=False,source_commit='98632899646f3eefa292b8a9f4c1b5123ae37ba0',attempt=str(P),input_sha256={name:sha(P/name) for name in ['manifest.json','terminal-records.json','summary.json','receipt.json']},equality_checks=checks,qualification=summary['qualification'],counts=dict(cases=3456,contexts=192,terminal_records=6944,native_traces=6912,mandatory_independent_traces=32,mandatory_independent_cases=16,missing=0,nonfinite=0,scalar_pass=3456,scalar_unresolved=0,undefined=0,native_waveform_pairs=3456,independent_waveform_pairs=48,all_required_waveform_flags_pass=all(c['all_required_waveform_flags'] for c in calc.values())),selected=dict(**summary['original_selected_unique'],assignments=384,policy_assignments=dict(collections.Counter(a['policy'] for a in plan['assignments'])),d2=dist([calc[c['id']]['d2'] for c in selected]),absolute_old_to_finest=dist([abs(calc[c['id']]['finest_minus_old']) for c in selected])),native_d1=dist([c['d1'] for c in calc.values()]),native_d2=dist([c['d2'] for c in calc.values()]),native_d2_over_d1=dist([c['d2']/c['d1'] for c in calc.values() if c['d1']!=0]),absolute_old_to_finest=dist([abs(c['finest_minus_old']) for c in calc.values()]),strata=strata,rank_comparisons=aggregates,near_envelope_contexts=near_envelope,sentinels=sentinels,waveform_aggregates=wave_aggregates,notes=['Waveform numerical values were independently replayed exactly from all saved arrays by sibling recent_results; this audit rechecks every scalar comparison and waveform threshold flag without importing the producer.', 'All required terminal keys were present, complete, and finite. No completed-only subset substituted for an all-required denominator.','Empirical two-level discrepancy envelopes do not bound exact-solution error or provide a task noninferiority tolerance.','No fitting, integration, source edit, original action reselection or new utility claim.'])
out=Path('/tmp/EXP-010-independent-ranking-audit.json');out.write_text(json.dumps(report,indent=2)+'\n');print('report',out,'bytes',out.stat().st_size,'sha256',sha(out));print('checks',checks);print(json.dumps({k:report[k] for k in ['counts','selected','native_d1','native_d2','native_d2_over_d1','absolute_old_to_finest','rank_comparisons']},indent=2));print('near envelope',json.dumps(near_envelope,indent=2))
