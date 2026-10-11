"""EXP011 saved-evidence statistical audit. No producer imports or refitting."""
import collections,hashlib,itertools,json,math,os,time
from pathlib import Path
import numpy as np
P=Path('/data2/neural-shrimp-data/raw/EXP-011/attempt-001')
RES=['rk4-0.00125','rk4-0.0025']; SCHEMAS=['shared','geometry']; CONTROL={'axes':[0,4,8,12],'diagonal':[2,6,10,14],'local_coverage':[0,8,16,17]};POL=SCHEMAS+list(CONTROL)
MET=['rmse','excess_rmse','winner_inclusion','any_minimizer_inclusion','best_selected_rank','mean_selected_rank']
read=lambda name:json.loads((P/name).read_text())
manifest=read('manifest.json');receipt=read('receipt.json');summary=read('summary.json');records=read('terminal-records.json');timings=read('timings.json')
checks=0

def eq(a,b,label):
 global checks
 checks+=1
 assert a==b,(label,a,b)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stats(x):
 a=np.asarray(x,float);return {'n':len(a),'minimum':float(a.min()),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),'maximum':float(a.max())}

rows=manifest['rows'];groups=sorted({r['group'] for r in rows});byid={r['id']:r for r in rows};byctx=collections.defaultdict(list);names={r['group']:r['anchor'] for r in rows}
for r in rows:byctx[(r['group'],r['phase'],r['radius'])].append(r)
for v in byctx.values():v.sort(key=lambda r:r['candidate']);eq([r['candidate'] for r in v],list(range(18)),'candidate census')
eq(len(groups),24,'24 origins');eq(len(byctx),192,'192 contexts');eq(len(rows),3456,'3456 rows');eq(len(records),24,'24 records');eq({r['case_id'] for r in records},set(groups),'origin records');eq(collections.Counter(r['status'] for r in records),{'complete':24},'complete records')
eq(sum(len(r['completed_recipes']) for r in records),96,'96 recipes');eq(summary['evaluations']['failures'],[],'no evaluation failures')
savedctx={tuple(c['context']):c for c in summary['evaluations']['contexts']};eq(set(savedctx),set(byctx),'complete evaluation contexts')

# Evaluation uses only sealed prediction values plus separately saved full-label reveals.
def outcome(entries,label,selected):
 eq(len(set(selected)),4,'four unique entries');order=sorted(range(18),key=lambda i:(label[i],i));winner=min(selected,key=lambda i:(label[i],i));minimum=label[order[0]];rank={i:j+1 for j,i in enumerate(order)}
 return {'selected':list(selected),'chosen_candidate':winner,'chosen_id':entries[winner]['id'],'rmse':label[winner],'full18_min_rmse':minimum,'full18_winner':order[0],'excess_rmse':label[winner]-minimum,'winner_inclusion':order[0] in selected,'any_minimizer_inclusion':any(label[i]==minimum for i in selected),'best_selected_rank':min(rank[i] for i in selected),'mean_selected_rank':float(np.mean([rank[i] for i in selected])),'required_preview_calls':4,'full18_preview_calls':18}

def diagnostic(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);ia=np.lexsort((np.arange(18),a)).tolist();ib=np.lexsort((np.arange(18),b)).tolist()
 return dict(signed_label_changes=(a-b).tolist(),empirical_max_absolute_score_change=float(np.max(np.abs(a-b))),envelope_is_certified_error_bound=False,canonical_best_second_gap=float(a[ia[1]]-a[ia[0]]),canonical_fourth_fifth_gap=float(a[ia[4]]-a[ia[3]]),comparison_best_second_gap=float(b[ib[1]]-b[ib[0]]),comparison_fourth_fifth_gap=float(b[ib[4]]-b[ib[3]]),canonical_order=ia,comparison_order=ib,full_ranking_changed=ia!=ib,oracle_argmin_changed=ia[0]!=ib[0],oracle_minimum_signed_change=float(a[ia[0]]-b[ib[0]]))

contexts={};packet_rows=[];methods_diagnostics=collections.defaultdict(list);resolution_diagnostics=collections.defaultdict(list);penalties=collections.defaultdict(collections.Counter);negative_predictions=collections.Counter();prediction_totals=collections.Counter();label_rank_counts=collections.Counter();fullwinner_counts=collections.Counter();selected_candidate_counts=collections.defaultdict(collections.Counter)
for rec in records:
 g=rec['case_id'];folder=Path('folds')/rec['id'];reveal=read(folder/'full18-label-reveals.json')['data'];packets={};pred={};shortlists={}
 for res,schema in itertools.product(RES,SCHEMAS):
  name=res+'--'+schema;pack=read(folder/(name+'.json'));packets[name]=pack
  eq(len(pack['inner_folds']),23,'23 inner folds');eq(len(pack['training_groups']),23,'23 training origins');eq(g not in pack['training_groups'],True,'held origin excluded');eq(pack['numerical_audit']['passed'],True,'packet numerical gate')
  for solver in ['svd','qr']:
   pred[name,solver]=dict(zip(pack['held_row_ids'],pack['held_predictions_'+solver]));shortlists[name,solver]={tuple(c['context']):c for c in pack['shortlists_'+solver]}
  penalties[name][pack['selected_penalty_svd']]+=1
  negative_predictions[name]+=sum(x<0 for x in pack['held_predictions_svd']);prediction_totals[name]+=len(pack['held_predictions_svd'])
  packet_rows.append(dict(group=g,anchor=names[g],recipe=name,selected_penalty_svd=pack['selected_penalty_svd'],selected_penalty_qr=pack['selected_penalty_qr'],mean_inner_mse_svd=pack['mean_inner_mse_svd'],numerical_audit=pack['numerical_audit']))
 for key in sorted(k for k in byctx if k[0]==g):
  entries=byctx[key];old=savedctx[key];lab={res:[float(reveal[res][r['id']]) for r in entries] for res in RES};pol={res:{} for res in RES};scores={}
  eq(old['qualified'],True,'qualified context')
  for res in RES:
   for schema in SCHEMAS:
    name=res+'--'+schema
    for solver in ['svd','qr']:
     z=[pred[name,solver][r['id']] for r in entries];scores[name,solver]=z;rank=np.lexsort((np.arange(18),z)).tolist();sp=shortlists[name,solver][key]
     eq(sp['candidate_order'],rank,'packet rank reconstruction');eq(sp['top4'],rank[:4],'packet top4 reconstruction');eq(sp['top4_ids'],[entries[i]['id'] for i in rank[:4]],'packet top4 IDs')
    selected=shortlists[name,'svd'][key]['top4'];pol[res][schema]=outcome(entries,lab[res],selected);alt=outcome(entries,lab[res],shortlists[name,'qr'][key]['top4'])
    eq(pol[res][schema],old['policies'][res][schema],'learned observed policy reconstruction');selected_candidate_counts[name].update(selected)
    diag=diagnostic(scores[name,'svd'],scores[name,'qr']);s=old['solver_sensitivity'][name];eq(diag,s['predicted_score_diagnostic'],'solver score diagnostics')
    sx=dict(numerical_gate=packets[name]['numerical_audit']['passed'],selected_penalty_svd=packets[name]['selected_penalty_svd'],selected_penalty_qr=packets[name]['selected_penalty_qr'],ordered_shortlist_changed=selected!=alt['selected'],shortlist_set_changed=set(selected)!=set(alt['selected']),observed_selected_error_signed_change=pol[res][schema]['rmse']-alt['rmse'],observed_winner_changed=pol[res][schema]['chosen_candidate']!=alt['chosen_candidate'])
    for field,value in sx.items():eq(s[field],value,'solver action diagnostic')
    methods_diagnostics[name].append(dict(context=key,**sx,prediction=diag))
   for policy,subset in CONTROL.items():pol[res][policy]=outcome(entries,lab[res],subset);eq(pol[res][policy],old['policies'][res][policy],'fixed policy reconstruction')
  labeldiag=diagnostic(lab[RES[0]],lab[RES[1]]);eq(labeldiag,old['numerical_resolution'],'label resolution')
  label_rank_counts['ranking_changes']+=labeldiag['full_ranking_changed'];label_rank_counts['argmin_changes']+=labeldiag['oracle_argmin_changed'];fullwinner_counts[pol[RES[0]]['shared']['full18_winner']]+=1
  for schema in SCHEMAS:
   a=pol[RES[0]][schema];b=pol[RES[1]][schema];fixed=outcome(entries,lab[RES[1]],a['selected']);pa=packets[RES[0]+'--'+schema];pb=packets[RES[1]+'--'+schema];diag=diagnostic(scores[RES[0]+'--'+schema,'svd'],scores[RES[1]+'--'+schema,'svd'])
   z=dict(ordered_shortlist_changed=a['selected']!=b['selected'],shortlist_set_changed=set(a['selected'])!=set(b['selected']),selected_action_changed=a['chosen_candidate']!=b['chosen_candidate'],selected_error_signed_change=a['rmse']-b['rmse'],fixed_canonical_shortlist_error_signed_change=a['rmse']-fixed['rmse'],fixed_canonical_shortlist_action_changed=a['chosen_candidate']!=fixed['chosen_candidate'],prediction_diagnostic=diag)
   eq(z,old['model_resolution_sensitivity'][schema],'full resolution sensitivity');resolution_diagnostics[schema].append(dict(context=key,selected_penalty_changed=pa['selected_penalty_svd']!=pb['selected_penalty_svd'],**z))
  contexts[key]=dict(context=list(key),anchor=names[g],role=entries[0]['role'],policies=pol,label_diagnostic=labeldiag)

# Reconstruct all 380 bootstrap intervals from independently reconstructed observations.
boot=np.random.default_rng(20260914).integers(0,24,size=(20000,24));interval_count=0

def interval(vals):
 global interval_count
 interval_count+=1;a=np.asarray(vals,float);eq(a.shape,(24,),'24 bootstrap units');drawmeans=a[boot].mean(axis=1)
 return dict(mean=float(a.mean()),descriptive_percentile_interval=np.percentile(drawmeans,[2.5,97.5]).tolist(),origins=24,seed=20260914,resamples=20000,interpretation='Descriptive resampling of these development origins, not confirmatory uncertainty')

aggregates={}
for res in RES:
 aggregates[res]={}
 for radius in [None,.00025,.001,.004,.016]:
  co=[c for k,c in sorted(contexts.items()) if radius is None or k[2]==radius];per_origin=[]
  for g in groups:
   selected=[c for c in co if c['context'][0]==g];eq(len(selected),8 if radius is None else 2,'complete origin contexts')
   metrics={p:{m:float(np.mean([c['policies'][res][p][m] for c in selected])) for m in MET} for p in POL}
   per_origin.append(dict(group=g,contexts=len(selected),policies=metrics,full18_min_rmse=float(np.mean([c['policies'][res]['shared']['full18_min_rmse'] for c in selected]))))
  result=dict(origins=24,contexts=len(co),per_origin=per_origin,policies={},paired_contrasts={})
  for policy in POL:result['policies'][policy]={m:interval([o['policies'][policy][m] for o in per_origin]) for m in MET}
  for a,b in [('shared','geometry')]+[(a,b) for a in CONTROL for b in SCHEMAS]:result['paired_contrasts'][a+'_minus_'+b]=interval([o['policies'][a]['rmse']-o['policies'][b]['rmse'] for o in per_origin])
  result['full18_min_rmse']=interval([o['full18_min_rmse'] for o in per_origin])
  target=summary['primary_equal_origin_readout' if res==RES[0] else 'comparison_resolution_readout'];saved=target['all_origins'] if radius is None else target['by_radius'][str(radius)]
  eq(result,saved,'complete per-origin aggregates and bootstrap intervals')
  aggregates[res]['all' if radius is None else str(radius)]=result

primary=aggregates[RES[0]]['all'];origin_contrasts=[]
for o in primary['per_origin']:
 p=o['policies'];origin_contrasts.append(dict(group=o['group'],anchor=names[o['group']],shared_minus_geometry=p['shared']['rmse']-p['geometry']['rmse'],axes_minus_shared=p['axes']['rmse']-p['shared']['rmse'],axes_minus_geometry=p['axes']['rmse']-p['geometry']['rmse'],policy_rmse={k:v['rmse'] for k,v in p.items()},full18_min_rmse=o['full18_min_rmse']))
origin_contrasts.sort(key=lambda o:o['shared_minus_geometry'],reverse=True);delta=np.array([o['shared_minus_geometry'] for o in origin_contrasts]);concentration=dict(positive_origins=int((delta>0).sum()),negative_origins=int((delta<0).sum()),tied_origins=int((delta==0).sum()),median_origin_contrast=float(np.median(delta)),net_sum=float(delta.sum()),positive_sum=float(delta[delta>0].sum()),negative_sum=float(delta[delta<0].sum()),top3=origin_contrasts[:3],top3_fraction_of_net=float(delta[:3].sum()/delta.sum()),remaining21_mean=float(delta[3:].mean()),scope='Post-result concentration diagnostic only; no origins excluded from the declared24-origin readout.',origin_contrasts=origin_contrasts)
context_contrasts=[]
for key,c in contexts.items():
 p=c['policies'][RES[0]];context_contrasts.append(dict(context=key,anchor=c['anchor'],shared_minus_geometry=p['shared']['rmse']-p['geometry']['rmse'],shared_rmse=p['shared']['rmse'],geometry_rmse=p['geometry']['rmse'],axes_rmse=p['axes']['rmse']))
context_contrasts.sort(key=lambda c:c['shared_minus_geometry'],reverse=True)
concentration['largest_context_benefits']=context_contrasts[:10];concentration['largest_context_harms']=context_contrasts[-10:]
concentration['context_counts']=dict(positive=sum(c['shared_minus_geometry']>0 for c in context_contrasts),negative=sum(c['shared_minus_geometry']<0 for c in context_contrasts),tied=sum(c['shared_minus_geometry']==0 for c in context_contrasts))

solver_summary={}
for name,diags in methods_diagnostics.items():
 solver_summary[name]=dict(contexts=len(diags),penalty_disagreements=sum(d['selected_penalty_svd']!=d['selected_penalty_qr'] for d in diags),ordered_shortlist_changes=sum(d['ordered_shortlist_changed'] for d in diags),set_changes=sum(d['shortlist_set_changed'] for d in diags),winner_changes=sum(d['observed_winner_changed'] for d in diags),full_ranking_changes=sum(d['prediction']['full_ranking_changed'] for d in diags),maximum_prediction_difference=max(d['prediction']['empirical_max_absolute_score_change'] for d in diags),maximum_selected_error_difference=max(abs(d['observed_selected_error_signed_change']) for d in diags))
resolution_summary={}
for schema,diags in resolution_diagnostics.items():
 resolution_summary[schema]=dict(contexts=len(diags),penalty_changes=sum(diags[i]['selected_penalty_changed'] for i in range(0,len(diags),8)),ordered_shortlist_changes=sum(d['ordered_shortlist_changed'] for d in diags),set_changes=sum(d['shortlist_set_changed'] for d in diags),winner_changes=sum(d['selected_action_changed'] for d in diags),full_predicted_ranking_changes=sum(d['prediction_diagnostic']['full_ranking_changed'] for d in diags),fixed_primary_shortlist_winner_changes=sum(d['fixed_canonical_shortlist_action_changed'] for d in diags),maximum_prediction_difference=max(d['prediction_diagnostic']['empirical_max_absolute_score_change'] for d in diags),maximum_selected_error_difference=max(abs(d['selected_error_signed_change']) for d in diags),maximum_fixed_primary_selected_error_difference=max(abs(d['fixed_canonical_shortlist_error_signed_change']) for d in diags))
 eq(summary['numerical_resolution_summary']['models'][schema],dict(ordered_shortlist_changes=resolution_summary[schema]['ordered_shortlist_changes'],shortlist_set_changes=resolution_summary[schema]['set_changes'],action_changes=resolution_summary[schema]['winner_changes'],maximum_selected_error_change=resolution_summary[schema]['maximum_selected_error_difference']),'model numerical summary')

topcost=summary['cost_contract'];eq(topcost['preview_calls_per_policy_context'],4,'four previews');eq(topcost['full_reference_calls_per_context'],18,'full18');eq(topcost['reduction_calls'],14,'14 reduction');eq(topcost['reduction_fraction'],14/18,'fraction');eq(topcost['new_preview_integrations'],0,'no acquisitions');eq(topcost['net_speedup_claim'],False,'no speedup claim');eq(topcost['prerequisite_dependencies']['full18_reference'],['reference orbit','saved 300-unit unperturbed sensitivity'],'full18 costs')
cost=dict(contract=topcost,nominal_calls_per_policy_full192_contexts=768,full18_calls_full192_contexts=3456,nominal_reduction_per_policy=2688,receipt_seconds=receipt['wall_seconds'],receipt_inventory_seconds=receipt['receipt_inventory_seconds'],stage_timings=timings,summed_fold_wall_seconds=sum(r['wall_seconds'] for r in records),summed_recipe_seconds=sum(c['seconds'] for r in records for c in r['completed_recipes']),summed_stage_seconds={k:sum(c['stage_timings'][k] for r in records for c in r['completed_recipes']) for k in ['grid_fit_seconds','inner_validation_scoring_seconds','held_prediction_and_shortlist_seconds']},original_sensitivity_call_seconds_sum=float(sum(timings['feature_adapter_detail']['original_sensitivity_call_seconds'])),interpretation='First-pass recipe/fold times overlap across4 workers; historical sensitivity call times belong to the oldhost; no new previews, atlas acquisition, sensitivity integration, or prospective end-to-end speedup measured.')
pids=sorted({r['pid'] for r in records});eq(len(pids),24,'unique owned worker PIDs')
# Post-run metadata amendment: sandbox /proc does not establish host absence.
# Bind the separate host-permission process check; do not perform a local substitute.
host_check_path=Path('/data2/neural-shrimp-data/audits/EXP-011/post-run-audit/EXP-011-host-process-check.json');host_check=json.loads(host_check_path.read_text())
eq(host_check['source_commit'],receipt['source_commit'],'host check source')
eq(host_check['expected_worker_pids'],pids,'host check exact worker census')
eq(host_check['expected_parent_pid'],receipt['parent_pid'],'host check producer parent')
eq(host_check['present_owned_process_lines'],[],'host check absent owned processes')
eq(host_check['all_owned_pids_absent'],True,'host check passed')
termination=dict(source='Separate require_escalated host process inventory checked by root; this statistics script does not query host processes.',report_path=str(host_check_path),report_sha256=sha(host_check_path),report_bytes=host_check_path.stat().st_size,report=host_check)
amendment=dict(reason='The initial audit used default-sandbox /proc and incorrectly labeled it host verification. Statistics were complete and unchanged. A separate host-permission check now establishes all24 workers plus producer absent.',initial_source_path='/tmp/audit_exp011_statistics_operational.initial-sandbox.py',initial_source_sha256='618361e6d8e6ccb3cb2353f464a5d1d460e7da2b5e2a9551e4f29948c0b43295',initial_report_path='/tmp/EXP-011-independent-statistics-audit.initial-sandbox.json',initial_report_sha256='715c3c351ad6063796cbe95bbf74bf92317446653faea8f9e295b9650e70fdd8',numerical_recipe_changed=False)

report=dict(audit='EXP-011 independent saved-evidence statistical reconstruction',producer_functions_imported=False,refitting=False,new_integration=False,source_commit=receipt['source_commit'],attempt=str(P),input_hashes={name:sha(P/name) for name in ['manifest.json','receipt.json','summary.json','terminal-records.json','timings.json']},checks=checks,bootstrap_intervals_reconstructed=interval_count,bootstrap_resamples_per_interval=20000,exact_aggregate_match=True,counts=dict(origins=24,contexts=192,rows=3456,model_packets=96,inner_folds=2208,final_grid_fits=96,all_complete=True,all_numerical_flags_pass=True,independent_cases=len(summary['evaluations']['independent'])),primary=aggregates[RES[0]],comparison_resolution=aggregates[RES[1]],concentration=concentration,solver_summary=solver_summary,resolution_summary=resolution_summary,label_resolution=dict(label_rank_counts,maximum_label_change=max(c['label_diagnostic']['empirical_max_absolute_score_change'] for c in contexts.values())),selected_penalty_counts={k:dict(v) for k,v in penalties.items()},full18_winner_candidate_counts=dict(fullwinner_counts),selected_candidate_counts={k:dict(v) for k,v in selected_candidate_counts.items()},negative_prediction_counts=dict(negative_predictions),prediction_counts=dict(prediction_totals),packet_numerical_audits=packet_rows,cost=cost,worker_termination=termination,post_run_metadata_amendment=amendment,recommendation=['Retain this complete retrospective result; the geometry-vs-shared mean improvement is concentrated and its descriptive origin interval spans zero.','Do not advance the present geometry learner as a proven improvement or preferential fresh confirmation: fixed axes have lower overall RMSE with fewer prerequisites.','Any new objective/shortlist rule belongs to a separately frozen development study. Prediction MSE and near-minimum action selection can differ; diagnose that mismatch without treating it as established causation.','Radius patterns and origin-deletion diagnostics are descriptive only; do not select a favorable radius or drop origins from the existing endpoint.','The75-unit feasibility result does not resolve300-unit utility, independent population generalization, atlas value, or net compute savings.'])
out=Path('/tmp/EXP-011-independent-statistics-audit.json');out.write_text(json.dumps(report,indent=2)+'\n');print('REPORT',out,'BYTES',out.stat().st_size,'SHA256',sha(out),'CHECKS',checks,'BOOTSTRAP_INTERVALS',interval_count)
print(json.dumps({k:report[k] for k in ['counts','solver_summary','resolution_summary','selected_penalty_counts','full18_winner_candidate_counts','negative_prediction_counts','worker_termination']},indent=2))
print('POLICY_MEANS', {k:v['rmse']['mean'] for k,v in primary['policies'].items()});print('GEOMETRY_CONTRAST',primary['paired_contrasts']['shared_minus_geometry']);print('CONCENTRATION', {k:v for k,v in concentration.items() if k not in ['origin_contrasts','largest_context_benefits','largest_context_harms','top3']});print('COST', {k:v for k,v in cost.items() if k not in ['contract','stage_timings']})
