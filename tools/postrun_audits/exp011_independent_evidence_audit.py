#!/usr/bin/env python3
"""Post-run independent EXP-011 audit: stdlib/NumPy only, no producer imports or refits."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
import collections, datetime, hashlib, json, math, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path('/data2/neural-shrimp')
RAW=Path('/data2/neural-shrimp-data/raw/EXP-011/attempt-001')
OUT=Path('/tmp/EXP-011-independent-evidence-audit.json')
SOURCE='64712453153ade24d2b22cf811f7a83811304de4'
RES=('rk4-0.00125','rk4-0.0025')
IND=('dop853-1e-11','dop853-1e-12')
SCHEMAS=('shared','geometry')
CONTROL={'axes':[0,4,8,12],'diagonal':[2,6,10,14],'local_coverage':[0,8,16,17]}
PEN=[1e-6,1e-4,.01,1.,100.]
audit_started=datetime.datetime.now(datetime.timezone.utc).isoformat()
start=time.monotonic(); failures=[]; checks=collections.Counter(); counts=collections.Counter(); maxima={}
def require(test,category,detail):
    checks[category]+=1
    if not bool(test): failures.append({'category':category,'detail':detail})
def equal(a,b,category,detail):
    require(a==b,category,detail)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()
def binding(path):
    p=Path(path);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p): return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON '+x)))
def lines(p): return [json.loads(line) for line in Path(p).read_text().splitlines()]
def check_binding(item,base=None,category='binding'):
    p=(base/item['path']) if base else Path(item['path'])
    b=binding(p)
    equal((b['sha256'],b['bytes']),(item['sha256'],item['bytes']),category,str(p))
    return p

def near(a,b,category,detail,atol=1e-12,rtol=1e-12):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    require(a.shape==b.shape,category,detail+' shape')
    if a.shape!=b.shape:return
    require(np.isfinite(a).all() and np.isfinite(b).all(),category,detail+' finite')
    delta=np.abs(a-b); value=float(np.max(delta)) if delta.size else 0.
    if value>maxima.get(category,{}).get('maximum_absolute_difference',-1):
        maxima[category]={'maximum_absolute_difference':value,'location':detail}
    require(np.all(delta<=atol+rtol*np.maximum(np.abs(a),np.abs(b))),category,detail+' tolerance')
    counts[category+'_values']+=int(a.size)

def context(row): return (row['group'],row['phase'],row['radius'])
def predict(grid,x,index,solver):
    # Elementwise reduction is deliberately independent of producer BLAS matvec.
    z=(x-np.asarray(grid['mean']))/np.asarray(grid['scale'])
    return np.sum(z*np.asarray(grid['fits'][index][solver])[None,:],axis=1)+grid['intercept']
def shortlists(rr,scores):
    by=collections.defaultdict(list)
    for row,score in zip(rr,scores):by[context(row)].append((float(score),row['candidate'],row['id']))
    answer=[]
    for ctx,items in sorted(by.items()):
        require(len(items)==18 and {i[1] for i in items}==set(range(18)),'shortlist_census',str(ctx))
        ranked=sorted(items)
        answer.append({'context':list(ctx),'candidate_order':[i[1] for i in ranked],
                       'top4':[i[1] for i in ranked[:4]],'top4_ids':[i[2] for i in ranked[:4]]})
    return answer

def score_gate(a,b):
    a=np.asarray(a);b=np.asarray(b); delta=np.abs(a-b)
    return {'passed':bool(np.all(delta<=1e-9+1e-7*np.maximum(np.abs(a),np.abs(b)))),
            'maximum_absolute_difference':float(np.max(delta))}

def policy(rr,labels,selected):
    data={r['candidate']:(r['id'],labels[r['id']]) for r in rr}
    order=sorted(data,key=lambda c:(data[c][1],c)); winner=min(selected,key=lambda c:(data[c][1],c))
    ranks=[order.index(c)+1 for c in selected]; best=order[0]
    return dict(selected=selected,chosen_candidate=winner,chosen_id=data[winner][0],rmse=data[winner][1],
        full18_min_rmse=data[best][1],full18_winner=best,excess_rmse=data[winner][1]-data[best][1],
        winner_inclusion=best in selected,any_minimizer_inclusion=any(data[c][1]==data[best][1] for c in selected),
        best_selected_rank=min(ranks),mean_selected_rank=sum(ranks)/4.,required_preview_calls=4,full18_preview_calls=18)

receipt=read(RAW/'receipt.json');plan=read(RAW/'manifest.json');summary=read(RAW/'summary.json')
records=read(RAW/'terminal-records.json'); feature=read(RAW/'feature-identity.json')
equal(receipt['source_commit'],SOURCE,'source','receipt source')
require(datetime.datetime.fromisoformat(receipt['remote_head_verified_at'])<=datetime.datetime.fromisoformat(receipt['started_at'])<datetime.datetime.fromisoformat(receipt['ended_at']),'receipt_utc_chronology','remote verification before start before end')
equal(receipt['manifest_sha256'],sha(RAW/'manifest.json'),'source','manifest receipt hash')
equal(receipt['manifest_sha256'],'a32c5283a2674ddaec01ba4cb508cf473f0be95f5b1724fa4ef7468a3c9b3f08','source','announced plan hash')
paths=[item['path'] for item in receipt['artifacts']]
equal(len(paths),len(set(paths)),'inventory','no duplicates')
equal(set(paths),{str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p.name!='receipt.json'},'inventory','exact file census')
for item in receipt['artifacts']:
    p=check_binding(item,RAW,'raw_artifact_binding')
    require(p.resolve().is_relative_to(RAW) and not p.is_symlink(),'inventory','safe path '+item['path'])
counts['receipt_artifacts']=len(paths);counts['receipt_artifact_bytes']=sum(i['bytes'] for i in receipt['artifacts'])
for item in plan['source_files']:
    check_binding(item,ROOT,'current_frozen_source_binding')
    blob=subprocess.check_output(['git','show',SOURCE+':'+item['path']],cwd=ROOT)
    equal((hashlib.sha256(blob).hexdigest(),len(blob)),(item['sha256'],item['bytes']),'committed_source_binding',item['path'])
blob=subprocess.check_output(['git','show',SOURCE+':experiments/hopfield/EXP-011-plan.json'],cwd=ROOT)
equal(blob,(RAW/'manifest.json').read_bytes(),'committed_source_binding','full frozen manifest bytes')
for item in plan['feature_source_bindings']:check_binding(item,ROOT,'feature_input_binding')
for item in plan['external_artifacts']:check_binding(item,category='external_evidence_binding')
check_binding(plan['qualification_certificate_binding'],ROOT,'qualification_binding')
equal(plan['qualification'],read(ROOT/plan['qualification_certificate_binding']['path']),'qualification_binding','certificate content')
counts.update(source_files=len(plan['source_files']),feature_input_files=len(plan['feature_source_bindings']),external_evidence_files=len(plan['external_artifacts']))
q=plan['qualification']
for k,v in {'status':'qualified_at_tested_resolutions','planned_traces':6944,'complete_traces':6944,'preview_cases':3456,'native_scalar_qualified':3456,'native_waveform_qualified':3456,'independent_cases':16,'independent_scalar_qualified':16,'independent_waveform_qualified':16,'contexts':192,'all_native_rankings_unchanged':True,'all_required_methods_complete':True,'all_scalar_and_waveform_checks_pass':True}.items():equal(q[k],v,'qualification_census',k)
prior=read('/data2/neural-shrimp-data/raw/EXP-010/attempt-001/receipt.json')
prior_inventory={i['path']:i for i in prior['artifacts']}
equal(plan['label_source_commit'],prior['source_commit'],'label_source','commit')
rows=plan['rows']; byid={r['id']:r for r in rows}; groups=sorted({r['group'] for r in rows});idx={r['id']:i for i,r in enumerate(rows)}
contexts=collections.defaultdict(list)
for row in rows:contexts[context(row)].append(row)
equal((len(rows),len(byid),len(groups),len(contexts)),(3456,3456,24,192),'feature_census','totals')
equal(dict(collections.Counter(r['group'] for r in rows)),dict.fromkeys(groups,144),'feature_census','group counts')
for ctx,rr in contexts.items():equal(sorted(r['candidate'] for r in rr),list(range(18)),'feature_census',str(ctx))
equal(feature,plan['feature_identity'],'feature_identity','saved vs plan');equal(feature['rows'],rows,'feature_identity','row identities')
with np.load(RAW/'feature-arrays.npz',allow_pickle=False) as z:
    equal(set(z.files),{'shared','geometry'},'feature_identity','array keys'); shared=z['shared'].copy();geometry=z['geometry'].copy()
for name,a,shape in [('shared',shared,(3456,37)),('geometry',geometry,(3456,6))]:
    equal(a.shape,shape,'feature_identity',name+' shape');require(np.isfinite(a).all(),'feature_identity',name+' finite')
    equal(hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest(),feature[name+'_sha256'],'feature_identity',name+' value hash')
for i,row in enumerate(rows):
    equal(row['group'],'hopfield-da-silva-rech-2022:'+float(shared[i,0]).hex()+':'+float(shared[i,1]).hex(),'physical_origin_identity',row['id'])
mat={'shared':shared,'geometry':np.column_stack((shared,geometry))}
labels={r:{} for r in (*RES,*IND)};sentinels=[]
equal(set(plan['label_records']),set(byid),'label_source','complete row identities')
for ident,mapping in plan['label_records'].items():
    equal(set(mapping),set(RES)| (set(IND) if ident in plan['independent_case_ids'] else set()),'label_source','method census '+ident)
    if all(r in mapping for r in IND):sentinels.append(ident)
    for resolution,item in mapping.items():
        p=check_binding(item,category='label_binding'); counts['label_records']+=1
        relative='records/'+ident+'--'+resolution+'.json'
        equal((p.name,item['sha256'],item['bytes']),(Path(relative).name,prior_inventory[relative]['sha256'],prior_inventory[relative]['bytes']),'label_source','prior receipt '+relative)
        rec=read(p)
        equal((rec['id'],rec['case_id'],rec['method'],rec['status']),(ident+'--'+resolution,ident,resolution,'complete'),'label_source',relative)
        val=rec['rmse'];require(type(val) in (float,int) and math.isfinite(val) and 0<=val<=2,'label_source','finite label '+relative)
        equal(val,rec['metrics']['aligned_rms']['aggregate'],'label_source','aggregate '+relative); labels[resolution][ident]=val
    counts['label_case_rows']+=1
equal(sentinels,plan['independent_case_ids'],'label_source','independent sentinel identities')
equal(len(sentinels),16,'label_source','independent sentinel count')
equal([r['id'] for r in records], [r['id'] for r in lines(RAW/'journal.jsonl')],'terminal_census','journal order')
equal(records,lines(RAW/'journal.jsonl'),'terminal_census','full journal records')
equal(set(r['id'] for r in records),{f'outer-{i:02d}' for i in range(24)},'terminal_census','24 exact tasks')
equal(len(records),24,'terminal_census','no duplicate tasks')
equal(receipt['status_counts'],dict(collections.Counter(r['status'] for r in records)),'terminal_census','statuses')
equal((receipt['expected_records'],receipt['observed_records'],receipt['status'],receipt['error']),(24,24,'complete',None),'terminal_census','complete receipt')
for k,v in read(RAW/'started.json').items():equal(receipt[k],v,'receipt_started_binding',k)
equal(summary['status'],'qualified_retrospective_feasibility','qualification','summary status')
equal(summary['evaluations']['failures'],[],'qualification','no silently omitted failures')
entrymap={tuple(e['context']):e for e in summary['evaluations']['contexts']}
equal(set(entrymap),set(contexts),'evaluation_census','context bijection');equal(len(summary['evaluations']['contexts']),192,'evaluation_census','no duplicate contexts')
all_pre=[];all_reveal=[];fold_reports=[];selected_policy_results={}
for record in sorted(records,key=lambda r:r['id']):
    fold=record['id'];group=groups[int(fold.split('-')[1])];folder=RAW/'folds'/fold
    equal((record['case_id'],record['method'],record['status']),(group,'nested-ridge-both-resolutions','complete'),'terminal_identity',fold)
    equal(record,read(RAW/'records'/f'{fold}.json'),'terminal_identity',fold+' individual record')
    require(record['numerical_agreement'] is True,'qualification',fold)
    held=[r for r in rows if r['group']==group];training=[r for r in rows if r['group']!=group]
    hi=[idx[r['id']] for r in held];ti=[idx[r['id']] for r in training];hids=[r['id'] for r in held];tids=[r['id'] for r in training];tg=[r['group'] for r in training];allowed=sorted(set(tg))
    seal=read(folder/'pre-reveal-seal.json'); sealhash=sha(folder/'pre-reveal-seal.json'); all_pre.append(seal['monotonic_ns'])
    equal((record['seal'],record['seal_sha256']),(f'folds/{fold}/pre-reveal-seal.json',sealhash),'seal_binding',fold)
    equal((seal['fold_id'],seal['held_group'],seal['held_ids'],seal['feature_identity_sha256'],seal['controls'],seal['pid']),(fold,group,hids,digest(feature),CONTROL,record['pid']),'seal_binding',fold+' identity')
    train_events=lines(folder/'label-access.jsonl');equal(len(train_events),2,'training_access',fold+' event count')
    for j,(res,event) in enumerate(zip(RES,train_events)):
        expected=dict(stage='training',resolution=res,ids=tids,groups=allowed,held_group=group,event_index=j)
        equal({k:v for k,v in event.items() if k!='monotonic_ns'},expected,'training_access',fold+'/'+res)
        require(event['monotonic_ns']<=seal['monotonic_ns'],'training_access',fold+' before seal')
        expectedtraining=dict(ids=tids,groups=tg,values=[labels[res][i] for i in tids],source='permitted EXP-010 per-case score records')
        equal(read(folder/f'training-{res}.json'),expectedtraining,'training_label_replay',fold+'/'+res)
        counts['training_label_values']+=len(tids)
    require([e['monotonic_ns'] for e in train_events]==sorted(e['monotonic_ns'] for e in train_events),'training_access',fold+' monotone')
    counts['training_access_events']+=len(train_events)
    packets={};shortmap={}; learned_names=[res+'--'+schema for res in RES for schema in SCHEMAS]
    equal([r['name'] for r in record['completed_recipes']],learned_names,'packet_census',fold+' terminal recipes')
    equal([a['path'] for a in seal['artifacts']],[n+'.json' for n in learned_names],'packet_census',fold+' sealed recipes')
    for item in seal['artifacts']:check_binding(item,folder,'packet_byte_binding')
    for res in RES:
        y=np.array([labels[res][i] for i in tids])
        for schema in SCHEMAS:
            name=res+'--'+schema;p=read(folder/(name+'.json'));packets[name]=p;xt=mat[schema][ti];xh=mat[schema][hi]
            equal(p['packet_sha256'],digest({k:v for k,v in p.items() if k!='packet_sha256'}),'packet_digest',fold+'/'+name)
            equal((p['training_groups'],p['held_group'],p['training_rows'],p['training_label_sha256'],p['held_row_ids'],p['penalty_grid']),
                  (allowed,group,3312,digest(y.tolist()),hids,PEN),'packet_training_identity',fold+'/'+name)
            equal(p['feature_names'],feature['shared_names']+(feature['geometry_names'] if schema=='geometry' else []),'packet_training_identity',fold+'/'+name+' features')
            equal([f['held_group'] for f in p['inner_folds']],allowed,'inner_fold_isolation',fold+'/'+name+' exact validation groups')
            cv={s:[] for s in ('svd','qr')};coeffmax=0.
            for inner in p['inner_folds']:
                vg=inner['held_group'];mask=np.asarray(tg)==vg;grid=inner['grid'];tag=fold+'/'+name+'/'+vg
                equal((inner['validation_rows'],inner['training_label_sha256'],grid['training_groups'],grid['training_rows']),
                      (144,digest(y[~mask].tolist()),[g for g in allowed if g!=vg],3168),'inner_fold_isolation',tag)
                equal([f['penalty'] for f in grid['fits']],PEN,'inner_fold_isolation',tag+' grid penalties')
                preds={s:np.stack([predict(grid,xt[mask],j,s) for j in range(5)]) for s in ('svd','qr')}
                for solver in ('svd','qr'):
                    mse=np.mean((preds[solver]-y[mask])**2,axis=1);cv[solver].append(mse)
                    near(mse,inner[solver+'_mse'],'inner_validation_mse_replay',tag+'/'+solver)
                    counts['inner_prediction_values']+=int(preds[solver].size)
                gate=score_gate(preds['svd'],preds['qr'])
                equal(gate['passed'],inner['score_agreement']['passed'],'numerical_gate_replay',tag)
                near(gate['maximum_absolute_difference'],inner['score_agreement']['maximum_absolute_difference'],'inner_score_delta_replay',tag)
                for fit in grid['fits']:coeffmax=max(coeffmax,float(np.max(np.abs(np.asarray(fit['svd'])-fit['qr']))))
                counts['inner_folds']+=1
            grid=p['final_grid']
            equal((grid['training_groups'],grid['training_rows'],[f['penalty'] for f in grid['fits']]),(allowed,3312,PEN),'final_grid_isolation',fold+'/'+name)
            for fit in grid['fits']:coeffmax=max(coeffmax,float(np.max(np.abs(np.asarray(fit['svd'])-fit['qr']))))
            for solver in ('svd','qr'):
                mean=np.mean(cv[solver],axis=0);selected=int(np.argmin(mean))
                near(mean,p['mean_inner_mse_'+solver],'mean_validation_mse_replay',fold+'/'+name+'/'+solver)
                equal((p['selected_index_'+solver],p['selected_penalty_'+solver]),(selected,PEN[selected]),'penalty_selection_replay',fold+'/'+name+'/'+solver)
                pred=predict(grid,xh,selected,solver)
                near(pred,p['held_predictions_'+solver],'held_prediction_replay',fold+'/'+name+'/'+solver)
                short=shortlists(held,pred)
                equal(short,p['shortlists_'+solver],'full_prediction_ranking_replay',fold+'/'+name+'/'+solver)
                counts['solver_context_rankings']+=len(short)
                if solver=='svd':shortmap[name]=short
            audit=p['numerical_audit'];scorecheck=score_gate(p['held_predictions_svd'],p['held_predictions_qr'])
            expectedaudit=dict(passed=all(f['score_agreement']['passed'] for f in p['inner_folds']) and scorecheck['passed'] and p['selected_index_svd']==p['selected_index_qr'] and all(a['top4']==b['top4'] for a,b in zip(p['shortlists_svd'],p['shortlists_qr'])),
                all_inner_scores_agree=all(f['score_agreement']['passed'] for f in p['inner_folds']),final_score_agreement=scorecheck,
                selected_penalty_agrees=p['selected_index_svd']==p['selected_index_qr'],shortlist_agrees=all(a['top4']==b['top4'] for a,b in zip(p['shortlists_svd'],p['shortlists_qr'])),
                full_ranking_agrees=all(a['candidate_order']==b['candidate_order'] for a,b in zip(p['shortlists_svd'],p['shortlists_qr'])),coefficient_max_absolute_difference=coeffmax)
            equal(audit,expectedaudit,'numerical_audit_replay',fold+'/'+name);require(audit['passed'],'qualification',fold+'/'+name)
            completed=next(c for c in record['completed_recipes'] if c['name']==name)
            equal((completed['packet_sha256'],completed['numerical_agreement']),(p['packet_sha256'],audit['passed']),'packet_digest',fold+'/'+name+' terminal link')
            counts['model_packets']+=1
    expectedshort={n:[c['top4_ids'] for c in cs] for n,cs in shortmap.items()}
    ctxorder=sorted({context(r) for r in held})
    for policyname,candidates in CONTROL.items():
        expectedshort[policyname]=[[next(r['id'] for r in contexts[ctx] if r['candidate']==c) for c in candidates] for ctx in ctxorder]
    equal(seal['shortlist_ids'],expectedshort,'selection_seal_replay',fold)
    selected=read(folder/'selected-label-reveals.json');winner=read(folder/'winner-seal.json');full=read(folder/'full18-label-reveals.json')
    expecteddata={};expectedwinners={}; expectedevents=[]
    for res in RES:
        for policyname in (*SCHEMAS,*CONTROL):
            name=res+'--'+policyname;source=name if policyname in SCHEMAS else policyname
            ids=[i for cc in expectedshort[source] for i in cc];vals=[labels[res][i] for i in ids]
            expecteddata[name]={'ids':ids,'values':vals};expectedwinners[name]=[]
            for j,ctx in enumerate(ctxorder):
                chunk=ids[j*4:j*4+4]; chosen=min(chunk,key=lambda i:(labels[res][i],byid[i]['candidate']))
                expectedwinners[name].append(dict(context=list(ctx),chosen_id=chosen,chosen_candidate=byid[chosen]['candidate'],rmse=labels[res][chosen]))
                obs=policy(contexts[ctx],labels[res],[byid[i]['candidate'] for i in chunk])
                equal(entrymap[ctx]['policies'][res][policyname],obs,'policy_score_replay',fold+'/'+name+'/'+str(ctx))
                selected_policy_results[(ctx,res,policyname)]=obs;counts['policy_context_scores']+=1
            expectedevents.append(dict(stage='reveal',purpose='selected_four',resolution=res,ids=ids,held_group=group,seal_sha256=sealhash,event_index=len(expectedevents)))
    equal(selected,{'seal_sha256':sealhash,'data':expecteddata},'selected_reveal_replay',fold)
    equal({k:v for k,v in winner.items() if k!='monotonic_ns'},dict(schema_version='exp011-four-observed-winner-v1',pre_reveal_seal_sha256=sealhash,selected_reveals_sha256=sha(folder/'selected-label-reveals.json'),winners=expectedwinners),'four_label_winner_seal_replay',fold)
    for res in RES:expectedevents.append(dict(stage='reveal',purpose='full18_evaluation',resolution=res,ids=hids,held_group=group,seal_sha256=sealhash,event_index=len(expectedevents)))
    independent={i:{res:labels[res][i] for res in IND} for i in hids if i in sentinels}
    for ident in hids:
        if ident in sentinels:
            for res in IND:expectedevents.append(dict(stage='reveal',purpose='independent_diagnostic',resolution=res,ids=[ident],held_group=group,seal_sha256=sealhash,event_index=len(expectedevents)))
    equal(full,{'seal_sha256':sealhash,'data':{res:{i:labels[res][i] for i in hids} for res in RES},'independent':independent},'oracle_reveal_replay',fold)
    events=lines(folder/'reveal-access.jsonl');equal([{k:v for k,v in e.items() if k!='monotonic_ns'} for e in events],expectedevents,'reveal_access_sequence',fold)
    times=[e['monotonic_ns'] for e in events];all_reveal.extend(times)
    require(times==sorted(times) and min(times)>=seal['monotonic_ns'],'seal_chronology',fold+' monotone after prediction seal')
    require(all((e['monotonic_ns']<=winner['monotonic_ns'] if e['purpose']=='selected_four' else e['monotonic_ns']>=winner['monotonic_ns']) for e in events),'seal_chronology',fold+' winner before oracle')
    require(seal['monotonic_ns']<=winner['monotonic_ns'],'seal_chronology',fold+' seal order')
    counts.update({'reveal_access_events':len(events),'selected_label_reveal_events':sum(e['purpose']=='selected_four' for e in events),'oracle_label_reveal_events':sum(e['purpose']=='full18_evaluation' for e in events),'independent_label_reveal_events':sum(e['purpose']=='independent_diagnostic' for e in events),'folds':1})
    for ctx in ctxorder:
        entry=entrymap[ctx]; rr=contexts[ctx]
        equal((entry['role'],entry['anchor'],entry['qualified']),(rr[0]['role'],rr[0]['anchor'],True),'evaluation_identity',str(ctx))
        equal(set(entry['policies']),set(RES),'evaluation_identity',str(ctx)+' resolutions')
        for res in RES:equal(set(entry['policies'][res]),set(SCHEMAS)|set(CONTROL),'evaluation_identity',str(ctx)+' policies')
    fold_reports.append(dict(fold_id=fold,held_group=group,packet_count=4,training_label_events=len(train_events),reveal_events=len(events),prediction_seal_ns=seal['monotonic_ns'],winner_seal_ns=winner['monotonic_ns'],first_reveal_ns=min(times),last_reveal_ns=max(times)))
    print('audited',fold,flush=True)
require(min(all_reveal)>max(all_pre),'global_phase_isolation','all 24 prediction seals precede any held-label reveal')
equal(read(RAW/'model-replay.json'),{'replayed_recipes':96,'expected_complete_recipes':96},'producer_replay_record','record only; independent audit does not refit')
expected_independent=[]
for ident in sentinels:
    row=byid[ident];ctx=context(row)
    expected_independent.append(dict(id=ident,group=row['group'],context=list(ctx),candidate=row['candidate'],native={res:labels[res][ident] for res in RES},independent={res:labels[res][ident] for res in IND},canonical_minus_independent={res:labels[RES[0]][ident]-labels[res][ident] for res in IND},selected_membership={res:{policyname:row['candidate'] in selected_policy_results[(ctx,res,policyname)]['selected'] for policyname in (*SCHEMAS,*CONTROL)} for res in RES},scope='Partial scalar sentinel only; no independent 18-entry oracle or independent retraining'))
equal(sorted(summary['evaluations']['independent'],key=lambda r:r['id']),sorted(expected_independent,key=lambda r:r['id']),'independent_panel_replay','all 16 sentinel values and memberships')
for name,value in {'model_packets':96,'folds':24,'inner_folds':2208,'solver_context_rankings':1536,'label_records':6944,'policy_context_scores':1920,'training_access_events':48,'reveal_access_events':320,'selected_label_reveal_events':240,'oracle_label_reveal_events':48,'independent_label_reveal_events':32}.items():equal(counts[name],value,'final_census',name)
report=dict(schema_version='EXP-011-independent-evidence-audit-v1',status='pass' if not failures else 'fail',source_commit=SOURCE,
    audit_started_utc=audit_started,audit_completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
    runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),thread_environment={n:os.environ[n] for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}),
    audit_script=binding(Path(__file__)),inputs={n:binding(RAW/n) for n in ['receipt.json','manifest.json','summary.json','terminal-records.json','feature-arrays.npz','model-replay.json']},
    raw_inventory_artifact_count=len(paths),raw_inventory_artifact_bytes=counts['receipt_artifact_bytes'],checks=dict(checks),counts=dict(counts),maxima=maxima,
    failure_count=len(failures),failures=failures,folds=fold_reports,
    all_prediction_seals_before_any_reveal=dict(passed=min(all_reveal)>max(all_pre),last_prediction_seal_ns=max(all_pre),first_reveal_ns=min(all_reveal),gap_seconds=(min(all_reveal)-max(all_pre))/1e9),
    tolerances=dict(independent_arithmetic_atol=1e-12,independent_arithmetic_rtol=1e-12,frozen_svd_qr_atol=1e-9,frozen_svd_qr_rtol=1e-7,discrete_rank_and_identity='exact'),
    scope=['Independent NumPy elementwise sum prediction replay from saved coefficient packets; no producer module imports, refitting, target integration or raw artifact writes.',
           'Inner validation MSE and penalty choices replayed from saved coefficients and permissible labels; coefficient fitting and original feature construction were not independently repeated.',
           'Label isolation conclusions use complete persisted access logs, group and label hashes, packet identities, seals, and previously audited frozen source; they are not an OS-level proof of absence of unlogged file reads.',
           'All 24 physical origins are retrospective development data. This audit establishes saved evidence consistency, not untouched confirmation, 300-unit target utility, geometry benefit or measured net speedup.',
           'Equal-origin aggregates, bootstrap intervals, resolution contrasts, costs and PID cleanup independently audited by the operational agent.'])
OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':str(OUT),'status':report['status'],'failure_count':len(failures),'counts':dict(counts),'maxima':maxima,'elapsed_seconds':report['elapsed_seconds']},indent=2))
