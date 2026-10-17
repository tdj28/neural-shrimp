"""Prepared C12 heldout adapter. Only own adjacent reducers/controls are imported.

The source is unexecuted. Exact card admission and owner supervision are external.
Scientific refusals survive independently for each component; they are not passes.
"""
import argparse
import ast
import collections
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time
import traceback

OUTPUT = '/data2/neural-shrimp-data/codex/task27-ready-review-20261017/run-001/c12'
LIMITS = {'cpu_seconds':55,'wall_seconds':60,'address_space_bytes':1073741824,
          'output_bytes':134217728,'input_bytes':67108864,'threads':1,'nice':10}
MISSING = object()


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x):return {'nonfinite':repr(x)}
    return x


class Writer:
    def __init__(self,out):self.out=out;self.files=[];self.used=0
    def write(self,name,value):
        b=(json.dumps(safe(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        if self.used+len(b)>LIMITS['output_bytes']-1048576:raise RuntimeError('output reserve exhausted')
        with (self.out/name).open('xb') as f:f.write(b)
        self.used+=len(b);self.files.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    def lines(self,name,rows):
        n=0;size=0;h=hashlib.sha256()
        with (self.out/name).open('xb') as f:
            for row in rows:
                b=(json.dumps(safe(row),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                if self.used+size+len(b)>LIMITS['output_bytes']-1048576:raise RuntimeError('output reserve exhausted')
                f.write(b);h.update(b);size+=len(b);n+=1
        self.used+=size;self.files.append({'path':name,'bytes':size,'sha256':h.hexdigest(),'records':n})


def own(name):
    p=Path(__file__).with_name(name+'.py');s=importlib.util.spec_from_file_location('c12_own_'+name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def check_pin(pin):
    p=Path(pin['path'])
    if not p.is_file() or p.stat().st_size!=pin['bytes'] or digest(p)!=pin['sha256']:
        raise ValueError('input pin mismatch: '+str(p))


def load_json(pin):return json.loads(Path(pin['path']).read_text())


def leaves(x,path=()):
    if isinstance(x,dict) and x:
        for k,v in x.items():yield from leaves(v,path+(str(k),))
    elif isinstance(x,list) and x:
        for i,v in enumerate(x):yield from leaves(v,path+(str(i),))
    else:yield path,x


def at(x,path):
    for k in path:
        try:x=x[int(k)] if isinstance(x,list) else x[k]
        except (KeyError,IndexError,ValueError,TypeError):return MISSING
    return x


def difference(path,ours,published,basis):
    z={'path':list(path),'basis':basis,'published':published}
    if ours is MISSING:z['disposition']='missing_own';return z
    z.update(own=ours,disposition='exact' if ours==published else 'different')
    if type(ours) in (int,float) and type(published) in (int,float):z['signed_difference']=ours-published
    return z


def source_evidence(inputs,pins,configs):
    groups=collections.defaultdict(list)
    for row in inputs['source_rows']:
        pin=pins[row['input_role']]
        groups[row['experiment'],row['relative_path']].append({'commit':row['commit'],'sha256':pin['sha256'],'path':pin['path']})
    identity=[{'experiment':exp,'relative_path':p,'all_saved_versions_equal':len({r['sha256'] for r in rows})==1,'versions':rows}
              for (exp,p),rows in groups.items()]
    sm=load_json(pins['source_metadata']);chronology=[]
    for item in sm['chronology']:
        if not item['path'].startswith(('docs/pi/c12-3','docs/pi-c12-4','docs/pi/c12-4c')):continue
        for line in item['log']:
            parts=line.split(' ',3);stamp=datetime.datetime.fromisoformat(parts[1])
            starts={k:c['history'][0]['started'] for k,c in configs.items()}
            selected=('c3',) if 'c12-3' in item['path'] else ('a','b')
            chronology.append({'source_path':item['path'],'git_log':line,'commit_time_utc':stamp.astimezone(datetime.timezone.utc).isoformat(),
                'recorded_start_comparisons':{k:{'start':starts[k],'commit_precedes_start':stamp<datetime.datetime.fromisoformat(starts[k].replace('Z','+00:00'))} for k in selected}})
    # Literal source-pool values; no generator execution, tokenizer or model.
    src=next(r for r in inputs['source_rows'] if r['experiment']=='c12-4c' and r['relative_path']=='tools/c12_steering_history.py')
    tree=ast.parse(Path(pins[src['input_role']]['path']).read_text());pools={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id.startswith('USERS_'):
                    pools[target.id]=ast.literal_eval(node.value)
    config_source=[]
    for key,conf in configs.items():
        exp='c12-3' if key=='c3' else 'c12-4c'
        tool='tools/c12_coupled_agents.py' if key=='c3' else 'tools/c12_steering_history.py'
        for h in conf['history']:
            for field,rel in (('tool_sha256',tool),('pi_harness_sha256','tools/pi_harness.py')):
                matches=[x for x in groups[exp,rel] if x['commit']==h['commit']]
                config_source.append({'run':key,'commit':h['commit'],'field':field,'recorded':h[field],
                    'captured':matches,'matches_exact_original_export':len(matches)==1 and matches[0]['sha256']==h[field]})
    return {'source_identity':identity,'config_source_links':config_source,'chronology':chronology,'source_user_pools':pools,
        'scope':'Original Git exports and run metadata only; no push-time, historical loaded weights, or never-used-before proof.'}


def do_c3(R,rows,config,problems):
    tests=[x for x in problems['problems'] if x['split']=='test'];uids=[x['uid'] for x in tests];gold={x['uid']:x for x in tests}
    by={};ledger=None;pairs=None;selfreplay={};parse_equal=True
    for S in (24,48):
        item=R.coupled_summary(rows,uids,S=S);ledger=item.pop('row_ledger')
        pairs=item.pop('pair_steps',None);selfreplay[str(S)]=item.pop('selfreplay_rows',[]);by[str(S)]=item
        parse_equal= parse_equal and all(all(x['stored_answer_matches'].values()) for x in ledger)
    extra=[]
    for i,row in enumerate(rows):
        inst=gold.get(row['uid']);issues=[]
        if inst is None:issues.append('unexpected_problem_uid')
        else:
            for k in ('id','gold'):
                if row[k]!=inst[k]:issues.append('problem_'+k)
            if str(inst['a']*inst['b'])!=inst['gold']:issues.append('problem_gold_product')
        if issues:extra.append({'line':i+1,'issues':issues})
    info=config['history'][-1];s=info['settings']
    fixed={'64_test_problems':len(uids)==64 and len(set(uids))==64,'rows8192':len(rows)==8192,
           'model_qwen1_5b':info['args']['model']=='qwen1.5b','test_split':info['args']['split']=='test',
           'phases12':info['args']['phases']==[1,2],'48_steps':info['args']['steps']==48,
           'readouts':s['readouts']==[24,48],'dtype':s['dtype']=='float32','settings':s==settings3(R)}
    ph=hashlib.sha256(json.dumps(problems['problems'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    fixed['canonical_problems_hash']=ph==problems['sha256']==info['problems_sha256']
    coverage=by['48']['coverage']
    qualified=all(fixed.values()) and not extra and not any(coverage[k] for k in ('missing','unexpected','duplicates')) and all(not x['issues'] and not x['nonfinite'] for x in ledger) and parse_equal
    if parse_equal:parsed={'reuses_literal':True,'basis':'Every saved answer equals the independently parsed complete saved text at both readouts.'}
    else:
        parsed={'reuses_literal':False,'by_readout':{str(S):R.coupled_summary(rows,uids,S=S,parsed=True)['summary'] for S in (24,48)}}
    return {'by_readout':by,'fixed_checks':fixed,'problem_issues':extra,'qualified_saved_rows':qualified,
        'parsed_projection':parsed,'selfreplay':selfreplay,'pair_steps':pairs,
        'scope':'Historical verdict strings are literal source labels, not guaranteed causal conclusions; no native state/message equality checked.'},ledger


def settings3(R):
    return {'coupling':.25,'strong_coupling':.5,'steps':48,'readouts':[24,48],'primary_readout':48,
        'sparse_every':4,'top_n':15,'temperature':.6,'answer_tokens':8,'start_words':list(R.STARTS3),
        'pairs':8,'mask_special':True,'dtype':'float32','problem_seed':12301,'boot_seed':12303,'n_boot':10000}


def do_c4(R,raw,configs,insts,old,inputs,pins,source):
    selected={k:R.completed4(raw[k]) for k in ('a','b')};infos={k:configs[k]['history'][-1] for k in ('a','b')}
    fixed={k:R.fixed_coverage4(selected[k]['rows'],infos[k],insts['instances'],k) for k in ('a','b')}
    ledger=[{'run':k,**x} for k in ('a','b') for x in selected[k].pop('row_ledger')]
    result=R.confirm4(selected['a']['rows'],selected['b']['rows'],infos['a'],infos['b'])
    parse_equal=all(all(r['stored_decision_matches'].values()) and all(r['forced_decision_matches'].values())
       for k in ('analysis_a','analysis_b') for r in result[k]['gate_diagnostics']['row_checks'])
    if parse_equal:parsed={'reuses_literal':True,'basis':'All complete saved readout texts, including saved forced readouts, agree with stored strict decisions.'}
    else:parsed={'reuses_literal':False,'result':R.confirm4(selected['a']['rows'],selected['b']['rows'],infos['a'],infos['b'],parsed=True)}
    ih=hashlib.sha256(json.dumps(insts['instances'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    projection=R.confirm_projection4(result,selected['a'],selected['b'],configs['a']['history'],configs['b']['history'],ih,pins['choices']['sha256'],
        '1aae51535d2485bb0c010ed9801a0374bb88179e01f735cfca1ff8bf56148475')
    checker=R.original_checker4(selected['a']['rows'],selected['b']['rows']);comp=R.checker_comparison4(checker,projection)
    fresh=R.freshness4(insts['instances'],old['instances']);pools=source['source_user_pools']
    fresh['source_pool_overlaps']={k:sorted(set(pools['USERS_CONF'])&set(v)) for k,v in pools.items() if k!='USERS_CONF'}
    metadata_checks={'canonical_instance_hash':ih==insts['sha256'],
        'both_instance_hashes':projection['gates']['instances_sha256_ok'],
        'both_conf_splits':projection['gates']['split_conf'],
        'choices_identity':projection['gates']['choices_equal_c12_4'],
        'recorded_v_hash':projection['gates']['vectors_a_ok'],
        'recorded_v2_hash':infos['b']['vectors_sha256']=='4554a8d1d816826baa2ed942df0eef8d0c550d89261087a6312d41b9adb161d5',
        'source_links':all(x['matches_exact_original_export'] for x in source['config_source_links'] if x['run'] in ('a','b')),
        'no_prior_table_matches':all(not x['prior_table_matches'] and not x['previous_new_table_matches'] for x in fresh['tables']),
        'gold_and_class':all(x['gold_matches'] and x['class_matches'] for x in fresh['tables']),
        'source_user_pools_disjoint':all(not v for v in fresh['source_pool_overlaps'].values())}
    qualifications={}
    for k,key in (('a','analysis_a'),('b','analysis_b')):
        g=fixed[k];diag=result[key]['gate_diagnostics']
        qualifications[k]=all(g['fixed_checks'].values()) and not g['identity_issues'] and all(not x['issues'] for x in g['row_issues']) and not selected[k]['selection_issues'] and not selected[k]['selected_duplicates'] and not diag['source_issues'] and not diag['additional_unexpected_rows'] and all(not x['nonfinite'] and all(x['stored_decision_matches'].values()) and all(x['forced_decision_matches'].values()) for x in diag['row_checks'])
    return {'selection':selected,'fixed_coverage':fixed,'qualified_saved_rows':qualifications,'result':result,
       'confirm_projection':projection,'checker':checker,'checker_comparison':comp,'parsed_projection':parsed,
       'freshness':fresh,'metadata_and_freshness_checks':metadata_checks,
       'frozen_data_gates_satisfied':all(metadata_checks.values()) and all(qualifications.values()),
       'scope':'Every historical verdict remains visible; own failed qualification prevents endorsing it. Native vectors/states and previous model use are not independently checked.'},ledger


def publication_rows(R,pubs,c3,c4,configs,inputs,pins):
    good3='by_readout' in c3;good4='confirm_projection' in c4
    d3={s:x['summary'] for s,x in c3.get('by_readout',{}).items()}
    conf=c4.get('confirm_projection',{});chk=c4.get('checker_comparison',{}).get('literal') or {}
    contexts={'publication3':{'experiment','settings','problems_sha256','status','model','choice','label','decision_3a','decision_3b','deviations','caveats','runs'},
      'publication4':{'card','label','status','tool','settings','deviations','caveats','freeze','instances','choices','choices_file','vectors','analyses','raw'},
      'receipt_spec':{'card','label','status','tool','instance_set','settings','deviations','caveats','choices_file','vectors_dir','analyses','raw_dirs','freeze'}}
    for doc,pub in pubs.items():
        for path,val in leaves(pub):
            expected=MISSING;basis=None;inherited=None
            if doc=='analysis3':
                expected=at({'run':str(Path(pins['c3_records']['path']).parent),'by_readout':d3},path);basis='own_C3_heldout'
            elif doc=='confirm':expected=at(conf,path);basis='own_C4_confirm'
            elif doc=='check':expected=at(chk,path);basis='own_C4_checker_sequential_stream'
            elif doc=='publication3':
                if path[:3]==('runs','test-1','results'):expected=at(d3,path[3:]);basis='own_C3_heldout'
                elif path and path[0]=='settings':expected=at(settings3(R),path[1:]);basis='frozen_source_constants'
                elif path and path[0] in ('decision_3a','decision_3b'):expected=at(d3.get('48',{}),path);basis='own_C3_literal_verdict'
                elif path==('problems_sha256',):expected=configs['c3']['history'][-1]['problems_sha256'];basis='recorded_config_hash'
                elif path==('model',):expected=configs['c3']['history'][-1]['args']['model'];basis='recorded_config_identity'
                elif len(path)>1 and path[0]=='runs' and path[1]!='test-1':inherited='Development numerical/receipt fields outside this heldout replay.'
                elif path[:2]==('runs','test-1'):
                    info=configs['c3']['history'];obj={'path':str(Path(pins['c3_records']['path']).parent),'commits':[h['commit'] for h in info],
                      'model':info[0]['args']['model'],'split':info[0]['args']['split'],
                      'files_sha256':{'records.jsonl':pins['c3_records']['sha256'],'run.json':pins['c3_config']['sha256'],'analysis.json':pins['analysis3']['sha256']}}
                    expected=at(obj,path[2:]);basis='pinned_metadata'
                    if expected is MISSING:inherited='Unselected native message or other original receipt file; no native tensor audit.';basis=None
            else:
                if doc=='publication4' and path[:2]==('analyses','confirm'):expected=at(conf,path[2:]);basis='own_C4_confirm'
                elif doc=='publication4' and path[:2]==('analyses','independent_check'):expected=at(chk,path[2:]);basis='own_C4_checker_sequential_stream'
                elif path and path[0]=='choices':expected=at(load_json(pins['choices']),path[1:]);basis='frozen_choices_source_equality_not_dev_replay'
                elif path[:2]==('choices_file','sha256'):expected=pins['choices']['sha256'];basis='frozen_choices_pin'
                elif path[:2]==('instances','sha256'):expected=c4.get('freshness',{}).get('canonical_list_sha256',MISSING);basis='own_canonical_table_hash'
                elif path[:2]==('instances','n'):expected=c4.get('freshness',{}).get('new_count',MISSING);basis='own_table_count'
                elif path and path[0]=='freeze':
                    m={'run_a_started_utc':configs['a']['history'][0]['started'],'run_a_finished_utc':configs['a']['history'][-1].get('finished'),
                       'run_b_started_utc':configs['b']['history'][0]['started'],'run_b_finished_utc':configs['b']['history'][-1].get('finished'),
                       'tool_sha256':configs['a']['history'][-1]['tool_sha256'],'pi_harness_sha256':configs['a']['history'][-1]['pi_harness_sha256']}
                    expected=at(m,path[1:]);basis='recorded_run_metadata'
                    if expected is MISSING:inherited='Frozen source/chronology declaration retained; not historical push or loaded-source proof.';basis=None
            if basis:
                z=difference(path,expected,val,basis)
                if expected is MISSING and ((basis.startswith('own_C3') and not good3) or (basis.startswith('own_C4') and not good4)):z['disposition']='unavailable_component_refusal'
            else:
                if inherited is None and path and path[0] in contexts.get(doc,set()):inherited='Source/receipt prose or native/development premise retained; no heldout arithmetic reproduction claimed.'
                z={'path':list(path),'published':val,'disposition':'inherited_scope' if inherited else 'unclassified_publication_leaf','reason':inherited}
            yield {'document':doc,**z}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    card=json.loads(Path(args.card).read_text());out=Path(args.out)
    if card.get('status')!='admitted' or str(out)!=OUTPUT or card['output']!=OUTPUT or card['limits']!=LIMITS:
        raise ValueError('prepared/wrong card or output: no execution admitted')
    for name,pin in card['source_files'].items():
        local=Path(__file__).with_name(name)
        if digest(local)!=pin['sha256'] or local.stat().st_size!=pin['bytes']:raise ValueError('source mismatch '+name)
    check_pin(card['inputs']);inputs=load_json(card['inputs']);pins={r['role']:r for r in inputs['files']}
    if len(pins)!=len(inputs['files']) or sum(x['bytes'] for x in pins.values())>LIMITS['input_bytes']:raise ValueError('input role/byte budget')
    if set(inputs['jsonl_expected'])!={'c3','a','b'}:raise ValueError('wrong heldout selection')
    out.mkdir(parents=False,exist_ok=False);W=Writer(out);started=time.monotonic()
    try:
        for p in pins.values():check_pin(p)
        for p in card['inherited_profile'].values():check_pin(p)
        controls=load_json(card['inherited_profile']['controls'])
        profile=load_json(card['inherited_profile']['report'])
        receipt=load_json(card['inherited_profile']['receipt'])
        closure=load_json(card['inherited_profile']['closure'])
        if (controls['failed'] or controls['passed']!=15 or profile.get('groups')!=15
            or not profile.get('computational_completed') or not profile.get('resource_profile_only')
            or receipt.get('returncode')!=0 or not receipt.get('worker_reaped')
            or closure.get('worker_returncode')!=0 or not closure.get('worker_reaped')):
            raise ValueError('inherited sole manufactured profile not complete/passed')
        W.write('controls.json',controls)  # Literal saved result; no controls rerun.
        R=own('reducers')
        raw={};configs={}
        for k in ('c3','a','b'):
            raw[k]=[json.loads(line) for line in Path(pins[k+'_records']['path']).read_text().splitlines()]
            if len(raw[k])!=inputs['jsonl_expected'][k]:raise ValueError('wrong immutable JSONL count '+k)
            configs[k]=load_json(pins[k+'_config'])
        evidence=source_evidence(inputs,pins,configs)
        components={};ledgers={}
        for key,fn in (('c12_3',lambda:do_c3(R,raw['c3'],configs['c3'],load_json(pins['problems']))),
                       ('c12_4c',lambda:do_c4(R,raw,configs,load_json(pins['instances']),load_json(pins['old_instances']),inputs,pins,evidence))):
            try:components[key],ledgers[key]=fn()
            except Exception as exc:
                components[key]={'refusal':repr(exc),'traceback':traceback.format_exc(),'qualified_saved_rows':False}
                ks=('c3',) if key=='c12_3' else ('a','b')
                ledgers[key]=[{'run':k,'line':i+1,'source':r,'analysis_refusal':True} for k in ks for i,r in enumerate(raw[k])]
        W.write('c12-3.json',components['c12_3']);W.write('c12-4c.json',components['c12_4c'])
        W.lines('c12-3-rows.jsonl',ledgers['c12_3']);W.lines('c12-4c-rows.jsonl',ledgers['c12_4c'])
        pubs={k:load_json(pins[k]) for k in ('publication3','publication4','confirm','check','receipt_spec','analysis3')}
        comparison_counts=collections.Counter()
        def counted():
            for row in publication_rows(R,pubs,components['c12_3'],components['c12_4c'],configs,inputs,pins):
                comparison_counts[row['document'],row['disposition']]+=1;yield row
        W.lines('comparisons.jsonl',counted())
        for p in pins.values():check_pin(p)
        check_pin(card['inputs'])
        for p in card['inherited_profile'].values():check_pin(p)
        report={'schema':'independent-c12-heldout-v1','computational_completed':True,'scientific_claims_accepted':False,
          'input_manifest':card['inputs'],'input_files':list(pins.values()),'source_evidence':evidence,
          'inherited_manufactured_profile':card['inherited_profile'],'controls_executed_in_this_run':False,
          'components':{k:{'completed':'refusal' not in v,'qualified_saved_rows':v.get('qualified_saved_rows'),'refusal':v.get('refusal')} for k,v in components.items()},
          'comparisons':[{'document':k[0],'disposition':k[1],'leaves':v} for k,v in sorted(comparison_counts.items())],
          'source_files':card['source_files'],'elapsed_through_rehash_before_report':time.monotonic()-started,
          'rss_kib_observed_before_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'limitations':['Unblinded saved-row arithmetic, not model replay or fresh confirmation.','Historical verdict wording is not guaranteed causal attribution.',
             'Native weights/state/message/vectors and original remote push time not independently verified.','C12-3 development values are inherited outside heldout scope.',
             'All failed coverage, parsing and nonfinite conditions retained; numerical equality is distinct from scientific qualification.']}
        W.write('report.json',report);W.write('manifest.json',{'files':list(W.files),'scope':'All prior own output files; manifest excludes itself.'})
    except Exception as exc:
        W.write('failure.json',{'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started,'computational_completed':False})
        raise


if __name__=='__main__':main()
