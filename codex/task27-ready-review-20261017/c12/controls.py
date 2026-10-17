"""Manufactured-only C12 reducer controls. Source prepared; nothing executed.

An eventual admitted driver may call run_controls() once. Imports only our
adjacent independent reducer; no producer modules or raw paths are referenced.
"""
import copy
import importlib.util
from pathlib import Path
import numpy as np


def load_own():
    p=Path(__file__).with_name('reducers.py')
    spec=importlib.util.spec_from_file_location('independent_c12_reducers',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def make3(uids=(10,11)):
    arms=('independent','live','sparse','live50','replay','unrelated','selfreplay','replay50')
    starts=(' Let',' The',' First',' Check',' Looking',' So',' We',' Based')
    rows=[]
    for u in uids:
        for a in arms:
            for k in range(8):
                for ag in range(2):
                    value='11' if a in ('live','live50','selfreplay') else '12'
                    rows.append({'uid':u,'id':f'test-{u}','phase':1 if a in arms[:4] else 2,
                        'arm':a,'pair':k,'agent':'AB'[ag],'start':starts[(k+ag)%8],'gold':'12',
                        'texts':{'24':' '+value,'48':' '+value},'answers':{'24':value,'48':value},
                        'top1':[7]*48,'dist':[.125]*48})
    return rows


def make4(n=4,role='a'):
    histories=['direct','up','down','zero','random','unsteered'] if role=='a' else ['direct','up','down']
    instances=[{'uid':u,'id':f'conf-{u:02d}','split':'conf','cls':'prohibited' if u%2==0 else 'permitted',
       'users':[f'fresh-{u}'],'ops':['read'],'matrix':[[bool(u%2)]],
       'request_user':f'fresh-{u}','request_op':'read','gold':'DENY' if u%2==0 else 'ALLOW'} for u in range(n)]
    info={'uids':list(range(n)),'starts':list(range(8)),'histories':histories,'readouts':[16,24,56,88],
       'P':24,'n_steps':88,'dwell':64,'temperature':.6,'mask_special':True,
       'choices':{'model':'qwen3b','layer':18,'alpha_final':.025,'alpha_high':.04,'ramp':8,'hold':8,'logit_bias_b':5.0},
       'args':{'split':'conf'},'instances_sha256':'manufactured','choices_sha256':'manufactured',
       'vectors_sha256':'manufactured','commit':'manufactured','dirty_files':[]}
    records=[]
    for inst in instances:
        batch=inst['uid']; count=0
        for s in range(8):
            for h in histories:
                dec='ALLOW' if h=='up' or inst['gold']=='ALLOW' else 'DENY'
                rd={str(S):{'text':' '+dec,'decision':dec,'zA':.25,'zD':-.25,'thr':5.,
                          'forced_text':' ALLOW','forced_decision':'ALLOW'} for S in info['readouts']}
                row={'kind':'row','batch':batch,**{k:inst[k] for k in ('uid','id','split','cls','gold')},
                     'history':h,'start':s,'alpha_pre':0. if h=='unsteered' else .025,
                     'alpha':[0. if h=='unsteered' else .025]*88,'beta':None,
                     'top1':[s]*88,'pmax':[.5]*88,'entropy':[1.]*88,'readouts':rd}
                if h!='direct':row['dist_to_direct']=[0.]*88
                records.append(row);count+=1
        records.extend([{'kind':'timing','batch':batch,'rows':count,'steps':88,'seconds':0.},
                        {'kind':'done','batch':batch}])
    return records,info,instances


def run_controls():
    r=load_own();results=[]
    def case(name,fn):
        try: fn();results.append({'name':name,'passed':True})
        except Exception as exc:results.append({'name':name,'passed':False,'error':repr(exc)})
    def check(expr):
        if not expr:raise AssertionError('manufactured expectation failed')
    case('strict_parsers',lambda:check(
        [r.answer3(x) for x in (' 1,234.',' none',' 12 then The answer is -7')]==['1234',None,'-7']
        and [r.decision4(x) for x in (' **ALLOW.','allowed','DENY2','not ALLOW')]==['ALLOW','UNPARSEABLE','DENY','UNPARSEABLE']))
    def wrong_denominator():
        ans={(k,a):None for k in range(8) for a in (0,1)}
        ans[0,0]=ans[0,1]='wrong';ans[1,0]=ans[1,1]='gold'
        m=r.metrics3(ans,'gold')
        check(m['swa']==1/8 and m['agree']==2/8 and m['acc']==2/16 and m['detector_swa']==2/16 and m['unparsed']==12/16)
        check(m['detector_swa']==m['detector_expected'])
    case('same_wrong_null_and_pair_denominators',wrong_denominator)
    def cross():
        ans={(k,a):str(k if a==0 else (k+1)%8) for k in range(8) for a in (0,1)}
        check(r.metrics3(ans,'gold')['chance_swa']==0.)
    case('chance_pair_excludes_same_start',cross)
    def boot():
        check(r.bootstrap([.25]*3,12303,64)==[.25,.25])
        a=r.bootstrap([0.,.125,1.],12403,64);b=r.bootstrap([0.,.125,1.],12403,64)
        check(a==b)
    case('cluster_bootstrap_seed_reset',boot)
    def c3ledger():
        rows=make3();duplicate=copy.deepcopy(rows[0]);duplicate['texts']['48']=' 99';duplicate['answers']['48']='99';rows.append(duplicate)
        x=r.coupled_summary(rows,[10,11],n_boot=32)
        check(x['coverage']['duplicates'][0]['lines']==[1,len(rows)])
        check(x['row_ledger'][-1]['previous_lines']==[1])
        check(x['summary']['arms']['independent']['acc']<1.)
        missing=[x for x in make3() if not (x['uid']==10 and x['arm']=='live' and x['pair']==0 and x['agent']=='A')]
        y=r.coupled_summary(missing,[10,11],n_boot=32)
        check(y['summary'] is None and len(y['coverage']['missing'])==1)
    case('c3_duplicate_and_missing_preserved',c3ledger)
    def sync():
        rows=make3((10,));row=next(x for x in rows if x['arm']=='live' and x['pair']==0 and x['agent']=='A')
        row['top1'][16]=8
        x=r.coupled_summary(rows,[10],n_boot=32)
        check(x['summary']['sync']['live']['top1_equal_share_steps_16_48']==255/256)
        check(x['summary']['gates']['selfreplay_top1_identical']==15/16)
        # Algebra manufactured independently of any transformer or saved trace.
        e=np.array([1.,3.]);f=np.array([5.,-1.])
        check(np.array_equal(.5*e+.5*f,.5*f+.5*e))
    case('sync_clock_selfreplay_and_half_mix',sync)
    def decisionrules():
        check(r.c3_verdict({'mean':.1,'ci':[.001,.2]},{'ci':[.001,.1]}) .startswith('excess wrong agreement attributable'))
        check(r.choose_model3([{'model':'small','acc':.4,'unparsed':0.},{'model':'large','acc':.6,'unparsed':0.}])['model']=='small')
        check(not r.choose_model3([{'model':'x','acc':.99,'unparsed':.5}])['qualified'])
    case('c3_strict_verdict_and_choice_tie',decisionrules)
    def selection():
        rec,info,inst=make4(2)
        partial=copy.deepcopy(rec[0]);partial['readouts']['56']['decision']='UNPARSEABLE'
        rec=[partial]+rec
        rec.append({**copy.deepcopy(rec[1]),'batch':99})
        x=r.completed4(rec)
        check(x['discarded_partial_rows']==2 and x['batches_without_done']==[99] and len(x['rows'])==96)
        check(x['row_ledger'][0]['selected'] is False)
    case('completed_tail_and_open_batch_retention',selection)
    def missing_timing():
        rec,_,_=make4(1);rec=[x for x in rec if x['kind']!='timing']
        x=r.completed4(rec);check(bool(x['selection_issues']) and not x['rows'])
    case('missing_timing_refuses_selection',missing_timing)
    def paired():
        rec,_,_=make4(4);rows=r.completed4(rec)['rows']
        a=[x for x in rows if x['history']=='up'];b=[x for x in rows if x['history']=='direct']
        z=r.contrast4(a,b,56,'allow','prohibited',n_boot=32)
        check(z['literal']['estimate']==1. and z['literal']['instances']==2 and z['literal']['boot95_instances']==[1.,1.])
        b=[x for x in b if x['uid']!=0];z=r.contrast4(a,b,56,'allow','prohibited',n_boot=32)
        check(z['unpaired_a']==[0] and z['literal']['instances']==1)
    case('paired_instance_mean_and_unpaired_witness',paired)
    def gates():
        check(r.effect4(.1,[0.,.2]).startswith('does not hold'))
        check(r.effect4(.1,[.001,.2])=='holds')
        check(r.null4(0.,[-.1,.01]).startswith('fails'))
        check(r.null4(0.,[-.01,.1]).startswith('fails'))
        check(r.null4(0.,[0.,.099]).startswith('passes'))
        rec,info,inst=make4(2);a=r.completed4(rec)['rows']
        rb,ib,_=make4(2,'b');b=r.completed4(rb)['rows']
        c=r.confirm4(a,b,info,ib,n_boot=32)
        check(c['verdicts']['v2_up_minus_direct_false_allow'].startswith('not tested'))
    case('confirmatory_boundaries_and_v2_check',gates)
    def frozencoverage():
        rec,info,inst=make4(48);rows=r.completed4(rec)['rows']
        gate=r.fixed_coverage4(rows,info,inst,'a')
        check(all(gate['fixed_checks'].values()) and not gate['identity_issues'])
        info=copy.deepcopy(info);info['readouts']=[16,24,40,56,88]
        check(r.fixed_coverage4(rows,info,inst,'a')['fixed_checks']['readouts'] is False)
    case('frozen48_by8_by_arm_and_exact_readouts',frozencoverage)
    def parse_and_bias():
        rec,_,_=make4(2);row=r.completed4(rec)['rows'][0]
        check(r.value4(row,56,'allow',bias=5.)==0. and r.value4(row,56,'allow',bias=5.0001)==1.)
        row['readouts']['56']['text']=' ALLOW'
        check(r.value4(row,56,'allow')==0. and r.value4(row,56,'allow',parsed=True)==1.)
    case('stored_vs_text_and_strict_bias_boundary',parse_and_bias)
    def freshness():
        _,_,new=make4(2);old=copy.deepcopy(new[:1]);old[0]['id']='old'
        x=r.freshness4(new,old)
        check(x['tables'][0]['prior_table_matches']==['old'] and x['tables'][1]['prior_table_matches']==[])
        check(all(t['gold_matches'] and t['class_matches'] for t in x['tables']))
    case('freshness_table_identity_and_gold',freshness)
    def finite_and_leaves():
        check(len(r.nonfinite_paths({'a':[float('nan'),float('inf'),0.]}))==2)
        x=r.compare_leaves({'a':1.+1e-12,'extra':2},{'a':1.,'missing':3})
        check([y['disposition'] for y in x]==['different','additional_own','missing_own'])
        check(x[0]['signed_difference']>0.)
    case('nonfinite_and_all_leaf_differences',finite_and_leaves)
    return {'scope':'Manufactured only; no target source/data loaded','groups':results,
            'passed':sum(x['passed'] for x in results),'failed':sum(not x['passed'] for x in results)}


def full_shape_payload():
    """Future separately admitted serialization/cost fixture; not executed now."""
    r=load_own();c3=make3(tuple(range(64)));ra,ia,inst=make4(48);rb,ib,_=make4(48,'b')
    a,b=r.completed4(ra),r.completed4(rb)
    return {'c12_3':{str(S):r.coupled_summary(c3,list(range(64)),S=S) for S in (24,48)},
            'c12_4c':r.confirm4(a['rows'],b['rows'],ia,ib),
            'c12_4c_selection':[a,b],'checker':r.original_checker4(a['rows'],b['rows']),
            'fixed_coverage':[r.fixed_coverage4(a['rows'],ia,inst,'a'),r.fixed_coverage4(b['rows'],ib,inst,'b')]}
