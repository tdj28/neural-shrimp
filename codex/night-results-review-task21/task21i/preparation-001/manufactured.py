"""Combined tiny controls and full-shape synthetic workload. No target input access."""
import argparse
import importlib.util
import json
import resource
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent

def module(name,file):
    sp=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

a=module('adapter21i','run.py'); c=module('controls21i','controls.py')


def fakepin(path): return {'path':path,'bytes':1,'sha256':'0'*64}


def fixtures():
    tasks={'arithmetic':[{'answer':i+1} for i in range(50)],'nq':[{'answers':['gold']} for i in range(50)],'mbpp':[{} for i in range(30)]}
    pairs={'pairs':[],'cells':[]};bz=[]
    order={'bz':{'runs':['llm-soft-loop/pod64/bz/pilot-'+str(i) for i in range(3)],'tasks':list(a.TASKS),'metrics':['d_acc','d_entropy','d_length']},
           'bp':{'label_runs':{'q05':['q05-label-'+str(i) for i in range(2)],'q7':['q7-label-'+str(i) for i in range(6)]},'temperature_order':list(a.TEMPERATURES),'subset_order':['all','greedy_correct']}}
    for p in range(30):
        lt=-.17+.01*p;alpha=.06+.00001*p
        pairs['pairs'].append({'pair':p,'map':'manufactured','log10_temperature':lt,'chaotic':{'alpha':alpha,'exponent':.1},'periodic':{'alpha':alpha+.0001,'exponent':-.1,'period':7}})
        for k,label in enumerate(('chaotic','periodic','unsteered')):
            ident=3*p+k;cell={'cell':ident,'pair':p,'label':label,'log10_temperature':lt,'alpha':alpha if k==0 else alpha+.0001 if k==1 else 0.}
            pairs['cells'].append(cell);groups={};meta={}
            for t in a.TASKS:
                rows=[];repeat=2 if t=='mbpp' else 3
                for i,item in enumerate(tasks[t]):
                    for j in range(repeat):
                        yes=(i+j+ident)%5!=0
                        reply=(str(item['answer']) if yes else '999999') if t=='arithmetic' else ('gold' if yes else 'wrong') if t=='nq' else 'pass'
                        rows.append({'item':i,'sample':j,'correct':yes,'entropy':.5+.01*((i+j+ident)%9),'length':3+(i+j)%8,'reply':reply})
                groups[t]={'accuracy':sum(r['correct'] for r in rows)/len(rows),'mean_entropy':float(a.np.mean([r['entropy'] for r in rows])),
                           'mean_length':float(a.np.mean([r['length'] for r in rows])),'rows':rows}
                meta[t]={'rows':len(rows),'row_key_order':list(rows[0]),'item_sample_identities':[[r['item'],r['sample']] for r in rows]}
            entry=fakepin(f'/manufactured/bz/pilot-{ident%3}/cell-{ident:03d}.json')|{'attempt':f'BZ{ident%3}','metadata':{'cell':cell,'task_order':list(a.TASKS),'groups':meta}}
            bz.append((entry,{'cell':cell,'groups':groups}))
    items=[{'prompt':p,'task':'arithmetic','answer':p+1} if p<100 else {'prompt':p,'task':'nq','answers':['gold']} for p in range(200)]
    bp={}
    for model,shards in (('q05',2),('q7',6)):
        labs=[]
        for shard in range(shards):
            records=[]
            for p in range(shard,200,shards):
                cells=[]
                for j,lt in enumerate(a.TEMPERATURES):
                    k=(p+j)%3;cls=('periodic','visible chaos','irregular, exponent not positive at both offsets')[k]
                    cells.append({'log10_temperature':lt,'class':cls,'lyapunov':[.1,.2] if k==1 else [-.2,-.1] if k==0 else [.2,-.1],
                                  'period':7 if k==0 else 0,'entropy_mean':.5})
                records.append({'prompt':p,'task':items[p]['task'],'prefix_tokens':20,'cells':cells,'seconds':.01})
            meta={'rows':len(records),'identifiers_in_file_order':[{'prompt':r['prompt'],'task':r['task'],'temperatures':list(a.TEMPERATURES)} for r in records]}
            labs.append((fakepin(f'/manufactured/bp/{model}-label-{shard}/labels.jsonl')|{'metadata':meta},records))
        gen={'greedy':{},'sampled':{str(lt):{} for lt in a.TEMPERATURES}}
        for p,it in enumerate(items):
            def reply(yes):return (str(it['answer']) if yes else '999999') if it['task']=='arithmetic' else 'gold' if yes else 'wrong'
            gen['greedy'][str(p)]={'correct':p%4!=0,'reply':reply(p%4!=0)}
            for j,lt in enumerate(a.TEMPERATURES):
                gen['sampled'][str(lt)][str(p)]=[{'correct':(p+n+j)%4!=0,'entropy':.3+.01*((p+n)%7),'length':4+n%3,'reply':reply((p+n+j)%4!=0)} for n in range(10)]
        meta={'greedy_prompt_order':list(gen['greedy']),'sampled_temperature_order':list(gen['sampled']),
              'sampled':{key:{'prompt_order':list(v),'sample_counts':{p:len(rs) for p,rs in v.items()}} for key,v in gen['sampled'].items()}}
        bp[model]=(labs,fakepin('/manufactured/bp/'+model+'-gen/generations.json')|{'metadata':meta},gen)
    return bz,pairs,tasks,{'items':items},bp,order


def refusal(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('expected refusal absent')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(exist_ok=False,parents=False)
    try:
        passed=c.controls()
        refusal(lambda:a.decode('{"prompt":1,"prompt":2}'))
        passed.append('duplicate_json_key_refused')
        for cell,mapped,valid in [({'period':7,'lyapunov':[.3,.4],'class':'periodic'},'periodic',True),
                                 ({'period':0,'lyapunov':[.3,.4],'class':'hidden chaos'},'chaotic',True),
                                 ({'period':0,'lyapunov':[.3,-.4],'class':'irregular, exponent not positive at both offsets'},'other',True),
                                 ({'period':-1,'lyapunov':[float('nan'),.4],'class':'visible chaos'},'other',False)]:
            d=a.label_diagnostic(cell);assert d['reconstructed_union']==mapped and d['finite_period_qualified']==valid
        passed.append('period_precedence_union_and_invalid_label_diagnostics')
        delta=a.compare('tiny',{'a':1.0000000000000002,'b':[2]}, {'a':1.,'b':[2],'c':3},'manufactured')
        assert len(delta)==3 and delta[0]['difference']>0 and not delta[2]['equal']
        passed.append('every_leaf_nonzero_and_missing_comparison_retained')
        bz,pairs,tasks,bptasks,bp,order=fixtures()
        invalid=list(bz);e,r=invalid[0];broken=dict(r);broken['groups']=dict(r['groups']);broken['groups']['arithmetic']=dict(r['groups']['arithmetic']);broken['groups']['arithmetic']['rows']=r['groups']['arithmetic']['rows'][:-1];invalid[0]=(e,broken)
        refusal(lambda:a.bz_reduce(invalid,pairs,tasks,order['bz'],lambda r:None,1))
        passed.append('BZ_missing_response_refused')
        lab,ge,g=bp['q05'];badlab=list(lab);badlab[0]=(lab[0][0],lab[0][1][:-1])
        refusal(lambda:a.bp_reduce(badlab,ge,g,bptasks,'q05',order['bp'],lambda r:None,1))
        passed.append('BP_missing_label_identity_refused')
        # Full fixed rows and RNG workload; separately named manufactured output files.
        started=time.monotonic();count=0
        with open(args.out/'responses.jsonl','x') as stream:
            def emit(row):
                nonlocal count
                stream.write(json.dumps(a.clean(row),allow_nan=False)+'\n');count+=1
            out={'bz':a.bz_reduce(bz,pairs,tasks,order['bz'],emit)}
            out['bp']={model:a.bp_reduce(*bp[model],bptasks,model,order['bp'],emit) for model in ('q05','q7')}
        assert count==44800 and out['bz']['response_rows']==32400 and all(r['response_rows']==6200 for r in out['bp'].values())
        a.write_json(args.out/'scientific-shape.json',out)
        elapsed=time.monotonic()-started
        passed.append('full_fixed_coverage_and_source_RNG_workload_serialized')
        files=[a.pin(args.out/n) for n in ('responses.jsonl','scientific-shape.json')]
        report={'passed':passed,'failed':0,'measurement':{'seconds_through_scientific_serialization':elapsed,
            'observed_peak_kib_before_final_report':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'response_rows':count,
            'BZ_pairs':30,'BZ_cells':90,'BP_models':2,'BP_prompt_temperature_rows':1200,
            'BZ_bootstrap_draws':90000,'BP_permutation_draws':120000,'files':files,'scientific_bytes':sum(r['bytes'] for r in files),
            'scope':'One manufactured full allocation and full declared RNG workload; original hash/I/O/decode and actual-value string lengths unmeasured; no worst-case claim.'},
            'sources':[a.pin(HERE/n) for n in ('statistics.py','run.py','controls.py','manufactured.py')]}
        a.write_json(args.out/'report.json',report)
        print(json.dumps({'passed':len(passed),'failed':0,'report':a.pin(args.out/'report.json')},indent=2))
    except BaseException as e:
        a.write_json(args.out/'failure.json',{'type':type(e).__name__,'message':str(e)});raise

if __name__=='__main__':main()
