"""Saved two-chart adapter; producer-authored wrapper, independent core reductions.

Every fixed scalar domain and the full global acquisition tree are reconstructed
here. All original curve roots/joins remain independently source-bound. No native
calls or producer root/tangent/quotient reducers execute in this audit.
"""
import os
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import gc,json,platform,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'tools'),str(ROOT/'src')]
import flint
from flint import ctx
import review_cubic_remaining as B
import review_cubic_center as C
import probe_native705_cubic_panel as P
need,read=B.need,B.read
OUT=Path('/data2/neural-shrimp-data/audits/native705-cubic-panel-independent/attempt001/run')
CARD='docs/plans/native705-cubic-panel-independent-inputs.json'
LIMITS=dict(seconds=2400,rss_bytes=4*1024**3,as_bytes=6*1024**3,raw_bytes=512*1024**2,native_calls=0)
FILES=sorted(set(B.FILES+P.FILES+[f'tools/reviews/{pre}{n}.py'for pre in('','native704_aux/')for n in('review_joined_period18_math','review_quadratic_connection_math','review_unit_pair_math')]+['tools/reviews/review_cubic_panel.py','tools/reviews/launch_cubic_panel_review.py','tests/reviews/test_cubic_panel_review.py','tests/reviews/test_cubic_exact_ranges.py','docs/plans/native705-cubic-panel-independent.md',CARD]))


def card_check(card):
    need(card['status']=='reviewed'and card['limits']==LIMITS and card['ordinals']==[33,34]and card['maximum_domains']==16 and card['maximum_centers']==96,'literal reviewed saved panel card')
    need(card['measured_producer']==P.MEASURED,'literal measured producer reference')
    for k in('producer','initial_independent'):
        need(set(card[k])=={'root','source','outer','worker'}and all(type(v)is str and v for v in card[k].values()),'completed literal '+k+' pins')


def fixed_domains(spec):
    need(spec['ordinal']in(33,34),'literal chart33/34')
    a,b=map(F,spec['parameter_endpoints']);need(b-a==F(1,16),'entire original chart width')
    return [dict(index=i+1,endpoints=[str(a+F(i,128)),str(a+F(i+1,128))])for i in range(8)]


def fixed_cell(domains,domain,depth,index):
    need(type(domain)is int and 1<=domain<=8 and type(depth)is int and 0<=depth<=3 and type(index)is int and 0<=index<2**depth,'literal depth3 binary cell')
    a,b=map(F,domains[domain-1]['endpoints']);width=(b-a)/2**depth;a+=index*width;b=a+width
    return dict(domain=domain,depth=depth,index=index,endpoints=[str(a),str(b)],center=str((a+b)/2),halfwidth=str((b-a)/2))


def exact_union(rows,endpoints):
    need(rows and all(r['passed']and r['complement_dimension']==11 for r in rows),'all independently passing11-mode leaves')
    spans=sorted(tuple(map(F,r['cell']['endpoints']))for r in rows)
    need([str(spans[0][0]),str(spans[-1][1])]==endpoints and all(a[1]==b[0]for a,b in zip(spans,spans[1:])),'complete closed original chart union without gaps/duplicate interiors')
    return dict(passed=True,endpoints=endpoints,total_cells=len(rows),complement_dimension=11)


def history(out,run,pc):
    need(out['passed']and out['status']=='two_cubic_charts_complement_covered'and out['domains_done']==16 and not out['pending_cells'],'complete two-chart producer')
    all_attempts=out['completed_attempts'];need(16<=len(all_attempts)<=96 and len(out['completed_charts'])==2,'fixed global96 center bound')
    B.pairs(run,0,144);cursor=144;index=0;selected=[];stages=[]
    for spec in pc['charts']:
        ordinal=spec['ordinal'];folder=run/f'charts/{ordinal:03d}';domains=fixed_domains(spec);accepted=[]
        for d in domains:
            part=folder/f"domains/{d['index']:03d}";dom=read(part/'domain-outcome.json')
            need(dom==dict(passed=True,domain=d,dimension=705,native_calls=90,first_call=cursor+1,last_call=cursor+90),'exact successive90-call domain')
            B.pairs(part,cursor,90);stages.append(dict(chart=ordinal,domain=d,start=cursor));cursor+=90;queue=[fixed_cell(domains,d['index'],0,0)]
            while queue:
                need(index<len(all_attempts),'complete literal binary subtree');cell=queue.pop(0);number=index+1;r=all_attempts[index];index+=1;part=folder/f'cells/{number:03d}'
                need(r['chart']==ordinal and r['ordinal']==number and r['cell']==cell and r['completed']and r['native_calls']==126 and r['dimension']==705 and r['complement_dimension']==11 and r['first_call']==cursor+1 and r['last_call']==cursor+126,'global ordered126-call full705 attempt')
                native={k:v for k,v in r.items()if k not in('chart','ordinal')}
                need(read(part/'cell-input.json')==cell and read(part/'cell-outcome.json')==native and read(run/f'completed-attempt-{number:03d}.json')==r,'literal durable attempt records')
                B.pairs(part,cursor,126)
                if r['passed']:
                    item=dict(chart=ordinal,ordinal=number,cell=cell,start=cursor);selected.append(item);accepted.append(item)
                else:
                    need(r['passed']is False and cell['depth']<3 and read(part/'complement.json')['passed']is False,'only complete negative metric split')
                    queue=[fixed_cell(domains,cell['domain'],cell['depth']+1,2*cell['index']+k)for k in(0,1)]+queue
                cursor+=126
        complete=read(run/f'complete-chart-{ordinal:03d}.json')
        need(complete==out['completed_charts'][ordinal-33]and complete['chart']==ordinal and complete['root_index']==spec['root_index'],'literal complete chart/source row')
        u=exact_union([dict(v,passed=True,complement_dimension=11)for v in accepted],spec['parameter_endpoints'])
        need(complete['union']==dict(passed=True,endpoints=u['endpoints'],accepted_cells=len(accepted),domain_count=8),'original exact full chart completion')
    need(index==len(all_attempts)and out['accepted_cells']==[all_attempts[r['ordinal']-1]for r in selected],'all accepted producer leaves, no selected subset')
    need(out['native_calls']==cursor==144+16*90+126*index<=13680 and out['profile_native_calls']==144 and out['target_native_calls']==cursor-144,'complete global13680 acquisition history')
    return stages,selected


def metadata(w,card):
    card_check(card)
    root,rt,out,inventory=B.A.S.bundle(card['producer'],w.tick);run=root/'run';pc=read(run/'source'/P.CARD);P.card_check(pc)
    need(pc['initial_independent']==card['initial_independent']and rt['caps']==P.CAPS,'same initial independent32 and reviewed producer caps')
    for n in P.G.B.NATIVE+P.G.M.T.PATH_FILES:need(rt['sources'][n]==B.A.S.entry(ROOT/n),'unchanged acquired native/path source')
    for n in('tools/probe_native705_cubic_panel.py','tools/bind_native705_cubic_panel.py','src/neural_shrimp/cubic_complement_panel.py'):
        need(rt['sources'][n]==B.A.S.entry(ROOT/n),'literal producer adapter grammar '+n)
    profile=read(run/'batch-profile.json');need(profile['passed']and profile['maximum_total_calls']==13680,'whole global budget profiled before acquisition')
    stages,cells=history(out,run,pc);w.prefix='';previous=P.initial(w,pc);contexts=[]
    for spec in pc['charts']:
        original=P.G.metadata(spec,previous,w.tick);saved=read(run/f"charts/{spec['ordinal']:03d}/context/input-binding.json")
        need(saved['index']==original['index']and saved['prior_card']==original['independent_binding']['card']and saved['prior_stable_authority']==previous and saved['prior_projection']==original['anchor_projection']and saved['predictor']==spec['predictor']and saved['coefficient_artifacts']==spec['coefficients'],'literal original full705 root/gauge/source context')
        completed=read(run/f"complete-chart-{spec['ordinal']:03d}.json");need(completed['predecessor']==previous,'literal completed prior-root link')
        contexts.append((spec,original));previous=dict(index=spec['root_index'],complete_complement_endpoints=spec['parameter_endpoints'],producer_prefix_ordinal=spec['ordinal'])
    w.save('authority-binding.json',dict(producer=card['producer'],initial_independent=card['initial_independent'],charts=pc['charts'],inventories=inventory,domains=stages,cells=cells,authorship='Producer-agent wrapper reviewed/executed by root; shared source/metadata and finite decoding, separately authored root/tangent/point/quotient arithmetic unchanged.'))
    return run,stages,cells,contexts


class Work:
    def __init__(self):self.start=time.monotonic();self.used=0;self.prefix='';self.rows=[];self.completed=[]
    def tick(self):need(time.monotonic()-self.start<2400 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<4*1024**2,'2400s4GiB independent panel')
    def save(self,name,value,closing=False):
        raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();need(self.used+len(raw)<(512 if closing else 508)*1024**2,'512MiB4MiB reserve');p=OUT/self.prefix/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.used+=len(raw)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,)*2);need(not subprocess.check_output(['git','-c','core.preloadIndex=false','-c','index.threads=1','status','--porcelain'],cwd=ROOT).strip()and os.getpriority(os.PRIO_PROCESS,0)>=10,'clean CPU1nice10 saved adapter');card=read(ROOT/CARD);card_check(card);OUT.mkdir(parents=True,exist_ok=False);w=Work();result=dict(passed=False,native_calls=0)
    def stop(*_):raise TimeoutError('2400s saved panel')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,2400)
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();w.save('runtime.json',dict(source_commit=source,pid=os.getpid(),sources={n:B.A.S.entry(ROOT/n)for n in FILES},precision=480,native_calls=0,python=platform.python_version(),python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,threads=dict.fromkeys(THREADS,'1')))
        for n in FILES:
            raw=(ROOT/n).read_bytes();need(raw==subprocess.check_output(['git','show',source+':'+n],cwd=ROOT),'exact committed full source');need(w.used+len(raw)<508*1024**2,'bounded source archive');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);w.used+=len(raw)
        with ctx.workprec(480):
            card=read(ROOT/CARD);before=time.monotonic();trial=B.A.A.synthetic(705,w.tick);seconds=time.monotonic()-before
            prior=B.P.OUT.parent;need(str(prior)==card['measured_producer']['root']and card['measured_producer']==P.MEASURED,'literal complete6domain30center producer cost')
            pin=card['initial_independent'];pr=Path(pin['root'])/'run';need(B.A.S.entry(pr/'manifest.json')['sha256']==pin['worker'],'completed measured independent32 inventory');pm=read(pr/'manifest.json')['files'];need(B.A.S.entry(pr/'outcome.json')==pm['outcome.json'],'literal completed independent cost');cost=read(pr/'outcome.json');need(cost['passed']and cost['total_cells']==18 and cost['native_calls']==0,'completed independent remaining32 reference')
            ratio=max(16/6,96/18);wall=8*seconds+2*ratio*cost['elapsed_seconds']+2*61.466+60;storage=w.used+2*ratio*sum(v['bytes']for v in pm.values())+64*1024**2;ok=trial['passed']and wall<2400 and storage<512*1024**2
            w.save('profile.json',dict(passed=ok,seconds=seconds,forecast_seconds=wall,forecast_bytes=storage,target_arrays_unopened=True));need(ok,'full705 and complete measured96leaf panel admission');gc.collect()
            run,stages,cells,contexts=metadata(w,card)
            for spec,original in contexts:
                ordinal=spec['ordinal'];w.prefix=f'charts/{ordinal:03d}/context';data=B.context(w,original,spec);w.prefix=f'charts/{ordinal:03d}';local=[]
                for domain in stages:
                    if domain['chart']!=ordinal:continue
                    shared=C.shared_domain(w,run/f'charts/{ordinal:03d}',domain,data)
                    for item in cells:
                        if item['chart']!=ordinal or item['cell']['domain']!=domain['domain']['index']:continue
                        row=C.one(w,run/f'charts/{ordinal:03d}',item,data,shared);row['chart']=ordinal;local.append(row);w.rows.append(row);w.save(f'prefix-{len(local):02d}.json',dict(rows=local));gc.collect()
                union=exact_union(local,spec['parameter_endpoints']);w.save('union.json',union);w.completed.append(dict(chart=ordinal,union=union,original_root=spec['root_index']));del data,shared;gc.collect()
            need(len(w.completed)==2 and len(w.rows)==len(cells),'entire independently replayed two-chart union');w.tick();result.update(passed=True,complement_dimension=11,scope='Independent full11 complementary stability on whole existing cubic33–34, joined by original full705 uniqueness to independently qualified whole32. Neutral critical pair retained; no final-junction or full-window claim.')
    except BaseException as exc:result.update(passed=False,error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);w.prefix='';result.update(completed_charts=w.completed,completed_cells=w.rows,elapsed_seconds=time.monotonic()-w.start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);w.save('outcome.json',result,True);w.save('manifest.json',dict(files={str(p.relative_to(OUT)):B.A.S.entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':raise SystemExit(main())
