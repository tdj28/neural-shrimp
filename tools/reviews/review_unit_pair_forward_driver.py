"""Card-bound saved384-bit independent native705 prefix and full-gauge joins; zero native calls."""
import os
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
for k in THREADS:os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,resource,signal,subprocess,sys,time,traceback
from pathlib import Path
import flint
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).parent)]
import review_unit_pair_pal_math as A
import review_pal_rechart_math as E
need=A.need
FILES=['tools/reviews/review_unit_pair_pal_math.py','tools/reviews/review_unit_pair_forward_driver.py','tools/reviews/launch_unit_pair_forward_review.py','tests/test_unit_pair_forward_review.py','tests/test_pal_rechart_review.py','docs/plans/native705-forward-independent.md']
FILES=sorted(set(FILES+[str(Path(m.__file__).resolve().relative_to(ROOT))for m in list(sys.modules.values())if getattr(m,'__file__',None)and Path(m.__file__).resolve().is_relative_to(ROOT)]))
def entry(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        while b:=f.read(1024**2):h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())
def read(p):
    if p.suffix=='.gz':
        with gzip.open(p,'rb')as f:b=f.read(256*1024**2+1)
        need(len(b)<=256*1024**2,'bounded256MiB decompression');return json.loads(b)
    return json.loads(p.read_bytes())
def inventory(root,pin):
    need(entry(root/'manifest.json')['sha256']==pin,'literal manifest');m=read(root/'manifest.json')['files']
    need(set(m)=={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()and p!=root/'manifest.json'},'complete raw inventory')
    for n,v in m.items():need(entry(root/n)==v,'immutable file '+n)

def validate_card(card):
    need(card['schema']=='native705-fixed-batch-audit-v1'and 1<=card['qualified_prefix']<=8 and card['qualified_prefix']<=card['expected_workers']<=8,'bounded complete fixed prefix')
    need(len(card['charts'])==card['qualified_prefix']and [v['unit']for v in card['charts']]==list(range(1,card['qualified_prefix']+1)),'ordered literal prefix')
    for row in card['charts']:need(row['path']==f"chart{row['unit']:03d}/run"and len(row['worker'])==64,'safe literal chart and worker pin')
    need(type(card.get('localize_prior',False))is bool,'explicit saved localization switch')
    need(len(card['source'])==40 and len(card['outer'])==len(card['prior']['outer'])==64,'complete literal source/inventory pins')
    need(Path(card['root']).is_relative_to('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/generalized-chain/unit-pair/pal')and Path(card['prior']['root']).is_relative_to('/data2/neural-shrimp-data/audits')and Path(card['out']).is_relative_to('/data2/neural-shrimp-data/audits/native705-forward-independent'),'fixed task-owned storage')
    need(all('..'not in Path(card[k]).parts for k in ['root','out'])and '..'not in Path(card['prior']['root']).parts,'no path escape')

def main(card_path):
    card_path=card_path.resolve();need(card_path.is_relative_to(ROOT/'docs/plans'),'committed plan path');card=read(card_path);validate_card(card);OUT=Path(card['out'])/'run'
    need(os.getpriority(os.PRIO_PROCESS,0)>=10 and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).strip(),'clean committed nice10 source')
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,)*2);OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();used=0;ticks=0;result=dict(passed=False)
    def stop(*_):raise TimeoutError('240s independent705 deadline')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.setitimer(signal.ITIMER_REAL,240)
    def tick():
        nonlocal ticks
        ticks+=1
        if time.monotonic()-start>=240:raise TimeoutError('240s independent705 deadline')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>=4*1024**2:raise MemoryError('4GiB independent705 RSS')
    def save(name,value,closing=False):
        nonlocal used
        data=(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        need(used+len(data)<(64 if closing else 60)*1024**2,'64MiB raw cap');used+=len(data);p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
    try:
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();identities={n:entry(ROOT/n)for n in FILES+[str(card_path.relative_to(ROOT))]}
        for n,v in identities.items():
            need(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==v['sha256'],'committed dependency');p=OUT/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());used+=p.stat().st_size
        save('runtime.json',dict(source_commit=source,sources=identities,precision=384,model_calls=0,derivative_calls=0,python=sys.version,python_flint=flint.__version__,flint=flint.__FLINT_VERSION__,pid=os.getpid(),threads=dict.fromkeys(THREADS,'1')))
        batch=Path(card['root']);inventory(batch,card['outer']);bo=read(batch/'outcome.json');bs=card['source']
        need(bo['commit']==bs and bo['connected_count']==card['qualified_prefix']and bo['owned_live_pids']==[]and len(bo['workers'])==card['expected_workers']and all(v['owned_live_pids']==[]for v in bo['workers']),'fixed complete producer ownership and qualified prefix')
        prior=Path(card['prior']['root']);inventory(prior,card['prior']['outer']);old=read(prior/'run/outcome.json');ce=read(prior/'run/connected-endpoint.json');oldbind=read(prior/'run/input-binding.json')
        need(old['passed']and old['connected']and old['connected_count']>=1 and old['proofs'][-1]['parameter_endpoints'][1]==old['connected_parameter_endpoints'][1]==ce['chart']['parameter_endpoints'][1],'independently proved connected starting endpoint')
        last=read(prior/f"run/chart-{old['connected_count']}/uniform.json")
        need(all(v for k,v in last['gates'].items()if k!='anchor_identified')and last['inclusions']==705 and ce['endpoint']==last['recomputed_endpoints'][1],'exact independently recomputed endpoint enclosure')
        previous=ce['chart'];endpoint=ce['endpoint'];meta=['shooting_dimension','normal','origin','parameter_center','parameter_endpoints','settings']
        previous_source=Path(oldbind['charts'][-1]['root']);need(entry(previous_source/'manifest.json')==oldbind['charts'][-1]['manifest'],'prior source map inventory pin')
        inventory(previous_source,oldbind['charts'][-1]['manifest']['sha256']);map_inputs=read(previous_source/'inputs.json')['map_inputs']
        if card.get('localize_prior',False):
            prior_chart=read(previous_source/'certificate.json.gz')
            need(all(prior_chart[k]==previous[k]for k in meta),'known endpoint matches exact prior chart geometry')
            pd=sorted((previous_source/'events').glob('*-direction_return.json.gz'));pp=sorted((previous_source/'events').glob('*-pal_path_phase_return.json.gz'))
            localized=E.localize_right(prior_chart,[read(p)for p in pd],[read(p)for p in pp],endpoint,384,tick)
            save('prior-endpoint-localization.json',localized);need(localized['passed']and localized['dimension']==705,'independent known endpoint localization')
            endpoint=localized['box'];del prior_chart,localized;gc.collect();tick()
            packet=read(batch/'input-card.json')
            need(packet['source']==bs and packet['worker_count']==card['expected_workers']and packet['map_inputs']==map_inputs and Path(packet['parent']['root'])/'run'==previous_source and packet['parent']['manifest']==entry(previous_source/'manifest.json'),'literal rotated predecessor and exact map')
        checked=set();proofs=[];joins=[];inputs=[]
        for row in card['charts']:
            unit=row['unit'];root=batch/row['path'];inventory(root,row['worker']);rt=read(root/'runtime.json');done=read(root/'outcome.json')
            need(bo['workers'][unit-1]['unit']==unit and bo['workers'][unit-1]['exit_code']==0,'selected worker success')
            need(rt['commit']==bs and done['unit']==unit and done['passed']and done['ticks']==253 and done['strict_inclusions']==705 and done['events']['direction_return']==144 and done['events']['pal_path_phase_return']==108,'exact uniform chart native acquisition')
            for name,v in rt['sources'].items():
                need(entry(root/'source'/name)==v,'archived producer source '+name);key=(name,v['sha256'])
                if key not in checked:
                    need(hashlib.sha256(subprocess.check_output(['git','show',bs+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'historical committed producer source '+name);checked.add(key)
            inp=read(root/'inputs.json');ci=read(root/'chart-inputs.json');need(inp['map_inputs']==map_inputs,'identical exact native map')
            if card.get('localize_prior',False):need(inp['card']==entry(batch/'input-card.json')and ci['normal']==packet['normal']and ci['origin']==packet['origin']and ci['weights']==packet['weights'],'exact rotated finite proposal and current card')
            directions=sorted((root/'events').glob('*-direction_return.json.gz'));paths=sorted((root/'events').glob('*-pal_path_phase_return.json.gz'))
            need(len(directions)==144 and len(paths)==108,'complete chart derivative inventory')
            chart=read(root/'certificate.json.gz');need(chart['normal']==ci['normal']and chart['origin']==ci['origin']and chart['weights']==ci['weights'],'same exact geometry inputs')
            retain={'schema','dimension','period','shooting_dimension','settings','parameter_center','parameter_endpoints','normal','origin','center','slope','curvature','weights','coordinate_radii','preconditioner','whole_box'}
            for key in list(chart):
                if key not in retain:del chart[key]
            gc.collect();tick()
            # Uniform proof first. A changed gauge requires the entire projected
            # old root interval; only an identical gauge admits exact-parameter use.
            same=previous['normal']==chart['normal']and previous['origin']==chart['origin']
            proof=A.check_chart(chart,[read(p)for p in directions],[read(p)for p in paths],None if same else endpoint,384,tick)
            save(f'chart-{unit}/uniform.json',proof);tick()
            need(all(v for k,v in proof['gates'].items()if k!='anchor_identified')and proof['inclusions']==705,'independent full705 uniform root, primitive period and critical frame')
            join=A.exact_gauge_join(previous,endpoint,chart,384)if same else dict(proof['anchor'],scope='Whole independently proved endpoint projected into new exact affine gauge; full705 strict membership.')
            save(f'chart-{unit}/join.json',join);need(join['passed']and join['coordinates']==705,'all705 strict same-root join coordinates')
            proofs.append(dict(unit=unit,inclusions=proof['inclusions'],contraction=proof['contraction'],parameter_endpoints=chart['parameter_endpoints'],same_gauge=same));joins.append(join)
            inputs.append(dict(root=str(root),manifest=entry(root/'manifest.json'),certificate=entry(root/'certificate.json.gz')))
            previous={k:chart[k]for k in meta};endpoint=proof['recomputed_endpoints'][1]
            del chart,proof;gc.collect();tick()
        save('input-binding.json',dict(card=card,prior=str(prior),prior_manifest=card['prior']['outer'],charts=inputs))
        save('connected-endpoint.json',dict(chart=previous,endpoint=endpoint))
        result.update(passed=True,connected=True,connected_count=len(proofs),connected_parameter_endpoints=[proofs[0]['parameter_endpoints'][0],proofs[-1]['parameter_endpoints'][1]],proofs=proofs,joins=joins,localized_prior=card.get('localize_prior',False),scope='Independent384-bit native705 uniform roots and strict full-state known-root joins for only the fixed qualified prefix. Parameter intervals refer to this packet; global scalar ordering is not inferred across changed gauges. Shared acquired analytic derivatives and Arb.')
    except BaseException as exc:result.update(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);result.update(seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,ticks=ticks,native_calls=0);save('outcome.json',result,True);save('manifest.json',dict(files={str(p.relative_to(OUT)):entry(p)for p in OUT.rglob('*')if p.is_file()}),True)
    print(json.dumps(result));return 0 if result['passed']else 2
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--card',type=Path,required=True);args=parser.parse_args();raise SystemExit(main(args.card))
