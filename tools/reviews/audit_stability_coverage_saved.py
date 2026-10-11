"""Metadata-only extension of the independently passing chart156 reducer."""
from audit_chart156_stability_saved import *
import audit_chart156_stability_saved as C
META=None
PARENT=BASE/'s2-period18-fold-small-march-2026-10-10/unit-001/run'
PARENT_PIN='a5399a6b1888006dc6ca2b32e8ae0f7beb367b4281c66d55ab20f33b7f8ec3d2'
PARENT_EXEC='e987c8669efb5fb9275046f401cd4cfa172cfa9b'
FIRST='tools/certify_s2_period18_preceding_chart_stability.py'
MARCH='tools/certify_s2_period18_stability_march.py'
FIRST_SOURCE='64dfb34a87b40cac876be9d9b556403d09fa696f'
MARCH_SOURCE='b6017dd790c38a52a45082ffa585e2a08fd7d95e'

def partition(index):
    need(type(index)is int and 0<=index<=8,'fixed chart index')
    lo=8*(35+index)
    return [[str(Fraction(k,1024)),str(Fraction(k+1,1024))]for k in range(lo+7,lo-1,-1)]

def coverage(rows,index):
    expected=partition(index);need(rows==expected,'exact descending eight-slice coverage')
    return [expected[-1][0],expected[0][1]]

def verify_metadata(path,pin):
    need(sha(path.read_bytes())==pin,'external metadata pin');m=read(path)
    need(set(m)=={'root','outer','worker','execution','index'},'exact pin fields');partition(m['index'])
    need(Path(m['root']).resolve().is_relative_to(BASE.resolve()),'fixed certificate tree')
    for k in ('outer','worker'):need(isinstance(m[k],str)and re.fullmatch('[0-9a-f]{64}',m[k])is not None,'full hash '+k)
    need(isinstance(m['execution'],str)and re.fullmatch('[0-9a-f]{40}',m['execution'])is not None,'full execution')
    return m

def source_inventory(rt,index):
    ref=FIRST_SOURCE if index==8 else MARCH_SOURCE;runner=FIRST if index==8 else MARCH
    expected_inventory='9041a7ef586d98130554b52740080395ab533db559984660d0a3794ee42ac05b' if index==8 else 'ef414651fb8663626172a224c64b401ee722aab2a46068ebd8c81ce95528730f'
    need(sha(json.dumps(rt['source_sha256'],sort_keys=True,separators=(',',':')).encode())==expected_inventory,'complete frozen source inventory')
    need(runner in rt['source_sha256'],'actual executed runner present')
    if index==8:need(MARCH not in rt['source_sha256'],'first adapter identity')
    for name,digest in rt['source_sha256'].items():
        need(sha(subprocess.check_output(['git','show',ref+':'+name],cwd=ROOT))==digest,'unchanged reviewed producer byte '+name)

def parent_chart(inp,index,inv):
    inv.append(inventory(PARENT,PARENT_PIN));rt=read(PARENT/'runtime-start.json');need(rt['source_commit']==PARENT_EXEC,'known partial parent source')
    for name,digest in rt['source_sha256'].items():need(sha(subprocess.check_output(['git','show',PARENT_EXEC+':'+name],cwd=ROOT))==digest,'historical root source')
    out=read(PARENT/'outcome.json');need(out['exit_code']==2 and not out['passed'],'partial root attempt stays partial')
    selected=inp['selected_artifact'];need(selected['root']==str(PARENT)and selected['manifest']==PARENT_PIN and selected['source_commit']==PARENT_EXEC and selected['index']==index,'same selected root artifact')
    path=PARENT/f'chart-{index:03d}.json.gz';need(selected['identity']=={'bytes':path.stat().st_size,'sha256':sha(path.read_bytes())},'selected chart bytes')
    chart=read(path);need(inp['chart']==chart and inp['model_inputs']==read(PARENT/'inputs.json')['model_inputs'],'same native root/model')
    need(chart['passed']and all(chart['gates'].values())and chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'qualified inherited primitive chart')
    need(chart['exact_parameter_endpoints']==coverage(partition(index),index),'exact selected chart interval')
    return chart

def audit():
    meta=META;root=Path(meta['root']);index=meta['index'];inv=[inventory(root,meta['outer'])]
    a,rt,out=authenticate(root/'run',meta['worker'],meta['execution']);inv.append(a);source_inventory(rt,index)
    launch=read(root/'launcher-result.json');need(launch['source_commit']==meta['execution']and launch['exit_code']==0 and not launch['owned_live_pids']and 0<launch['elapsed_seconds']<397,'bounded completed launch')
    need(out['passed']and out['completed_slices']==8 and out['stages']==1288 and len(out['rows'])==8,'complete eight-slice outcome')
    need(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'full derivative acquisition counts')
    inp=read(root/'run/inputs.json.gz');expected=partition(index);need(inp['slices']==expected and inp['coverage_endpoints']==out['complete_coverage']==coverage(expected,index),'fixed coverage domain')
    chart=parent_chart(inp,index,inv)
    predecessor=None
    if index<8:
        predecessor=inp['coverage_predecessor'];need(predecessor['index']==index+1,'immediate predecessor index')
        pp=Path(predecessor['root']);need(pp.resolve().is_relative_to(BASE.resolve()),'predecessor path')
        inv.extend([inventory(pp.parent,predecessor['outer_manifest']),inventory(pp,predecessor['worker_manifest'])])
        pr=read(pp/'inputs.json.gz');po=read(pp/'outcome.json');pl=read(pp.parent/'launcher-result.json')
        need(po['passed']and po['exit_code']==pl['exit_code']==0 and not pl['owned_live_pids']and po['completed_slices']==8,'complete predecessor scope')
        need(pr['selected_artifact']['index']==index+1 and pr['model_inputs']==inp['model_inputs']and pr['chart']==read(PARENT/f'chart-{index+1:03d}.json.gz'),'same native adjacent predecessor')
        need(chart['normal']==pr['chart']['normal']and chart['origin']==pr['chart']['origin']and chart['exact_parameter_endpoints'][1]==pr['chart']['exact_parameter_endpoints'][0],'same gauge adjacency')
    rows=[]
    for i,ends in enumerate(expected):
        report=read(root/f'run/slice-{i:02d}-result.json.gz');records=C.grouped(root/'run',i);result=C.reduce_slice(inp,report,records,ends);put(AUDIT_OUT/f'case-{i:02d}.json',result);rows.append(result);del report,records;gc.collect()
    return {'passed':True,'index':index,'inventories':inv,'cases':rows,'covered_exact_chart':coverage([r['exact_endpoints']for r in rows],index),'coverage_predecessor':predecessor,'scope':'Independent saved true-tangent and full norm inequalities on this fixed chart only. Shared Arb and authenticated primitive-root/localization/analytic derivative premises; root chain not freshly replayed, no neutral-boundary/whole-shrimp/utility claim.'}
def main(out):
    global AUDIT_OUT
    AUDIT_OUT=out
    need(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed review')
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();status={'passed':False};code=1
    put(out/'runtime-start.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'checker_sha256':sha(Path(__file__).read_bytes()),'helper_sha256':sha(Path(A.__file__).read_bytes()),'tangent_helper_sha256':sha(Path(R.__file__).read_bytes()),'chart_helper_sha256':sha(Path(C.__file__).read_bytes()),'pins':META,'pid':os.getpid(),'ppid':os.getppid(),'nice':os.getpriority(os.PRIO_PROCESS,0),'precision':256,'internal_seconds':60,'raw_cap_bytes':8*1024**2,'rss_cap_kib':2*1024**2,'post_outcome':True})
    def stop(*_):raise TimeoutError('fixed saved-audit deadline')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,60)
    try:
        need(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10')
        with ctx.workprec(256):status=audit()
        code=0
    except BaseException as e:status.update(error={'type':type(e).__name__,'message':str(e)})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);status.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,exit_code=code)
        if status['peak_rss_kib']>=2*1024**2:status.update(passed=False,exit_code=1,error={'type':'MemoryError','message':'audit RSS exceeded'});code=1
        put(out/'result.json',status)
        files={str(p.relative_to(out)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}
        need(sum(v['bytes'] for v in files.values())<8*1024**2,'raw audit cap');put(out/'manifest.json',{'files':files})
    print(json.dumps(status));return code
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--pins',required=True,type=Path);p.add_argument('--pins-sha256',required=True);v=p.parse_args();META=verify_metadata(v.pins,v.pins_sha256);raise SystemExit(main(v.out))
