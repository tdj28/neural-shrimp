"""Only original charts143 then142, each its own capped eight-slice unit."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_localized_predecessor_stability as base
import certify_s2_period18_preceding_chart_stability as core
import certify_s2_period18_fold_endpoint as endpoint
import certify_s2_period18_fold_small_march as native
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_two_predecessor_stability.py'
RECEIPT='docs/reviews/s2-period18-localized-predecessor-stability-outcome.json'
RAW=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-two-predecessor-stability-2026-10-10')
FIRST=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-localized-predecessor-stability-2026-10-10/attempt001-target/run')
ENDS={13:['29/128','31/128'],12:['27/128','29/128']}
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_two_predecessor_stability.py','tests/test_s2_period18_two_predecessor_stability.py','docs/plans/s2-period18-two-predecessor-stability.md',RECEIPT]))

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(data)
    return result

def paths(index):
    require(type(index)is int and index in ENDS,'only143 then142')
    return (FIRST if index==13 else RAW/'unit001/run'),RAW/f'unit{14-index:03d}/run'

def completed(index,path,pin,outer):
    expected,_=paths(index);require(path==expected,'fixed immediate completed predecessor')
    if index==13:
        rec=io.read(ROOT/RECEIPT);require(Path(rec['root'])/'run'==path and rec['worker_manifest']['sha256']==pin and rec['outer_manifest']['sha256']==outer,'fixed first completed144 receipt')
    for root,digest in [(path,pin),(path.parent,outer)]:
        require(io.entry(root/'manifest.json')['sha256']==digest,'exact predecessor manifest');io.verify_inventory(root,io.read(root/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');launch=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'exact predecessor settings')
    driver=base.CURRENT if index==13 else CURRENT
    require(rt['source_sha256'].get(driver)==io.entry(ROOT/driver)['sha256'],'exact predecessor runner')
    require(rt['source_commit']==launch['source_commit'] and launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'complete predecessor execution')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'historical predecessor source')
    require(out['passed'] and out['completed_slices']==8 and out['stages']==1288 and len(out['rows'])==8 and all(r['passed']for r in out['rows']),'complete predecessor attraction')
    require(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete predecessor calls')
    require(inp['selected_artifact']['index']==index+1 and inp['selected_artifact']['global_chart']==131+index,'exact predecessor chart index')
    require(core.coverage([r['exact_endpoints']for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'complete predecessor partition')
    return inp,{'root':str(path),'worker_manifest':io.entry(path/'manifest.json'),'outer_manifest':io.entry(path.parent/'manifest.json')}

def select(index,value,adjacent,identity,known,chart,nextchart):
    require(index in ENDS and chart['exact_parameter_endpoints']==ENDS[index],'fixed current chart interval')
    require(adjacent['chart']==nextchart and value['model_inputs']==adjacent['model_inputs']==known['model_inputs'],'same exact native map and next chart')
    native.joined(native.compact_chart(chart),nextchart)
    value.update(chart=chart,slices=core.partition(chart['exact_parameter_endpoints']),coverage_endpoints=chart['exact_parameter_endpoints'],coverage_predecessor=identity,
      selected_artifact={'root':str(endpoint.PARENT),'manifest':endpoint.PIN,'source_commit':endpoint.EXECUTION,'identity':io.entry(endpoint.PARENT/f'chart-{index:03d}.json.gz'),'global_chart':130+index,'index':index})
    # The old144-to145 localized join remains a predecessor fact, not this
    # ordinary143-to144 or142-to143 native endpoint join.
    value.pop('localized_join',None)
    return value

def input_for(mode,index=None,previous=None,pin=None,outer=None):
    value=base.input_for(mode)
    if mode=='profile':return value
    adjacent,identity=completed(index,previous,pin,outer)
    known=endpoint.partial_input()
    return select(index,value,adjacent,identity,known,io.read(endpoint.PARENT/f'chart-{index:03d}.json.gz'),io.read(endpoint.PARENT/f'chart-{index+1:03d}.json.gz'))

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None,index=None,previous=None,pin=None,outer=None):
    require(out==(RAW/'attempt001-profile/run' if mode=='profile' else paths(index)[1]),'fixed exclusive output')
    old=core.sources,core.input_for
    core.sources=sources;core.input_for=lambda mode:input_for(mode,index,previous,pin,outer)
    try:return core.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:core.sources,core.input_for=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--index',type=int);p.add_argument('--previous',type=Path);p.add_argument('--previous-manifest');p.add_argument('--previous-outer-manifest');a=p.parse_args()
    required=(a.profile,a.profile_manifest,a.profile_outer_manifest,a.index,a.previous,a.previous_manifest,a.previous_outer_manifest)
    require(all(v is not None for v in required)if a.mode=='target'else all(v is None for v in required),'exact target predecessor/profile arguments only')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest,a.index,a.previous,a.previous_manifest,a.previous_outer_manifest))
