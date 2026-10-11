"""Fixed chart144 coverage through its authenticated localized endpoint."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_preceding_quarter_stability as base
import certify_s2_period18_preceding_chart_stability as core
import certify_s2_period18_fold_rechart as localized
import certify_s2_period18_fold_endpoint as endpoint
import certify_s2_period18_fold_saved_rebox as saved
import certify_s2_period18_fold_small_march as native
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_localized_predecessor_stability.py'
RECEIPT='docs/reviews/s2-period18-preceding-quarter-stability-outcome.json'
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_localized_predecessor_stability.py','tests/test_s2_period18_localized_predecessor_stability.py','docs/plans/s2-period18-localized-predecessor-stability.md',RECEIPT]))
ORIGINAL_INPUT=base.input_for

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(data)
    return result

def completed_adjacent():
    receipt=io.read(ROOT/RECEIPT);path=Path(receipt['root'])/'run'
    require(path==Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-preceding-quarter-stability-2026-10-10/attempt001-target/run'),'fixed completed chart145')
    for root,pin in [(path,receipt['worker_manifest']),(path.parent,receipt['outer_manifest'])]:
        require(io.entry(root/'manifest.json')==pin,'exact adjacent manifest');io.verify_inventory(root,io.read(root/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');launch=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'exact adjacent settings')
    require(rt['source_sha256'].get(base.CURRENT)==io.entry(ROOT/base.CURRENT)['sha256'],'exact adjacent four-piece runner')
    require(rt['source_commit']==launch['source_commit'] and launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'complete adjacent execution')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'historical adjacent source')
    require(out['passed'] and out['completed_slices']==4 and out['stages']==644 and len(out['rows'])==4 and all(r['passed']for r in out['rows']),'complete adjacent attraction')
    require(out['counts']=={'map_attempted':648,'map_returned':648,'unit_attempted':1296,'unit_returned':1296,'mixed_attempted':144,'mixed_returned':144},'complete adjacent calls')
    require(inp['selected_artifact']['global_chart']==145 and out['complete_coverage']==inp['coverage_endpoints']==['33/128','17/64'],'exact chart145 coverage')
    require(base.base.coverage([r['exact_endpoints']for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'exact adjacent quarter partition')
    return inp,{'root':str(path),'worker_manifest':receipt['worker_manifest'],'outer_manifest':receipt['outer_manifest']}

def select(value,adjacent,identity,known,rechart,chart,old):
    require(value['chart']==adjacent['chart'] and value['model_inputs']==adjacent['model_inputs']==known['model_inputs']==rechart['model_inputs']==old['model_inputs'],'same exact map and adjacent chart')
    require(native.compact_chart(chart)==known['previous']==old['previous'],'exact qualified original chart144')
    require(chart['exact_parameter_endpoints']==['31/128','33/128'],'exact native width and endpoint')
    require(adjacent['chart']['anchor']==rechart['anchor'],'exact full localized235D anchor')
    require(chart['normal']==adjacent['chart']['normal']==rechart['normal'] and chart['origin']==adjacent['chart']['origin']==rechart['origin'],'same gauge through localization')
    require(rechart['anchor']['exact_parameter']==chart['exact_parameter_endpoints'][1]==adjacent['chart']['exact_parameter_endpoints'][0]=='33/128','same exact shared parameter')
    value.update(chart=chart,slices=core.partition(chart['exact_parameter_endpoints']),coverage_endpoints=chart['exact_parameter_endpoints'],coverage_predecessor=identity,
      selected_artifact={'root':str(endpoint.PARENT),'manifest':endpoint.PIN,'source_commit':endpoint.EXECUTION,'identity':io.entry(endpoint.PARENT/'chart-014.json.gz'),'global_chart':144,'index':14},
      localized_join={'root':str(localized.PARENT),'manifest':localized.PIN,'source_commit':localized.EXECUTION,'result':io.entry(localized.PARENT/'result.json'),'scope':'same root retained by three conditional intersections; full localized anchor equality'})
    return value

def input_for(mode):
    value=ORIGINAL_INPUT(mode)
    if mode=='profile':return value
    adjacent,identity=completed_adjacent()
    known=endpoint.partial_input() # authenticates all15 original charts and retained failed tail
    rechart=localized.prepare() # authenticates exact three-intersection localization of chart144 endpoint
    return select(value,adjacent,identity,known,rechart,io.read(endpoint.PARENT/'chart-014.json.gz'),io.read(localized.PARENT/'inputs.json'))

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    old=core.sources,core.input_for
    core.sources=sources;core.input_for=input_for
    try:return core.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:core.sources,core.input_for=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
