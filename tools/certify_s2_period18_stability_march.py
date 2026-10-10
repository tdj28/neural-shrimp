"""Binding-only fixed descending coverage of remaining small-march charts."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,subprocess,sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_preceding_chart_stability as base
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_stability_march.py'
FIRST='tools/certify_s2_period18_preceding_chart_stability.py'
CARD='docs/plans/s2-period18-stability-march.md'
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_stability_march.py','tests/test_s2_period18_stability_march.py',CARD]))
ORIGINAL_INPUT=base.input_for


def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    return bound


def actual_runner(runtime,index):
    """Presence of the new executed adapter cannot hide behind its old import."""
    inventory=runtime['source_sha256']
    if CURRENT in inventory:
        require(index<7 and inventory[CURRENT]==io.entry(ROOT/CURRENT)['sha256'],'exact previous march runner')
    else:
        require(index==7 and inventory.get(FIRST)==io.entry(ROOT/FIRST)['sha256'],'exact first predecessor runner')


def previous_coverage(path,pin,outer_pin,index):
    require(type(index)is int and 0<=index<=7,'fixed descending remaining chart index')
    require(io.entry(path.parent/'manifest.json')['sha256']==outer_pin,'preceding outer pin');io.verify_inventory(path.parent,io.read(path.parent/'manifest.json')['files'])
    require(io.entry(path/'manifest.json')['sha256']==pin,'preceding worker pin');io.verify_inventory(path,io.read(path/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');launch=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    actual_runner(rt,index)
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'preceding exact settings')
    require(launch['source_commit']==rt['source_commit'] and launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'preceding execution/ownership')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'preceding historical source')
    require(out['passed'] and out['completed_slices']==8 and out['stages']==1288 and len(out['rows'])==8 and all(r['passed'] for r in out['rows']),'complete qualified preceding coverage')
    require(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete preceding acquisition')
    require(inp['selected_artifact']['index']==index+1 and out['complete_coverage']==inp['coverage_endpoints']==inp['chart']['exact_parameter_endpoints'],'immediate exact preceding coverage index')
    require(base.coverage([r['exact_endpoints'] for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'preceding complete exact partition')
    return inp,{'root':str(path),'worker_manifest':pin,'outer_manifest':outer_pin,'source_commit':rt['source_commit'],'index':index+1}


def input_for(mode,index=None,previous=None,pin=None,outer_pin=None):
    if mode=='profile':return ORIGINAL_INPUT(mode)
    predecessor,identity=previous_coverage(previous,pin,outer_pin,index)
    parent=io.io.parent;value=parent.partial_input()
    path=parent.PARENT/f'chart-{index:03d}.json.gz';chart=parent.read(path)
    next_chart=parent.read(parent.PARENT/f'chart-{index+1:03d}.json.gz')
    require(predecessor['chart']==next_chart and predecessor['model_inputs']==value['model_inputs'],'same exact predecessor root/map')
    parent.io.joined(parent.io.compact_chart(chart),next_chart)
    require(chart['passed'] and all(chart['gates'].values()) and chart['statement']['period']==18 and chart['statement']['phase_dimension']==13,'qualified native chart')
    require(chart['normal']==next_chart['normal'] and chart['origin']==next_chart['origin'],'same native gauge')
    ends=[str(Fraction(35+index,128)),str(Fraction(36+index,128))]
    require(chart['exact_parameter_endpoints']==ends and ends[1]==next_chart['exact_parameter_endpoints'][0],'fixed original adjacent chart endpoints')
    value.update(chart=chart,slices=base.partition(ends),coverage_endpoints=ends,coverage_predecessor=identity,
      selected_artifact={'root':str(parent.PARENT),'manifest':parent.PIN,'source_commit':parent.EXECUTION,'index':index,'identity':io.entry(path)})
    return value


def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None,index=None,previous=None,previous_pin=None,previous_outer_pin=None):
    require((mode=='target')==(index is not None and previous is not None and previous_pin is not None and previous_outer_pin is not None),'exact predecessor only in target')
    if mode=='target':require(type(index)is int and 0<=index<=7,'fixed remaining index')
    old_sources,old_input=base.sources,base.input_for
    base.sources=sources;base.input_for=lambda mode:input_for(mode,index,previous,previous_pin,previous_outer_pin)
    try:return base.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:base.sources,base.input_for=old_sources,old_input

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--index',type=int)
    p.add_argument('--previous',type=Path);p.add_argument('--previous-manifest');p.add_argument('--previous-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'exact profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest,a.index,a.previous,a.previous_manifest,a.previous_outer_manifest))
