"""One full earlier chart, unchanged true-tangent and norm-cocycle proof."""
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_preceding_chart_stability as base
import resume_s2_period18_stability_march as previous
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_whole_chart_stability.py'
RECEIPT='docs/reviews/s2-period18-stability-resume-unit004.json'
SOURCES=list(dict.fromkeys(previous.SOURCES+[CURRENT,'tools/launch_s2_period18_whole_chart_stability.py','tests/test_s2_period18_whole_chart_stability.py','docs/plans/s2-period18-whole-chart-stability.md',RECEIPT]))
ORIGINAL_INPUT=base.input_for

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    return bound

def completed_adjacent():
    receipt=io.read(ROOT/RECEIPT);path=Path(receipt['root'])/'run'
    require(path==Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-stability-resume-2026-10-10/unit004/run'),'fixed immediate completed coverage')
    for root,pin in [(path,receipt['worker_manifest']),(path.parent,receipt['outer_manifest'])]:
        require(io.entry(root/'manifest.json')==pin,'exact completed adjacent manifest');io.verify_inventory(root,io.read(root/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');launch=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'exact adjacent settings')
    require(rt['source_sha256'].get(previous.CURRENT)==io.entry(ROOT/previous.CURRENT)['sha256'],'exact adjacent runner')
    require(rt['source_commit']==launch['source_commit'] and launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'complete adjacent execution')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'historical adjacent source')
    require(out['passed'] and out['completed_slices']==8 and out['stages']==1288 and len(out['rows'])==8 and all(r['passed'] for r in out['rows']),'complete adjacent attraction')
    require(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete adjacent calls')
    require(inp['selected_artifact']['index']==0 and out['complete_coverage']==inp['coverage_endpoints']==['35/128','9/32'],'exact chart147 coverage')
    require(base.coverage([r['exact_endpoints'] for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'adjacent partition')
    return inp,{'root':str(path),'worker_manifest':receipt['worker_manifest'],'outer_manifest':receipt['outer_manifest']}

def input_for(mode):
    if mode=='profile':return ORIGINAL_INPUT(mode)
    adjacent,identity=completed_adjacent();parent=io.io.parent
    value=parent.partial_input() # full native10-chart root chain and retained failures
    source=parent.io;initial,compact=source.initial_anchor()
    path=source.INITIAL/'certificate.json.gz';chart=source.read(path)
    require(source.compact_chart(chart)==compact,'complete selected predecessor chart')
    source.joined(compact,adjacent['chart'])
    require(adjacent['model_inputs']==initial['model_inputs']==value['model_inputs'],'same exact map')
    require(chart['exact_parameter_endpoints']==['17/64','35/128'],'exact chart146 full width')
    value.update(chart=chart,slices=[chart['exact_parameter_endpoints']],coverage_endpoints=chart['exact_parameter_endpoints'],coverage_predecessor=identity,
      selected_artifact={'root':str(source.INITIAL),'manifest':source.INITIAL_PIN,'source_commit':source.INITIAL_EXECUTION,'identity':io.entry(path),'global_chart':146})
    return value

def single_coverage(rows,ends):
    require(len(rows)==1 and rows[0]==ends,'one entire exact chart');return ends

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    saved=base.sources,base.input_for,base.coverage
    base.sources=sources;base.input_for=input_for
    # completed_adjacent needs the original8-slice validator before the stream.
    def load(mode):
        value=input_for(mode);base.coverage=single_coverage;return value
    base.input_for=load
    try:return base.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:base.sources,base.input_for,base.coverage=saved

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
