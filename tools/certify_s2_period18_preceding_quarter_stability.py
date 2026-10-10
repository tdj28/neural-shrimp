"""Fixed chart145 predecessor after complete four-piece chart146 attraction."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_quarter_chart_stability as base
import certify_s2_period18_fold_saved_rebox as saved
import certify_s2_period18_fold_small_march as native
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_preceding_quarter_stability.py'
RECEIPT='docs/reviews/s2-period18-quarter-chart-stability-outcome.json'
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_preceding_quarter_stability.py','tests/test_s2_period18_preceding_quarter_stability.py','docs/plans/s2-period18-preceding-quarter-stability.md',RECEIPT]))
ORIGINAL_INPUT=base.input_for

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(data)
    return result

def completed_adjacent():
    receipt=io.read(ROOT/RECEIPT);path=Path(receipt['root'])/'run'
    require(path==Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-quarter-chart-stability-2026-10-10/attempt001-target/run'),'fixed completed chart146')
    for root,pin in [(path,receipt['worker_manifest']),(path.parent,receipt['outer_manifest'])]:
        require(io.entry(root/'manifest.json')==pin,'exact adjacent manifest');io.verify_inventory(root,io.read(root/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');launch=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1'),'exact adjacent settings')
    require(rt['source_sha256'].get(base.CURRENT)==io.entry(ROOT/base.CURRENT)['sha256'],'exact adjacent four-piece runner')
    require(rt['source_commit']==launch['source_commit'] and launch['exit_code']==out['exit_code']==0 and launch['owned_live_pids']==[],'complete adjacent execution')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'historical adjacent source')
    require(out['passed'] and out['completed_slices']==4 and out['stages']==644 and len(out['rows'])==4 and all(r['passed']for r in out['rows']),'complete adjacent attraction')
    require(out['counts']=={'map_attempted':648,'map_returned':648,'unit_attempted':1296,'unit_returned':1296,'mixed_attempted':144,'mixed_returned':144},'complete adjacent calls')
    require(inp['selected_artifact']['global_chart']==146 and out['complete_coverage']==inp['coverage_endpoints']==['17/64','35/128'],'exact chart146 coverage')
    require(base.coverage([r['exact_endpoints']for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'exact adjacent quarter partition')
    return inp,{'root':str(path),'worker_manifest':receipt['worker_manifest'],'outer_manifest':receipt['outer_manifest']}

def select(value,adjacent,identity,prepared,pair):
    require(value['chart']==adjacent['chart'] and value['model_inputs']==adjacent['model_inputs']==prepared['model_inputs'],'same exact adjacent map/chart')
    failed=saved.target_pair(pair,saved.read(saved.PARENT/'inputs.json'))
    require(failed==prepared['chart'],'same retained failed second chart')
    first=pair['charts'][0]
    require(first['exact_parameter_endpoints']==['33/128','17/64'],'fixed predecessor chart145')
    native.joined(native.compact_chart(first),adjacent['chart'])
    require(first['normal']==adjacent['chart']['normal'] and first['origin']==adjacent['chart']['origin'],'unchanged native gauge')
    value.update(chart=first,slices=base.partition(first['exact_parameter_endpoints']),coverage_endpoints=first['exact_parameter_endpoints'],coverage_predecessor=identity,
      selected_artifact={'root':str(saved.PARENT),'manifest':saved.PIN,'source_commit':saved.EXECUTION,'identity':io.entry(saved.PARENT/'certificate.json.gz'),'global_chart':145,'certificate_chart_index':0})
    return value

def input_for(mode):
    value=ORIGINAL_INPUT(mode)
    if mode=='profile':return value
    adjacent,identity=completed_adjacent()
    prepared=saved.prepare() # authenticates archived source, first proved chart and immutable failed second
    return select(value,adjacent,identity,prepared,saved.read(saved.PARENT/'certificate.json.gz'))

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    old=base.sources,base.input_for
    base.sources=sources;base.input_for=input_for
    try:return base.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:base.sources,base.input_for=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
