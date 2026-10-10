"""One measured whole-chart141 attempt; unchanged correlated proof kernel."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_two_predecessor_stability as previous
import certify_s2_period18_preceding_chart_stability as core
ROOT=previous.ROOT;io=previous.io;require=previous.require;CAP=previous.CAP;THREADS=previous.THREADS
CURRENT='tools/certify_s2_period18_earlier_whole_stability.py'
CARD='docs/plans/s2-period18-earlier-whole-stability.md'
RECEIPT='docs/reviews/s2-period18-two-predecessor-native-unit002.json'
INDEPENDENT='docs/reviews/s2-attraction-remaining-batch003.json'
RAW=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-earlier-whole-stability-2026-10-10')
ENDS=['25/128','27/128']
SOURCES=list(dict.fromkeys(previous.SOURCES+[CURRENT,'tools/launch_s2_period18_earlier_whole_stability.py','tests/test_s2_period18_earlier_whole_stability.py',CARD,RECEIPT,INDEPENDENT]))

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');bound={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(b)
    return bound

def completed_adjacent():
    rec=io.read(ROOT/RECEIPT);path=Path(rec['root'])/'run';require(path==previous.RAW/'unit002/run','fixed immediately adjacent chart142')
    for root,pin in((path,rec['worker_manifest']),(path.parent,rec['outer_manifest'])):
        require(io.entry(root/'manifest.json')==pin,'exact adjacent complete manifest');io.verify_inventory(root,io.read(root/'manifest.json')['files'])
    rt=io.read(path/'runtime-start.json');out=io.read(path/'outcome.json');la=io.read(path.parent/'launcher-result.json');inp=io.read(path/'inputs.json.gz')
    require(rt['source_commit']==la['source_commit']==rec['source_commit']and la['exit_code']==out['exit_code']==0 and la['owned_live_pids']==[],'completed adjacent owned execution')
    require(rt['mode']=='target'and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(THREADS,'1')and rt['source_sha256'][previous.CURRENT]==io.entry(ROOT/previous.CURRENT)['sha256'],'unchanged exact adjacent source/settings')
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',rt['source_commit']+':'+name],cwd=ROOT))==digest,'exact historical adjacent source')
    require(out['passed']and out['completed_slices']==8 and out['stages']==1288 and len(out['rows'])==8 and all(r['passed']for r in out['rows']),'entire adjacent producer cover')
    require(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'all adjacent acquisition counts')
    require(inp['selected_artifact']['index']==12 and inp['selected_artifact']['global_chart']==142 and core.coverage([r['exact_endpoints']for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage']==['27/128','29/128'],'exact adjacent full partition')
    independent=io.read(ROOT/INDEPENDENT);audit=Path(independent['root']);require(io.entry(audit/'manifest.json')==independent['outer_manifest'],'fixed independent batch receipt');io.verify_inventory(audit,io.read(audit/'manifest.json')['files'])
    proof=io.read(audit/'chart142/run/outcome.json');binding=io.read(audit/'chart142/run/input-binding.json')
    require(proof['passed']and proof['chart']==142 and proof['completed_slices']==8 and proof['coverage']==['27/128','29/128'],'independently proved adjacent attraction')
    require(binding['root']==rec['root']and binding['outer_manifest']==rec['outer_manifest']and binding['worker_manifest']==rec['worker_manifest'],'independent proof binds actual adjacent producer')
    return inp,{'root':str(path),'worker_manifest':rec['worker_manifest'],'outer_manifest':rec['outer_manifest'],'independent_batch_manifest':independent['outer_manifest']}

def select(adjacent,identity,known,chart,nextchart):
    require(chart['exact_parameter_endpoints']==ENDS and adjacent['chart']==nextchart and nextchart['exact_parameter_endpoints']==['27/128','29/128'],'fixed exact native141→142')
    require(adjacent['model_inputs']==known['model_inputs'],'same exact actual map')
    previous.native.joined(previous.native.compact_chart(chart),nextchart)
    value=dict(adjacent);value.update(chart=chart,slices=[ENDS],coverage_endpoints=ENDS,coverage_predecessor=identity,selected_artifact={'root':str(previous.endpoint.PARENT),'manifest':previous.endpoint.PIN,'source_commit':previous.endpoint.EXECUTION,'identity':io.entry(previous.endpoint.PARENT/'chart-011.json.gz'),'global_chart':141,'index':11});value.pop('localized_join',None)
    return value

def input_for(mode):
    if mode=='profile':return previous.base.input_for(mode)
    adjacent,identity=completed_adjacent();known=previous.endpoint.partial_input()
    return select(adjacent,identity,known,io.read(previous.endpoint.PARENT/'chart-011.json.gz'),io.read(previous.endpoint.PARENT/'chart-012.json.gz'))

def single_coverage(rows,ends):require(ends==ENDS and rows==[ENDS],'one exact entire chart141');return ends

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    require(out==RAW/f'attempt001-{mode}/run','one fixed exclusive profile/target path')
    saved=core.sources,core.input_for,core.coverage;core.sources=sources
    def load(which):
        value=input_for(which);core.coverage=single_coverage;return value
    core.input_for=load
    try:return core.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:core.sources,core.input_for,core.coverage=saved
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only for target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
