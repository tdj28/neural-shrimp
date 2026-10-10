"""Seal a completed worker whose launcher completion receipt was not persisted.

Never synthesizes a launcher exit, peak group RSS, or final elapsed time.
"""
import argparse, json, math, os, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_stability_march as old
io=old.io;require=old.require;ROOT=old.ROOT
ORIGINAL=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-stability-march-2026-10-10/unit004')
EXECUTION='ee304621ea74e8a340c0134becab3bb0a50e8f07'
RECOVERY=Path('/data2/neural-shrimp-data/audits/s2-stability-launcher-recovery/attempt-001')

def worker_checks(rt,out,inp,launch):
    require(rt['source_commit']==launch['source_commit']==EXECUTION,'exact interrupted-launcher execution')
    require(rt['mode']=='target' and rt['precision']==256 and rt['nice']>=10 and rt['threads']==dict.fromkeys(old.THREADS,'1'),'exact worker settings')
    require(launch['mode']=='target' and launch['source_sha256']==rt['source_sha256'],'launcher/source identity')
    require(out['status']=='fixed_chart_slices_completed' and out['passed'] and out['exit_code']==0 and out['completed_slices']==8 and out['stages']==1288,'complete worker outcome')
    require(len(out['rows'])==8 and all(r['passed'] for r in out['rows']),'all slices pass')
    require(out['counts']=={'map_attempted':1296,'map_returned':1296,'unit_attempted':2592,'unit_returned':2592,'mixed_attempted':288,'mixed_returned':288},'complete derivative counts')
    for key,cap in [('elapsed_seconds',360),('max_rss_kib',2*1024**2)]:
        v=out[key];require(type(v) in (int,float) and math.isfinite(v) and 0<v<cap,'worker bounded '+key)
    require(inp['selected_artifact']['index']==4 and out['complete_coverage']==inp['coverage_endpoints']==inp['chart']['exact_parameter_endpoints']==['39/128','5/16'],'exact recovered chart')
    require(old.base.coverage([r['exact_endpoints'] for r in out['rows']],inp['coverage_endpoints'])==out['complete_coverage'],'complete partition')

def owned_absent(launch,rt):
    exact={launch['launcher_pid'],launch['owned_group_pid'],rt['pid'],rt['ppid']};live=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split()
            if stat[0]!='Z' and (int(p.name) in exact or int(stat[2])==launch['owned_group_pid']):live.append(int(p.name))
        except (OSError,ValueError):continue
    require(not live,'original PIDs and process group absent');return sorted(exact)

def verify_original():
    p=ORIGINAL/'run';require(not (ORIGINAL/'launcher-result.json').exists() and not (ORIGINAL/'manifest.json').exists(),'missing original outer completion remains explicit')
    io.verify_inventory(p,io.read(p/'manifest.json')['files'])
    rt=io.read(p/'runtime-start.json');out=io.read(p/'outcome.json');inp=io.read(p/'inputs.json.gz');launch=io.read(ORIGINAL/'launcher-start.json')
    worker_checks(rt,out,inp,launch)
    for name,digest in rt['source_sha256'].items():require(io.sha256(subprocess.check_output(['git','show',EXECUTION+':'+name],cwd=ROOT))==digest,'original historical source')
    require(io.entry(ORIGINAL/'launcher-source.py')['sha256']==launch['launcher_source_sha256']==rt['source_sha256']['tools/launch_s2_period18_stability_march.py'],'original launcher source')
    require(io.total(ORIGINAL)<514*1024**2,'original aggregate cap')
    pids=owned_absent(launch,rt)
    return rt,out,inp,launch,pids

def recover():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean committed recovery source')
    rt,out,inp,launch,pids=verify_original();RECOVERY.mkdir(parents=True,exist_ok=False)
    inventory=io.inventory(ORIGINAL)
    # Change permissions only; preserve every original byte and missing receipt.
    for p in sorted(ORIGINAL.rglob('*'),reverse=True):p.chmod(0o444 if p.is_file() else 0o555)
    ORIGINAL.chmod(0o555);io.verify_inventory(ORIGINAL,inventory)
    receipt={'status':'completed_worker_recovered_without_launcher_receipt','original_root':str(ORIGINAL),'worker_manifest':io.entry(ORIGINAL/'run/manifest.json'),'original_inventory':inventory,'execution':EXECUTION,'worker_outcome':out,'original_pids_observed_absent':pids,'observed_utc':io.now(),'recovery_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'recovery_source':io.entry(Path(__file__)),'launcher_exit_code':None,'launcher_peak_group_rss_kib':None,'launcher_final_elapsed_seconds':None,'limitation':'Original launcher final receipt is unavailable. Worker complete outcome, full immutable inventory, self RSS/time and current absence of owned processes are established; historical group peak and parent exit are not reconstructed.'}
    io.save(RECOVERY/'recovery.json',receipt);io.save(RECOVERY/'manifest.json',{'files':io.inventory(RECOVERY)})
    for p in RECOVERY.iterdir():p.chmod(0o444)
    RECOVERY.chmod(0o555)
    print(json.dumps({'worker':receipt['worker_manifest'],'recovery_manifest':io.entry(RECOVERY/'manifest.json'),'coverage':out['complete_coverage']}))

if __name__=='__main__':recover()
