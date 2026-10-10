"""Bind new unchanged-method coverage to transparently recovered unit004."""
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_stability_march as old
import recover_s2_stability_completion as recovery
ROOT=old.ROOT;io=old.io;require=old.require;CAP=old.CAP;THREADS=old.THREADS
CURRENT='tools/resume_s2_period18_stability_march.py'
SOURCES=old.SOURCES+[CURRENT,'tools/launch_s2_period18_stability_resume.py','tools/recover_s2_stability_completion.py','tests/test_s2_stability_recovery.py','docs/plans/s2-period18-stability-recovery.md']
ORIGINAL_PREVIOUS=old.previous_coverage

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');bound={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes();require(data==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(data)
    return bound

def previous_coverage(path,pin,outer_pin,index):
    require(type(index)is int and 0<=index<=3,'fixed remaining index')
    if index<3:
        rt=io.read(path/'runtime-start.json')
        require(rt['source_sha256'].get(CURRENT)==io.entry(ROOT/CURRENT)['sha256'],'exact resumed predecessor runner')
        return ORIGINAL_PREVIOUS(path,pin,outer_pin,index)
    require(path==recovery.ORIGINAL/'run','only exact recovered predecessor')
    manifest=recovery.RECOVERY/'manifest.json'
    require(io.entry(manifest)['sha256']==outer_pin,'separate recovery manifest pin')
    io.verify_inventory(recovery.RECOVERY,io.read(manifest)['files'])
    saved=io.read(recovery.RECOVERY/'recovery.json')
    require(saved['status']=='completed_worker_recovered_without_launcher_receipt' and saved['original_root']==str(recovery.ORIGINAL) and saved['execution']==recovery.EXECUTION,'explicit recovery identity')
    require(saved['launcher_exit_code'] is None and saved['launcher_peak_group_rss_kib'] is None,'no invented launcher completion')
    require(saved['recovery_source']['sha256']==io.entry(ROOT/'tools/recover_s2_stability_completion.py')['sha256'],'unchanged reviewed recovery validator')
    io.verify_inventory(recovery.ORIGINAL,saved['original_inventory'])
    rt,out,inp,launch,pids=recovery.verify_original()
    require(io.entry(path/'manifest.json')['sha256']==pin==saved['worker_manifest']['sha256'],'recovered complete worker pin')
    require(out==saved['worker_outcome'],'unchanged recovered worker')
    return inp,{'root':str(path),'worker_manifest':pin,'recovery_manifest':outer_pin,'source_commit':rt['source_commit'],'index':4,'launcher_receipt_missing':True}

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None,index=None,previous=None,previous_pin=None,previous_outer_pin=None):
    if mode=='target':require(type(index)is int and 0<=index<=3,'remaining index after recovery')
    saved=old.sources,old.previous_coverage
    old.sources=sources;old.previous_coverage=previous_coverage
    try:return old.execute(out,mode,profile,profile_pin,profile_outer_pin,index,previous,previous_pin,previous_outer_pin)
    finally:old.sources,old.previous_coverage=saved

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--index',type=int);p.add_argument('--previous',type=Path);p.add_argument('--previous-manifest');p.add_argument('--previous-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest,a.index,a.previous,a.previous_manifest,a.previous_outer_manifest))
