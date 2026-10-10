"""Fixed four-piece attempt after the retained full-chart146 bound failed."""
import argparse,subprocess,sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_whole_chart_stability as base
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_quarter_chart_stability.py'
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_quarter_chart_stability.py','tests/test_s2_period18_quarter_chart_stability.py','docs/plans/s2-period18-quarter-chart-stability.md','docs/reviews/s2-period18-whole-chart-stability-outcome.json']))
ORIGINAL_INPUT=base.input_for

def sources():
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');result={}
    for name in SOURCES:
        b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');result[name]=io.sha256(b)
    return result

def partition(ends):
    lo,hi=map(Fraction,ends);require(lo<hi,'nonzero interval');w=(hi-lo)/4
    return [[str(lo+i*w),str(lo+(i+1)*w)]for i in range(3,-1,-1)]

def coverage(rows,ends):
    require(rows==partition(ends),'exact four closed descending intervals');return ends

def input_for(mode):
    value=ORIGINAL_INPUT(mode)
    if mode=='target':
        require(value['coverage_endpoints']==['17/64','35/128'] and value['selected_artifact']['global_chart']==146,'same exact full parent')
        value['slices']=partition(value['coverage_endpoints'])
    return value

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
    old=base.sources,base.input_for,base.single_coverage
    base.sources=sources;base.input_for=input_for;base.single_coverage=coverage
    try:return base.execute(out,mode,profile,profile_pin,profile_outer_pin)
    finally:base.sources,base.input_for,base.single_coverage=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True)
    p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
    require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
    raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
