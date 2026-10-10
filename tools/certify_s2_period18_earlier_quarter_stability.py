"""Fixed four-quarter chart141 attempt after retained whole-chart wrapping."""
import argparse,subprocess,sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import certify_s2_period18_earlier_whole_stability as base
ROOT=base.ROOT;io=base.io;require=base.require;CAP=base.CAP;THREADS=base.THREADS
CURRENT='tools/certify_s2_period18_earlier_quarter_stability.py'
CARD='docs/plans/s2-period18-earlier-quarter-stability.md'
RECEIPT='docs/reviews/s2-period18-earlier-whole-stability-outcome.json'
RAW=Path('/data2/neural-shrimp-data/explore/certificates/s2-period18-earlier-quarter-stability-2026-10-10')
SOURCES=list(dict.fromkeys(base.SOURCES+[CURRENT,'tools/launch_s2_period18_earlier_quarter_stability.py','tests/test_s2_period18_earlier_quarter_stability.py',CARD,RECEIPT]))
ORIGINAL_INPUT=base.input_for;ORIGINAL_RAW=base.RAW

def sources():
 require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'clean source');bound={}
 for name in SOURCES:
  b=(ROOT/name).read_bytes();require(b==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),'committed source');bound[name]=io.sha256(b)
 return bound

def partition(ends):
 require(ends==base.ENDS,'literal native chart141');lo,hi=map(Fraction,ends);w=(hi-lo)/4
 return [[str(lo+i*w),str(lo+(i+1)*w)]for i in range(3,-1,-1)]

def coverage(rows,ends):
 require(rows==partition(ends),'exact four closed descending quarters');return ends

def input_for(mode):
 value=ORIGINAL_INPUT(mode)
 if mode=='target':
  require(value['coverage_endpoints']==base.ENDS and value['selected_artifact']['global_chart']==141,'same original141 after full142 admission')
  rec=io.read(ROOT/RECEIPT);path=Path(rec['root']);require(path==ORIGINAL_RAW/'attempt001-target','literal retained whole141 failure')
  require(io.entry(path/'manifest.json')==rec['outer_manifest'],'immutable whole141 receipt');io.verify_inventory(path,io.read(path/'manifest.json')['files'])
  old=io.read(path/'run/inputs.json.gz');done=io.read(path/'run/outcome.json')
  require(old['chart']==value['chart'] and old['model_inputs']==value['model_inputs'] and not done['passed'] and done['completed_slices']==1 and done['stages']==161,'unchanged completed insufficient whole-chart domain')
  value['slices']=partition(value['coverage_endpoints']);value['retained_whole_failure']={'root':str(path),'outer_manifest':rec['outer_manifest']}
 return value

def execute(out,mode,profile=None,profile_pin=None,profile_outer_pin=None):
 require(out==RAW/f'attempt001-{mode}/run','one exclusive quarter profile/target')
 old=base.sources,base.input_for,base.single_coverage,base.RAW
 base.sources=sources;base.input_for=input_for;base.single_coverage=coverage;base.RAW=RAW
 try:return base.execute(out,mode,profile,profile_pin,profile_outer_pin)
 finally:base.sources,base.input_for,base.single_coverage,base.RAW=old

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');a=p.parse_args()
 require((a.mode=='target')==(a.profile is not None and a.profile_manifest is not None and a.profile_outer_manifest is not None),'profile pins only target')
 raise SystemExit(execute(a.out,a.mode,a.profile,a.profile_manifest,a.profile_outer_manifest))
