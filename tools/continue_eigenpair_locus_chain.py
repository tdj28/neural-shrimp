#!/usr/bin/env python3
"""Endpoint/source-pinned adapter around the unchanged bounded march worker."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,gc,gzip,hashlib,json,re,subprocess,sys
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import continue_eigenpair_locus as M
from neural_shrimp.rigorous_eigenpair_march import compact
from neural_shrimp.rigorous_eigenpair_locus import require
BASE=Path('/data2/neural-shrimp-data/explore/eigenpair-locus-prototype')
EXTRA=['tools/continue_eigenpair_locus_chain.py','tools/launch_eigenpair_chain.py','tests/test_eigenpair_chain.py','docs/plans/eigenpair-quadratic-chain.md','docs/reviews/eigenpair-march-unit001.json']
ORIGINAL_SOURCES=list(M.SOURCES)


def input_path(parent,name):
    p=Path(name);require(not p.is_absolute() and '..' not in p.parts,'relative input descriptor')
    return parent/'run'/p


def validate_completion(outcome,result,accepted,chart,expected_right):
    require(outcome['passed'] is True and outcome['status']=='four_joined_charts' and result['completed'] is True,'completed parent unit')
    require(len(outcome['progress']['accepted'])==len(result['accepted'])==4 and 4<=result['calls']<=5 and result['calls']==outcome['calls'],'fixed parent cardinality')
    last=outcome['progress']['accepted'][-1]
    require(outcome['progress']['last_accepted']==last and outcome['progress']['active_call'] is None,'durable last accepted parent')
    require(accepted['call']==last['call'] and accepted['index']==last['index']==3,'accepted descriptor join')
    full=compact(chart)
    require(full==accepted['chart']==result['last_chart'],'full chart/accepted/summary identity')
    require(full['period']==18 and full['sigma']==1 and full['dimension']==13 and full['shooting_dimension']==470,'native convention')
    require(full['normal']==['0']*469+['1'] and full['settings']['halfwidth_exponent']==-15 and full['settings']['precision']==256 and full['settings']['newton_steps']==2,'same gauge/width/precision')
    require(full['parameter_endpoints'][1]==last['right']==expected_right,'prospective exact endpoint')
    require(Fraction(expected_right)>Fraction(3,32768),'advance beyond original pair')
    return full


def install(args):
    original=M.bind_parent
    M.SOURCES=list(dict.fromkeys(ORIGINAL_SOURCES+EXTRA))
    if args.mode=='profile':return
    require(type(args.unit)is int and 2<=args.unit<=64,'bounded explicit unit index')
    expected=BASE/f'attempt-{args.unit-1:03d}-march-target'
    require(args.parent==expected and not args.parent.is_symlink(),'exact prior unit directory')
    require(re.fullmatch('[0-9a-f]{40}',args.parent_source or '') is not None,'explicit parent execution commit')
    require(all(re.fullmatch('[0-9a-f]{64}',v or '') for v in (args.parent_manifest,args.parent_run_manifest)),'parent manifest pins')
    require(args.expected_right==str(Fraction(8*(args.unit-1)+3,32768)),'prospective fixed-grid endpoint')
    def bound(profile_root,profile_pin,profile_outer_pin):
        inp,original_parent,bindings=original(profile_root,profile_pin,profile_outer_pin)
        parent=args.parent
        bindings['march_parent']={'path':str(parent),'source_commit':args.parent_source,'outer':M.Q.B.inventory(parent,args.parent_manifest),'run':M.Q.B.inventory(parent/'run',args.parent_run_manifest),'expected_right':args.expected_right}
        rt=M.read(parent/'run/runtime.json');la=M.read(parent/'launcher-result.json');out=M.read(parent/'run/outcome.json')
        require(rt['commit']==la['commit']==args.parent_source and rt['mode']==la['mode']=='target' and rt['precision']==256 and rt['nice']>=10,'parent runtime/source/mode')
        require(la['exit_code']==0 and not la['owned_live_pids'] and la['stop_reason'] is None and M.finite(la['elapsed_seconds'],322),'parent successful owned exit')
        require(M.finite(out['elapsed_seconds'],300) and M.finite(out['peak_rss_kib'],2*1024**2) and out['ticks']<=650,'parent finite resource caps')
        for key in ('python','numpy','flint','flint_library','threads'):
            pr=M.read(profile_root/'runtime.json');require(rt[key]==pr[key],'parent/profile environment '+key)
        require(set(ORIGINAL_SOURCES)<=set(rt['sources'])<=set(M.SOURCES),'transitive parent source inventory')
        for name,v in rt['sources'].items():
            require(M.entry(ROOT/name)==M.entry(parent/'run/source'/name)==v,'same parent source '+name)
            require(hashlib.sha256(subprocess.check_output(['git','show',args.parent_source+':'+name],cwd=ROOT)).hexdigest()==v['sha256'],'committed parent source '+name)
        result=M.read(parent/'run/march.json');last=out['progress']['accepted'][-1]
        with gzip.open(input_path(parent,last['accepted_file']),'rt') as f:accepted=json.load(f)
        with gzip.open(input_path(parent,last['full_chart_file']),'rt') as f:chart=json.load(f)
        previous=validate_completion(out,result,accepted,chart,args.expected_right)
        require(previous['origin']==original_parent['origin'],'same original gauge origin')
        pi=M.read(parent/'run/inputs.json');require(pi['map_inputs']==inp,'same exact map/data')
        require(pi['parent']['parameter_endpoints'][1]==str(Fraction(args.expected_right)-Fraction(8,32768)),'fixed four-chart advance from recorded prior endpoint')
        bindings['march_parent'].update(last_full_chart=last['full_chart_file'],last_full_chart_identity=M.entry(input_path(parent,last['full_chart_file'])),last_accepted_identity=M.entry(input_path(parent,last['accepted_file'])))
        del result,chart,accepted,original_parent;gc.collect()
        return inp,previous,bindings
    M.bind_parent=bound


def parser():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['profile','target'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',type=Path);p.add_argument('--profile-manifest');p.add_argument('--profile-outer-manifest');p.add_argument('--unit',type=int);p.add_argument('--parent',type=Path);p.add_argument('--parent-source');p.add_argument('--parent-manifest');p.add_argument('--parent-run-manifest');p.add_argument('--expected-right');return p
if __name__=='__main__':
    args=parser().parse_args();install(args);raise SystemExit(M.main(args))
