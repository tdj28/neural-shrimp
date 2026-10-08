"""Exact first S1R five-output pilot; target execution requires root signoff.

No configurable system, seed, tolerances or trajectory fallback. Output must
be a new attempt directory. A completed pilot may still contain explicit gaps.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.period2_branch import Budget,Options,PilotStop,Continuation,phase_match
from neural_shrimp.period2_branch_io import Store,file_entry,identity,digest,canonical
from neural_shrimp.gdmaps import GridMap
from neural_shrimp.trainability import SYSTEMS

PLAN='docs/s1r-branch-association-plan.json'
SOURCES=(PLAN,'src/neural_shrimp/period2_branch.py','src/neural_shrimp/period2_branch_io.py',
 'tools/explore_s1r_branch.py','tools/check_s1r_branch.py','src/neural_shrimp/gdmaps.py',
 'src/neural_shrimp/trainability.py','src/neural_shrimp/models.py')


def load_plan():
    plan=json.loads((ROOT/PLAN).read_text())
    if plan['system']!='S1R' or plan['seed']!=28002 or plan['row_index']!=2 or plan['lineage']!='criticality_saved_row_2':
        raise ValueError('outside reviewed development target')
    if plan['limits']!={'threads':1,'nice':10,'internal_seconds':120,'external_seconds':150,
       'kill_grace_seconds':5,'phase_corrector_calls':160,'raw_bytes':67108864,
       'newton_iterations_per_corrector':20,'line_search_halvings':12}:raise ValueError('outside reviewed budgets')
    if plan['output_offsets']!=[0,.001,.002,-.001,-.002] or plan['hidden_backoff']!=.002 or plan['hidden_max_step']!=.0005:
        raise ValueError('outside reviewed pilot geometry')
    if plan['max_link_halvings']!=4 or plan['min_output_step']!=6.25e-5:raise ValueError('outside reviewed subdivisions')
    expected={k:v for k,v in asdict(Options()).items() if k not in ('newton_iterations_per_corrector','line_search_halvings')}
    if plan['tolerances']!=expected:raise ValueError('outside reviewed gates')
    return plan


def bound_inputs(plan):
    for item in plan['inputs'].values():
        path=Path(item['path'])
        if file_entry(path)!={k:item[k] for k in ('bytes','sha256')}:raise ValueError('planned input mismatch')
    saved=json.loads(Path(plan['inputs']['saved_anchor']['path']).read_text())
    row=saved['rows'][plan['row_index']]
    if row['log10_output']!=plan['log10_output_anchor'] or row['log10_hidden_last_attracting']!=plan['log10_hidden_anchor']:
        raise ValueError('anchor parameter mismatch')
    seed=np.asarray(row['orbit_point'],dtype=np.float64)
    if seed.shape!=(72,) or not np.isfinite(seed).all():raise ValueError('invalid full anchor')
    return seed


def sources():return {name:file_entry(ROOT/name) for name in SOURCES}


def run(output):
    plan=load_plan();seed=bound_inputs(plan)
    # These read-only checks precede output creation and every model evaluation.
    subprocess.run(['git','ls-files','--error-unmatch',*SOURCES],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
    subprocess.run(['git','diff','--quiet','HEAD','--',*SOURCES],cwd=ROOT,check=True)
    if os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('launch with nice -n 10 or lower priority')
    store=Store(output,plan['limits']['raw_bytes'])
    started=time.monotonic();budget=Budget(120,160)
    runtime=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      sources=sources(),plan_sha256=digest((ROOT/PLAN).read_bytes()),inputs=plan['inputs'],
      started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0),
      python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cpu_threads=1,
      thread_env={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
      load_average=list(os.getloadavg()),memory_available_kib=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')),
      gpu=False,paid_compute=False,limits=plan['limits'])
    store.json('runtime-start.json',runtime)
    result=dict(schema='s1r-branch-result-v1',status='running',lineage=plan['lineage'],
      historical_family_association='unresolved',historical_families=plan['historical_families'],
      requested_outputs=plan['output_offsets'],anchor=None,hidden_links=[],segments=[],
      evidence='exploratory numerical local continuation; no uniqueness or ancestry proof')
    old_handler=signal.getsignal(signal.SIGTERM)
    def stop(signum,frame):raise PilotStop('external_signal_'+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        net,x,y,theta0=SYSTEMS['S1R'](28002)
        budget.check()
        store.emit('inputs',dict(model=asdict(net),x=x,y=y,theta0=theta0,seed=seed,
             identity={k:identity(v) for k,v in dict(x=x,y=y,theta0=theta0,seed=seed).items()}))
        def family(params):
            return GridMap(net,x,y,[10.**float(params[0])],[10.**float(params[1])]).cell_map(0)
        engine=Continuation(family,Options(),budget,store.emit)
        rates=np.array([plan['log10_hidden_anchor'],plan['log10_output_anchor']])
        store.emit('rate_identity',dict(log10=rates,rates=np.array([10.**float(v) for v in rates])))
        anchor_started=time.monotonic()
        anchor=engine.node(rates,[seed,None],'anchor');result['anchor']=anchor['id']
        anchor_seconds=time.monotonic()-anchor_started
        if not anchor['accepted']:
            result.update(status='anchor_failed',reason=anchor['reason'])
            return result
        timing=dict(anchor_pair_seconds=anchor_seconds,remaining_seconds=budget.deadline-time.monotonic(),
                    nominal_remaining_pairs=20,remaining_time_estimate=20*anchor_seconds,
                    limitation='linear timing screen only; corrector difficulty and retries may differ')
        store.emit('anchor_feasibility',timing);result['anchor_feasibility']=timing
        if timing['remaining_time_estimate']>timing['remaining_seconds']:
            result.update(status='resource_stop',reason='anchor_timing_infeasible');return result
        current=anchor
        for i in range(1,5):
            params=np.array([rates[0]-.002*i/4,rates[1]])
            child=engine.node(params,current['pair'],'hidden_backoff')
            match=engine.align(current,child)
            link=dict(parent=current['id'],parent_phase=current.get('carried_phase',0),child=child['id'],
                      match=match,status='accepted' if match['passed'] else 'unresolved_gap')
            store.emit('hidden_link',link);result['hidden_links'].append(link)
            if not match['passed']:
                result.update(status='hidden_backoff_gap',reason=child['reason']);return result
            current=child.copy();current['carried_phase']=match['phase']
            if match['phase']:current['pair']=current['pair'][::-1]
        central=current
        for direction in (1,-1):
            current=central;previous=None
            for offset in (.001,.002):
                target=np.array([central['params'][0],rates[1]+direction*offset])
                segment,current,previous=engine.segment(current,target,plan['min_output_step'],plan['max_link_halvings'],previous)
                segment['direction']=direction;result['segments'].append(segment)
                store.emit('segment',segment)
                if segment['status']!='connected':break
        result['status']='completed_with_gaps' if any(s['status']!='connected' for s in result['segments']) else 'completed_local_segment'
        return result
    except PilotStop as error:
        result.update(status='resource_stop',reason=str(error));return result
    except BaseException as error:
        result.update(status='failed_or_interrupted',reason=type(error).__name__+': '+str(error));raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)
        result.update(phase_corrector_calls=budget.calls,elapsed_seconds=time.monotonic()-started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        store.finish(result)
        for path in store.root.iterdir():path.chmod(0o444)
        store.root.chmod(0o555)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.out)
    print(canonical(result).decode())
    return 0 if result['status']=='completed_local_segment' else 2


if __name__=='__main__':raise SystemExit(main())
