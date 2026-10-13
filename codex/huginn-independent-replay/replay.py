"""One bounded independent saved-raw Huginn replay; no producer/model execution.

External source/card/private-push admission and a process-owning hard wall
watchdog remain the launcher's obligations. This driver cannot start from a
prepared card, follows no retry loop, and preserves a failed attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time
import warnings

OWNED = Path('/data2/neural-shrimp-data/codex/huginn-independent-replay')
GIB = 1024**3
SOURCES = {'admission.py', 'statistics.py', 'comparison.py', 'replay.py'}
ENVELOPE = {'cpu_threads': 1, 'supervisor_slots': 1, 'nice': 10, 'address_space_gib': 4,
            'timeout_seconds': 300, 'startup_available_gib': 20, 'runtime_available_gib': 16,
            'output_bytes': 64*1024**2}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def validate_card(card, directory):
    if (card.get('schema') != 'codex-huginn-independent-replay-v1'
            or card.get('status') != 'accepted_by_root' or not card.get('acceptance_reference')
            or card.get('part') not in ('A', 'B') or card.get('attempt') != 'part-'+card['part']+'-001'):
        raise ValueError('only the externally admitted single complete part can run')
    env = card.get('envelope')
    if not isinstance(env, dict) or set(env) != set(ENVELOPE) or any(type(env[k]) is not int or env[k] != v for k,v in ENVELOPE.items()):
        raise ValueError('fixed measured envelope required')
    if set(card.get('sources', {})) != SOURCES:
        raise ValueError('complete source closure required')
    for name, expected in card['sources'].items():
        if sha(directory/name) != expected:
            raise ValueError('source drift: '+name)
    if not isinstance(card.get('source_commit'), str) or len(card['source_commit']) != 40 or any(c not in '0123456789abcdef' for c in card['source_commit']):
        raise ValueError('exact source commit required')
    if type(card.get('max_input_bytes')) is not int or card['max_input_bytes'] <= 0:
        raise ValueError('actual measured input byte bound required')
    if set(card.get('inputs', {})) != {'path','sha256','bytes'}:
        raise ValueError('external complete-input manifest pin required')
    if not card.get('local_receipt_review_reference'):
        raise ValueError('source/receipt/closure review reference required')


def validate_producer_args(record, part):
    expected = ({'command':'items','start':0,'count':64,'split':'test','seed':0,
                 'lams':[1.0,1.05,1.08,1.10],'fixed_tol':1e-4,'boot':2000} if part=='A' else
                {'command':'generation','start':0,'count':200,'split':'test','steps':32,
                 'prompt_style':'cot8','seed':0,'arms':['1.0','1.05','1.15','1.20'],
                 'base':'1.0','collapse_arm':'1.20','break_arm':'1.15','safe_arm':'1.05',
                 'margin':0.04,'max_new':256})
    args = record.get('args')
    if not isinstance(args,dict):raise ValueError('producer frozen invocation absent')
    for key,value in expected.items():
        if key not in args or json.dumps(args[key],sort_keys=True,allow_nan=False) != json.dumps(value,sort_keys=True,allow_nan=False):
            raise ValueError('producer invocation differs: '+key)
    return expected


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _write(path, value):
    with path.open('x') as handle:
        # Explicit NaN is retained for frozen undefined statistical outcomes.
        json.dump(value,handle,indent=2);handle.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic();directory=Path(__file__).resolve().parent
    card_bytes=args.card.read_bytes();card=json.loads(card_bytes);validate_card(card,directory)
    if os.environ.get('TASK8_CARD_SHA256') != hashlib.sha256(card_bytes).hexdigest() or os.environ.get('TASK8_SOURCE_COMMIT') != card['source_commit']:
        raise ValueError('external source/card admission missing')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':raise ValueError('one-thread environment required before numerical imports')
    if os.getpriority(os.PRIO_PROCESS,0) != 10 or resource.getrlimit(resource.RLIMIT_AS) != (4*GIB,4*GIB):
        raise ValueError('exact nice10/hard4GiB AS required before imports')
    if available()<20*GIB:raise ValueError('startup headroom below20GiB')
    out=OWNED/card['attempt'];out.mkdir(parents=True,exist_ok=False)
    (out/'card.json').write_bytes(card_bytes)
    for name in card['sources']:(out/name).write_bytes((directory/name).read_bytes())
    minimum=[available()];events=[];captured=[]
    status={'status':'running','computational_completed':False,'source_commit':card['source_commit'],
            'card_sha256':hashlib.sha256(card_bytes).hexdigest(),'part':card['part'],
            'producer_or_model_executed':False,'certificate_passed':False}
    def event(stage,**fields):
        row={'stage':stage,'seconds':time.monotonic()-start,**fields};events.append(row)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    def guard(signum=None,frame=None):
        minimum[0]=min(minimum[0],available())
        if minimum[0]<16*GIB:raise MemoryError('runtime headroom below16GiB')
        if time.monotonic()-start>=300:raise TimeoutError('300s replay deadline')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>ENVELOPE['output_bytes']-2*1024**2:
            raise OSError('64MiB output budget including closing reserve')
    signal.signal(signal.SIGALRM,guard);signal.setitimer(signal.ITIMER_REAL,1,1)
    def interrupted(signum,frame):raise InterruptedError('external termination signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        event('sources_captured')
        admission=_load(out/'admission.py','task8_run_admission')
        manifest=admission.pinned_path(card['inputs'],admission.DATA_ROOT)
        (out/'inputs.json').write_bytes(manifest.read_bytes())
        evidence=admission.authenticate(manifest,card['inputs']['sha256'],card['part'],max_input_bytes=card['max_input_bytes'])
        invocation=validate_producer_args(evidence['producer'],card['part'])
        _write(out/'admission.json',{'authentication':evidence['report'],'producer_args_checked':invocation})
        event('complete_inputs_admitted',files=evidence['report']['file_count'])
        reducer=_load(out/'statistics.py','task8_run_statistics')
        comparison=_load(out/'comparison.py','task8_run_comparison')
        with warnings.catch_warnings(record=True) as found:
            warnings.simplefilter('always')
            if card['part']=='A':
                result=reducer.analyze_a(evidence['spectra'],evidence['rows'])
            else:result=reducer.analyze_b(evidence['rows'])
            captured=[{'category':type(w.message).__name__,'message':str(w.message)} for w in found]
        _write(out/'independent.json',result);_write(out/'producer.json',evidence['producer'])
        event('independent_statistics_completed')
        compared=(comparison.compare_a if card['part']=='A' else comparison.compare_b)(evidence['producer'],result)
        _write(out/'comparison.json',compared);event('all_fields_compared')
        admission.recheck(manifest,card['inputs']['sha256'],evidence['manifest'])
        validate_card(card,directory);guard()
        status.update(status='replay_completed',computational_completed=True,
                      interpretation='Field comparison and scientific gates remain separate; no silent retry or outcome-dependent change.')
    except BaseException as exc:
        status.update(status='replay_failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        status.update(seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      minimum_available_bytes=minimum[0],events=events,warnings=captured)
        _write(out/'status.json',status)


if __name__=='__main__':main()
