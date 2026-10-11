"""EXP-016 fork of the reviewed owned-worker execute lifecycle.

Only the journal/result terminal commit is additionally shielded from handled
signals. Source: fresh_preview_runtime.execute. Ownership, bounded workers,
partial records, start shielding and cleanup semantics remain the same.
No workers, files, network calls or target calculations run at import.
"""
from __future__ import annotations
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import fresh_preview_runtime as work
from fresh_preview_runtime import append, write, identity
from fresh_preview_runtime import legacy


def execute(specs, contexts, directory, libraries, *, workers, deadline, per_task, worker_fn=work.worker):
    """Reuse reviewed EXP009 owned-entry lifecycle; retain identity and every status."""
    if not 1<=workers<=4:raise ValueError('Worker cap 1..4')
    directory=Path(directory);directory.mkdir();(directory/'payloads').mkdir();(directory/'records').mkdir()
    if len({t['id'] for t in specs})!=len(specs):raise ValueError('Duplicate tasks')
    pending=iter(specs);active={};results=[];exhausted=False;process_context=mp.get_context('fork')
    def record(task,row,expected_pid=None):
        if (not isinstance(row,dict) or row.get('id')!=task['id'] or row.get('case_id')!=task['case']['id']
                or row.get('method')!=task['method']['id'] or row.get('kind')!=task['kind']
                or (expected_pid is not None and row.get('pid')!=expected_pid)):
            row=dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='invalid_worker_identity',rmse=None)
        # Commit both durable journal and in-memory terminal state before a
        # pending handled interruption can reach cleanup and record this task
        # again. SIGKILL and IO failure remain explicitly unsealable failures.
        previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,
            {signal.SIGINT,signal.SIGTERM,signal.SIGALRM})
        try:
            append(directory/'journal.jsonl',row)
            results.append(row)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
        if len(results)%100==0:
            try:print(json.dumps(dict(stage=directory.name,terminal=len(results),required=len(specs))),flush=True)
            except OSError:pass  # A closed progress consumer must not alter research execution.
    try:
        while active or not exhausted:
            expired=time.monotonic()>=deadline
            while not expired and not exhausted and len(active)<workers:
                task=next(pending,None)
                if task is None:exhausted=True;break
                process=process_context.Process(target=legacy.owned_entry,args=(worker_fn,task,contexts[task['case']['id']],directory,libraries,os.getpid()))
                active[task['id']]=(process,task,time.monotonic());mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGALRM})
                try:
                    try:process.start()
                    finally:
                        if process.pid is not None:append(directory/'processes.jsonl',dict(task_id=task['id'],pid=process.pid,identity=identity(process.pid),started_at=legacy.now()))
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            for ident,(process,task,started) in list(active.items()):
                timed_out=expired or time.monotonic()-started>=per_task
                if process.is_alive() and not timed_out:continue
                if timed_out:legacy.terminate_owned(process)
                else:process.join()
                p=directory/'records'/f'{ident}.json'
                if p.exists() and not timed_out and process.exitcode==0:
                    try:row=json.loads(p.read_text())
                    except (ValueError,OSError):row=dict(id=ident,case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],pid=process.pid,status='invalid_worker_record',rmse=None)
                else:
                    row=dict(id=ident,case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],pid=process.pid,
                        status='whole_attempt_timeout' if expired else 'task_timeout' if timed_out else 'worker_failed',rmse=None,exitcode=process.exitcode)
                    write(directory/'records'/f'{ident}-termination.json',row)
                record(task,row,process.pid);del active[ident]
            if expired and not exhausted:
                for task in pending:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='not_started_deadline',rmse=None))
                exhausted=True
            if active:time.sleep(.01)
    except BaseException:
        recorded={r['id'] for r in results}
        for process,task,_ in active.values():
            legacy.terminate_owned(process)
            if task['id'] not in recorded:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='interrupted',pid=process.pid,rmse=None))
        for task in pending:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='not_started_interruption',rmse=None))
        raise
    finally:
        for process,_,_ in active.values():legacy.terminate_owned(process)
        write(directory/'terminal-records.json',results)
        identities=[json.loads(line) for line in (directory/'processes.jsonl').read_text().splitlines()] if (directory/'processes.jsonl').exists() else []
        checks=[]
        for row in identities:
            current=identity(row['pid'])
            checks.append(dict(**row,current_identity=current,original_identity_absent=current is None or (row['identity'] is not None and current!=row['identity'])))
        write(directory/'cleanup.json',dict(workers=checks,all_registered_workers_joined=not any(p.is_alive() for p,_,_ in active.values()),all_original_identities_absent=all(c['original_identity_absent'] for c in checks)))
    return results
