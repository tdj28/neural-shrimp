"""Sequence the authorized local FGR3 stages, without rentals or GPU work.

One exclusive lock; no retries, no duplicate workers, no Git mutations. A
failure stops the sequence and preserves its evidence. Publication is a later
human-readable audit/authoring task, never an automated success claim here.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import sha, write_json, backup_file


def owned_process_active(pid, command):
    try:
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        cmd = [x.decode() for x in cmd if x]
        state = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
    except FileNotFoundError:
        return False
    if state == 'Z': return False
    if cmd != command:
        raise ValueError('PID identity changed; do not touch or wait for an unrelated process')
    return True


def main(args):
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    with open(root/'controller.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'launch.json').exists(): raise FileExistsError('Controller already launched; do not duplicate')
        source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        runtime = ['fold_return_followup.py','fold_return_surrogate.py','fold_return_growth.py',
                   'audit_fold_return_followup.py','audit_fold_return_growth.py','fold_return_study_controller.py',
                   'fold_return_curve.py','fold_return_line.py','fold_global_return.py','fold_panel_data.py',
                   'audit_fold_return_line.py','audit_fold_global_return.py','explore_llm_soft_loop.py',
                   'explore_llm_attractors.py','explore_llm_phase_map.py']
        hashes = {str(ROOT/'tools'/p):sha(ROOT/'tools'/p) for p in runtime}
        plan = ROOT/'docs/fold-return-autonomous-plan.json'; hashes[str(plan)] = sha(plan)
        launch = {'pid':os.getpid(),'source_commit':source,'runtime_hashes':hashes,
                  'training_launch':args.training_launch,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
        write_json(root/'launch.json',launch)
        paths = {
            'training':Path('/data2/neural-shrimp-data/explore/llm-soft-loop/global-return/20261113-fgr3-training-01/run'),
            'training_audit':Path('/data2/neural-shrimp-data/audits/fold-global-return/20261113-fgr3-training-01'),
            'validation':Path('/data2/neural-shrimp-data/explore/llm-soft-loop/global-return/20261113-fgr3-validation-01/run'),
            'validation_audit':Path('/data2/neural-shrimp-data/audits/fold-global-return/20261113-fgr3-validation-01'),
            'growth':Path('/data2/neural-shrimp-data/explore/llm-soft-loop/global-return/20261113-fgr3-growth-01/run'),
            'growth_audit':Path('/data2/neural-shrimp-data/audits/fold-global-return/20261113-fgr3-growth-01')}
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                   OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
        python = str(ROOT/'.venv/bin/python')
        stages = []
        def status(stage, **extra):
            write_json(root/'status.json',{'stage':stage,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                           'completed_stages':stages,**extra})
        def checked_stage(name, command):
            for path,digest in hashes.items():
                if sha(path)!=digest: raise ValueError(f'Frozen stage source changed: {path}')
            with open(root/(name+'-stdout.txt'),'xb') as log:
                p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                write_json(root/(name+'-launch.json'),{'pid':p.pid,'command':command,'source_commit':source})
                while p.poll() is None:
                    status(name,pid=p.pid)
                    time.sleep(30)
                if p.returncode:
                    raise RuntimeError(f'{name} exited with code {p.returncode}; no retry')
            stages.append(name);status(name+' complete')
        try:
            training_launch=json.loads(Path(args.training_launch).read_text())
            while owned_process_active(training_launch['pid'],training_launch['command']):
                status('training',pid=training_launch['pid'])
                time.sleep(30)
            if not (paths['training']/'complete.json').exists() or (paths['training']/'failure.json').exists():
                raise RuntimeError('Training ended without verified completion')
            checked_stage('training-audit',[python,'tools/audit_fold_return_followup.py',
                '--run',str(paths['training']),'--out',str(paths['training_audit']),
                '--backup','/archive/neural-shrimp-backups/20261113-fgr3-training-01/audit'])
            freeze={'model':str(paths['training']/'model.json'),'model_sha256':sha(paths['training']/'model.json'),
                    'plan_sha256':sha(plan),'training_audit_sha256':sha(paths['training_audit']/'audit.json'),
                    'source_commit':source,'frozen_before_fresh_validation_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    'status':'Exploratory prospective validation; no validation outputs yet',
                    'paths':{k:str(v) for k,v in paths.items()}}
            if paths['validation'].exists():raise FileExistsError('Fresh validation path already exists')
            write_json(root/'validation-freeze.json',freeze)
            backup_file(root/'validation-freeze.json',root,Path('/archive/neural-shrimp-backups/20261113-fgr3-controller-01'))
            checked_stage('validation',[python,'tools/fold_return_followup.py','validate',
                '--model',freeze['model'],'--out',str(paths['validation']),
                '--backup','/archive/neural-shrimp-backups/20261113-fgr3-validation-01/run'])
            checked_stage('validation-audit',[python,'tools/audit_fold_return_followup.py',
                '--run',str(paths['validation']),'--out',str(paths['validation_audit']),
                '--backup','/archive/neural-shrimp-backups/20261113-fgr3-validation-01/audit'])
            checked_stage('growth',[python,'tools/fold_return_growth.py',
                '--validation',str(paths['validation']),'--audit',str(paths['validation_audit']),
                '--out',str(paths['growth']),'--backup','/archive/neural-shrimp-backups/20261113-fgr3-growth-01/run'])
            checked_stage('growth-audit',[python,'tools/audit_fold_return_growth.py',
                '--run',str(paths['growth']),'--out',str(paths['growth_audit']),
                '--backup','/archive/neural-shrimp-backups/20261113-fgr3-growth-01/audit'])
            status('ready_for_publication',paths={k:str(v) for k,v in paths.items()})
            write_json(root/'complete.json',{'numerics_and_audits_complete':True,'publication_and_visual_review_pending':True,
                                             'completed_stages':stages,'paths':{k:str(v) for k,v in paths.items()}})
        except BaseException as exc:
            status('failed',error=type(exc).__name__,message=str(exc))
            write_json(root/'failure.json',{'type':type(exc).__name__,'message':str(exc),'completed_stages':stages})
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--training-launch',required=True)
    main(p.parse_args())
