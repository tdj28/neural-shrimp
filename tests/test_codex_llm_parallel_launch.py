"""Launcher integration under mocked Git/process boundaries; no target jobs."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('certificate_launcher',ROOT/'codex/certify-llm/launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def case(tmp_path,monkeypatch,*,timeout=False,unpublished=False,dirty=False,driver="center_continue"):
    repo=tmp_path/'repo';owned=tmp_path/'owned';source=repo/'codex/certify-llm';source.mkdir(parents=True)
    blobs={
        'codex/certify-llm/launch.py':b'# mocked launcher snapshot\n',
        'codex/certify-llm/pilot.py':b'# historical pilot\n',
        'codex/certify-llm/center_continue.py':b'# never executed\n',
        'codex/certify-llm/assembly.py':b'# mocked assembly, never executed\n',
        'codex/certify-llm/parallel-plan.json':b'{"mock_only":true}\n',
    }
    for name,data in blobs.items():(repo/name).write_bytes(data)
    if dirty:(source/'center_continue.py').write_bytes(b'# dirty\n')
    monkeypatch.setattr(launch,'__file__',str(source/'launch.py'))
    monkeypatch.setattr(launch,'ROOT',owned)
    calls=[];signals=[]
    def git_output(args,**kwargs):
        assert args[0]=='git';args=args[1:]
        if args==['rev-parse','HEAD'] or args==['rev-parse','origin/main']:return 'a'*40+'\n'
        if args[:2]==['ls-tree','-r']:return '\n'.join(blobs)+'\n'
        if args[0]=='show':return blobs[args[1].split(':',1)[1]]
        raise AssertionError(args)
    def git_run(args,**kwargs):
        assert args[0]=='git' and kwargs.get('check') is True
        calls.append(args)
        if unpublished and args[1]=='merge-base':raise subprocess.CalledProcessError(1,args)
        return SimpleNamespace(returncode=0)
    class Child:
        pid=987654
        def __init__(self,command,**kwargs):
            self.command=command;self.kwargs=kwargs;self.waits=0;calls.append(self)
        def wait(self,timeout=None):
            self.waits+=1
            if self.waits==1 and timeout_case:raise subprocess.TimeoutExpired(self.command,timeout)
            return 0
    timeout_case=timeout
    monkeypatch.setattr(launch.subprocess,'check_output',git_output)
    monkeypatch.setattr(launch.subprocess,'run',git_run)
    monkeypatch.setattr(launch.subprocess,'Popen',Child)
    monkeypatch.setattr(launch.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    monkeypatch.setattr(sys,'argv',['launch.py','--attempt','parallel-001','--driver',driver,
        '--plan','parallel-plan.json','--seconds','3600'if driver=='assembly'else '1800','--threads','1'])
    return owned,calls,signals


def test_parallel_launch_preserves_cpu1_and_records_worker_envelope(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch)
    with pytest.raises(SystemExit) as outcome:launch.main()
    assert outcome.value.code==0
    path=owned/'parallel-001';manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['threads']==1 and manifest['parallel_workers']==4 and manifest['worker_threads']==1
    assert manifest['memory_gib']==18 and manifest['aggregate_rss_watchdog_gib']==32
    assert manifest['sampled_rss_is_hard_cap'] is False and manifest['minimum_available_gib']==12
    assert manifest['worker_timeout_seconds']==600 and manifest['timeout_seconds']==1800
    child=next(c for c in calls if hasattr(c,'kwargs'))
    assert child.command[1:4]==['-I','-B','-u'] and child.kwargs['start_new_session']
    env=child.kwargs['env']
    assert all(env[k]=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    assert env['CUDA_VISIBLE_DEVICES']=='' and env['HF_HUB_OFFLINE']=='1' and env['CERT_SOURCE_COMMIT']=='a'*40
    assert child.command[-2:] == ['--plan',str(path/'source/codex/certify-llm/parallel-plan.json')]
    receipt=json.loads((path/'receipt.json').read_text());assert receipt['exit_code']==0 and not receipt['timed_out']
    assert all(launch.sha(path/a['path'])==a['sha256'] for a in receipt['artifacts'])
    assert all(not (p.stat().st_mode & 0o222) for p in path.rglob('*') if p.is_file())
    assert signals==[]


def test_parallel_timeout_signals_only_owned_controller_group(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch,timeout=True)
    with pytest.raises(SystemExit):launch.main()
    assert signals==[(987654,signal.SIGTERM)]
    receipt=json.loads((owned/'parallel-001/receipt.json').read_text());assert receipt['timed_out']
    child=next(c for c in calls if hasattr(c,'kwargs'));assert child.waits==2


@pytest.mark.parametrize('fault',['unpublished','dirty'])
def test_unauthenticated_source_never_creates_attempt_or_job(tmp_path,monkeypatch,fault):
    owned,calls,signals=case(tmp_path,monkeypatch,unpublished=fault=='unpublished',dirty=fault=='dirty')
    with pytest.raises((ValueError,subprocess.CalledProcessError)):launch.main()
    assert not owned.exists() and not any(hasattr(c,'kwargs') for c in calls) and not signals


@pytest.mark.parametrize('arguments',[
    ['--seconds','3600'],['--seconds','1799'],['--threads','4'],['--threads','2'],
    ['--skip-box'],['--plan','../outside.json'],
])
def test_invalid_parallel_envelope_refused_before_git_or_jobs(tmp_path,monkeypatch,arguments):
    owned,calls,signals=case(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',sys.argv+arguments)
    with pytest.raises(ValueError):launch.main()
    assert not calls and not owned.exists() and not signals


def test_parallel_requires_committed_plan(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch)
    monkeypatch.setattr(sys,'argv',['launch.py','--attempt','parallel-001','--driver','center_continue','--seconds','1800'])
    with pytest.raises(ValueError):launch.main()
    assert not calls and not owned.exists() and not signals


def test_assembly_launch_is_cpu1_hour_and_captures_explicit_plan(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch,driver='assembly')
    with pytest.raises(SystemExit)as outcome:launch.main()
    assert outcome.value.code==0
    path=owned/'parallel-001';manifest=json.loads((path/'manifest.json').read_text())
    assert manifest['threads']==1 and manifest['timeout_seconds']==3600 and manifest['memory_gib']==18
    assert 'parallel_workers'not in manifest
    child=next(c for c in calls if hasattr(c,'kwargs'))
    assert child.command[1:4]==['-I','-B','-u']
    assert child.command[4]==str(path/'source/codex/certify-llm/assembly.py')
    assert child.command[-2:]==['--plan',str(path/'source/codex/certify-llm/parallel-plan.json')]
    assert child.kwargs['start_new_session'] and child.kwargs['env']['CERT_SOURCE_COMMIT']=='a'*40
    assert all(child.kwargs['env'][k]=='1'for k in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    assert child.kwargs['env']['CUDA_VISIBLE_DEVICES']=='' and child.kwargs['env']['HF_HUB_OFFLINE']=='1'
    receipt=json.loads((path/'receipt.json').read_text())
    assert receipt['exit_code']==0 and receipt['timed_out']is False
    assert all(launch.sha(path/r['path'])==r['sha256']for r in receipt['artifacts'])
    assert signals==[]


@pytest.mark.parametrize('arguments',[
    ['--seconds','3599'],['--seconds','3601'],['--seconds','1800'],['--seconds','0'],
    ['--threads','2'],['--threads','4'],['--skip-box'],['--plan','../outside.json'],
])
def test_bad_assembly_scope_refused_before_git_or_process(tmp_path,monkeypatch,arguments):
    owned,calls,signals=case(tmp_path,monkeypatch,driver='assembly')
    monkeypatch.setattr(sys,'argv',sys.argv+arguments)
    with pytest.raises(ValueError):launch.main()
    assert not calls and not owned.exists() and not signals


def test_assembly_requires_explicit_plan_before_git(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch,driver='assembly')
    monkeypatch.setattr(sys,'argv',['launch.py','--attempt','parallel-001','--driver','assembly','--seconds','3600'])
    with pytest.raises(ValueError):launch.main()
    assert not calls and not owned.exists() and not signals


def test_assembly_timeout_only_signals_owned_group_and_retains_receipt(tmp_path,monkeypatch):
    owned,calls,signals=case(tmp_path,monkeypatch,driver='assembly',timeout=True)
    with pytest.raises(SystemExit):launch.main()
    assert signals==[(987654,signal.SIGTERM)]
    receipt=json.loads((owned/'parallel-001/receipt.json').read_text())
    assert receipt['timed_out']is True
    assert next(c for c in calls if hasattr(c,'kwargs')).waits==2
