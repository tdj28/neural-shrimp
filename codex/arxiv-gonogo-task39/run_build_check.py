"""One bounded read-only-producer arXiv packaging check; writes only fresh scratch output."""
import datetime, hashlib, json, os, pathlib, resource, signal, subprocess, time
ROOT=pathlib.Path('/data2/neural-shrimp-data/scratch/audit-codex-1019/task39-build-001')
SOURCE='4ca5b5e35bc766581a5e2d326858fcd0edc312bb'
BUILDER='0641e45a9031cdcc1b5f21bdd210869689e68a4a42a203d51d0f7eb8c309581b'
def run():
    repo=ROOT/'article-source';out=ROOT/'bundle';receipt=ROOT/'build-execution.json'
    assert not out.exists() and not receipt.exists(), 'fresh allocation required'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==SOURCE
    assert hashlib.sha256((repo/'build_arxiv_bundle.py').read_bytes()).hexdigest()==BUILDER
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert mem>=20*1024**3, '20GiB startup headroom required for small packaging lane'
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
    def limits():
        os.setsid();os.nice(10);os.sched_setaffinity(0,{1})
        resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(300,300))
        resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2))
    start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    command=['/usr/bin/python3','-B',str(repo/'build_arxiv_bundle.py'),'--out',str(out)]
    reason=None
    with (ROOT/'build.stdout').open('xb') as stdout,(ROOT/'build.stderr').open('xb') as stderr:
        p=subprocess.Popen(command,cwd=repo,env=env,stdout=stdout,stderr=stderr,preexec_fn=limits)
        while p.poll() is None:
            elapsed=time.monotonic()-start
            current=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
            if elapsed>360 or current<16*1024**3:
                reason='wall_limit' if elapsed>360 else 'available_RAM_floor'
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(1)
        rc=p.wait()
    outputs={}
    for path in ROOT.rglob('*'):
        if path.is_file() and '.git' not in path.parts and 'article-source' not in path.relative_to(ROOT).parts:
            b=path.read_bytes();outputs[str(path.relative_to(ROOT))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
    record={'schema':'codex-task39-build-execution-v1','start_utc':utc,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,'source_commit':SOURCE,'command':command,'pid':p.pid,'returncode':rc,'stop_reason':reason,'initial_available_bytes':mem,'limits':{'wall_seconds':360,'cpu_seconds_per_process':300,'address_space_bytes_per_process':4*1024**3,'single_file_bytes':128*1024**2,'cpu_affinity':[1],'nice':10,'numerical_threads':1,'runtime_available_floor_bytes':16*1024**3},'outputs':outputs,'scientific_model_calls':0,'producer_checkout_modified':False}
    receipt.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='outputs'},indent=2))
    raise SystemExit(rc if rc>=0 else 1)
if __name__=='__main__':run()
