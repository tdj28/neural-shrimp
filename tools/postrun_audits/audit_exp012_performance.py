#!/usr/bin/env python3
"""Post-run EXP-012 timing/receipt/resource audit; never imports producer code.

Read-only access to immutable raw evidence and source. Writes only a new external
audit directory. Does not integrate, score, fit, load CUDA or signal processes.
Use --host-scope-confirmed only when launched outside the filesystem/PID sandbox.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time
import zipfile

SOURCE = 'c3a88b9db0a3e82b65d5d55a58d155b9a3275d5f'
PLAN_SHA = '74cef32c3eedf20cba0f3e9b05dc44f1a3f29f6b04c865cf60bee4712750360e'
ROOT = Path('/data2/neural-shrimp')
PLAN_NAME = 'experiments/hopfield/EXP-012-plan.json'
GIB = 1024**3


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def binding(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)}


def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def schedule():
    jobs = [dict(id='cold-'+arm, arm=arm, dt=.0025, size=18, kind='cold', rep=None, layout='canonical')
            for arm in ('cpu', 'gpu')]
    for dt in (.0025, .00125):
        for n in (18, 144, 576, 3456):
            for rep in range(3):
                for arm in (('cpu','gpu') if rep % 2 == 0 else ('gpu','cpu')):
                    jobs.append(dict(id=f'{arm}-dt{dt:g}-n{n}-r{rep}', arm=arm, dt=dt,
                                     size=n, kind='warm', rep=rep, layout='canonical'))
        for layout in ('reverse', 'chunked'):
            jobs.append(dict(id=f'gpu-dt{dt:g}-{layout}', arm='gpu', dt=dt,
                             size=3456, kind='layout', rep=None, layout=layout))
    return jobs


def proc_identity(pid):
    try:
        tail = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=tail[19], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def gpu_rows(text):
    result=[]
    for line in text.splitlines():
        fields=[p.strip() for p in line.split(',')]
        if len(fields) != 8:
            raise ValueError('Unexpected GPU resource CSV')
        result.append(dict(index=int(fields[0]), uuid=fields[1], name=fields[2], driver=fields[3],
                           total_mib=int(fields[4]), used_mib=int(fields[5]), free_mib=int(fields[6]), utilization=int(fields[7])))
    return result


def gpu_pids(text):
    return sorted({int(line.split(',',1)[0].strip()) for line in text.splitlines() if line.strip()})


def npy_member(z, name):
    """Read .npy metadata; only small status/progress payloads are decoded."""
    with z.open(name+'.npy') as stream:
        if stream.read(6) != b'\x93NUMPY':
            raise ValueError('Invalid npy magic')
        version=stream.read(2)
        count = 2 if version[0] == 1 else 4
        length=int.from_bytes(stream.read(count), 'little')
        header=ast.literal_eval(stream.read(length).decode('latin1').strip())
        data=stream.read() if name in ('statuses','progress') else None
    item=z.getinfo(name+'.npy')
    return header, data, item.file_size-(8+count+length)


class Checks:
    def __init__(self):
        self.count=0
        self.failures=[]
    def check(self, value, label, detail=None):
        self.count+=1
        if not value:
            self.failures.append(dict(check=label, detail=detail))
        return bool(value)
    def equal(self, actual, expected, label):
        return self.check(type(actual) is type(expected) and actual == expected, label,
                          None if actual == expected and type(actual) is type(expected) else dict(actual=actual, expected=expected))


def audit(args):
    begin=time.monotonic()
    raw=args.raw.resolve(); out=args.out.resolve()
    if out.is_relative_to(raw) or out.is_relative_to(ROOT) or out.exists():
        raise ValueError('Output must be new, outside raw evidence and repository')
    out.mkdir(parents=True)
    check=Checks(); report=dict(schema='EXP-012-independent-performance-v1', created_at=datetime.now(timezone.utc).isoformat(),
        execution_scope='host' if args.host_scope_confirmed else 'sandbox-or-unconfirmed',
        scope='Independent saved-record timing, receipt, packet schema, resource and owned-process audit. No producer imports, numerical metric replay, integration, fitting, CUDA context or signaling.',
        script=binding(__file__), raw=str(raw), expected_source=SOURCE, expected_plan_sha256=PLAN_SHA)
    try:
        receipt=read(raw/'receipt.json'); manifest=read(raw/'manifest.json'); plan=manifest['plan']
        summary=read(raw/'summary.json'); records=read(raw/'terminal-records.json')
        startup=read(raw/'startup.json'); builds=read(raw/'builds.json'); cleanup=read(raw/'cleanup.json')
        events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
        original_audit=read(raw/'audit.json'); postflight=read(raw/'postflight.json')
        report['raw_receipt']=binding(raw/'receipt.json')
        frozen=subprocess.check_output(['git','-C',str(ROOT),'show',f'{SOURCE}:{PLAN_NAME}'])
        check.equal(hashlib.sha256(frozen).hexdigest(), PLAN_SHA, 'frozen plan hash')
        check.equal(json.loads(frozen), plan, 'manifest plan equals exact Git plan')
        for name,obj in [('receipt',receipt),('manifest',manifest)]:
            check.equal(obj['source_commit'],SOURCE,name+' source')
            check.equal(obj['plan_sha256'],PLAN_SHA,name+' plan')
        check.equal(receipt['summary'],summary,'receipt summary identity')
        check.equal(receipt['directory'],str(raw),'receipt directory')
        check.equal(receipt['experiment_id'],'EXP-012','receipt experiment')
        check.equal(plan['experiment_id'],'EXP-012','plan experiment')
        check.equal(summary['error'],None,'no run error')
        check.equal(postflight['source_and_inputs_unchanged'],True,'saved postflight source check')
        check.check(stamp(manifest['remote_head_verified_at']) <= stamp(manifest['started_at']), 'remote verification before run')
        check.check(stamp(manifest['started_at']) <= stamp(summary['ended_at']) <= stamp(receipt['inventory_sealed_at']), 'run/seal chronology')
        sources=[]
        check.equal(len(plan['source_files']),22,'complete source bindings')
        check.equal(len({x['path'] for x in plan['source_files']}),22,'unique source bindings')
        for row in plan['source_files']+[plan['old_plan']]:
            content=subprocess.check_output(['git','-C',str(ROOT),'show',f"{SOURCE}:{row['path']}"])
            check.equal(hashlib.sha256(content).hexdigest(),row['sha256'],'frozen source hash '+row['path'])
            check.equal(len(content),row['bytes'],'frozen source bytes '+row['path'])
            sources.append(row)
        receipt_ref=plan['label_receipt']
        check.equal(digest(receipt_ref['path']),receipt_ref['sha256'],'upstream receipt hash')
        check.equal(Path(receipt_ref['path']).stat().st_size,receipt_ref['bytes'],'upstream receipt bytes')
        inventory=receipt['artifacts']; inv={i['path']:i for i in inventory}
        check.equal(len(inv),len(inventory),'unique receipt paths')
        actual={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}
        check.equal(actual,set(inv)|{'receipt.json'},'complete receipt census')
        artifact_bytes=0
        for relative,item in inv.items():
            p=raw/relative
            check.check(p.resolve().is_relative_to(raw) and not p.is_symlink(),'safe artifact path '+relative)
            check.equal(p.stat().st_size,item['bytes'],'artifact bytes '+relative)
            check.equal(digest(p),item['sha256'],'artifact hash '+relative)
            artifact_bytes+=p.stat().st_size
        report['inventory']=dict(artifacts=len(inventory),artifact_bytes=artifact_bytes,source_files=22)
        jobs=schedule(); ids=[c['id'] for c in plan['census']]
        check.equal(plan['jobs'],jobs,'independent 54-job schedule')
        check.equal(len(jobs),54,'54 jobs')
        check.equal(sum(j['size'] for j in jobs),64188,'64188 case integrations')
        check.equal(len(ids),3456,'3456 cases')
        check.equal(len(set(ids)),3456,'unique cases')
        check.equal([r['id'] for r in records],[j['id'] for j in jobs],'ordered complete terminal census')
        lookup={r['id']:r for r in records}
        check.equal(len(lookup),54,'unique terminal records')
        for n in (18,144,576,3456):
            check.equal(plan['subsets'][str(n)],ids[:n],'subset '+str(n))
            encoded=json.dumps(ids[:n],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            check.equal(hashlib.sha256(encoded).hexdigest(),plan['subset_hashes'][str(n)],'subset hash '+str(n))
        dispatches=[e for e in events if e['event']=='dispatch']
        terminals=[e for e in events if e['event']=='terminal']
        starts=[e for e in events if e['event']=='worker_started']
        check.equal([e['job'] for e in dispatches],jobs,'dispatch order and alternating pairs')
        check.equal([e['id'] for e in terminals],[j['id'] for j in jobs],'terminal event order')
        check.equal([e['arm'] for e in starts],['cpu','gpu'],'two owned worker starts')
        check.equal([e['event'] for e in events],['worker_started']*2+['dispatch','terminal']*54,'complete journal event sequence')
        identities={e['arm']:e['identity'] for e in starts}
        report['workers']=[]
        for arm in ('cpu','gpu'):
            r=startup[arm]; ident=identities[arm]
            check.equal(r['status'],'ready',arm+' startup ready')
            check.equal(r['pid'],ident['pid'],arm+' startup identity')
            check.check(finite(r['startup_seconds'],True) and finite(r['parent_observed_startup_seconds'],True),arm+' startup finite')
            check.check(r['parent_observed_startup_seconds']>=r['startup_seconds'],arm+' startup wall includes initialization')
            own=next(c for c in cleanup['workers'] if c['arm']==arm)
            check.equal(own['initial_identity'],ident,arm+' cleanup identity')
            check.equal(own['alive'],False,arm+' saved worker not alive')
            check.equal(own['original_identity_absent'],True,arm+' saved original identity absent')
            check.equal(own['exitcode'],0,arm+' clean worker exit')
            check.equal(own['current_identity'],None,arm+' saved PID absent')
        check.equal(len(cleanup['workers']),2,'two cleanup rows')
        check.equal(cleanup['all_owned_workers_stopped'],True,'all saved owned workers stopped')
        check.equal(manifest['packages'],{'numpy':'2.5.3','scipy':'1.18.1'},'runtime package pins')
        check.equal(set(manifest['thread_env']),{'OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'},'six thread caps')
        check.check(all(v=='1' for v in manifest['thread_env'].values()),'all thread caps one')
        for arm,pathkey in [('cpu','path'),('gpu','library')]:
            b=builds[arm]
            check.check(finite(b['wall_seconds'],True),arm+' build positive time')
            check.equal(digest(b[pathkey]),b['library_sha256'],arm+' compiled library binding')
        for k,v in dict(returncode=0,architecture='sm_89',dtype='float64',fmad=False,fast_math=False,ftz=False,precise_division=True,precise_sqrt=True,gpu_initialized=False).items():
            check.equal(builds['gpu'][k],v,'GPU build '+k)
        check.equal(startup['gpu']['device']['library_sha256'],builds['gpu']['library_sha256'],'loaded GPU library hash')
        resource_rows=[]; packet_rows=[]; chunk_rows=[]; prior_end=None
        expected_array_names={'times','states','statuses','final','progress','alpha','gamma','initial','delta'}
        for index,job in enumerate(jobs):
            name=job['id']; row=lookup[name]; base=raw/'batches'/name
            for key,value in job.items():
                check.equal(row[key],value,name+' job '+key)
            check.equal(row['status'],'complete',name+' status')
            check.equal(row['pid'],identities[job['arm']]['pid'],name+' worker ownership')
            check.equal(row['ids'],ids[:job['size']],name+' exact ordered IDs')
            packet=read(base/'record.json')
            check.equal({k:v for k,v in row.items() if k!='resources_before'},packet,name+' worker/terminal record identity')
            check.equal({p.name for p in base.iterdir() if p.is_file()},{'record.json','scores.json','traces.npz'},name+' three-file packet census')
            for key,file in [('raw','traces.npz'),('scores','scores.json')]:
                b=row['artifacts'][key]
                check.equal(b,{'path':file,'bytes':inv[f'batches/{name}/{file}']['bytes'],'sha256':inv[f'batches/{name}/{file}']['sha256']},name+' '+key+' receipt binding')
            scores=read(base/'scores.json')
            check.equal(scores['raw'],row['artifacts']['raw'],name+' scores bind traces')
            check.equal(len(scores['scores']),job['size'],name+' score count')
            shapes={'times':(751,),'states':(job['size'],751,3),'statuses':(job['size'],),'final':(job['size'],3),'progress':(job['size'],2),'alpha':(job['size'],),'gamma':(job['size'],),'initial':(job['size'],3),'delta':(job['size'],2)}
            with zipfile.ZipFile(base/'traces.npz') as z:
                check.equal(set(z.namelist()),{n+'.npy' for n in expected_array_names},name+' array census')
                for member,shape in shapes.items():
                    meta,data,bytecount=npy_member(z,member)
                    dtype='<i4' if member=='statuses' else '<i8' if member=='progress' else '<f8'
                    check.equal(meta,{'descr':dtype,'fortran_order':False,'shape':shape},name+' '+member+' schema')
                    check.equal(bytecount,math.prod(shape)*int(dtype[-1]),name+' '+member+' payload bytes')
                    if data is not None:
                        vals=struct.unpack('<'+('i' if member=='statuses' else 'q')*math.prod(shape),data)
                        expected=(751,)*job['size'] if member=='statuses' else (round(75/job['dt']),751)*job['size']
                        check.equal(vals,expected,name+' '+member+' complete values')
            elapsed=row['end_to_end_seconds']; integrated=row['integration_scoring_seconds']; storage=row['storage_verification_seconds']
            for key in ['end_to_end_seconds','input_preparation_seconds','integration_scoring_seconds','storage_verification_seconds','max_rss_kib']:
                check.check(finite(row[key],True),name+' finite positive '+key)
            check.check(elapsed<=180,name+' batch ceiling')
            check.equal(storage,elapsed-integrated,name+' E2E decomposition')
            check.check(row['input_preparation_seconds']<integrated<elapsed,name+' timer nesting')
            phase=row['phase_seconds']
            check.equal(set(phase),{'write_and_hash_seconds','readback_seconds','score_replay_seconds'},name+' storage phase census')
            check.check(all(finite(v,True) for v in phase.values()),name+' positive storage phases')
            check.check(sum(phase.values()) <= storage+1e-9,name+' storage phases fit wall')
            wall=stamp(row['ended_at'])-stamp(row['started_at'])
            check.check(wall>=elapsed-2e-6,name+' worker timestamp contains timed batch')
            if prior_end is not None:
                check.check(stamp(row['started_at'])>=prior_end,name+' no batch overlap')
            prior_end=stamp(row['ended_at'])
            check.check(stamp(dispatches[index]['at'])<=stamp(row['started_at'])<=stamp(row['ended_at'])<=stamp(terminals[index]['at']),name+' event chronology')
            res=row['resources_before']
            check.equal(res,dispatches[index]['resources'],name+' dispatched resource identity')
            check.equal(res['nvidia_smi_status'],0,name+' GPU inventory success')
            check.equal(res['gpu_process_query_status'],0,name+' process inventory success')
            check.check(res['available_ram_bytes']>=4*GIB,name+' RAM working set plus reserve')
            check.check(res['free_disk_bytes']>=2*GIB,name+' disk reserve')
            check.equal(res['estimated_task_peak_bytes'],2*GIB,name+' working set estimate')
            check.check(stamp(res['at'])<=stamp(dispatches[index]['at']),name+' resource query before timed batch')
            gpu=gpu_rows(res['nvidia_smi']); owners=gpu_pids(res['gpu_processes'])
            resource_rows.append(dict(id=name,available_ram_bytes=res['available_ram_bytes'],free_disk_bytes=res['free_disk_bytes'],gpu=gpu,gpu_owner_pids=owners))
            solver=scores['solver']
            if job['arm']=='cpu':
                check.check(finite(solver['integration_and_scoring_seconds'],True),name+' CPU combined phase')
                check.check(all(finite(solver['sum_case_seconds'][key],True) for key in ('integration_seconds','scoring_seconds')),name+' CPU overlapped case sums')
                kernel=None
            else:
                sizes=[job['size']] if job['layout']!='chunked' else [257]*13+[115]
                check.equal(len(solver['chunks']),len(sizes),name+' GPU chunk count')
                kernel=0.
                for c,size in zip(solver['chunks'],sizes):
                    m=c['memory']; required=24*size*751+84*size
                    check.equal(m['requested_device_bytes'],required,name+' requested device bytes')
                    check.equal(m['device_byte_ceiling'],GIB,name+' fixed GPU cap')
                    check.check(required<=GIB and required<=m['free_device_bytes_before']//2,name+' GPU allocation margins')
                    check.equal(c['device']['library_sha256'],builds['gpu']['library_sha256'],name+' chunk library')
                    check.check(all(finite(v) for v in c['timings'].values()),name+' finite GPU phase times')
                    kernel+=c['timings'].get('kernel_seconds',0.)
                    chunk_rows.append(dict(id=name,size=size,requested_bytes=required,free_bytes=m['free_device_bytes_before'],timings=c['timings']))
                check.check(finite(solver['scoring_seconds'],True),name+' GPU CPU scoring phase')
            packet_rows.append(dict(id=name,arm=job['arm'],size=job['size'],dt=job['dt'],end_to_end_seconds=elapsed,integration_scoring_seconds=integrated,storage_verification_seconds=storage,phase_seconds=phase,max_rss_kib=row['max_rss_kib'],kernel_seconds=kernel))
        report['census']=dict(batches=len(records),case_integrations=sum(r['size'] for r in records),arms=dict(Counter(r['arm'] for r in records)),statuses=dict(Counter(r['status'] for r in records)))
        report['packets']=packet_rows
        timing=[]
        for dt in (.0025,.00125):
            for size in (18,144,576,3456):
                pairs=[]
                for rep in range(3):
                    cpu=lookup[f'cpu-dt{dt:g}-n{size}-r{rep}']['end_to_end_seconds']
                    gpu=lookup[f'gpu-dt{dt:g}-n{size}-r{rep}']['end_to_end_seconds']
                    pairs.append(dict(rep=rep,cpu_seconds=cpu,gpu_seconds=gpu,speedup=cpu/gpu))
                ratios=[p['speedup'] for p in pairs]
                timing.append(dict(dt=dt,size=size,pairs=pairs,median_speedup=statistics.median(ratios),cpu_median_seconds=statistics.median([p['cpu_seconds'] for p in pairs]),gpu_median_seconds=statistics.median([p['gpu_seconds'] for p in pairs]),speed_gate=min(ratios)>1 and statistics.median(ratios)>=1.25))
        check.equal(timing,summary['timing'],'all independent timing fields equal summary')
        check.equal(timing,original_audit['timing'],'all independent timing fields equal full audit')
        cold={arm:lookup['cold-'+arm]['end_to_end_seconds']+startup[arm]['parent_observed_startup_seconds'] for arm in ('cpu','gpu')}
        target=next(r for r in timing if r['dt']==.00125 and r['size']==3456)
        warm={'cpu':target['cpu_median_seconds'],'gpu':target['gpu_median_seconds']}
        extra=max(0.,cold['gpu']+builds['gpu']['wall_seconds']-cold['cpu']-builds['cpu']['wall_seconds'])
        saving=warm['cpu']-warm['gpu']
        cost=dict(cold_pipeline_seconds=cold,warm_full_batch_median_seconds=warm,additional_gpu_setup_seconds=extra,median_full_batch_saving_seconds=saving,break_even_full_batches=math.ceil(extra/saving) if saving>0 else None)
        for key,value in cost.items():
            check.equal(value,summary['cost'][key],'independent cost '+key)
        check.equal(summary['gpu_deployment_qualified'],bool(summary['all_parity_pass'] and target['speed_gate']),'deployment rule conditional on recorded numerical parity')
        check.equal(summary['all_parity_pass'],original_audit['all_parity_pass'],'audit/summary recorded parity identity')
        check.equal(summary['gpu_deployment_qualified'],original_audit['gpu_deployment_qualified'],'audit/summary deployment identity')
        report['timing']=timing; report['cost']=cost
        report['decision']=dict(independently_verified_full_batch_speed_gate=target['speed_gate'],recorded_all_parity_pass=summary['all_parity_pass'],recorded_gpu_deployment_qualified=summary['gpu_deployment_qualified'],numerical_parity_recomputed_here=False)
        report['resources']=dict(minimum_available_ram_bytes=min(r['available_ram_bytes'] for r in resource_rows),minimum_free_disk_bytes=min(r['free_disk_bytes'] for r in resource_rows),maximum_worker_rss_kib=max(r['max_rss_kib'] for r in records),maximum_device_requested_bytes=max(r['requested_bytes'] for r in chunk_rows),minimum_free_device_bytes_before=min(r['free_bytes'] for r in chunk_rows),gpu_utilization_percent_range=[min(g['utilization'] for r in resource_rows for g in r['gpu']),max(g['utilization'] for r in resource_rows for g in r['gpu'])],all_54_resource_snapshots_checked=True,gpu_chunk_count=len(chunk_rows),initial_gpu_owner_pids=gpu_pids(manifest['initial_resources']['gpu_processes']),postflight_gpu_owner_pids=gpu_pids(postflight['resources']['gpu_processes']),gpu_device=startup['gpu']['device'])
        report['gpu_chunks']=chunk_rows
        check.check(sum(r['end_to_end_seconds'] for r in records) < summary['wall_seconds'], 'attempt wall includes batch timing')
        check.check(summary['wall_seconds']<=3600,'whole attempt ceiling')
        report['wall']=dict(started_at=manifest['started_at'],ended_at=summary['ended_at'],inventory_sealed_at=receipt['inventory_sealed_at'],through_summary_seconds=summary['wall_seconds'],analysis_seconds=summary['analysis_seconds'],sealing_seconds=receipt['sealing_seconds'],sum_batch_e2e_seconds=sum(r['end_to_end_seconds'] for r in records))
        if args.host_scope_confirmed:
            for arm,identity in identities.items():
                current=proc_identity(identity['pid'])
                report['workers'].append(dict(arm=arm,original_identity=identity,current_identity=current,original_identity_absent=current!=identity))
                check.check(current!=identity,arm+' current host owned identity absent')
            parent_current=proc_identity(manifest['pid'])
            report['parent_pid_current']=dict(pid=manifest['pid'],current_identity=parent_current,note='Manifest did not bind parent start ticks; only current PID presence is reported.')
            query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
            owners=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
            check.equal(query.returncode,0,'current host GPU inventory query')
            check.equal(owners.returncode,0,'current host GPU owner query')
            current_owners=gpu_pids(owners.stdout) if owners.returncode==0 else []
            owned_pids={i['pid'] for i in identities.values()}|{manifest['pid']}
            check.check(not owned_pids.intersection(current_owners),'no owned parent/worker PID currently owns GPU')
            report['host_inventory']=dict(at=datetime.now(timezone.utc).isoformat(),gpu=gpu_rows(query.stdout) if query.returncode==0 else [],gpu_owner_pids=current_owners,owned_pids=sorted(owned_pids),owned_gpu_pids=sorted(owned_pids.intersection(current_owners)),gpu_query_status=query.returncode,gpu_owner_query_status=owners.returncode)
        else:
            report['host_inventory']=dict(status='not verified: execution scope unconfirmed')
        report['limitations']=['This audit reconstructs timing and data/resource provenance; numerical parity and score values require the separate saved-array audit.', 'Shared-host timings apply to the recorded workload and concurrent load. No fresh science or new case inference follows.', 'Saved worker start/boot identity plus a host-visible final check establishes owned-worker cleanup. The parent manifest lacks start ticks, so parent identity reuse cannot be excluded if that PID is present.', 'No other process was signaled, terminated or modified. GPU owner inventory is read-only.']
    except BaseException as error:
        check.check(False,'audit raised exception',repr(error))
        report['exception']=repr(error)
    report.update(status='pass' if not check.failures else 'fail',checks=check.count,failures=check.failures,elapsed_seconds=time.monotonic()-begin)
    target=out/'report.json'
    target.write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda obj:sorted(obj) if isinstance(obj,set) else str(obj))+'\n')
    receipt_out=dict(created_at=datetime.now(timezone.utc).isoformat(),report=binding(target),script=binding(__file__),immutable_raw_modified=False)
    (out/'receipt.json').write_text(json.dumps(receipt_out,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],failures=report['failures'],report=receipt_out['report']),indent=2))
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path('/data2/neural-shrimp-data/raw/EXP-012/attempt-001'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--host-scope-confirmed',action='store_true')
    sys.exit(audit(parser.parse_args()))
