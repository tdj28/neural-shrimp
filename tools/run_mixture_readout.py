#!/usr/bin/env python3
"""EXP020 immutable saved-evidence decoder pipeline; no neural integration.

Metadata preparation hashes target files without decoding them. Every classifier
is serialized/reloaded before finite observation access or probability workers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import fresh_preview_runtime as work
import run_hopfield_refinement as legacy
import run_phase_memory_recovery as codec
import two_write_runtime as runtime
import run_refined_two_write_memory as previous
import gaussian_mixture_readout as core
import mixture_readout_evaluation as evaluation
import numpy as np
import flint

ROOT=work.ROOT
OLD=Path('/data2/neural-shrimp-data/raw/EXP-019/attempt-001')
OLD_SOURCE='7eeba42182184a209dbc2f718e8469a5cd97e4a9'
OLD_PLAN_SHA='86ae6924a1a89c1d8ae878742f0b012680443fa2cad8746c1853a52d4b6c0543'
OLD_RECEIPT_SHA='ef9d6020c6c13f3b7ea89fddff4c4eabb112275c6c8b98724f2e56d0735c5001'
COMPACT='docs/EXP-019-summary.json'
COMPACT_SHA='1fdffac18c5c5de47fe721c042dcd437a4b33ef1545537ce083dac3d4020ad7a'
CERTIFICATE='docs/EXP-019-reuse-qualification.json'
CERTIFICATE_SHA='80fcd32b678395976a768b460b1643b9fab471f3e73ebcce6eaf6ab6bd17c7d9'
PLAN_REL='experiments/hopfield/EXP-020-plan.json'
REVIEW_REL='docs/EXP-020-local-review.json'
NEW_SOURCES=['tools/run_mixture_readout.py','tools/gaussian_mixture_readout.py',
 'tools/mixture_readout_evaluation.py','tests/test_mixture_readout_runner.py',
 'tests/test_gaussian_mixture_readout.py','tests/test_mixture_readout_evaluation.py',
 'tests/test_mixture_readout_integration.py','docs/EXP-020-protocol.md',
 'requirements-exp020.lock','docs/EXP-020-runtime-provenance.json',CERTIFICATE,COMPACT]
SOURCE_PATHS=list(dict.fromkeys(NEW_SOURCES+previous.SOURCE_PATHS))
GIB=1024**3
CONTRACT=dict(experiment_id='EXP-020',schema='saved-mixture-analysis-v1',exploratory=True,
 contexts=48,phase_count=24,phase_seed=2026091701,reused_regression_phases=True,fresh_holdout=False,
 cell_count=144,classifier_definitions=432,finite_class_method_rows=1200,probability_calculations=4704,
 primary_finite_calculations=1200,secondary_finite_calculations=1200,alternate_fine_calculations=576,
 reference_controls=1728,auxiliary_linear_reference_calculations=1728,mixture_plus_auxiliary_probability_calculations=6432,primary_method_comparisons=624,secondary_method_comparisons=624,
 reference_comparisons=576,method_context_window_groups=300,windows=['k2','k4','k8'],
 settings=['core','failed-near-edge'],recipes=['primary','alternate','one_mode'],arm='identity',signs=[-1,1],
 epsilon=1e-4,sigma=1e-4,workers=4,thread_limit=1,producer_seconds=900,per_cell_seconds=30,
 sealing_seconds=120,launch_ram_bytes=16*GIB,reserve_ram_bytes=8*GIB,owned_rss_bytes=8*GIB,
 raw_budget_bytes=512*1024**2,neural_integrations=0,paid_compute=False,paid_review=False,gpu=False,
 required_packages={'numpy':'2.5.3','scipy':'1.18.1','python-flint':'0.9.0'},flint_native='3.6.0',
 original_studies_requalified=False,global_definition_barrier=True,
 primary='Equal24 near-edge k4 fine primary-mixture-minus-original-linear BA; both settings must qualify')

class ProducerStop(BaseException):pass


def read(path):
    return json.loads(Path(path).read_text())

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def binding(path, root=ROOT):
    path=Path(path)
    return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=legacy.sha(path))

def inventory(directory):
    directory=Path(directory);result=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Evidence symlinks are forbidden')
        if path.is_file() and path != directory/'receipt.json':result.append(binding(path,directory))
    return result

def source_guard(plan,path,expected_head):
    if not re.fullmatch('[0-9a-f]{40}',expected_head):raise ValueError('Exact expected commit required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    commit=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    if commit!=expected_head or not branch.startswith('codex/'):raise ValueError('Source HEAD/branch differs')
    if Path(path).resolve()!=ROOT/PLAN_REL:raise ValueError('Canonical absolute-resolved plan path required')
    for relative in [*SOURCE_PATHS,PLAN_REL,REVIEW_REL]:
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{relative}'])
        if hashlib.sha256(content).hexdigest()!=legacy.sha(ROOT/relative):raise ValueError('Uncommitted source/plan/review')
    review=read(ROOT/REVIEW_REL)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(path):raise ValueError('Exact-plan local review required')
    if legacy.git('remote','get-url','origin')!='git@github-personal:tdj28/neural-shrimp.git':raise ValueError('Expected private origin required')
    remote=legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]
    if remote!=commit:raise ValueError('Exact private source not pushed')
    return dict(source_commit=commit,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(ROOT/REVIEW_REL))

def external_attempt(data_root,attempt):
    if not isinstance(attempt,str) or re.fullmatch(r'attempt-\d+',attempt) is None:raise ValueError('Named attempt-N required')
    root=Path(data_root).resolve()
    if root==ROOT or root.is_relative_to(ROOT):raise ValueError('Raw evidence must be external')
    for ancestor in (root,*root.parents):
        if (ancestor/'receipt.json').exists() or (re.fullmatch(r'attempt-\d+',ancestor.name) and ancestor.parent.name.startswith('EXP-')):
            raise ValueError('Never extend an existing historical attempt')
    for protected in (OLD,previous.OLD,Path('/data2/neural-shrimp-data/raw/EXP-016/attempt-001')):
        if root==protected or root.is_relative_to(protected):raise ValueError('Never write into historical evidence')
    root.mkdir(parents=True,exist_ok=True);parent=root/'EXP-020';parent.mkdir(exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Experiment directory alias forbidden')
    directory=parent/attempt;directory.mkdir()
    return directory

def resource_check(directory,*,launch=False):
    directory=Path(directory);state=codec.resources(directory)
    threshold=CONTRACT['launch_ram_bytes'] if launch else CONTRACT['reserve_ram_bytes']
    if state['available_ram_bytes']<threshold:raise RuntimeError('Shared RAM reserve unavailable')
    owned={};pending=[os.getpid()]
    while pending:
        pid=pending.pop()
        if pid in owned:continue
        identity=work.identity(pid)
        if identity is None:continue
        owned[pid]=identity
        try:pending.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except FileNotFoundError:pass
    rss=0
    for pid,identity in owned.items():
        try:
            if work.identity(pid)==identity:rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:pass
    raw_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    state.update(owned_rss_bytes=rss,raw_bytes=raw_bytes,owned_identities=list(owned.values()),shared_pages_counted_per_process=True)
    work.append(directory/'resource-monitor.jsonl',state)
    if rss>CONTRACT['owned_rss_bytes'] or raw_bytes>CONTRACT['raw_budget_bytes']:raise RuntimeError('Owned memory/raw cap exceeded')
    if state['disk_free_bytes']<CONTRACT['raw_budget_bytes']:raise RuntimeError('Disk reserve unavailable')
    return state

def event(directory,name,**fields):
    work.append(Path(directory)/'events.jsonl',dict(event=name,at=legacy.now(),monotonic_ns=time.monotonic_ns(),**fields))

def normalized(value):
    arrays={};tree=codec._encode(value,arrays)
    return codec._decode(tree,arrays,set())

def phases():
    values=np.random.Generator(np.random.PCG64(CONTRACT['phase_seed'])).random(24)
    return [dict(phase_index=i,phase_fraction=float(v),split='development' if i<8 else 'heldout') for i,v in enumerate(values)]


def old_inputs():
    """Hash finite evidence without parsing it; read only metadata certificate."""
    if legacy.sha(OLD/'receipt.json')!=OLD_RECEIPT_SHA:raise ValueError('Immutable EXP019 receipt changed')
    receipt=read(OLD/'receipt.json')
    if receipt.get('source_commit')!=OLD_SOURCE or receipt.get('plan_sha256')!=OLD_PLAN_SHA or receipt.get('status')!='complete':raise ValueError('Complete exact prior source required')
    inventory={r['path']:r for r in receipt['artifacts']}
    if len(inventory)!=len(receipt['artifacts']):raise ValueError('Duplicate prior inventory')
    relatives=['manifest.json','summary.json','analysis.json','analysis.npz','input-seal.json','input-seal.npz',
        'reference-seal-index.json','global-reference-gate.json']
    for setting in CONTRACT['settings']:
        for phase in range(24):
            cid=f'{setting}--p{phase:02d}'
            relatives.extend(f'seals/{cid}.{suffix}' for suffix in ('json','npz'))
            for method in ('dop853','rk45'):
                relatives.extend(f'references/payloads/{cid}--reference--{method}/packet.{suffix}' for suffix in ('json','npz'))
    entries=[dict(root=str(OLD),**binding(OLD/'receipt.json',OLD))]
    for relative in relatives:
        item=binding(OLD/relative,OLD)
        if inventory.get(relative)!=item:raise ValueError('Prior receipt membership differs: '+relative)
        entries.append(dict(root=str(OLD),**item))
    if legacy.sha(ROOT/COMPACT)!=COMPACT_SHA or legacy.sha(ROOT/CERTIFICATE)!=CERTIFICATE_SHA:raise ValueError('Final compact qualification binding differs')
    cert=read(ROOT/CERTIFICATE)
    if (cert['source_commit']!=OLD_SOURCE or cert['plan_sha256']!=OLD_PLAN_SHA or cert['raw_receipt']['sha256']!=OLD_RECEIPT_SHA
        or cert['compact_summary']['sha256']!=COMPACT_SHA or cert['raw_summary']['sha256']!=inventory['summary.json']['sha256']
        or cert.get('no_performance_outcomes') is not True):raise ValueError('Metadata-only qualification identity differs')
    q=cert['qualification']
    expected=dict(complete=True,source_valid=True,contexts_planned=48,contexts_qualified=48,scores_planned=2700,scores_complete=2700,
        comparisons_planned=1404,comparisons_qualified=1404,all_primary_setting_prerequisites_qualified=True,
        global_reference_barrier_pass=True,frozen_saved_replay_pass=True,owned_process_cleanup_verified=True,
        reference_evidence_and_gate_pass=True,finite_evidence_and_gate_pass=True,finite_strict_arithmetic_pass=True,
        reference_alternate_strict_arithmetic_pass=False,reference_alternate_failed_comparisons=165,strict_thresholds_unchanged=True)
    if any(type(q.get(k)) is not type(v) or q[k]!=v for k,v in expected.items()):raise ValueError('Full prior numerical qualification required; retain strict alternate failures')
    for audit in cert['audit_bindings']:
        p=Path(audit['path'])
        if p.stat().st_size!=audit['bytes'] or legacy.sha(p)!=audit['sha256']:raise ValueError('Prior audit binding differs')
    manifest=read(OLD/'manifest.json')
    if manifest['source_commit']!=OLD_SOURCE or manifest['plan_sha256']!=OLD_PLAN_SHA or manifest['plan']['phases']!=phases():raise ValueError('Original regression phase provenance differs')
    for relative in (COMPACT,CERTIFICATE):entries.append(dict(root=str(ROOT),**binding(ROOT/relative)))
    if len(entries)!=299:raise ValueError('Exact299 historical input files required')
    return entries


def tasks():
    cells=evaluation.cell_census(phases())
    return [dict(id=c['id'],kind='cell',case=c,method=dict(id='certified-mixture')) for c in cells]


def build_plan():
    specs=tasks();cells=[t['case'] for t in specs]
    slots={c['id']:evaluation.calculation_census(c) for c in cells}
    return dict(**CONTRACT,date='2026-09-17',phases=phases(),cells=cells,
        historical_inputs=old_inputs(),source_files=[binding(ROOT/p) for p in SOURCE_PATHS],
        prior_source_commit=OLD_SOURCE,prior_plan_sha256=OLD_PLAN_SHA,prior_receipt_sha256=OLD_RECEIPT_SHA,
        evaluation_contract=evaluation.contract(),core_contract=core.contract(),
        task_census_sha256=hashlib.sha256(canonical(specs)).hexdigest(),
        calculation_census={cid:dict(count=len(rows),sha256=hashlib.sha256(canonical(rows)).hexdigest()) for cid,rows in slots.items()},
        runtime_provenance=read(ROOT/'docs/EXP-020-runtime-provenance.json'))


def validate(plan,*,bindings=True,reconstruct=True):
    if any(canonical(plan.get(k))!=canonical(v) for k,v in CONTRACT.items()):raise ValueError('Frozen controller contract differs')
    specs=tasks();cells=[t['case'] for t in specs]
    if plan['phases']!=phases() or plan['cells']!=cells or len(cells)!=144 or len({c['id'] for c in cells})!=144:raise ValueError('Full144 regression cells required')
    counts=[len(evaluation.calculation_census(c)) for c in cells]
    if Counter(counts)!={32:132,40:12} or sum(counts)!=4704:raise ValueError('Fixed4704 probability slots required')
    if plan['task_census_sha256']!=hashlib.sha256(canonical(specs)).hexdigest():raise ValueError('Task census differs')
    if [r['path'] for r in plan['source_files']]!=SOURCE_PATHS:raise ValueError('Complete source manifest differs')
    if bindings:
        if plan['historical_inputs']!=old_inputs():raise ValueError('Prior input manifest changed')
        if any(binding(ROOT/r['path'])!=r for r in plan['source_files']):raise ValueError('Source binding changed')
    if reconstruct and plan!=build_plan():raise ValueError('Metadata-only exact plan reconstruction differs')
    return dict(cells=144,classifier_definitions=432,probability_calculations=4704,finite_class_method_rows=1200,neural_integrations=0)


def reference_input(cell):
    cid=cell['context_id']
    return (codec.load_packet(OLD/'seals'/f'{cid}.json'),
        codec.load_packet(OLD/'references/payloads'/f'{cid}--reference--rk45'/'packet.json'))


def reconstruct_definitions(cells):
    result={}
    for cell in cells:
        outer,alternate=reference_input(cell)
        result[cell['id']]=evaluation.seal_cell(cell,outer,alternate)
    return result


def definition_identity(cell,packet):
    if packet.get('id')!=cell['id'] or packet.get('cell')!=cell or packet.get('context_id')!=cell['context_id']:raise ValueError('Classifier cell identity differs')
    if set(packet.get('definitions',{}))!=set(CONTRACT['recipes']):raise ValueError('All3 classifier rows must remain present')


def definition_index(directory,cells,definitions):
    result={}
    for cell in cells:
        ident=cell['id'];packet=definitions[ident];definition_identity(cell,packet)
        p=Path(directory)/'definitions'/f'{ident}.json'
        loaded=codec.load_packet(p);definition_identity(cell,loaded)
        if codec.packet_digest(loaded)!=codec.packet_digest(packet):raise ValueError('Classifier disk readback semantic identity differs')
        result[ident]=dict(binding=binding(p,Path(directory)),array_binding=binding(p.with_suffix('.npz'),Path(directory)),semantic_sha256=codec.packet_digest(loaded),definition_count=3)
    return result


def barrier_packet(directory,cells,definitions):
    if len(cells)!=144 or len(definitions)!=144 or set(definitions)!={c['id'] for c in cells}:raise ValueError('Exact144 definition packets required')
    index=definition_index(directory,cells,definitions)
    path=Path(directory)/'definition-index.json'
    if read(path)!=index:raise ValueError('Definition index readback differs')
    return dict(schema='all-classifiers-before-observations-v1',definition_packets=144,classifier_definitions=432,
        all_declared_definitions_serialized=True,input_index_sha256=legacy.sha(Path(directory)/'input-index.json'),
        definition_index_sha256=legacy.sha(path),passed=True,
        scope='Availability and scientific qualification remain per recipe; this barrier proves the complete declared classifier census is fixed before finite observations.')


def require_barrier(directory,cells,definitions):
    path=Path(directory)/'global-definition-gate.json';expected=barrier_packet(directory,cells,definitions)
    if read(path)!=expected:raise ValueError('Global432-definition barrier differs')
    events=[json.loads(s) for s in (Path(directory)/'events.jsonl').read_text().splitlines()]
    gates=[e for e in events if e['event']=='classifiers_sealed']
    if len(gates)!=1 or gates[0]['sha256']!=legacy.sha(path):raise ValueError('Missing prior global classifier seal event')
    return gates[0]


def finite_observations(plan,directory,cells,definitions):
    require_barrier(directory,cells,definitions)
    wanted={r['path']:r for r in plan['historical_inputs'] if r['root']==str(OLD)}
    for relative in ('analysis.json','analysis.npz'):
        actual=dict(root=str(OLD),**binding(OLD/relative,OLD))
        if actual!=wanted[relative]:raise ValueError('Finite evidence changed')
    saved=codec.load_packet(OLD/'analysis.json')
    scores=saved['scores']
    if not isinstance(scores,dict) or len(scores)!=2700:raise ValueError('Complete2700 original score mapping required')
    result={}
    for cell in cells:
        bymethod={};methods=['rk4-coarse','rk4-fine']+(['dop853-independent'] if cell['phase_index'] in (0,8) else [])
        for method in methods:
            rows=[]
            for first in (-1,1):
                for second in (-1,1):
                    ident=f'{cell["context_id"]}--identity--s1{first:+d}--s2{second:+d}--{method}'
                    row=scores[ident];t=row['task']
                    expected=dict(id=ident,context_id=cell['context_id'],setting=cell['setting'],phase_index=cell['phase_index'],
                        phase_fraction=cell['phase_fraction'],split=cell['split'],arm='identity',first_sign=first,second_sign=second,method=method)
                    if any(t.get(k)!=v for k,v in expected.items()) or row.get('task_id')!=ident or row.get('complete') is not True or row.get('qualified') is not True:raise ValueError('Original finite score identity/qualification differs')
                    rows.append(row)
            bymethod[method]=rows
        result[cell['id']]=bymethod
    return result


def worker(task,context,directory,libraries,parent_pid):
    del libraries
    directory=Path(directory);dest=directory/'payloads'/task['id'];dest.mkdir()
    start=time.monotonic();cpu=time.process_time();cell=task['case']
    row=dict(id=task['id'],kind='cell',case_id=cell['id'],method=task['method']['id'],pid=os.getpid(),
        identity=work.identity(os.getpid()),started_at=legacy.now(),status='failed',qualified=False,rmse=None)
    def cancelled(sig,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned calculation worker interrupted; parent cleanup remains authoritative')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    packet=None;progress=[]
    try:
        sealed=codec.load_packet(context['definition_path']);observed=codec.load_packet(context['observation_path'])
        if codec.packet_digest(sealed)!=context['definition_semantic_sha256'] or codec.packet_digest(observed)!=context['observation_semantic_sha256']:raise ValueError('Worker sealed input identity differs')
        definition_identity(cell,sealed)
        if observed['cell']!=cell:raise ValueError('Worker observation cell differs')
        def record(slot):
            # A returned calculation is durably retained even if the next one times out.
            work.append(dest/'calculation-journal.jsonl',slot);progress.append(slot)
        try:
            packet=evaluation.evaluate_cell(cell,sealed,observed['scores'],progress_fn=record)
        except BaseException as error:
            row['error']=repr(error)
            packet=evaluation.assemble_cell(cell,sealed,observed['scores'],progress)
            packet=dict(packet,worker_interrupted=True,worker_error=repr(error))
        if not row.get('error'):
            row.update(status='complete' if packet.get('complete') is True else packet.get('status','failed'),qualified=packet.get('qualified') is True)
    except BaseException as error:row['error']=repr(error)
    if packet is not None:
        try:
            saved=codec.save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(saved['path']).relative_to(directory)),packet_sha256=saved['sha256'],packet_semantic_sha256=saved['semantic_sha256'])
        except BaseException as error:row.update(status='serialization_failed',qualified=False,error=repr(error))
    row.update(calculations_returned=len(progress),ended_at=legacy.now(),wall_seconds=time.monotonic()-start,
        cpu_seconds=time.process_time()-cpu,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def saved_worker_packet(directory,task,parent,*,include_partial=False):
    expected=dict(id=task['id'],kind='cell',case_id=task['case']['id'],method=task['method']['id'])
    if any(parent.get(k)!=v for k,v in expected.items()):raise ValueError('Parent task identity differs')
    if parent['status']!='complete' and not include_partial:return None
    row=parent
    if row.get('packet_path') is None:
        p=Path(directory)/'records'/f'{task["id"]}.json'
        if not p.exists():return None
        row=read(p)
        if any(row.get(k)!=v for k,v in expected.items()) or row.get('pid')!=parent.get('pid'):raise ValueError('Child prefix identity differs')
        if row.get('packet_path') is None:return None
    path=Path(directory)/f'payloads/{task["id"]}/packet.json'
    if row['packet_path']!=str(path.relative_to(directory)) or path.resolve()!=path or legacy.sha(path)!=row['packet_sha256']:raise ValueError('Worker packet byte binding differs')
    packet=codec.load_packet(path)
    if codec.packet_digest(packet)!=row['packet_semantic_sha256']:raise ValueError('Worker packet semantic binding differs')
    if packet.get('cell')!=task['case'] or packet.get('id')!=task['id']:raise ValueError('Worker result cell differs')
    if parent['status']=='complete' and packet.get('complete') is not True:raise ValueError('Parent completion lacks complete result')
    return packet


def collect(directory,specs,rows):
    byid={r['id']:r for r in rows};results={};partial=[]
    for task in specs:
        row=byid[task['id']];packet=saved_worker_packet(Path(directory)/'cells',task,row,include_partial=True)
        journal=Path(directory)/'cells/payloads'/task['id']/'calculation-journal.jsonl'
        slots=[json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
        wanted={s['id'] for s in evaluation.calculation_census(task['case'])}
        if len({s['id'] for s in slots})!=len(slots) or any(s['id'] not in wanted for s in slots):raise ValueError('Calculation prefix census differs')
        if row['status']=='complete':
            if packet is None:raise ValueError('Completed cell lacks packet')
            codec.compare(packet.get('calculations'),normalized(slots),'complete calculation journal')
            if len(slots)!=len(wanted):raise ValueError('Complete cell lacks full calculation journal')
            results[task['id']]=packet
        elif packet is not None or slots:
            partial.append(dict(id=task['id'],parent_status=row['status'],qualified=False,
                packet_present=packet is not None,packet_complete=None if packet is None else packet.get('complete'),
                returned_calculation_ids=[s['id'] for s in slots],planned_calculations=len(wanted)))
    return results,partial


def runtime_check():
    if any(importlib.metadata.version(k)!=v for k,v in CONTRACT['required_packages'].items()) or flint.__FLINT_VERSION__!=CONTRACT['flint_native']:raise ValueError('Pinned numerical runtime differs')


def run(plan_path,data_root,attempt,expected_head):
    plan_path=Path(plan_path).resolve();plan=read(plan_path);validate(plan);provenance=source_guard(plan,plan_path,expected_head)
    runtime_check()
    if any(os.environ.get(k)!='1' for k in legacy.THREAD_ENV):raise ValueError('Six single numerical thread settings required')
    directory=external_attempt(data_root,attempt);start=time.monotonic();deadline=start+CONTRACT['producer_seconds'];specs=tasks();cells=[t['case'] for t in specs]
    rows=[];definitions={};results={};partial=[];summary=None;error=None;postflight=None;stage='setup';times={};barrier=None
    manifest=dict(**provenance,experiment_id='EXP-020',plan=plan,plan_sha256=legacy.sha(plan_path),producer_pid=os.getpid(),producer_identity=work.identity(os.getpid()),
        started_at=legacy.now(),executable=sys.executable,argv=list(sys.argv),cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},flint_native=flint.__FLINT_VERSION__,thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV})
    work.write(directory/'manifest.json',manifest);work.write(directory/'planned-ledger.json',specs)
    work.write(directory/'planned-calculations.json',{c['id']:evaluation.calculation_census(c) for c in cells})
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    def stop(sig,frame):
        if sig!=signal.SIGALRM or time.monotonic()>=deadline:raise ProducerStop('Producer deadline/interruption')
        try:resource_check(directory)
        except Exception as exc:raise ProducerStop('Resource limit: '+repr(exc)) from exc
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,1.,1.)
    try:
        resource_check(directory,launch=True);stage='classifier_sealing';mark=time.monotonic()
        work.write(directory/'input-index.json',dict(historical_inputs=plan['historical_inputs'],no_finite_observations_loaded=True))
        event(directory,'reference_inputs_bound',sha256=legacy.sha(directory/'input-index.json'))
        for cell in cells:
            packet=evaluation.seal_cell(cell,*reference_input(cell));definition_identity(cell,packet)
            codec.save_packet(directory/'definitions',cell['id'],packet)
            loaded=codec.load_packet(directory/'definitions'/f'{cell["id"]}.json');codec.compare(loaded,normalized(packet),'classifier readback')
            definitions[cell['id']]=loaded
        index=definition_index(directory,cells,definitions);work.write(directory/'definition-index.json',index)
        expected=barrier_packet(directory,cells,definitions);work.write(directory/'global-definition-gate.json',expected)
        barrier=read(directory/'global-definition-gate.json')
        if barrier!=expected:raise ValueError('Global classifier gate readback differs')
        event(directory,'classifiers_sealed',sha256=legacy.sha(directory/'global-definition-gate.json'),classifier_definitions=432)
        times[stage]=time.monotonic()-mark
        stage='finite_observation_access';mark=time.monotonic();require_barrier(directory,cells,definitions)
        event(directory,'finite_evidence_read_started',analysis_sha256=legacy.sha(OLD/'analysis.json'))
        observations=finite_observations(plan,directory,cells,definitions);contexts={}
        for cell in cells:
            ident=cell['id'];saved=codec.save_packet(directory/'observations',ident,dict(cell=cell,scores=observations[ident]))
            loaded=codec.load_packet(saved['path']);codec.compare(loaded,normalized(dict(cell=cell,scores=observations[ident])),'observation readback')
            contexts[ident]=dict(definition_path=str(directory/'definitions'/f'{ident}.json'),definition_semantic_sha256=codec.packet_digest(definitions[ident]),
                observation_path=saved['path'],observation_semantic_sha256=saved['semantic_sha256'])
        work.write(directory/'observation-index.json',contexts)
        if read(directory/'observation-index.json')!=contexts:raise ValueError('Observation index readback differs')
        event(directory,'observations_sealed',sha256=legacy.sha(directory/'observation-index.json'),finite_class_method_rows=1200)
        observations=None;gc.collect();times[stage]=time.monotonic()-mark
        stage='cells';mark=time.monotonic();resource_check(directory,launch=True);event(directory,'cells_started',planned=144)
        rows=runtime.execute(specs,contexts,directory/'cells',{},workers=CONTRACT['workers'],deadline=deadline,per_task=CONTRACT['per_cell_seconds'],worker_fn=worker)
        times[stage]=time.monotonic()-mark
        stage='aggregation';mark=time.monotonic();results,partial=collect(directory,specs,rows)
        summary=evaluation.summarize(cells,definitions,results);times[stage]=time.monotonic()-mark
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def sealstop(sig,frame):raise ProducerStop('Sealing grace exhausted; existing evidence retained')
        signal.signal(signal.SIGALRM,sealstop);signal.setitimer(signal.ITIMER_REAL,CONTRACT['sealing_seconds'])
        try:
            if not rows and (directory/'cells/terminal-records.json').exists():rows=read(directory/'cells/terminal-records.json')
            rows=codec.fill_terminal(specs,rows,'not_started_producer_or_prerequisite_failure');work.write(directory/'all-terminal-records.json',rows)
            collection_error=None
            try:
                # A signal may follow an exclusive packet save but precede the in-memory assignment.
                definitions=scan_definitions(directory,cells)
                results,partial=collect(directory,specs,rows)
            except Exception as exc:collection_error=repr(exc)
            if summary is None:
                try:summary=evaluation.summarize(cells,definitions,results)
                except Exception as exc:summary=dict(complete=False,qualified=False,status='unavailable',error=repr(exc),planned_cells=144,planned_calculations=4704)
            codec.save_packet(directory,'analysis',dict(summary=summary,partial_worker_packets=partial,collection_error=collection_error))
            try:
                validate(plan)
                if legacy.git('rev-parse','HEAD')!=expected_head or legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Frozen checkout changed')
            except Exception as exc:postflight=repr(exc)
            work.write(directory/'postflight.json',dict(source_bindings_unchanged=postflight is None,error=postflight))
            complete=error is None and postflight is None and collection_error is None and barrier is not None and all(r['status']=='complete' for r in rows) and summary.get('complete') is True
            qualified=complete and summary.get('qualified') is True
            final=dict(experiment_id='EXP-020',complete=bool(complete),qualified=bool(qualified),source_valid=postflight is None,error=error,postflight_error=postflight,
                collection_error=collection_error,status_counts=dict(Counter(r['status'] for r in rows)),analysis=summary,
                stage_wall_seconds=times,total_wall_seconds=time.monotonic()-start,global_definition_gate=barrier,
                original_studies_requalified=False,neural_integrations=0,reused_regression_phases=True,fresh_holdout=False)
            work.write(directory/'summary.json',final);event(directory,'producer_finished',complete=complete,qualified=qualified)
            artifacts=inventory(directory);receipt=dict(**provenance,experiment_id='EXP-020',directory=str(directory),plan_sha256=legacy.sha(plan_path),
                status='complete' if complete else 'incomplete_or_invalid',qualified=bool(qualified),artifacts=artifacts,artifact_bytes=sum(r['bytes'] for r in artifacts),sealed_at=legacy.now())
            work.write(directory/'receipt.json',receipt)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return dict(directory=str(directory),receipt_sha256=legacy.sha(directory/'receipt.json'),summary=final)
def scan_definitions(directory,cells):
    byid={c['id']:c for c in cells};result={};base=Path(directory)/'definitions'
    for path in sorted(base.glob('*.json')) if base.exists() else []:
        if path.stem not in byid:raise ValueError('Unplanned classifier packet')
        packet=codec.load_packet(path);definition_identity(byid[path.stem],packet);result[path.stem]=packet
    return result


def verify_lifecycle(directory,specs,rows):
    expected={t['id']:t for t in specs}
    if len(rows)!=144 or len({r['id'] for r in rows})!=144 or set(expected)!={r['id'] for r in rows}:raise ValueError('Full144 terminal ledger required')
    for row in rows:
        task=expected[row['id']]
        if any(row.get(k)!=v for k,v in dict(kind='cell',case_id=task['case']['id'],method=task['method']['id']).items()):raise ValueError('Terminal identity differs')
    stage=Path(directory)/'cells'
    if not stage.exists():
        if any(r.get('pid') is not None or r['status']=='complete' for r in rows):raise ValueError('Worker row lacks owned stage')
        return []
    cleanup=read(stage/'cleanup.json')
    if cleanup.get('all_registered_workers_joined') is not True or cleanup.get('all_original_identities_absent') is not True:raise ValueError('Owned cleanup incomplete')
    procs=[json.loads(x) for x in (stage/'processes.jsonl').read_text().splitlines()] if (stage/'processes.jsonl').exists() else []
    if len(procs)!=len({p['task_id'] for p in procs}) or len(procs)!=len(cleanup['workers']):raise ValueError('Owned census differs')
    for a,b in zip(procs,cleanup['workers'],strict=True):
        if any(b.get(k)!=v for k,v in a.items()) or b.get('original_identity_absent') is not True:raise ValueError('Owned identity/cleanup differs')
    inner=read(stage/'terminal-records.json');journal=[json.loads(x) for x in (stage/'journal.jsonl').read_text().splitlines()] if (stage/'journal.jsonl').exists() else []
    if inner!=journal or len(inner)!=len({r['id'] for r in inner}):raise ValueError('Owned terminal/journal mismatch or duplicate')
    outer={r['id']:r for r in rows};registered={p['task_id']:p for p in procs};index={r['id']:r for r in inner}
    if not set(index).issubset(outer) or not set(registered).issubset(index):raise ValueError('Executed task census differs')
    for row in rows:
        if row.get('pid') is not None or row['status']=='complete':
            if index.get(row['id'])!=row or row['id'] not in registered or registered[row['id']]['pid']!=row['pid']:raise ValueError('Authoritative row lacks exact ownership/journal')
    for row in inner:
        if outer[row['id']]!=row:raise ValueError('Stage/outer terminal differs')
        if row.get('packet_path') is not None or row['status']=='complete':
            if read(stage/'records'/f'{row["id"]}.json')!=row:raise ValueError('Worker record differs')
    return procs


def verify(directory):
    """Full saved decoder replay; missing probability slots are never computed."""
    runtime_check()
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=inventory(directory) or receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):raise ValueError('Immutable attempt inventory differs')
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['experiment_id']!='EXP-020':raise ValueError('Attempt source identity differs')
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+PLAN_REL])
    if hashlib.sha256(committed).hexdigest()!=receipt['plan_sha256'] or json.loads(committed)!=plan:raise ValueError('Committed plan differs')
    for item in plan['source_files']+[manifest['review']]:
        b=subprocess.check_output(['git','-C',str(ROOT),'show',receipt['source_commit']+':'+item['path']])
        if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('Committed source/review differs')
    validate(plan);specs=tasks();cells=[t['case'] for t in specs];rows=read(directory/'all-terminal-records.json')
    if read(directory/'planned-ledger.json')!=specs or read(directory/'planned-calculations.json')!={c['id']:evaluation.calculation_census(c) for c in cells}:raise ValueError('Complete fixed census differs')
    procs=verify_lifecycle(directory,specs,rows)
    events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
    def event_rows(name):return [e for e in events if e['event']==name]
    if any(b['monotonic_ns']<=a['monotonic_ns'] for a,b in zip(events,events[1:])):raise ValueError('Nonmonotone lifecycle events')
    for name in ('reference_inputs_bound','classifiers_sealed','finite_evidence_read_started','observations_sealed','cells_started','producer_finished'):
        if len(event_rows(name))>1:raise ValueError('Duplicate lifecycle stage '+name)
    definitions=scan_definitions(directory,cells)
    input_path=directory/'input-index.json'
    if input_path.exists():
        if read(input_path)!=dict(historical_inputs=plan['historical_inputs'],no_finite_observations_loaded=True):raise ValueError('Reference-only input index differs')
        declared=event_rows('reference_inputs_bound')
        if declared and declared[0]['sha256']!=legacy.sha(input_path):raise ValueError('Input index event differs')
        if definitions and len(declared)!=1:raise ValueError('Definition construction lacks bound input event')
    elif definitions or procs:raise ValueError('Prepared objects lack bound inputs')
    for cell in cells:
        if cell['id'] in definitions:
            replay=evaluation.seal_cell(cell,*reference_input(cell))
            codec.compare(definitions[cell['id']],normalized(replay),'classifier.'+cell['id'])
    gate_path=directory/'global-definition-gate.json';barrier=None
    if gate_path.exists():
        barrier=read(gate_path)
        if barrier!=barrier_packet(directory,cells,definitions):raise ValueError('Saved global barrier differs')
        if event_rows('classifiers_sealed') and event_rows('classifiers_sealed')[0]['sha256']!=legacy.sha(gate_path):raise ValueError('Global barrier event binding differs')
    elif event_rows('classifiers_sealed'):raise ValueError('Classifier event lacks gate')
    observation_paths=list((directory/'observations').glob('*.json')) if (directory/'observations').exists() else []
    need_observations=bool(observation_paths or procs or event_rows('finite_evidence_read_started'))
    observations={}
    if need_observations:
        boundary=require_barrier(directory,cells,definitions)
        access=event_rows('finite_evidence_read_started')
        if len(access)!=1 or access[0]['monotonic_ns']<=boundary['monotonic_ns'] or access[0]['analysis_sha256']!=legacy.sha(OLD/'analysis.json'):raise ValueError('Finite data access lacks prior complete classifier seal')
        observations=finite_observations(plan,directory,cells,definitions)
        byid={c['id']:c for c in cells}
        for path in observation_paths:
            if path.stem not in byid:raise ValueError('Unplanned observation packet')
            codec.compare(codec.load_packet(path),normalized(dict(cell=byid[path.stem],scores=observations[path.stem])),'observations.'+path.stem)
    index_path=directory/'observation-index.json'
    if index_path.exists():
        index=read(index_path)
        if set(index)!={c['id'] for c in cells} or len(observation_paths)!=144:raise ValueError('Full observation index required')
        for cell in cells:
            ident=cell['id'];expected=dict(definition_path=str(directory/'definitions'/f'{ident}.json'),definition_semantic_sha256=codec.packet_digest(definitions[ident]),
                observation_path=str(directory/'observations'/f'{ident}.json'),observation_semantic_sha256=codec.packet_digest(codec.load_packet(directory/'observations'/f'{ident}.json')))
            if index[ident]!=expected:raise ValueError('Observation index binding differs')
    if procs:
        sealed=event_rows('observations_sealed');began=event_rows('cells_started')
        if len(sealed)!=1 or len(began)!=1 or not index_path.exists() or sealed[0]['sha256']!=legacy.sha(index_path) or began[0]['planned']!=144:raise ValueError('Workers lack complete observation seal')
        if not (event_rows('classifiers_sealed')[0]['monotonic_ns']<event_rows('finite_evidence_read_started')[0]['monotonic_ns']<sealed[0]['monotonic_ns']<began[0]['monotonic_ns']):raise ValueError('Classifier/observation/worker chronology differs')
        if any(p['started_at']<sealed[0]['at'] for p in procs):raise ValueError('Worker preceded sealed observations')
    results,partial=collect(directory,specs,rows);byid={c['id']:c for c in cells};replayed_cells=0;replayed_prefix=0
    for task in specs:
        ident=task['id'];cell=task['case']
        if ident in results:
            replay=evaluation.evaluate_cell(cell,definitions[ident],observations[ident])
            codec.compare(results[ident],normalized(replay),'cell.'+ident);replayed_cells+=1
        else:
            path=directory/'cells/payloads'/ident/'calculation-journal.jsonl'
            if path.exists():
                wanted={s['id']:s for s in evaluation.calculation_census(cell)}
                for line in path.read_text().splitlines():
                    record=json.loads(line);replay=evaluation.evaluate_calculation(cell,definitions[ident],observations[ident],wanted[record['id']])
                    codec.compare(record,normalized(replay),'prefix.'+record['id']);replayed_prefix+=1
    saved=codec.load_packet(directory/'analysis.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    if saved['collection_error'] is not None:raise ValueError('Producer retained invalid collection; no verified numerical claim')
    if saved['summary'].get('status')=='unavailable':
        if not (directory/'producer-error.json').exists():raise ValueError('Missing aggregation requires retained producer failure')
    else:
        replay=evaluation.summarize(cells,definitions,results)
        codec.compare(saved['summary'],normalized(replay),'summary')
    codec.compare(saved['partial_worker_packets'],normalized(partial),'retained partials')
    codec.compare(summary['analysis'],work.baseline_engine.safe(saved['summary']),'summary JSON')
    error=read(directory/'producer-error.json')['error'] if (directory/'producer-error.json').exists() else None
    if summary['error']!=error or summary['postflight_error']!=post['error'] or post['source_bindings_unchanged'] is not (post['error'] is None) or summary['source_valid'] is not (post['error'] is None):raise ValueError('Failure/source status differs')
    if summary['global_definition_gate']!=barrier or summary['status_counts']!=dict(Counter(r['status'] for r in rows)):raise ValueError('Global barrier/status summary differs')
    complete=error is None and post['error'] is None and barrier is not None and all(r['status']=='complete' for r in rows) and saved['summary'].get('complete') is True
    qualified=complete and saved['summary'].get('qualified') is True
    if summary['complete'] is not bool(complete) or summary['qualified'] is not bool(qualified) or receipt['qualified'] is not bool(qualified) or receipt['status']!=('complete' if complete else 'incomplete_or_invalid'):raise ValueError('Completion/qualification conclusion differs')
    return dict(complete=bool(complete),qualified=bool(qualified),planned_cells=144,planned_calculations=4704,replayed_cells=replayed_cells,
        replayed_returned_partial_calculations=replayed_prefix,receipt_sha256=legacy.sha(directory/'receipt.json'),neural_integrations=0,
        all_original_owned_identities_absent_now=all(work.identity(p['pid']) is None or work.identity(p['pid'])!=p['identity'] for p in procs))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('validate');p.add_argument('--plan',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('run');p.add_argument('--plan',default=str(ROOT/PLAN_REL));p.add_argument('--data-root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--expected-head',required=True)
    p=sub.add_parser('verify');p.add_argument('--attempt-dir',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();work.write(Path(args.output),plan);answer=validate(plan)
    elif args.command=='validate':answer=validate(read(args.plan))
    elif args.command=='verify':answer=verify(args.attempt_dir)
    else:
        result=run(args.plan,args.data_root,args.attempt,args.expected_head);s=result['summary']
        answer=dict(directory=result['directory'],receipt_sha256=result['receipt_sha256'],complete=s['complete'],qualified=s['qualified'],
            status_counts=s['status_counts'],error=s['error'],primary=s['analysis'].get('primary'))
    try:print(json.dumps(work.baseline_engine.safe(answer),sort_keys=True,allow_nan=False))
    except OSError:pass

if __name__=='__main__':main()
