"""Dispatch only the remaining fixed Qwen P8 derivative shards; never certify.

No plan is generated here. A separately successful, independently audited first
shard is mandatory. Each unchanged launch.py retains its own snapshot/receipt.
Linux /proc + pidfds are required. Sampled RSS is a watchdog, not a hard cap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OWNED = Path('/data2/neural-shrimp-data/codex/llm-p8-certificate')
SCHEMA = 'codex-fixed-derivative-campaign-v1'
RECOVERY_SCHEMA = 'codex-fixed-derivative-campaign-recovery-v2'
AUDIT_SCHEMA = 'codex-first-final-shard-measured-audit-v1'
ASSIGNMENTS = [(p, s) for p in range(8) for s in range(0, 896, 64)]
ENVELOPE = dict(cpu_threads=1, nice=10, memory_gib=18, timeout_seconds=3600)
EVALUATOR_FILES = tuple('codex/certify-llm/'+n+'.py' for n in (
    'derivative_shard', 'streamed_head', 'ball_jet', 'ball_qwen', 'ball_linear',
    'cache_io', 'phase_domain', 'derivative_coverage', 'shooting_gates', 'proof_gates')) + ('codex/crosscheck/qwen_map.py',)
FILES = EVALUATOR_FILES + ('codex/certify-llm/launch.py', 'codex/certify-llm/derivative_campaign.py')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def same(a, b, label):
    if canonical(a) != canonical(b): raise ValueError(label+' differs')


def read(path, maximum=16*1024**2):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path or path.stat().st_size > maximum:
        raise ValueError('original bounded regular JSON required')
    def unique(pairs):
        answer = {}
        for k, v in pairs:
            if k in answer: raise ValueError('duplicate JSON key')
            answer[k] = v
        return answer
    return json.loads(path.read_bytes(), object_pairs_hook=unique)


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as f:
        f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def digest(value, length=64):
    if type(value) is not str or len(value) != length or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('literal lowercase source/hash required')


def integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high: raise ValueError(label+' outside fixed bounds')


def name(value):
    if type(value) is not str or not value or value.startswith('.') or Path(value).name != value or '/' in value:
        raise ValueError('unique direct-child attempt name required')
    return value


def pin(record, *, parent=None):
    if type(record) is not dict or set(record) != {'path', 'sha256'}: raise ValueError('exact external file pin required')
    digest(record['sha256']); p = Path(record['path'])
    if not p.is_absolute() or p.resolve() != p or p.is_symlink() or not p.is_file(): raise ValueError('original pinned file required')
    if parent is not None and not p.is_relative_to(parent): raise ValueError('file outside owned evidence')
    if sha(p) != record['sha256']: raise ValueError('original pinned bytes differ')
    return p


def common(plan):
    return {k: v for k, v in plan.items() if k not in ('phase_index', 'start_column', 'purpose', 'basis')}


def shard_plan(record, repo=REPO):
    if set(record) != {'name', 'sha256'} or name(record['name']) != record['name'] or not record['name'].endswith('.json'):
        raise ValueError('committed shard plan basename/hash required')
    digest(record['sha256']); p = repo/'codex/certify-llm'/record['name']
    if sha(p) != record['sha256']: raise ValueError('shard plan bytes differ')
    return read(p)


def validate(plan, repo=REPO, owned=OWNED):
    fields = {'schema', 'status', 'execution_checkout', 'first', 'common', 'assignments', 'sources', 'resources'}
    recovery = type(plan) is dict and plan.get('schema') == RECOVERY_SCHEMA
    if recovery: fields.add('recovery')
    if type(plan) is not dict or set(plan) != fields or plan['schema'] not in (SCHEMA, RECOVERY_SCHEMA) or plan['status'] != 'reviewed_measured':
        raise ValueError('complete measured reviewed campaign plan required; no provisional launch')
    same(plan['execution_checkout'], str(repo.resolve()), 'dedicated frozen execution checkout')
    if set(plan['sources']) != set(FILES): raise ValueError('complete dispatcher/launcher/evaluator source pins required')
    for file, h in plan['sources'].items():
        digest(h)
        if sha(repo/file) != h: raise ValueError('current pinned source differs: '+file)
    c = plan['common']
    fields = {'schema', 'definition_path', 'definition_sha256', 'map_sha256', 'backend_sha256', 'source_sha256',
              'evaluator_sha256', 'prefix_attempt', 'prefix_receipt_sha256', 'prefix_map_source_sha256',
              'prefix_cache_sha256', 'precision', 'token_chunk', 'envelope'}
    if type(c) is not dict or set(c) != fields or c['schema'] != 'codex-llm-final-derivative-shard-plan-v1':
        raise ValueError('unchanged complete derivative common plan required')
    same(c['envelope'], ENVELOPE, 'CPU1/nice10/18GiB/3600s'); same(c['precision'], 1024, 'precision'); same(c['token_chunk'], 1024, 'token chunk')
    same(c['source_sha256'], {f: plan['sources'][f] for f in EVALUATOR_FILES}, 'all evaluator sources')
    same(c['evaluator_sha256'], hashlib.sha256(canonical(dict(schema='codex-derivative-evaluator-source-bundle-v1', source_sha256=c['source_sha256'])).encode()).hexdigest(), 'source bundle identity')
    for k in ('definition_sha256', 'map_sha256', 'backend_sha256', 'evaluator_sha256', 'prefix_receipt_sha256', 'prefix_map_source_sha256', 'prefix_cache_sha256'): digest(c[k])
    pin({'path': c['definition_path'], 'sha256': c['definition_sha256']}, parent=owned)
    prefix = Path(c['prefix_attempt'])
    if prefix.parent != owned or prefix.resolve() != prefix or not prefix.is_dir(): raise ValueError('original owned prefix required')
    first = plan['first']
    if set(first) != {'attempt', 'receipt_sha256', 'plan', 'audit', 'phase_index', 'start_column'}: raise ValueError('independent first-shard authority required')
    name(first['attempt']); digest(first['receipt_sha256'])
    assignment = (first['phase_index'], first['start_column'])
    if any(type(v) is not int for v in assignment) or assignment not in ASSIGNMENTS: raise ValueError('first assignment invalid')
    old = shard_plan(first['plan'], repo)
    same(common(old), c, 'first original common plan'); same((old['phase_index'], old['start_column']), assignment, 'first plan assignment')
    rows = plan['assignments']; expected = [x for x in ASSIGNMENTS if x != assignment]
    if recovery:
        history = recovery_metadata(plan, repo, owned)
        retained = {pair(r['row']) for r in history['completed']}
        expected = [x for x in expected if x not in retained]
    if type(rows) is not list or len(rows) != (94 if recovery else 111): raise ValueError('exact remaining assignment count required')
    attempts = {first['attempt']}; plan_names = {first['plan']['name']}
    for row, assignment_pair in zip(rows, expected):
        if set(row) != {'attempt', 'phase_index', 'start_column', 'plan'}: raise ValueError('complete canonical row required')
        if any(type(row[k]) is not int for k in ('phase_index', 'start_column')): raise ValueError('integer assignment required')
        same((row['phase_index'], row['start_column']), assignment_pair, 'canonical remaining assignment')
        name(row['attempt'])
        if row['attempt'] in attempts or row['plan']['name'] in plan_names: raise ValueError('attempt/plan reused')
        attempts.add(row['attempt']); plan_names.add(row['plan']['name'])
        if (owned/row['attempt']).exists(): raise ValueError('all new attempts must be absent; no retry/resume')
        new = shard_plan(row['plan'], repo)
        same(common(new), c, 'same frozen map/domain/backend/prefix/evaluator')
        same((new['phase_index'], new['start_column']), assignment_pair, 'committed assignment')
    if recovery: recovery_partition(plan, history, repo)
    r = plan['resources']
    if set(r) != {'concurrency', 'cpu_total', 'aggregate_rss_bytes', 'minimum_available_bytes', 'timeout_seconds',
                  'registration_seconds', 'receipt_seconds', 'sample_seconds', 'raw_bytes', 'minimum_disk_bytes',
                  'time_headroom_milli', 'rss_headroom_milli', 'storage_headroom_milli'}:
        raise ValueError('complete fixed resource/watchdog card required')
    integer(r['concurrency'], 1, 10, 'concurrency')
    integer(r['cpu_total'], r['concurrency']+2, 12, 'workers plus registration/controller CPU reservation')
    integer(r['timeout_seconds'], 1, 432000, 'finite whole campaign deadline')
    for k in ('registration_seconds', 'receipt_seconds'): integer(r[k], 1, 120, k)
    integer(r['sample_seconds'], 1, 5, 'sampling interval')
    for k in ('aggregate_rss_bytes', 'minimum_available_bytes', 'raw_bytes', 'minimum_disk_bytes'): integer(r[k], 1, 2**50, k)
    for k in ('time_headroom_milli', 'rss_headroom_milli', 'storage_headroom_milli'): integer(r[k], 1000, 4000, k)
    return plan


def inventory(attempt, expected_receipt=None):
    """Whole byte inventory only; no derivative-array parsing or evaluation."""
    receipt_path = attempt/'receipt.json'
    if expected_receipt is not None and sha(receipt_path) != expected_receipt: raise ValueError('receipt pin differs')
    receipt = read(receipt_path); files = {}
    for row in receipt['artifacts']:
        if set(row) != {'path', 'bytes', 'sha256'}: raise ValueError('inventory row malformed')
        rel = Path(row['path']); path = attempt/rel; digest(row['sha256'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in files or str(rel) == 'receipt.json' or type(row['bytes']) is not int:
            raise ValueError('inventory path/size invalid')
        if path.is_symlink() or path.resolve() != path or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('original artifact bytes differ')
        files[str(rel)] = path
    observed = set()
    for path in attempt.rglob('*'):
        if path.is_symlink(): raise ValueError('symlink in attempt')
        if path.is_file(): observed.add(str(path.relative_to(attempt)))
    if observed != set(files)|{'receipt.json'}: raise ValueError('whole attempt inventory differs')
    return receipt, files


def completed(attempt, row, c, source=None, expected_receipt=None):
    receipt, files = inventory(attempt, expected_receipt)
    if type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0 or receipt.get('timed_out') is not False:
        raise ValueError('shard failed or timed out; no retry')
    if source is not None: same(receipt['source_commit'], source, 'captured campaign source')
    manifest = read(files['manifest.json']); inputs = read(files['run/inputs.json']); status = read(files['run/status.json'])
    same(manifest['source_commit'], receipt['source_commit'], 'source'); same(inputs['source_commit'], receipt['source_commit'], 'input source')
    same(inputs['plan_sha256'], row['plan']['sha256'], 'actual committed shard plan')
    same(sha(files['source/codex/certify-llm/'+row['plan']['name']]), row['plan']['sha256'], 'archived actual plan bytes')
    same(common(inputs['plan']), c, 'actual same domain/evaluator/backend')
    same([inputs['plan']['phase_index'], inputs['plan']['start_column']], [row['phase_index'], row['start_column']], 'actual assignment')
    for k, v in {'status': 'final_derivative_shard_completed', 'certificate_passed': False, 'phase_index': row['phase_index'],
                 'direction_indices': list(range(row['start_column'], row['start_column']+64)), 'columns_completed': 64,
                 'full_state_dimension': 896, 'full_vocabulary': 151936, 'precision': 1024}.items(): same(status[k], v, 'complete shard '+k)
    same(status['shard_sha256'], sha(files['run/derivative-shard.json']), 'complete saved shard')
    for rel, h in c['source_sha256'].items():
        if sha(files['source/'+rel]) != h: raise ValueError('actual captured evaluator differs')
    return dict(receipt_sha256=sha(attempt/'receipt.json'), source_commit=receipt['source_commit'],
                elapsed_seconds=receipt['elapsed_seconds'], max_rss_kib=status['max_rss_kib'],
                bytes=sum(p.stat().st_size for p in files.values())+(attempt/'receipt.json').stat().st_size)


def pair(row):
    return row['phase_index'], row['start_column']


def recovery_metadata(plan, repo=REPO, owned=OWNED):
    """Authenticate the closed controller's small metadata, without target decode.

    The separate authority pass below authenticates every retained/failed shard.
    The closure audit is externally reviewed evidence, not a proof from hashes.
    """
    recovery = plan['recovery']
    if set(recovery) != {'attempt', 'source_commit', 'receipt_sha256', 'closure', 'closure_source'}:
        raise ValueError('complete original recovery authority required')
    name(recovery['attempt']); digest(recovery['source_commit'], 40); digest(recovery['receipt_sha256'])
    prior = owned/recovery['attempt']
    receipt, files = inventory(prior, recovery['receipt_sha256'])
    old = read(files['original-plan.json']); admission = read(files['admission.json'])
    start = read(files['admission-started.json']); state = read(files['run/state.json'])
    status = read(files['status.json']); closure = read(pin(recovery['closure'], parent=owned))
    pin(recovery['closure_source'], parent=owned)
    if old.get('schema') != SCHEMA or old.get('status') != 'reviewed_measured':
        raise ValueError('only original closed v1 campaign may be recovered')
    same(admission['plan'], old, 'original admitted plan')
    for value in (admission, start, closure):
        same(value['source_commit'], recovery['source_commit'], 'original campaign source')
        same(value['plan_sha256'], sha(files['original-plan.json']), 'original campaign plan bytes')
    same(closure['original_receipt_sha256'], recovery['receipt_sha256'], 'closure receipt')
    same(closure['controller_artifacts_verified'], len(files), 'closure inventory count')
    same(closure['status'], status, 'closure status')
    same(read(files['run/result.json']), status, 'inner/outer failed status')
    same(receipt['status'], 'campaign_failed', 'prior failed campaign')
    for k, v in dict(status='campaign_failed', completed=17, started=21, undispatched=90,
                     uncertain_ownership=[], certificate_passed=False).items():
        same(status[k], v, 'fixed closed history '+k)
    same(state['uncertain_ownership'], [], 'no prior unresolved ownership')
    same(state['stop_reason'], status['stop_reason'], 'retained stop reason')
    same(closure['all43_original_pid_and_session_groups_absent'], True, 'independent closure')
    same(closure['tool_reaped_exit'], 1, 'controller reaped failure')
    same(old['first'], plan['first'], 'same independently audited first shard')
    same(old['common'], plan['common'], 'unchanged exact evaluator/domain/backend/prefix')
    same({k:v for k,v in old['resources'].items() if k!='timeout_seconds'},
         {k:v for k,v in plan['resources'].items() if k!='timeout_seconds'}, 'unchanged recovery resource envelope')
    if type(plan['resources']['timeout_seconds']) is not int or not 1<=plan['resources']['timeout_seconds']<=old['resources']['timeout_seconds']:
        raise ValueError('recovery deadline must not enlarge original envelope')
    same({f: old['sources'][f] for f in EVALUATOR_FILES}, plan['common']['source_sha256'], 'original native sources')
    if set(old['sources']) != set(FILES): raise ValueError('original source closure differs')
    expected = [a for a in ASSIGNMENTS if a != pair(plan['first'])]
    if len(old['assignments']) != 111 or [pair(r) for r in old['assignments']] != expected:
        raise ValueError('original111 canonical assignments differ')
    old_attempts = [r['attempt'] for r in old['assignments']]+[old['first']['attempt']]
    if len(set(old_attempts)) != 112: raise ValueError('duplicate original attempt')
    for row in old['assignments']:
        if set(row) != {'attempt','phase_index','start_column','plan'}: raise ValueError('original row malformed')
        name(row['attempt'])
        original = shard_plan(row['plan'], repo)
        same(common(original), plan['common'], 'original row domain/evaluator')
        same(pair(original), pair(row), 'original row assignment')
    rows = state['rows']
    if len(rows) != 21: raise ValueError('all21 original dispatched histories required')
    completed_rows = []; failed_rows = []
    identities = [dict(start['dispatcher_identity'], role='controller')]
    for row, original in zip(rows, old['assignments'][:21]):
        same(row['row'], original, 'literal dispatched prefix')
        same(row['registered'], True, 'original registration')
        same(row['worker_termination_verified'], True, 'original worker termination')
        for role in ('launcher_identity','worker_identity'):
            identities.append(dict(row[role], role=role, attempt=original['attempt']))
        if row['state'] == 'completed':
            same(row['launcher_exit'], 0, 'completed launcher exit'); completed_rows.append(row)
        elif row['state'] == 'failed':
            same(row['launcher_exit'], 1, 'interrupted launcher exit'); failed_rows.append(row)
        else: raise ValueError('unclosed dispatched state')
    if len(completed_rows) != 17 or len(failed_rows) != 4: raise ValueError('exact17 retained/four interrupted required')
    same(closure['identities'], identities, 'all43 literal birth-bound closure identities')
    if len(closure['failed_shards']) != 4: raise ValueError('all four failed receipt pins required')
    for row, failed in zip(failed_rows, closure['failed_shards']):
        for k in ('attempt','plan'): same(failed[k], row['row'][k], 'closed failure identity')
        same(failed['worker_termination_verified'], True, 'closed failure worker')
        same(failed['launcher_exit'], 1, 'closed failure launcher')
    existing_bytes=sum(p.stat().st_size for p in files.values())+(prior/'receipt.json').stat().st_size
    return dict(old=old, state=state, closure=closure, completed=completed_rows, failed=failed_rows, existing_bytes=existing_bytes)


def recovery_partition(plan, history, repo=REPO):
    """No auto retry: only four explicitly reviewed new attempt identities."""
    old = history['old']; completed_pairs = {pair(r['row']) for r in history['completed']}
    failures = {pair(r['row']):r['row'] for r in history['failed']}
    pending = {pair(r):r for r in old['assignments'][21:]}
    original_names = {r['attempt'] for r in [old['first']]+old['assignments']}
    original_plans = {r['plan']['name'] for r in [old['first']]+old['assignments']}
    expected = [r for r in old['assignments'] if pair(r) not in completed_pairs]
    if len(plan['assignments']) != 94 or [pair(r) for r in plan['assignments']] != [pair(r) for r in expected]:
        raise ValueError('exact canonical94 complement required; no retained rerun')
    for row in plan['assignments']:
        a = pair(row)
        if a in pending:
            same(row, pending[a], 'original90 undispatched attempt/plan identity')
        elif a in failures:
            if row['attempt'] in original_names or row['plan']['name'] in original_plans:
                raise ValueError('interrupted attempt requires new explicit identity')
            same(shard_plan(row['plan'],repo), shard_plan(failures[a]['plan'],repo), 'replacement exact original plan payload')
        else: raise ValueError('unexpected recovered assignment')


def recovery_authority(plan, repo=REPO, owned=OWNED):
    """Whole original bytes and status only; no mathematical derivative audit."""
    h = recovery_metadata(plan, repo, owned); recovery_partition(plan,h,repo)
    source = plan['recovery']['source_commit']; c = plan['common']; retained = []; existing_bytes=h['existing_bytes']
    for record in h['completed']:
        row = record['row']; expected = record['receipt']['receipt_sha256']
        result = completed(owned/row['attempt'],row,c,source,expected)
        same(result, record['receipt'], 'original complete receipt and measured cost')
        retained.append(dict(row=row, receipt=result)); existing_bytes+=result['bytes']
    failures = []
    for record, closed in zip(h['failed'],h['closure']['failed_shards']):
        row=record['row']; attempt=owned/row['attempt']
        receipt,files=inventory(attempt,closed['receipt.json']['sha256'])
        same(receipt['source_commit'],source,'failed shard source')
        same(receipt['exit_code'],-15,'original interrupted worker exit')
        same(receipt['timed_out'],False,'original campaign interruption')
        same(sha(files['manifest.json']),closed['manifest.json']['sha256'],'original interrupted manifest')
        manifest=read(files['manifest.json'])
        same(manifest['source_commit'],source,'failed manifest source')
        same(manifest['owned_pid'],record['worker_identity']['pid'],'failed worker identity')
        same(manifest['command'],expected_command(attempt,row),'failed exact native command')
        same(sha(files['source/codex/certify-llm/'+row['plan']['name']]),row['plan']['sha256'],'failed exact archived plan')
        failures.append(dict(row=row,receipt_sha256=closed['receipt.json']['sha256']))
        existing_bytes+=sum(p.stat().st_size for p in files.values())+(attempt/'receipt.json').stat().st_size
    # Captured operational source is distinct from the new dispatcher source.
    # Verify every old source (including launcher/controller) in every snapshot.
    for record in h['completed']+h['failed']:
        attempt=owned/record['row']['attempt']
        for rel,d in h['old']['sources'].items():
            if sha(attempt/'source'/rel)!=d: raise ValueError('original captured source differs')
    table=process_table()
    for identity in h['closure']['identities']:
        for q in table:
            if q['state']=='Z': continue
            if (q['pid']==identity['pid'] and q['start']==identity['start'] and q['uid']==identity['uid']) or (q['sid']==identity['sid'] and q['pgid']==identity['pgid']):
                raise ValueError('original campaign identity/group is still live or reused; refuse recovery')
    return dict(retained=retained, interrupted=failures, undispatched_unchanged=90, existing_bytes=existing_bytes,
                planned_new_assignments=94, known_completed_total_including_first=18,
                prior_receipt_sha256=plan['recovery']['receipt_sha256'],
                closure_sha256=plan['recovery']['closure']['sha256'],
                scientific_derivative_audit_pending=True, certificate_passed=False)


def first_authority(plan, owned=OWNED, *, existing_bytes=0):
    first = plan['first']; result = completed(owned/first['attempt'], first, plan['common'], expected_receipt=first['receipt_sha256'])
    audit = read(pin(first['audit'], parent=owned))
    expected = dict(schema=AUDIT_SCHEMA, passed=True, certificate_passed=False, attempt=first['attempt'],
        receipt_sha256=first['receipt_sha256'], plan_sha256=first['plan']['sha256'],
        phase_index=first['phase_index'], start_column=first['start_column'],
        **{k: plan['common'][k] for k in ('definition_sha256', 'map_sha256', 'backend_sha256', 'evaluator_sha256')})
    for k, v in expected.items(): same(audit[k], v, 'independent first-shard audit '+k)
    if not audit.get('reviewer') or not audit.get('review_source_sha256'): raise ValueError('explicit independent audit source/reviewer required')
    digest(audit['review_source_sha256'])
    same(audit['measured_elapsed_seconds'], result['elapsed_seconds'], 'actual cold first-shard time')
    same(audit['measured_max_rss_kib'], result['max_rss_kib'], 'actual first-shard RSS')
    seconds = result['elapsed_seconds']; rss = result['max_rss_kib']
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0 or type(rss) is not int or rss <= 0:
        raise ValueError('positive actual first-shard measurements required')
    r = plan['resources']; count = len(plan['assignments']); batches = math.ceil(count/r['concurrency'])
    predicted_seconds = seconds*r['time_headroom_milli']/1000
    forecast = batches*predicted_seconds + count*r['registration_seconds'] + r['receipt_seconds']
    if predicted_seconds > 3600 or forecast > r['timeout_seconds']: raise ValueError('measured conservative time forecast exceeds fixed caps')
    if math.ceil(r['rss_headroom_milli']*rss*1024*r['concurrency']/1000)+512*1024**2 > r['aggregate_rss_bytes']: raise ValueError('measured aggregate RSS forecast exceeds watchdog')
    history_bytes=existing_bytes+(result['bytes'] if plan['schema']==RECOVERY_SCHEMA else 0)
    forecast_bytes=math.ceil(r['storage_headroom_milli']*count*result['bytes']/1000)+64*1024**2+history_bytes
    if forecast_bytes > r['raw_bytes']: raise ValueError('measured storage forecast including retained history exceeds fixed cap')
    return dict(first=result, audit_sha256=first['audit']['sha256'], forecast_seconds=forecast,
                forecast_bytes=forecast_bytes, existing_history_bytes=history_bytes,
                sampled_rss_is_hard_cap=False, audit_truth_is_external=True)


def git(repo, *args):
    # Keep dirty/untracked checks intact; avoid optional threaded lstat preload.
    scoped = ['-c', 'core.preloadIndex=false'] if args and args[0] == 'status' else []
    return subprocess.check_output(['git', *scoped, *args], cwd=repo, text=True, timeout=30).strip()


def frozen(repo, plan_path, plan_hash, source, plan, *, remote=False, check_blobs=True):
    digest(source, 40); digest(plan_hash)
    same(git(repo, 'rev-parse', 'HEAD'), source, 'pinned HEAD')
    if git(repo, 'status', '--porcelain', '--untracked-files=all'): raise ValueError('dirty worktree; Git owner must freeze it')
    if sha(plan_path) != plan_hash: raise ValueError('campaign plan changed')
    names = list(plan['sources'])+['codex/certify-llm/'+r['plan']['name'] for r in [plan['first']]+plan['assignments']]
    names += [str(plan_path.relative_to(repo))]
    for rel in names if check_blobs else []:
        data = subprocess.check_output(['git', 'show', source+':'+rel], cwd=repo, timeout=30)
        if (repo/rel).read_bytes() != data: raise ValueError('not exact committed bytes: '+rel)
    if remote:
        fields = git(repo, 'ls-remote', '--exit-code', 'origin', 'refs/heads/main').split()
        if len(fields) != 2 or fields[1] != 'refs/heads/main': raise ValueError('remote main unresolved')
        digest(fields[0], 40)
        # Read-only; missing local remote objects require the Git owner to sync.
        subprocess.run(['git', 'merge-base', '--is-ancestor', source, fields[0]], cwd=repo, check=True, timeout=30)
        return fields[0]


def proc(pid, *, command=True):
    """Linux birth identity; a missing process is not an unknown worker proof."""
    try:
        directory = Path('/proc')/str(pid); stat = (directory/'stat').read_text()
        fields = stat[stat.rfind(')')+2:].split(); status = (directory/'status').read_text().splitlines()
        info = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in status if ':' in line}
        return dict(pid=pid, state=fields[0], ppid=int(fields[1]), pgid=int(fields[2]), sid=int(fields[3]),
            start=int(fields[19]), nice=int(fields[16]), uid=int(info['Uid'].split()[0]), rss=int(info.get('VmRSS', '0 kB').split()[0])*1024,
            command=(directory/'cmdline').read_bytes().split(b'\0')[:-1] if command else [])
    except (FileNotFoundError, ProcessLookupError): return None


def serial_identity(value):
    return {k: ([os.fsdecode(v) for v in x] if k == 'command' else x) for k, x in value.items()}


class Identity:
    def __init__(self, value):
        if value is None: raise RuntimeError('missing birth identity')
        self.value = value; self.fd = os.pidfd_open(value['pid'])
        now = proc(value['pid'])
        if now is None or any(now[k] != value[k] for k in ('pid', 'start', 'uid', 'sid', 'pgid')):
            os.close(self.fd); raise RuntimeError('process changed during pidfd registration')
    def live(self):
        value = proc(self.value['pid'])
        return value if value and all(value[k] == self.value[k] for k in ('pid', 'start', 'uid', 'sid', 'pgid')) and value['state'] != 'Z' else None
    def send(self, sig):
        if self.live(): signal.pidfd_send_signal(self.fd, sig)
    def close(self): os.close(self.fd)


def group(identity):
    value = identity.value
    current = proc(value['pid'], command=False)
    if current and any(current[k] != value[k] for k in ('start', 'uid', 'sid', 'pgid')):
        # A reused numeric leader PID cannot still identify the old session.
        return []
    # A recorded session/group remains owned after its leader exits. Linux
    # retains that group/session identity while descendants remain members.
    if value['sid'] != value['pid'] or value['pgid'] != value['pid']: raise RuntimeError('not an owned session leader')
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        q = proc(int(entry.name), command=False)
        if q and q['pgid'] == value['pid'] and q['state'] != 'Z':
            if q['sid'] != value['pid'] or q['uid'] != value['uid'] or q['start'] < value['start']: raise RuntimeError('group ownership uncertain')
            members.append(q)
    return members


def signal_group(identity, sig):
    # Bind each currently verified member to a pidfd before signalling. This
    # avoids numeric killpg PID-reuse races; no unrelated session can be hit.
    handles = []
    try:
        for value in group(identity):
            try: handles.append(Identity(value))
            except (ProcessLookupError, RuntimeError):
                now = proc(value['pid'], command=False)
                if now and now['state'] != 'Z' and now['start'] == value['start']: raise
        for handle in handles: handle.send(sig)
    finally:
        for handle in handles: handle.close()


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable unavailable')


def process_table():
    return [q for e in Path('/proc').iterdir() if e.name.isdigit() and (q := proc(int(e.name), command=False)) is not None]


def expected_command(attempt, row):
    return [sys.executable, '-I', '-B', '-u', str(attempt/'source/codex/certify-llm/derivative_shard.py'),
            '--output', str(attempt/'run'), '--plan', str(attempt/'source/codex/certify-llm'/row['plan']['name'])]


class Dispatcher:
    def __init__(self, plan, out, source, *, owned=OWNED, repo=REPO, before_launch=None, command=None, complete=None, existing_bytes=0):
        self.plan, self.out, self.source, self.owned, self.repo = plan, Path(out), source, Path(owned), Path(repo)
        self.before_launch = before_launch or (lambda: None)
        self.command = command or self.launch_command; self.complete = complete or self.check_complete
        self.started = time.monotonic(); self.children = []; self.stop_reason = None; self.uncertain = []
        self.r = plan['resources']; self.peak_rss = 0; self.existing_bytes=existing_bytes; self.out.mkdir(exist_ok=False)
        self.save()
    def launch_command(self, row):
        return [sys.executable, '-I', '-B', '-u', str(self.repo/'codex/certify-llm/launch.py'),
                '--attempt', row['attempt'], '--driver', 'derivative_shard', '--plan', row['plan']['name'], '--threads', '1', '--seconds', '3600']
    def check_complete(self, child):
        return completed(child['attempt'], child['row'], self.plan['common'], self.source)
    def save(self):
        rows = []
        for c in self.children:
            rows.append({k: v for k, v in c.items() if k in ('row', 'state', 'registered', 'launcher_pid', 'launcher_command', 'launcher_identity', 'launcher_command_admitted', 'launcher_command_admitted_at', 'launcher_command_observations', 'worker_identity', 'receipt', 'error', 'launcher_exit', 'worker_termination_verified')})
        write(self.out/'state.json', dict(certificate_passed=False, started_monotonic=self.started, elapsed_seconds=time.monotonic()-self.started,
            stop_reason=self.stop_reason, uncertain_ownership=self.uncertain, sampled_peak_aggregate_rss=self.peak_rss,
            sampled_rss_is_hard_cap=False, rows=rows))
    def start(self, row):
        self.before_launch()
        self.sample()
        if time.monotonic()-self.started > self.r['timeout_seconds']: raise RuntimeError('whole campaign deadline before spawn')
        attempt = self.owned/row['attempt']
        if attempt.exists(): raise RuntimeError('attempt appeared before launch')
        index = len(self.children); stdout = (self.out/f'launcher-{index:03d}.stdout').open('xb'); stderr = (self.out/f'launcher-{index:03d}.stderr').open('xb')
        env = os.environ.copy(); env.update({k: '1' for k in THREAD_ENV}); env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        command = self.command(row)
        p = subprocess.Popen(command, cwd=self.repo, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        # Retain the Popen child immediately, before /proc or pidfd operations
        # that can fail. The unreaped direct child is still ours on that path.
        child = dict(row=row, attempt=attempt, p=p, leader=None, worker=None, stdout=stdout, stderr=stderr,
            born=time.monotonic(), state='spawned_unregistered', registered=False, launcher_pid=p.pid,
            launcher_command=command, launcher_command_admitted=False, launcher_command_observations=[], worker_termination_verified=False)
        self.children.append(child); self.save()
        self.ensure_leader(child); child['state'] = 'registering'; self.save()
        return child
    def ensure_leader(self, c):
        if c.get('launcher_command_admitted'): return True
        q = proc(c['p'].pid)
        if q is None or q['ppid'] != os.getpid() or q['uid'] != os.getuid() or q['sid'] != q['pid'] or q['pgid'] != q['pid']:
            raise RuntimeError('direct launcher child identity unavailable')
        if c['leader'] is None:
            # Birth/pidfd ownership is established immediately for cleanup even
            # when /proc exposes a transient empty or stale post-exec command.
            c['leader'] = Identity(q); c['launcher_identity'] = serial_identity(q); self.save()
        elif any(q[k] != c['leader'].value[k] for k in ('pid', 'start', 'uid', 'sid', 'pgid')):
            raise RuntimeError('launcher birth identity changed before command admission')
        if q['state'] == 'Z': raise RuntimeError('launcher exited before exact command admission')
        actual = [os.fsdecode(v) for v in q['command']]
        observations = c['launcher_command_observations']
        if not observations or observations[-1]['command'] != actual:
            observations.append(dict(elapsed_seconds=time.monotonic()-c['born'], command=actual))
        if time.monotonic()-c['born'] >= self.r['registration_seconds']:
            self.save()
            if actual != c['launcher_command']:
                raise RuntimeError('persistent launcher command mismatch at registration deadline')
            raise RuntimeError('exact launcher command arrived after registration deadline')
        if actual != c['launcher_command']:
            self.save(); return False
        c['launcher_command_admitted'] = True
        c['launcher_command_admitted_at'] = time.monotonic()
        self.save(); return True
    def register(self, c):
        if not self.ensure_leader(c): return False
        path = c['attempt']/'manifest.json'
        if not path.exists(): return False
        try: m = read(path)
        except json.JSONDecodeError: return False  # launch.py writes registration non-atomically
        same(m['source_commit'], self.source, 'registered source'); same(m['threads'], 1, 'worker CPU')
        same(m['memory_gib'], 18, 'worker AS'); same(m['timeout_seconds'], 3600, 'worker deadline')
        command = expected_command(c['attempt'], c['row']); same(m['command'], command, 'literal archived command')
        pid = m['owned_pid']; integer(pid, 1, 2**31-1, 'worker pid'); q = proc(pid)
        if q is None or q['state'] == 'Z': raise RuntimeError('worker finished before live ownership registration; uncertain ownership')
        parent = c['leader'].live()
        if not parent or q['ppid'] != parent['pid'] or q['start'] < parent['start'] or q['uid'] != parent['uid'] or q['sid'] != pid or q['pgid'] != pid:
            raise RuntimeError('worker ownership identity mismatch')
        same([os.fsdecode(v) for v in q['command']], command, 'actual live worker command')
        same(q['nice'], 10, 'actual worker nice level')
        limit = next(line.split()[-3:-1] for line in (Path('/proc')/str(pid)/'limits').read_text().splitlines() if line.startswith('Max address space'))
        same(limit, [str(18*1024**3)]*2, 'actual worker AS soft/hard cap')
        c['worker'] = Identity(q); c['worker_identity'] = serial_identity(q); c['registered'] = True; c['state'] = 'running'; self.save(); return True
    def sample(self):
        table = process_table(); roots = {c['leader'].value['pid']: c['leader'].value for c in self.children if c['leader'] and c['leader'].live()}
        roots.update({c['worker'].value['pid']: c['worker'].value for c in self.children if c['worker'] and c['worker'].live()})
        selected = {p for p in roots}; changed = True
        while changed:
            old = len(selected)
            selected.update(q['pid'] for q in table if q['ppid'] in selected)
            changed = len(selected) != old
        total = sum(q['rss'] for q in table if q['pid'] in selected) + (proc(os.getpid()) or {}).get('rss', 0)
        self.peak_rss = max(self.peak_rss, total)
        if total > self.r['aggregate_rss_bytes']: raise RuntimeError('sampled aggregate RSS watchdog')
        if available() < self.r['minimum_available_bytes']: raise RuntimeError('MemAvailable watchdog')
        disk = os.statvfs(self.owned)
        if disk.f_bavail*disk.f_frsize < self.r['minimum_disk_bytes']: raise RuntimeError('free storage admission/watchdog')
        raw = sum(p.stat().st_size for root in [self.out]+[c['attempt'] for c in self.children] if root.exists() for p in root.rglob('*') if p.is_file())
        if raw+self.existing_bytes > self.r['raw_bytes']: raise RuntimeError('raw storage watchdog')
    def inspect(self):
        for c in self.children:
            if c['state'] in ('completed', 'failed'): continue
            if not c['registered']: self.register(c)
            rc = c['p'].poll()
            if rc is not None:
                c['launcher_exit'] = rc
                if c['worker'] is None: raise RuntimeError('launcher exited without established worker ownership')
                if c['worker'].live(): raise RuntimeError('launcher exited while owned worker remains live')
                # The reviewed worker does not spawn subprocesses; any retained
                # live group after its leader exits is an explicit failure.
                if any(q['pgid'] == c['worker'].value['pid'] and q['state'] != 'Z' for q in process_table()): raise RuntimeError('worker group remains live')
                c['worker_termination_verified'] = True
                if rc != 0: raise RuntimeError('failed shard launcher exit')
                c['receipt'] = self.complete(c); c['state'] = 'completed'; self.save()
            elif not c['registered'] and time.monotonic()-c['born'] > self.r['registration_seconds']:
                raise RuntimeError('bounded registration deadline')
    def cancel_unregistered(self, c):
        """Never assume missing manifest means no worker. Freeze only our parent.

        While the verified single-threaded launcher is stopped it cannot fork a
        new worker. A live direct child with the exact command can be recovered.
        Other-session unknown children make ownership unresolved, not successful.
        """
        try:
            if c['leader'] is None: self.ensure_leader(c)
        except Exception as exc:
            c['error'] = repr(exc)
            if c['leader'] is None:
                self.uncertain.append(dict(attempt=c['row']['attempt'], launcher_pid=c['p'].pid, reason='launcher identity still unavailable', error=repr(exc))); return
        # A command-unadmitted but birth-verified direct Popen child is still
        # ours for cleanup; it never gains scientific/dispatch admission here.
        parent = c['leader'].live()
        if not parent:
            self.uncertain.append(dict(attempt=c['row']['attempt'], reason='launcher died before registration; possible orphan not certified terminated')); return
        try:
            c['leader'].send(signal.SIGSTOP)
            deadline = time.monotonic()+1
            while time.monotonic() < deadline:
                q = c['leader'].live()
                if q and q['state'] in ('T', 't'): break
                time.sleep(.01)
            else:
                self.uncertain.append(dict(attempt=c['row']['attempt'], reason='launcher stop not confirmed')); return
            # Allow a child already between fork and exec to establish its final
            # command; the parent remains stopped throughout this bounded handoff.
            until = time.monotonic()+1; matches = []; unknown = []
            while True:
                children = [live for q in process_table() if q['ppid'] == parent['pid'] and q['state'] != 'Z'
                            and (live := proc(q['pid'])) is not None and live['ppid'] == parent['pid']]
                matches = [q for q in children if q['start'] >= parent['start'] and q['uid'] == parent['uid'] and
                    q['sid'] == q['pid'] == q['pgid'] and [os.fsdecode(v) for v in q['command']] == expected_command(c['attempt'], c['row'])]
                # A forked Python child may not yet have called setsid/exec.
                # It is NOT evidence that no worker exists, even while it still
                # shares the launcher's session. Only an already exec'd Git
                # child is the reviewed launcher's pre-worker registration path.
                unknown = [q for q in children if q not in matches and not (
                    q['sid'] == parent['sid'] and q['command'] and
                    os.fsdecode(q['command'][0]) == 'git')]
                if not unknown or time.monotonic() >= until: break
                time.sleep(.02)
            if len(matches) == 1 and not unknown:
                c['worker'] = Identity(matches[0]); c['worker_identity'] = serial_identity(matches[0]); c['registered'] = True
                c['state'] = 'recovered_for_cancellation_only'; self.save()
            elif matches or unknown:
                self.uncertain.append(dict(attempt=c['row']['attempt'], reason='unresolved pre-registration child', observed=[{k:v for k,v in q.items() if k != 'command'} for q in children]))
            else:
                # No worker existed while the sole spawning parent was stopped.
                # Kill its own registration session (Git helpers included), not
                # any unrelated worker. No worker receipt is invented.
                signal_group(c['leader'], signal.SIGKILL); c['state'] = 'registration_aborted_no_worker'
                c['error'] = 'registration aborted while parent stopped; no worker existed in confirmed direct-child snapshot'
        finally:
            c['leader'].send(signal.SIGCONT)
    def shutdown(self):
        # Cleanup failures for one owned child cannot skip cleanup of its peers.
        def act(c, label, fn, fallback=None):
            try: return fn()
            except BaseException as exc:
                self.uncertain.append(dict(attempt=c['row']['attempt'], reason=label, error=repr(exc)))
                return fallback
        # The first grace permits an in-progress launcher to finish registration.
        limit = time.monotonic()+self.r['registration_seconds']
        while any(not c['registered'] and c['p'].poll() is None for c in self.children) and time.monotonic() < limit:
            for c in self.children:
                if not c['registered']:
                    try: self.register(c)
                    except Exception as exc: c['error'] = repr(exc)
            time.sleep(.05)
        for c in self.children:
            if not c['registered']: act(c, 'unregistered cancellation failed', lambda: self.cancel_unregistered(c))
        for c in self.children:
            if c['worker']: act(c, 'owned worker TERM failed', lambda: signal_group(c['worker'], signal.SIGTERM))
        until = time.monotonic()+10
        while time.monotonic() < until:
            live = [c for c in self.children if c['worker'] and act(c, 'owned group inspection failed', lambda: group(c['worker']), True)]
            if not live: break
            time.sleep(.05)
        for c in self.children:
            if c['worker']: act(c, 'owned worker final KILL/check failed', lambda: signal_group(c['worker'], signal.SIGKILL) if group(c['worker']) else None)
        # Keep existing launchers alive while they collect exits/hash/write receipts.
        until = time.monotonic()+self.r['receipt_seconds']
        while any(c['p'].poll() is None for c in self.children) and time.monotonic() < until: time.sleep(.05)
        for c in self.children:
            c['launcher_exit'] = c['p'].poll()
            if c['launcher_exit'] is None:
                self.uncertain.append(dict(attempt=c['row']['attempt'], reason='launcher receipt collection still active', launcher_pid=c['p'].pid))
            if c['worker']:
                alive = act(c, 'owned worker termination check failed', lambda: c['worker'].live() or group(c['worker']), True)
                c['worker_termination_verified'] = not bool(alive)
                if alive: self.uncertain.append(dict(attempt=c['row']['attempt'], reason='owned worker termination unverified'))
            if c['state'] != 'completed': c['state'] = 'failed'
        self.save()
    def run(self):
        old = {}
        def stop(signum, frame): self.stop_reason = 'signal '+str(signum)
        try:
            for s in (signal.SIGTERM, signal.SIGINT): old[s] = signal.signal(s, stop)
            next_row = 0
            while True:
                if self.stop_reason: raise RuntimeError(self.stop_reason)
                if time.monotonic()-self.started > self.r['timeout_seconds']: raise RuntimeError('whole campaign deadline')
                self.sample(); self.inspect()
                if next_row == len(self.plan['assignments']) and all(c['state'] == 'completed' for c in self.children): break
                active = [c for c in self.children if c['state'] != 'completed']
                if next_row < len(self.plan['assignments']) and len(active) < self.r['concurrency'] and all(c['registered'] for c in active):
                    self.start(self.plan['assignments'][next_row]); next_row += 1
                time.sleep(self.r['sample_seconds'])
            outcome = 'campaign_completed'
        except BaseException as exc:
            self.stop_reason = self.stop_reason or repr(exc)
            try: write(self.out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), certificate_passed=False))
            except BaseException as logging_error:
                self.uncertain.append(dict(reason='failure log write failed; cleanup still required', error=repr(logging_error)))
            try: self.shutdown()
            except BaseException as cleanup:
                self.uncertain.append(dict(reason='cleanup exception', error=repr(cleanup))); self.save()
            outcome = 'campaign_failed_uncertain_ownership' if self.uncertain else 'campaign_failed'
        finally:
            for s, handler in old.items(): signal.signal(s, handler)
            for c in self.children:
                c['stdout'].close(); c['stderr'].close()
                if c['leader']: c['leader'].close()
                if c['worker']: c['worker'].close()
        result = dict(status=outcome, certificate_passed=False, completed=sum(c['state'] == 'completed' for c in self.children),
            started=len(self.children), undispatched=len(self.plan['assignments'])-len(self.children),
            stop_reason=self.stop_reason, uncertain_ownership=self.uncertain, sampled_rss_is_hard_cap=False,
            scientific_derivative_audit_pending=True)
        write(self.out/'result.json', result); return result


def seal(out, result):
    """Seal only after no owned writer remains; uncertainty is kept unsealed."""
    if result.get('uncertain_ownership'):
        write(out/'finalization-pending.json', dict(certificate_passed=False,
            reason='Uncertain ownership or receipt writer remains; preserve live files without claiming an immutable seal.'))
        return
    rows = [dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha(p))
            for p in sorted(out.rglob('*')) if p.is_file() and p != out/'receipt.json']
    write(out/'receipt.json', dict(certificate_passed=False, status=result['status'], artifacts=rows,
        scientific_derivative_audit_pending=True, owned_process_checks='See complete retained run/state.json; no uncertain state promoted.'))
    for p in out.rglob('*'):
        if p.is_file(): p.chmod(0o444)


def require_controller_nice10():
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError('campaign controller must be launched at exact nice10')


def main():
    require_controller_nice10()
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--source', required=True); parser.add_argument('--attempt', required=True)
    args = parser.parse_args(); name(args.attempt)
    out = OWNED/args.attempt; out.mkdir(exist_ok=False)
    write(out/'admission-started.json', dict(source_commit=args.source, plan_path=str(args.plan), plan_sha256=args.plan_sha256,
        dispatcher_source_sha256=sha(__file__), dispatcher_identity=serial_identity(proc(os.getpid())), certificate_passed=False))
    dispatcher = None
    try:
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'): raise RuntimeError('Linux pidfd support required')
        if args.plan.resolve().parent != HERE or args.plan.is_symlink(): raise ValueError('committed campaign plan required')
        if sha(args.plan) != args.plan_sha256: raise ValueError('literal original campaign plan hash required')
        for k in THREAD_ENV: os.environ[k] = '1'
        hard = resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard != resource.RLIM_INFINITY and hard < 18*1024**3: raise RuntimeError('inherited hard AS too small for unchanged launcher')
        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 18*1024**3))
        plan = validate(read(args.plan), REPO, OWNED)
        (out/'original-plan.json').write_bytes(args.plan.read_bytes())
        remote = frozen(REPO, args.plan, args.plan_sha256, args.source, plan, remote=True)
        recovered = recovery_authority(plan) if plan['schema']==RECOVERY_SCHEMA else None
        authority = first_authority(plan, existing_bytes=recovered['existing_bytes'] if recovered else 0)
        if recovered is not None: authority['recovery']=recovered
        dispatcher = Dispatcher(plan, out/'run', args.source, existing_bytes=authority['existing_history_bytes'],
            before_launch=lambda: frozen(REPO, args.plan, args.plan_sha256, args.source, plan, check_blobs=False))
        dispatcher.started = began
        write(out/'admission.json', dict(source_commit=args.source, plan_sha256=args.plan_sha256, plan=plan,
            authority=authority, remote_main=remote, dispatcher_pid=os.getpid(), certificate_passed=False))
        result = dispatcher.run()
        if plan['schema'] == RECOVERY_SCHEMA:
            result.update(retained_prior_shards=17, first_audited_shards=1,
                          total_completed_shards=18+result['completed'],
                          recovery_attempts_only=True)
        if result['status'] == 'campaign_completed':
            frozen(REPO, args.plan, args.plan_sha256, args.source, plan, check_blobs=False)
    except BaseException as exc:
        uncertain = list(dispatcher.uncertain) if dispatcher is not None else []
        if dispatcher is not None:
            for c in dispatcher.children:
                if c['p'].poll() is None or (c['worker'] and group(c['worker'])):
                    uncertain.append(dict(reason='owned process still active after exceptional finalization', launcher_pid=c['p'].pid, attempt=c['row']['attempt']))
        result = dict(status='campaign_admission_or_finalization_failed', certificate_passed=False, error=repr(exc), uncertain_ownership=uncertain)
        write(out/'failure.json', dict(result, traceback=traceback.format_exc()))
    write(out/'status.json', result); seal(out, result)
    print(canonical(result), flush=True)
    return 0 if result['status'] == 'campaign_completed' else 1

if __name__ == '__main__': sys.exit(main())
