"""Pure decoded-record consistency for the first Task33 tube execution only.

No filesystem, Git, process, backend, model, scalar helper or CLI access. Caller
pins must later be authenticated against the SAME bytes that supplied these
records. Even a consistent successful record is not original-execution authority
or an admitted operator bound. Closed failures refuse this success-only join.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re


ROOT = '/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp'
FOLDER = ROOT + '/codex/coexistence-task33'
BASE = '/data2/neural-shrimp-data/codex/coexistence'
OUTPUT = BASE + '/tube-target-001'
SUPERVISION = OUTPUT + '-supervision'
CARD = FOLDER + '/tube-target-001-card.json'
PLAN = FOLDER + '/tube-target-001-plan.json'
DRIVER = FOLDER + '/tube_target_driver_002.py'
PYTHON = '/data2/neural-shrimp/.venv/bin/python'
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
SAVED_FLAGS = dict(trajectory_enclosure=False, certificate_passed=False,
                   reachability_proved=False, conditional_operator_bound_only=True)
NAMES = ('admission.json', 'selection.json', 'prefix-binding.json', 'tube-inputs.json',
         'actual-receiver.json', 'tube-operator.json', 'coverage.json', 'status.json')
SOURCE_HASHES = {
    'codex/coexistence-task33/tube_receiver.py': 'de18915bec02aa27b84fa9f5a08768c33731efb94f6cf1a6f0dcdc2e42a76b49',
    'codex/coexistence-task33/tube_operator.py': '4e837348df63dbea4a1cef752b30d3d73bce23a0a279a35696a502245d804722',
    'codex/coexistence-task33/prefix_import.py': 'b4f82802a9bf77b8d90e399c5c7e9b2635106d39532079852539ee227f1211e8',
    'codex/certify-llm/streamed_head.py': 'f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a',
    'codex/certify-llm/ball_jet.py': '46b14d8bd65ed6c80174758daf37d00110cfa908813917bdb46c13749f999719',
    'codex/certify-llm/ball_qwen.py': 'fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80',
    'codex/certify-llm/ball_linear.py': '5d54200f5880d818eabc97e7eed2f1cf02b83b617b39380bef38d49a7ac45ff2',
    'codex/certify-llm/cache_io.py': 'f4e25b2134bef8a71c27d3156baf01be9d25c593d7b780f90a66b9c1141abdef',
    'codex/crosscheck/qwen_map.py': '493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e',
    'codex/certify-chaos/supervise.py': '306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44',
    'codex/coexistence-task33/tube_target_driver_002.py': '2c8e6ae4cb86b5d06b4e6a534d2e6f8a518d19f55fcab92b9b5d0b25210befc9',
    'codex/coexistence-task33/tube-target-001-regularity-review.md': '388c2d7038e86eaeddfb55d16df6832152a91ba23fde233f9d94c5d7f8ef79eb',
}
REGULARITY_CONTRACT = (
    'The authenticated fixed-prefix real map is C1 on the nominal closed convex cube; '
    'the accepted interval-Jet graph encloses F(x) and DF(x)v for every admitted x '
    'and independently varying v in [-1,1]^896; every interval denominator and '
    'positive-root gate succeeds over the entire containing input enclosure.'
)
PREFIX = dict(
    prefix_receipt=BASE + '/prefix-001-supervision/receipt.json',
    prefix_receipt_sha256='0e787d0ebd6fb505a202c96fb3cc2429803aa8cc50c5271f728dc3da59af6f12',
    prefix_source_commit='8c5170bce23a3cf054de8638ba047dcabc893518',
    prefix_output=BASE + '/prefix-001',
    prefix_admission_path=BASE + '/prefix-001-supervision/admission.json',
    prefix_admission_sha256='84be8d159949dc60aa0f3909c6a16ee5a967dab0e40cd717dda2738331bcfa28',
    prefix_map_source_sha256='fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80',
)
EXPECTED_KEYS = {'source_commit', 'card', 'plan', 'receipt', 'admission', 'worker_startup',
                 'host_closure', 'source_files', 'limits', 'cpu_ids'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ': exact integer required')
    return value


def digest(value, size=64):
    require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(size) + '}', value), 'full lowercase hash required')
    return value


def path(value):
    require(type(value) is str and '\0' not in value and value.startswith('/') and
            not value.startswith('//') and str(PurePosixPath(value)) == value and
            '..' not in PurePosixPath(value).parts, 'canonical absolute path required')
    return value


def pin(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'exact decoded pin schema required')
    path(value['path']); integer(value['bytes'], 'pin bytes'); digest(value['sha256'])
    # No file-reader size cap: metadata includes the ~988 MB checkpoint/cache.
    return value.copy()


def pins(values):
    require(type(values) is list and 0 < len(values) <= 128, 'bounded nonempty pin list required')
    result = {}
    for value in values:
        record = pin(value)
        require(record['path'] not in result, 'duplicate pin path')
        result[record['path']] = record
    return result


def flags(value):
    require(type(value) is dict and all(value.get(k) is v for k, v in SAVED_FLAGS.items()), 'saved operator-only scope flags differ')


def limits(value):
    require(type(value) is dict and set(value) == {'as_bytes', 'cpu_seconds', 'wall_seconds', 'output_bytes'}, 'exact limit fields required')
    for key, number in value.items():
        integer(number, key, 1)
    return value


def cpus(value):
    require(type(value) is list and len(value) == 1, 'one explicitly allocated CPU required')
    integer(value[0], 'CPU id')
    return value


def identity(value):
    require(type(value) is dict, 'process identity record required')
    result = {k: integer(value[k], k, 1) for k in ('pid', 'starttime', 'pgid', 'sid')}
    return result


def worker_runtime(value):
    for key, number in (('nice', 10), ('threads', 1)):
        require(integer(value[key], key) == number, 'worker nice/thread metadata differs')
    line = value['uid_line']
    require(type(line) is str and re.fullmatch(r'Uid:[ \t]+[0-9]+[ \t]+[0-9]+[ \t]+[0-9]+[ \t]+[0-9]+', line)
            and [int(x) for x in line.split()[1:]] == [1000] * 4, 'exact four worker UID 1000 fields required')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(value):
    require(type(value) is dict and set(value) == {'source_commit', 'card', 'plan', 'supervision_admission',
            'source_sha256', 'regularity_review', *SAVED_FLAGS}, 'exact internal admission binding required')
    digest(value['source_commit'], 40)
    for key in ('card', 'plan', 'supervision_admission', 'regularity_review'):
        pin(value[key])
    flags(value)
    require(value['source_sha256'] == SOURCE_HASHES, 'internal reviewed source hashes differ')
    return value


def check_target_execution(expected, card, plan, receipt, admission, worker_startup,
                           internal_admission, status, host_closure):
    """Join only supplied records; successful consistency grants no authority.

    expected independently supplies future original pins/source/limits/CPU choice.
    host_closure is a claimed actual-host observation whose file pin is explicit;
    this pure helper neither performs nor authenticates that observation.
    """
    require(type(expected) is dict and set(expected) == EXPECTED_KEYS, 'explicit complete future original expectations required')
    source = digest(expected['source_commit'], 40)
    fixed = {'card': CARD, 'plan': PLAN, 'receipt': SUPERVISION + '/receipt.json',
             'admission': SUPERVISION + '/admission.json', 'worker_startup': SUPERVISION + '/worker-startup.json'}
    for key, name in fixed.items():
        require(pin(expected[key])['path'] == name, 'wrong original ' + key + ' allocation')
    closure_pin = pin(expected['host_closure'])
    limit = limits(expected['limits']); cpu = cpus(expected['cpu_ids'])
    require(type(receipt) is dict and type(receipt.get('returncode')) is int and
            receipt['returncode'] == 0 and receipt.get('failure', 'missing') is None,
            'nonzero, missing or failed outer execution cannot enter the successful join')
    require(all(receipt.get(k) is True for k in ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')),
            'saved outer closure flags missing')
    require(card['schema'] == 'codex-task33-one-tube-card-v1' and card['status'] == 'reviewed_for_one_tube_execution'
            and card['repository'] == ROOT and card['output'] == OUTPUT, 'reviewed exact tube001 card required')
    flags(card)
    for key, value in {'target_calls': 1, 'precision': 2048, 'directions': 1, 'dimension': 896,
                       'layers': 24, 'vocabulary': 151936, 'token_chunk': 1024}.items():
        require(integer(card[key], key) == value, 'exact full target allocation differs')
    require(card['supervision_plan'] == PLAN and card['supervision_admission'] == fixed['admission'], 'card supervision paths differ')
    require(limits(card['limits']) == limit, 'independently specified card limits differ')
    require(plan['status'] == 'reviewed_for_execution' and plan['repository'] == ROOT and
            plan['raw_root'] == BASE and plan['python'] == PYTHON and type(plan['lanes']) is dict and
            set(plan['lanes']) == {'tube'}, 'reviewed exact single-lane original plan required')
    require(integer(plan['cpu_threads'], 'CPU threads') == 1 and cpus(plan['cpu_ids']) == cpu, 'CPU/thread allocation differs')
    require(integer(plan['startup_ram_gib'], 'startup RAM GiB') == 32, 'reviewed startup RAM floor differs')
    require(integer(admission['capacity']['available_ram_bytes'], 'admission RAM bytes') >= 32 * 2**30 and
            integer(admission['capacity']['disk_free_bytes'], 'admission disk bytes') >= 16 * 2**30,
            'recorded admission capacity below original floors')
    lane = plan['lanes']['tube']
    arguments = ['--card', CARD, '--card-sha256', expected['card']['sha256'], '--out', OUTPUT]
    argv = [PYTHON, '-I', '-B', DRIVER, *arguments]
    require(lane['entry'] == DRIVER and lane['arguments'] == arguments and lane['output'] == OUTPUT and
            lane['supervision'] == SUPERVISION and limits(lane['limits']) == limit, 'exact target argv/allocation/limits differ')
    for record in (receipt, admission):
        require(record['source_commit'] == source and record['lane'] == 'tube' and record['argv'] == argv,
                'original source/lane/argv differs')
    admitted_plan = dict(admission['plan'])
    if admitted_plan.get('path') == 'codex/coexistence-task33/tube-target-001-plan.json':
        admitted_plan['path'] = PLAN
    require(pin(admitted_plan) == expected['plan'] and limits(admission['limits']) == limit, 'original plan pin or admitted limits differ')
    require(type(admission['threads']) is dict and set(admission['threads']) == set(THREADS) and
            all(admission['threads'][key] == '1' for key in THREADS), 'exact four one-thread environment records required')

    reviewed, planned = pins(expected['source_files']), pins(plan['source_files'])
    require(planned == reviewed and planned.get(CARD) == expected['card'], 'complete reviewed source set/card presence differs')
    require(card['source_sha256'] == SOURCE_HASHES and card['regularity_contract'] == REGULARITY_CONTRACT,
            'reviewed driver/core source hashes or regularity contract differs')
    review = pin(card['regularity_review'])
    require(review['path'] == FOLDER + '/tube-target-001-regularity-review.md' and planned.get(review['path']) == review,
            'regularity review pin missing from original sources')
    for relative, sha in SOURCE_HASHES.items():
        require(planned.get(ROOT + '/' + relative, {}).get('sha256') == sha, 'reviewed loaded source missing or changed')
    require(type(admission['source_blobs']) is list and len(admission['source_blobs']) == len(planned), 'complete source blob list required')
    blob_pins = []
    for row in admission['source_blobs']:
        require(type(row) is dict and set(row) == {'relative_path', 'bytes', 'sha256'}, 'exact source blob schema required')
        relative = row['relative_path']
        require(type(relative) is str and relative and not relative.startswith('/') and
                str(PurePosixPath(relative)) == relative and '..' not in PurePosixPath(relative).parts, 'canonical source-relative path required')
        blob_pins.append(dict(path=ROOT + '/' + relative, bytes=row['bytes'], sha256=row['sha256']))
    require(pins(blob_pins) == planned, 'original admission source blob inventory differs')
    require(all(PurePosixPath(name).is_relative_to(ROOT) and name != ROOT for name in planned), 'source file escaped repository')

    # Structural preflight inclusion only. These upstream executions/assets are
    # deliberately NOT authenticated or interpreted by this target-only unit.
    preflight = pins(lane['preflight_pins'])
    required = [pin(card['audit_result']), review, *pins(card['input_files']).values()]
    for name in ('thin_execution', 'audit_execution', 'prefix_execution'):
        require(type(card[name]) is dict, 'upstream execution description required')
        required.extend(pin(card[name][key]) for key in ('card', 'plan', 'receipt', 'admission'))
    require(all(preflight.get(row['path']) == row for row in required), 'required declared input/authority pin missing from preflight')
    for key in ('map', 'backend'):
        require(type(card[key]) is dict and digest(card[key + '_sha256']) == canonical_hash(card[key]),
                'declared map/backend canonical hash differs')
    require(type(card['prefix']) is dict and card['prefix'] == PREFIX, 'reviewed literal prefix declaration differs')
    reconstructed = dict(source_commit=source, card=expected['card'], plan=expected['plan'],
                         supervision_admission=expected['admission'], source_sha256=SOURCE_HASHES.copy(),
                         regularity_review=review, **SAVED_FLAGS)
    require(binding(internal_admission) == reconstructed, 'internal original admission differs from reconstructed binding')
    flags(status)
    require(status['schema'] == 'codex-task33-one-tube-status-v1' and status['status'] == 'operator_measurement_completed'
            and integer(status['target_calls'], 'status target calls') == 1 and status['full_domain_gates_succeeded'] is True,
            'complete one-call target status required')
    require(binding(status['binding']) == reconstructed, 'status admission binding differs')
    outputs = pins(receipt['output_files'])
    require(set(outputs) == {OUTPUT + '/' + name for name in NAMES}, 'complete eight original output pins required')
    require(integer(receipt['actual_output_bytes'], 'actual output bytes') == sum(row['bytes'] for row in outputs.values())
            and receipt['actual_output_bytes'] <= limit['output_bytes'], 'recorded output total/cap differs')
    require(pins(status['artifacts']) == {p: v for p, v in outputs.items() if p != OUTPUT + '/status.json'} and
            pin(status['selection']) == outputs[OUTPUT + '/selection.json'], 'status preceding-artifact/selection join differs')

    worker = identity(worker_startup['identity'])
    postexec = identity(receipt['worker_observed_after_exec'])
    supervisor = identity(admission['supervisor'])
    require(worker == postexec and integer(receipt['worker_pid'], 'receipt worker PID', 1) == worker['pid'] and
            worker['pid'] == worker['pgid'] == worker['sid'] and worker['pid'] != supervisor['pid'], 'original worker birth/session differs')
    require(identity(receipt['supervisor']) == supervisor, 'supervisor birth differs')
    require(worker_startup['planned_argv'] == argv and cpus(worker_startup['affinity']) == cpu and
            receipt['worker_observed_after_exec']['cmdline'] == ' '.join(argv) + ' ', 'startup/postexec argv or affinity differs')
    worker_runtime(worker_startup['identity'])
    worker_runtime(receipt['worker_observed_after_exec'])
    for key, field in (('address_space', 'as_bytes'), ('cpu_seconds', 'cpu_seconds'), ('file_size', 'output_bytes')):
        row = worker_startup[key]
        require(type(row) is list and len(row) == 2 and all(type(v) is int for v in row) and row == [limit[field]] * 2,
                'actual startup rlimit metadata differs')
    _host_closure(host_closure, expected, worker, supervisor)
    return dict(status='target_execution_records_consistent_only', recorded_outer_outcome='success',
                reconstructed_binding=reconstructed, output_pins=outputs,
                original_source_commit_claim=source, host_closure_pin=closure_pin,
                declared_map_sha256=card['map_sha256'], declared_backend_sha256=card['backend_sha256'],
                scientific_verdict_admitted=False, operator_bound_admitted=False,
                conditional_operator_bound_admitted=False, trajectory_enclosure=False,
                reachability_proved=False, certificate_passed=False,
                original_file_authentication_performed=False, original_source_authentication_performed=False,
                original_git_commit_authenticated=False, original_execution_authenticated=False,
                actual_host_observation_authenticated=False,
                unchecked=['same-byte file/Git origin authentication', 'upstream thin and audit002 execution',
                           'prefix/cache/checkpoint/config/vector authority and literal asset aliases',
                           'map/backend/regularity semantics and scalar/coverage arithmetic',
                           'live runtime/capacity and actual host-observation provenance',
                           'supervisor group/session and descendant closure (may be shared)'])


def _host_closure(record, expected, worker, supervisor):
    keys = {'schema', 'source_commit', 'receipt', 'admission', 'worker_startup', 'observed_at_utc',
            'view', 'observation_reference', 'worker_identity', 'supervisor_identity',
            'worker_identity_absent', 'supervisor_identity_absent', 'worker_group_members', 'worker_session_members'}
    require(type(record) is dict and set(record) == keys and record['schema'] == 'codex-task33-tube-host-closure-v1',
            'explicit exact external host-closure record required')
    require(record['source_commit'] == expected['source_commit'] and
            all(pin(record[k]) == expected[k] for k in ('receipt', 'admission', 'worker_startup')), 'host closure original pins differ')
    require(record['view'] == 'actual_host' and type(record['observation_reference']) is str and
            0 < len(record['observation_reference'].strip()) <= 4096, 'claimed actual-host observation reference required')
    stamp = record['observed_at_utc']
    require(type(stamp) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z', stamp),
            'exact UTC observation timestamp required')
    require(datetime.fromisoformat(stamp.replace('Z', '+00:00')).utcoffset() == timezone.utc.utcoffset(None), 'UTC observation required')
    require(identity(record['worker_identity']) == worker and identity(record['supervisor_identity']) == supervisor,
            'host closure identities/births differ')
    require(record['worker_identity_absent'] is True and record['supervisor_identity_absent'] is True and
            type(record['worker_group_members']) is list and record['worker_group_members'] == [] and
            type(record['worker_session_members']) is list and record['worker_session_members'] == [],
            'both original identities and worker group/session must be recorded absent')
