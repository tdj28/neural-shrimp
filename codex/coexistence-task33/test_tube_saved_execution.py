"""NEW synthetic decoded-record controls only; no original evidence or execution.

These fixtures make no claim that their invented pins authenticate real files.
The caller must separately authenticate bytes before using the pure join.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location('tube_saved_execution', Path(__file__).with_name('tube_saved_execution.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_pin(path, size=100, sha=None):
    return dict(path=path, bytes=size, sha256=sha or hashlib.sha256(path.encode()).hexdigest())


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fixture():
    """Entirely invented future records, using only literal source-contract hashes."""
    limit = dict(as_bytes=18 * 2**30, cpu_seconds=3600, wall_seconds=3900, output_bytes=64 * 2**20)
    expected = dict(source_commit='1' * 40, card=fake_pin(M.CARD), plan=fake_pin(M.PLAN),
                    receipt=fake_pin(M.SUPERVISION + '/receipt.json'),
                    admission=fake_pin(M.SUPERVISION + '/admission.json'),
                    worker_startup=fake_pin(M.SUPERVISION + '/worker-startup.json'),
                    host_closure=fake_pin('/tmp/synthetic-tube-host-closure.json'),
                    limits=copy.deepcopy(limit), cpu_ids=[0])
    source_files = [fake_pin(M.ROOT + '/' + name, sha=sha) for name, sha in M.SOURCE_HASHES.items()]
    source_files += [copy.deepcopy(expected['card']), fake_pin(M.FOLDER + '/synthetic-extra-review.md')]
    expected['source_files'] = copy.deepcopy(source_files)
    review = next(copy.deepcopy(row) for row in source_files if row['path'].endswith('regularity-review.md'))
    card = dict(schema='codex-task33-one-tube-card-v1', status='reviewed_for_one_tube_execution',
                repository=M.ROOT, output=M.OUTPUT, target_calls=1, precision=2048, directions=1,
                dimension=896, layers=24, vocabulary=151936, token_chunk=1024,
                supervision_plan=M.PLAN, supervision_admission=expected['admission']['path'],
                limits=copy.deepcopy(limit), source_sha256=copy.deepcopy(M.SOURCE_HASHES),
                regularity_contract=M.REGULARITY_CONTRACT, regularity_review=review,
                prefix=copy.deepcopy(M.PREFIX), map={'synthetic': 'map'}, backend={'synthetic': 'backend'},
                audit_result=fake_pin('/tmp/synthetic-upstream-audit-result.json'),
                input_files=[fake_pin('/tmp/synthetic-model.safetensors', 988097824),
                             fake_pin('/tmp/synthetic-prefix-cache.json.gz', 68579136)], **M.SAVED_FLAGS)
    for name in ('thin_execution', 'audit_execution', 'prefix_execution'):
        card[name] = {key: fake_pin('/tmp/synthetic-' + name + '-' + key + '.json')
                      for key in ('card', 'plan', 'receipt', 'admission')}
    card['map_sha256'] = object_hash(card['map']); card['backend_sha256'] = object_hash(card['backend'])
    preflight = [copy.deepcopy(card['audit_result']), copy.deepcopy(review), *copy.deepcopy(card['input_files'])]
    for name in ('thin_execution', 'audit_execution', 'prefix_execution'):
        preflight.extend(copy.deepcopy(list(card[name].values())))
    arguments = ['--card', M.CARD, '--card-sha256', expected['card']['sha256'], '--out', M.OUTPUT]
    argv = [M.PYTHON, '-I', '-B', M.DRIVER, *arguments]
    lane = dict(entry=M.DRIVER, arguments=arguments, output=M.OUTPUT, supervision=M.SUPERVISION,
                limits=copy.deepcopy(limit), preflight_pins=preflight)
    plan = dict(status='reviewed_for_execution', repository=M.ROOT, raw_root=M.BASE, python=M.PYTHON,
                cpu_threads=1, cpu_ids=[0], startup_ram_gib=32, source_files=copy.deepcopy(source_files), lanes={'tube': lane})
    supervisor = dict(pid=700, starttime=900, pgid=600, sid=600)
    worker = dict(pid=701, starttime=901, pgid=701, sid=701, nice=10, threads=1,
                  uid_line='Uid:\t1000\t1000\t1000\t1000', cmdline='synthetic preexec supervisor argv ')
    admission = dict(source_commit=expected['source_commit'], lane='tube', argv=copy.deepcopy(argv),
                     plan=copy.deepcopy(expected['plan']), limits=copy.deepcopy(limit),
                     supervisor=copy.deepcopy(supervisor), threads={key: '1' for key in M.THREADS},
                     capacity=dict(available_ram_bytes=32 * 2**30, disk_free_bytes=16 * 2**30),
                     source_blobs=[dict(relative_path=row['path'][len(M.ROOT) + 1:], bytes=row['bytes'], sha256=row['sha256'])
                                   for row in source_files])
    worker_startup = dict(identity=copy.deepcopy(worker), planned_argv=copy.deepcopy(argv), affinity=[0],
                          address_space=[limit['as_bytes']] * 2, cpu_seconds=[limit['cpu_seconds']] * 2,
                          file_size=[limit['output_bytes']] * 2)
    output_files = [fake_pin(M.OUTPUT + '/' + name) for name in M.NAMES]
    postexec = copy.deepcopy(worker); postexec['cmdline'] = ' '.join(argv) + ' '
    receipt = dict(source_commit=expected['source_commit'], lane='tube', argv=copy.deepcopy(argv),
                   returncode=0, failure=None, worker_reaped=True, worker_identity_absent=True,
                   worker_group_session_absent=True, output_files=output_files, actual_output_bytes=800,
                   worker_observed_after_exec=postexec, worker_pid=701, supervisor=copy.deepcopy(supervisor))
    internal_admission = dict(source_commit=expected['source_commit'], card=copy.deepcopy(expected['card']),
                              plan=copy.deepcopy(expected['plan']), supervision_admission=copy.deepcopy(expected['admission']),
                              source_sha256=copy.deepcopy(M.SOURCE_HASHES), regularity_review=copy.deepcopy(review), **M.SAVED_FLAGS)
    status = dict(schema='codex-task33-one-tube-status-v1', status='operator_measurement_completed', target_calls=1,
                  full_domain_gates_succeeded=True, binding=copy.deepcopy(internal_admission),
                  artifacts=copy.deepcopy(output_files[:-1]), selection=copy.deepcopy(output_files[1]), **M.SAVED_FLAGS)
    host_closure = dict(schema='codex-task33-tube-host-closure-v1', source_commit=expected['source_commit'],
                        receipt=copy.deepcopy(expected['receipt']), admission=copy.deepcopy(expected['admission']),
                        worker_startup=copy.deepcopy(expected['worker_startup']), observed_at_utc='2026-10-19T13:00:00Z',
                        view='actual_host', observation_reference='Synthetic fixture, not a real observation',
                        worker_identity={key: worker[key] for key in ('pid', 'starttime', 'pgid', 'sid')},
                        supervisor_identity=copy.deepcopy(supervisor), worker_identity_absent=True,
                        supervisor_identity_absent=True, worker_group_members=[], worker_session_members=[])
    return dict(expected=expected, card=card, plan=plan, receipt=receipt, admission=admission,
                worker_startup=worker_startup, internal_admission=internal_admission, status=status, host_closure=host_closure)


class ExecutionControls(unittest.TestCase):
    def check(self, value):
        return M.check_target_execution(**value)

    def refuse_changes(self, cases):
        for keys, replacement in cases:
            with self.subTest(path=keys, replacement=replacement):
                value = fixture(); target = value
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = replacement
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    self.check(value)

    def test_01_consistency_only_and_no_input_mutation(self):
        value = fixture(); original = copy.deepcopy(value); result = self.check(value)
        self.assertEqual(value, original)
        self.assertEqual(result['status'], 'target_execution_records_consistent_only')
        for key in ('scientific_verdict_admitted', 'operator_bound_admitted', 'conditional_operator_bound_admitted',
                    'trajectory_enclosure', 'reachability_proved', 'certificate_passed',
                    'original_file_authentication_performed', 'original_source_authentication_performed',
                    'original_git_commit_authenticated', 'original_execution_authenticated', 'actual_host_observation_authenticated'):
            self.assertIs(result[key], False)
        self.assertNotIn('L', result)
        self.assertEqual(len(result['output_pins']), 8)
        self.assertTrue(any('supervisor group/session' in item for item in result['unchecked']))
        self.assertTrue(any('upstream thin' in item for item in result['unchecked']))

    def test_02_relative_plan_pin_and_preexec_cmdline_allowed(self):
        value = fixture()
        value['admission']['plan']['path'] = 'codex/coexistence-task33/tube-target-001-plan.json'
        value['worker_startup']['identity']['cmdline'] = 'different preexec command '
        self.check(value)

    def test_03_large_metadata_pins_extra_review_source_and_shared_supervisor_group(self):
        value = fixture()
        self.assertGreater(value['card']['input_files'][0]['bytes'], 64 * 2**20)
        self.assertGreater(value['card']['input_files'][1]['bytes'], 64 * 2**20)
        self.assertGreater(len(value['expected']['source_files']), len(value['card']['source_sha256']))
        self.assertNotEqual(value['admission']['supervisor']['pid'], value['admission']['supervisor']['pgid'])
        self.check(value)

    def test_04_outer_failure_never_becomes_success(self):
        self.refuse_changes([(('receipt', 'returncode'), x) for x in (1, -9, True, None)] +
                            [(('receipt', 'failure'), x) for x in ('wall deadline', '', False)] +
                            [(('receipt', key), 1) for key in ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')])

    def test_05_expectations_required_and_strict_pin_schema(self):
        for key in M.EXPECTED_KEYS:
            with self.subTest(missing=key):
                value = fixture(); del value['expected'][key]
                with self.assertRaises(ValueError): self.check(value)
        self.refuse_changes([(('expected', 'source_commit'), x) for x in ('1' * 39, 'A' * 40, True)] +
                            [(('expected', 'card', 'bytes'), True), (('expected', 'card', 'sha256'), 'A' * 64),
                             (('expected', 'host_closure', 'path'), '/tmp/../closure.json'),
                             (('expected', 'receipt', 'path'), '/tmp/wrong-receipt.json')])

    def test_06_card_status_root_and_fixed_dimensions(self):
        self.refuse_changes([(('card', 'status'), 'held'), (('card', 'repository'), '/tmp/other'),
                             (('card', 'output'), '/tmp/other'), (('card', 'schema'), 'other')] +
                            [(('card', key), True) for key in ('target_calls', 'precision', 'directions', 'dimension', 'layers', 'vocabulary', 'token_chunk')] +
                            [(('card', 'directions'), 2), (('card', 'dimension'), 895), (('card', 'precision'), 1024)])

    def test_07_plan_lane_argv_and_original_pins(self):
        self.refuse_changes([(('plan', 'status'), 'held'), (('plan', 'raw_root'), '/tmp'),
                             (('plan', 'python'), '/usr/bin/python3'), (('plan', 'lanes'), {}),
                             (('plan', 'lanes', 'tube', 'entry'), '/tmp/other.py'),
                             (('plan', 'lanes', 'tube', 'arguments'), []),
                             (('admission', 'plan', 'sha256'), '0' * 64),
                             (('receipt', 'argv'), []), (('admission', 'lane'), 'other'),
                             (('receipt', 'source_commit'), '2' * 40), (('card', 'supervision_admission'), '/tmp/other')])

    def test_08_limits_threads_and_cpu_types(self):
        self.refuse_changes([(('expected', 'limits', 'cpu_seconds'), True), (('card', 'limits', 'as_bytes'), 1),
                             (('plan', 'lanes', 'tube', 'limits', 'output_bytes'), 1),
                             (('admission', 'limits', 'wall_seconds'), 1), (('plan', 'cpu_threads'), True),
                             (('plan', 'cpu_ids'), [True]), (('expected', 'cpu_ids'), [0, 1]),
                             (('worker_startup', 'affinity'), [1]), (('admission', 'threads', M.THREADS[0]), 1)])

    def test_09_capacity_metadata_floors(self):
        self.refuse_changes([(('plan', 'startup_ram_gib'), x) for x in (True, 31, 33)] +
                            [(('admission', 'capacity', 'available_ram_bytes'), x) for x in (True, 32 * 2**30 - 1)] +
                            [(('admission', 'capacity', 'disk_free_bytes'), x) for x in (True, 16 * 2**30 - 1)])

    def test_10_exact_reviewed_source_and_card_presence(self):
        for target in ('expected', 'plan'):
            value = fixture(); value[target]['source_files'].pop()
            with self.assertRaises(ValueError): self.check(value)
        value = fixture()
        for target in ('expected', 'plan'):
            value[target]['source_files'] = [row for row in value[target]['source_files'] if row['path'] != M.CARD]
        value['admission']['source_blobs'] = [row for row in value['admission']['source_blobs'] if not row['relative_path'].endswith('/tube-target-001-card.json')]
        with self.assertRaises(ValueError): self.check(value)

    def test_11_matched_tampered_core_hash_is_not_authority(self):
        value = fixture(); relative = next(iter(M.SOURCE_HASHES)); bad = '0' * 64
        value['card']['source_sha256'][relative] = bad
        for target in ('expected', 'plan'):
            for row in value[target]['source_files']:
                if row['path'] == M.ROOT + '/' + relative: row['sha256'] = bad
        for row in value['admission']['source_blobs']:
            if row['relative_path'] == relative: row['sha256'] = bad
        value['internal_admission']['source_sha256'][relative] = bad
        value['status']['binding']['source_sha256'][relative] = bad
        with self.assertRaises(ValueError): self.check(value)

    def test_12_source_blob_duplicate_foreign_or_changed(self):
        value = fixture(); value['admission']['source_blobs'][1] = copy.deepcopy(value['admission']['source_blobs'][0])
        with self.assertRaises(ValueError): self.check(value)
        self.refuse_changes([(('admission', 'source_blobs', 0, 'relative_path'), x) for x in ('../escape', '/absolute', 'codex//file', 'codex/./file')] +
                            [(('admission', 'source_blobs', 0, 'bytes'), True),
                             (('admission', 'source_blobs', 0, 'sha256'), '0' * 64)])

    def test_13_bounded_unique_pin_sets(self):
        value = fixture(); value['expected']['source_files'] *= 10
        with self.assertRaises(ValueError): self.check(value)
        for location in ('source_files',):
            value = fixture(); value['plan'][location].append(copy.deepcopy(value['plan'][location][0]))
            with self.assertRaises(ValueError): self.check(value)
        value = fixture(); value['plan']['lanes']['tube']['preflight_pins'].append(copy.deepcopy(value['plan']['lanes']['tube']['preflight_pins'][0]))
        with self.assertRaises(ValueError): self.check(value)

    def test_14_preflight_requires_all_declared_inputs(self):
        for index in (0, 1, 2, 5, 15):
            value = fixture(); value['plan']['lanes']['tube']['preflight_pins'].pop(index)
            with self.assertRaises(ValueError): self.check(value)
        self.refuse_changes([(('card', 'input_files', 0, 'bytes'), True),
                             (('card', 'thin_execution', 'receipt', 'sha256'), '0' * 64)])

    def test_15_literal_prefix_declaration(self):
        self.refuse_changes([(('card', 'prefix'), {}),
                             (('card', 'prefix', 'prefix_source_commit'), '2' * 40),
                             (('card', 'prefix', 'prefix_receipt_sha256'), '0' * 64)])
        value = fixture(); value['card']['prefix']['prefix_receipt_sha256'] = '0' * 64
        value['card']['prefix_execution']['receipt']['sha256'] = '0' * 64
        for row in value['plan']['lanes']['tube']['preflight_pins']:
            if row['path'] == value['card']['prefix_execution']['receipt']['path']: row['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.check(value)

    def test_16_canonical_map_backend_and_contract(self):
        self.refuse_changes([(('card', 'map_sha256'), '0' * 64), (('card', 'backend_sha256'), '0' * 64),
                             (('card', 'map'), {'bad': float('nan')}), (('card', 'regularity_contract'), 'unchecked'),
                             (('card', 'regularity_review', 'bytes'), 1)])

    def test_17_reconstructed_internal_and_status_bindings(self):
        self.refuse_changes([((record, key, 'sha256'), '0' * 64)
                             for record in ('internal_admission',) for key in ('card', 'plan', 'supervision_admission', 'regularity_review')] +
                            [(('internal_admission', 'source_commit'), '2' * 40),
                             (('status', 'binding', 'card', 'sha256'), '0' * 64)])

    def test_18_operator_only_flags_and_complete_status(self):
        self.refuse_changes([((record, key), int(flag)) for record in ('card', 'internal_admission', 'status')
                             for key, flag in M.SAVED_FLAGS.items()] +
                            [(('status', 'status'), 'failed'), (('status', 'target_calls'), True),
                             (('status', 'full_domain_gates_succeeded'), 1), (('status', 'schema'), 'other')])

    def test_19_complete_exact_outputs_and_status_artifacts(self):
        for record, key in (('receipt', 'output_files'), ('status', 'artifacts')):
            value = fixture(); value[record][key].pop()
            with self.assertRaises(ValueError): self.check(value)
        value = fixture(); value['receipt']['output_files'].append(fake_pin(M.OUTPUT + '/failure.json'))
        with self.assertRaises(ValueError): self.check(value)
        self.refuse_changes([(('status', 'selection', 'sha256'), '0' * 64),
                             (('receipt', 'output_files', 0, 'path'), M.OUTPUT + '/sub/admission.json')])

    def test_20_output_byte_total_and_cap(self):
        self.refuse_changes([(('receipt', 'actual_output_bytes'), x) for x in (True, 799, 801)])
        value = fixture()
        for target in (value['expected'], value['card'], value['admission'], value['plan']['lanes']['tube']):
            target['limits']['output_bytes'] = 799
        value['worker_startup']['file_size'] = [799, 799]
        with self.assertRaises(ValueError): self.check(value)

    def test_21_worker_and_supervisor_birth_session_identity(self):
        self.refuse_changes([(('receipt', 'worker_observed_after_exec', 'starttime'), 902),
                             (('worker_startup', 'identity', 'pid'), True), (('receipt', 'worker_pid'), True),
                             (('receipt', 'supervisor', 'starttime'), 901)])
        value = fixture()
        for identity in (value['worker_startup']['identity'], value['receipt']['worker_observed_after_exec'], value['host_closure']['worker_identity']):
            identity['pgid'] = 702
        with self.assertRaises(ValueError): self.check(value)

    def test_22_worker_runtime_uid_nice_and_threads(self):
        for identity_path in (('worker_startup', 'identity'), ('receipt', 'worker_observed_after_exec')):
            self.refuse_changes([(identity_path + ('nice',), x) for x in (True, 0)] +
                                [(identity_path + ('threads',), x) for x in (True, 2)] +
                                [(identity_path + ('uid_line',), x) for x in ('Uid: 1000 1000 1000', 'Uid: 1000 0 1000 1000',
                                                                            'Uid: 1000 1000 1000 1000 extra', True)])

    def test_23_actual_startup_rlimits_and_exact_postexec_argv(self):
        self.refuse_changes([(('worker_startup', key), [True, True]) for key in ('address_space', 'cpu_seconds', 'file_size')] +
                            [(('worker_startup', 'planned_argv'), []),
                             (('receipt', 'worker_observed_after_exec', 'cmdline'), 'wrong command ')])

    def test_24_host_closure_requires_exact_record_and_original_pins(self):
        value = fixture(); value['host_closure']['all_closed'] = True
        with self.assertRaises(ValueError): self.check(value)
        for key in ('receipt', 'admission', 'worker_startup'):
            self.refuse_changes([(('host_closure', key, 'sha256'), '0' * 64)])
        self.refuse_changes([(('host_closure', 'source_commit'), '2' * 40),
                             (('host_closure', 'view'), 'namespace'), (('host_closure', 'observation_reference'), ' ')])

    def test_25_host_utc_calendar_and_identity_closure(self):
        self.refuse_changes([(('host_closure', 'observed_at_utc'), x) for x in ('2026-10-19T13:00:00+00:00', '2026-02-30T13:00:00Z', 'today')] +
                            [(('host_closure', 'worker_identity', 'starttime'), 902),
                             (('host_closure', 'supervisor_identity', 'pid'), 702),
                             (('host_closure', 'worker_identity_absent'), 1),
                             (('host_closure', 'supervisor_identity_absent'), False),
                             (('host_closure', 'worker_group_members'), [703]),
                             (('host_closure', 'worker_session_members'), [703])])


if __name__ == '__main__':
    unittest.main()
