"""Prepared audit002-bound positive-radius Task33 tube measurement; not a launch card.

Install at the reviewed repository path only after independent source review.
No target access occurs on import. Main requires a new externally hash-pinned
card, committed supervision plan, original audit and thin execution authorities,
an original 2048-bit prefix, and exact backend/map/regularity bindings.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import time
import traceback

R = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp')
D = R / 'codex/coexistence-task33'
B = Path('/data2/neural-shrimp-data/codex/coexistence')
SELF = D / 'tube_target_driver_002.py'
PYTHON = '/data2/neural-shrimp/.venv/bin/python'
THIN_RAW = B / 'thin-pilot-001'
AUDIT_RESULT = B / 'thin-pilot-001-audit-002/result.json'
THIN_CARD = D / 'thin-pilot-001-card.json'
THIN_CARD_SHA = '33922427cb39bb72662122ae06c461acdbfc9314344e4e9f8a468d2b47164213'
AUDITOR_SHA = 'b2a84b5022f71cadfce91e6fc21fa4ad9953bb0a1a0b0c1d62d24d43f0c7e4d5'
AUDIT_SOURCE = '70b043192b775f2cc67860c4709bfc867de096eb'
AUDIT_CARD = D / 'thin-pilot-001-audit-002-card.json'
AUDIT_CARD_SHA = '538e6ea648fa1e115dc35eceb4392d0798da9bc19f4c77c91098ac227fed8dd7'
AUDIT_PLAN = D / 'thin-pilot-001-audit-002-plan.json'
AUDIT_PLAN_SHA = 'd719c9366fc84380fa4983fd9c579a425ed8faeb99c33aad29c6373c3f0292a6'
DIMENSION, LAYERS, VOCABULARY, PRECISION, CHUNK = 896, 24, 151936, 2048, 1024
ALPHA, LOG10_TEMPERATURE = .06468, -.148
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
FLAGS = dict(trajectory_enclosure=False, certificate_passed=False,
             reachability_proved=False, conditional_operator_bound_only=True)
ASSETS = {
    'model.safetensors': 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
    'config.json': '18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
    'vectors.npz': 'b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41',
}
PREFIX = dict(
    prefix_receipt=str(B / 'prefix-001-supervision/receipt.json'),
    prefix_receipt_sha256='0e787d0ebd6fb505a202c96fb3cc2429803aa8cc50c5271f728dc3da59af6f12',
    prefix_source_commit='8c5170bce23a3cf054de8638ba047dcabc893518',
    prefix_output=str(B / 'prefix-001'),
    prefix_admission_path=str(B / 'prefix-001-supervision/admission.json'),
    prefix_admission_sha256='84be8d159949dc60aa0f3909c6a16ee5a967dab0e40cd717dda2738331bcfa28',
    prefix_map_source_sha256='fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80',
)
PREFIX_FILES = {
    'inputs.json': '726c125e1f20d85ab9396b70bc5aad6dab8ad6589c24d2a71e05e976796cf67d',
    'prefix-cache.json.gz': '52ead55348240b014ab3d0d41a925a58dfc477ceb8fdd2d3801e4ff7c5996634',
}
CORE_SHA = {
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
}
REGULARITY_CONTRACT = (
    'The authenticated fixed-prefix real map is C1 on the nominal closed convex cube; '
    'the accepted interval-Jet graph encloses F(x) and DF(x)v for every admitted x '
    'and independently varying v in [-1,1]^896; every interval denominator and '
    'positive-root gate succeeds over the entire containing input enclosure.'
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(items):
    result = {}
    for key, value in items:
        need(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON constant')))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def hash_string(value, length=64):
    need(type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value),
         'full hexadecimal hash required')
    return value


def pin(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink()
         and path.is_file() and stat.S_ISREG(path.stat().st_mode), 'original regular absolute file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())



CHECKPOINT_SNAPSHOT = Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors')
CHECKPOINT_RESOLVED = Path('/data2/neural-shrimp-data/models/hf/hub/blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7')
CONFIG_SNAPSHOT = CHECKPOINT_SNAPSHOT.parent / 'config.json'
CONFIG_RESOLVED = Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f')
CONFIG_LINK_TARGET = '../../blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f'
CONFIG_BYTES = 659
CHECKPOINT_LINKS = (
    (CHECKPOINT_SNAPSHOT, '../../blobs/fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'),
    (Path('/data2/neural-shrimp-data/models/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'),
     '../../blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7'))


def checkpoint_definition():
    return dict(snapshot_path=str(CHECKPOINT_SNAPSHOT),
                links=[dict(path=str(link), link_target=target) for link, target in CHECKPOINT_LINKS],
                resolved_path=str(CHECKPOINT_RESOLVED), expected_content_sha256=ASSETS['model.safetensors'])


def checkpoint_authority(path):
    """Only auditor002's literal immutable HF chain; ordinary pin stays strict."""
    path = Path(path)
    need(path == CHECKPOINT_SNAPSHOT, 'exact original checkpoint snapshot path required')
    need(len(CHECKPOINT_LINKS) == 2 and CHECKPOINT_LINKS[0][0] == path,
         'exact two-link checkpoint chain required')
    for link, target in CHECKPOINT_LINKS:
        need(link.is_symlink() and str(link.readlink()) == target, 'fixed checkpoint symlink target differs')
    need(path.resolve(strict=True) == CHECKPOINT_RESOLVED and
         all(link.resolve(strict=True) == CHECKPOINT_RESOLVED for link, _ in CHECKPOINT_LINKS),
         'canonical checkpoint destination differs')
    need(CHECKPOINT_RESOLVED.is_file() and not CHECKPOINT_RESOLVED.is_symlink()
         and CHECKPOINT_RESOLVED.resolve(strict=True) == CHECKPOINT_RESOLVED,
         'canonical checkpoint must be regular')
    return CHECKPOINT_RESOLVED, checkpoint_definition()


def checkpoint_binding(inputs, embedding, loader_path):
    """Bind actual loader alias, independently audited row and full content pin."""
    resolved, authority = checkpoint_authority(loader_path)
    need(embedding.get('snapshot_authority') == authority
         and embedding['asset']['path'] == str(resolved)
         and inputs.get(str(resolved)) == embedding['asset'], 'audited canonical checkpoint/snapshot join differs')
    asset = verify(embedding['asset'])
    need(asset['sha256'] == ASSETS['model.safetensors'], 'full original checkpoint content differs')
    need(checkpoint_authority(loader_path) == (resolved, authority), 'checkpoint chain changed during hashing')
    return dict(asset=asset, snapshot_authority=authority)



def config_authority(path):
    """Only the original literal one-link configuration cache mapping."""
    path = Path(path)
    need(path == CONFIG_SNAPSHOT and path.is_symlink() and str(path.readlink()) == CONFIG_LINK_TARGET,
         'exact original configuration snapshot/link required')
    need(path.resolve(strict=True) == CONFIG_RESOLVED and CONFIG_RESOLVED.is_file()
         and not CONFIG_RESOLVED.is_symlink() and CONFIG_RESOLVED.resolve(strict=True) == CONFIG_RESOLVED,
         'canonical configuration must be the original regular file')
    return CONFIG_RESOLVED, dict(snapshot_path=str(CONFIG_SNAPSHOT),
        links=[dict(path=str(CONFIG_SNAPSHOT), link_target=CONFIG_LINK_TARGET)],
        resolved_path=str(CONFIG_RESOLVED), expected_bytes=CONFIG_BYTES,
        expected_content_sha256=ASSETS['config.json'])


def config_binding(inputs, loader_path):
    resolved, authority = config_authority(loader_path)
    need(str(resolved) in inputs, 'canonical configuration missing from preflight inputs')
    asset = verify(inputs[str(resolved)])
    need(asset['path'] == str(resolved) and asset['bytes'] == CONFIG_BYTES
         and asset['sha256'] == ASSETS['config.json'], 'full original configuration content differs')
    need(config_authority(loader_path) == (resolved, authority), 'configuration chain changed during hashing')
    return dict(asset=asset, snapshot_authority=authority)


def verify(record):
    need(type(record) is dict and set(record) == {'path', 'bytes', 'sha256'}
         and type(record['bytes']) is int and record['bytes'] >= 0, 'complete exact file pin required')
    hash_string(record['sha256'])
    need(pin(record['path']) == record, 'original file pin differs: ' + record['path'])
    return record


def git_blob(source, path):
    hash_string(source, 40)
    path = Path(path)
    need(path.is_absolute() and path.is_relative_to(R), 'source path outside owned repository')
    return subprocess.check_output(['git', 'show', source + ':' + str(path.relative_to(R))], cwd=R)


def source_pin(source, record, current=False):
    blob = git_blob(source, record['path'])
    need(len(blob) == record['bytes'] and hashlib.sha256(blob).hexdigest() == record['sha256'],
         'original committed source differs')
    if current:
        verify(record)
    return record


def by_path(records):
    need(type(records) is list, 'file inventory list required')
    answer = {row['path']: row for row in records}
    need(len(answer) == len(records), 'duplicate file inventory path')
    return answer


def original_execution(authority, require_success):
    """Join separately pinned card/plan/receipt/admission, not report booleans."""
    required = {'source_commit', 'card', 'plan', 'receipt', 'admission', 'lane', 'output', 'argv'}
    need(type(authority) is dict and set(authority) == required, 'complete independent execution authority required')
    source = hash_string(authority['source_commit'], 40)
    for key in ('card', 'plan', 'receipt', 'admission'):
        verify(authority[key])
    for key in ('card', 'plan'):
        source_pin(source, authority[key])
    card, plan, receipt, admission = [read(authority[key]['path']) for key in ('card', 'plan', 'receipt', 'admission')]
    lane = plan['lanes'][authority['lane']]
    command = [plan['python'], '-I', '-B', lane['entry'], *lane['arguments']]
    need(plan['status'] == 'reviewed_for_execution' and plan['repository'] == str(R)
         and plan['raw_root'] == str(B) and plan['python'] == PYTHON and plan['cpu_threads'] == 1,
         'original reviewed owned CPU1 plan required')
    need(command == authority['argv'] == receipt['argv'] == admission['argv']
         and authority['lane'] == receipt['lane'] == admission['lane']
         and receipt['source_commit'] == admission['source_commit'] == source,
         'original exact command/source/lane differs')
    need(lane['output'] == authority['output'] and Path(authority['output']).is_relative_to(B)
         and authority['receipt']['path'] == str(Path(lane['supervision']) / 'receipt.json')
         and authority['admission']['path'] == str(Path(lane['supervision']) / 'admission.json'),
         'original exact allocation differs')
    original_plan = dict(admission['plan'])
    if not Path(original_plan['path']).is_absolute():
        original_plan['path'] = str(R / original_plan['path'])
    need(original_plan == authority['plan'] and admission['limits'] == lane['limits']
         and all(admission['threads'].get(key) == '1' for key in THREADS), 'original plan/limits/thread binding differs')
    need(type(receipt['returncode']) is int and all(receipt[key] is True for key in
         ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')), 'original worker closure absent')
    need(receipt['failure'] is None or type(receipt['failure']) is str, 'invalid original failure field')
    success = receipt['returncode'] == 0 and receipt['failure'] is None
    need(not require_success or success, 'upstream audit did not complete successfully')
    planned = by_path(plan['source_files'])
    blobs = {str(R / row['relative_path']): {key: row[key] for key in ('bytes', 'sha256')}
             for row in admission['source_blobs']}
    need(len(blobs) == len(admission['source_blobs']) and
         {key: {field: value[field] for field in ('bytes', 'sha256')} for key, value in planned.items()} == blobs,
         'complete original source inventory differs')
    need(planned.get(authority['card']['path']) == authority['card'], 'execution card missing from original source inventory')
    for record in planned.values():
        source_pin(source, record)
    files = by_path(receipt['output_files'])
    root = Path(authority['output'])
    for record in files.values():
        need(Path(record['path']).is_relative_to(root), 'output escaped original allocation')
        verify(record)
    need({str(path) for path in root.rglob('*') if path.is_file()} == set(files),
         'whole original output inventory differs')
    return card, plan, receipt, admission, files, success


def qrecord(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def rational(record):
    need(type(record) is dict and set(record) == {'numerator', 'denominator'}
         and all(type(record[key]) is int for key in record) and record['denominator'] > 0,
         'exact rational record required')
    answer = F(record['numerator'], record['denominator'])
    need(qrecord(answer) == record, 'canonical rational required')
    return answer


def dyadic(pair):
    need(type(pair) is list and len(pair) == 2 and all(type(value) is int for value in pair),
         'native integer midpoint/radius pair required')
    return F(pair[0]) * F(2) ** pair[1]


def bounds(record):
    need(type(record) is dict and set(record) == {'mid', 'rad'}, 'original native Arb record required')
    mid, radius = dyadic(record['mid']), dyadic(record['rad'])
    need(radius >= 0, 'negative native radius')
    return mid, radius, mid - radius, mid + radius


def vector(records, exact=False):
    need(type(records) is list and len(records) == DIMENSION, 'all 896 native coordinates required')
    rows = [bounds(record) for record in records]
    if exact:
        need(all(row[1] == 0 for row in rows), 'exact original centres required')
        return [row[0] for row in rows]
    return rows


def false_flags(record):
    need(record.get('trajectory_enclosure') is False and record.get('certificate_passed') is False,
         'false trajectory/certificate flags required')


def candidate(card):
    thin = card['thin_execution']
    tc, tp, tr, ta, thin_files, thin_success = original_execution(thin, False)
    need(thin['card']['path'] == str(THIN_CARD) and thin['card']['sha256'] == THIN_CARD_SHA
         and thin['plan']['path'] == str(D / 'thin-pilot-001-plan.json')
         and thin['output'] == str(THIN_RAW) and thin['lane'] == 'thin', 'original thin authority differs')
    thin_command = [PYTHON, '-I', '-B', str(D / 'thin_pilot.py'), '--card', str(THIN_CARD), '--out', str(THIN_RAW)]
    need(thin['argv'] == thin_command and all(tc.get(key) == value for key, value in PREFIX.items()),
         'thin source cell/prefix command differs')
    need(tc['precision'] == 2048 and tc['steps'] == 20 and tc['initial_token_id'] == 198
         and tc['alpha'] == ALPHA and tc['log10T'] == LOG10_TEMPERATURE, 'thin science card differs')
    original_hashes = {row['relative_path']: row['sha256'] for row in ta['source_blobs']}
    for relative, expected in {
        'codex/coexistence-task33/thin_pilot.py': 'a12fd306fb49392c0d4922b6c4d7aabec1dd56e0b0b835373ecf74287ff05cbc',
        'codex/coexistence-task33/thin_pseudo_orbit.py': '1e9974f2bd5cddfc0ddfd6b1038c7d046df7f8abdb6f061b67fb764a6189369e',
        'codex/crosscheck/qwen_map.py': CORE_SHA['codex/crosscheck/qwen_map.py'],
    }.items():
        need(original_hashes.get(relative) == expected, 'original thin science source differs')
    audit = card['audit_execution']
    need(audit['source_commit'] == AUDIT_SOURCE and audit['card']['path'] == str(AUDIT_CARD)
         and audit['card']['sha256'] == AUDIT_CARD_SHA and audit['plan']['path'] == str(AUDIT_PLAN)
         and audit['plan']['sha256'] == AUDIT_PLAN_SHA, 'exact original audit002 source/card/plan required')
    ac, ap, ar, aa, audit_files, _ = original_execution(audit, True)
    audit_source = str(D / 'audit_thin_pilot_002.py')
    need(ac['schema'] == 'codex-task33-thin-audit-execution-card-v1'
         and ac['auditor_sha256'] == AUDITOR_SHA and ac['original_execution'] == thin
         and ac['result_path'] == str(AUDIT_RESULT), 'independent audit execution card differs')
    need({row['relative_path']: row['sha256'] for row in aa['source_blobs']}.get(
         'codex/coexistence-task33/audit_thin_pilot_002.py') == AUDITOR_SHA, 'unreviewed original saved-audit implementation')
    audit_command = [PYTHON, '-I', '-B', audit_source, '--source', thin['source_commit'],
                     '--card', str(THIN_CARD), '--card-sha256', THIN_CARD_SHA,
                     '--plan', thin['plan']['path'], '--plan-sha256', thin['plan']['sha256'],
                     '--receipt-sha256', thin['receipt']['sha256'],
                     '--admission-sha256', thin['admission']['sha256'], '--output', str(AUDIT_RESULT)]
    need(audit['argv'] == audit_command and audit['output'] == str(AUDIT_RESULT.parent), 'audit allocation/arguments differ')
    report_pin = verify(card['audit_result'])
    need(report_pin['path'] == str(AUDIT_RESULT) and audit_files.get(str(AUDIT_RESULT)) == report_pin,
         'audit report absent from original successful audit receipt')
    report = read(AUDIT_RESULT)
    need(report['schema'] == 'codex-task33-thin-pilot-saved-audit-v1', 'saved audit schema differs')
    false_flags(report)
    need(report['reachability_proved'] is False and report['original_partial_evidence_preserved'] is True
         and report['status'] == ('complete_success' if thin_success else 'closed_partial_failure'),
         'saved audit scope or original outcome differs')
    original = report['original_execution']
    for key in ('source_commit', 'receipt', 'admission', 'card', 'plan'):
        need(original[key] == thin[key], 'audit original authority join differs')
    need(original['raw_root'] == str(THIN_RAW) and original['output_files'] == tr['output_files']
         and original['source_blobs'] == ta['source_blobs'] and original['returncode'] == tr['returncode']
         and original['failure'] == tr['failure'] and original['host_original_groups_absent'] is True,
         'audit original execution receipt join differs')
    initial, entry = report['initial_state'], report['entry_candidate']
    need(type(initial) is dict and type(entry) is dict and report['steps_checked'] >= 1
         and len(report['steps']) == report['steps_checked'], 'original audited first candidate absent')
    initial_pin, step_pin = thin_files[str(THIN_RAW / 'centre-000.json')], thin_files[str(THIN_RAW / 'step-001.json')]
    need(initial['file'] == initial_pin and entry['file'] == step_pin and entry['step'] == 1
         and initial['component_count'] == DIMENSION and initial['exact_checkpoint_row_match'] is True
         and initial['source_authority'] == tc['producer_initial_state_authority'], 'initial row/step pin join differs')
    initial_record, step = read(initial_pin['path']), read(step_pin['path'])
    false_flags(initial_record); false_flags(step)
    need(initial_record['step'] == 0 and step['step'] == 1, 'original first-step indices differ')
    c0, c1 = vector(initial_record['centre'], True), vector(step['next_centre'], True)
    need(vector(step['input_centre'], True) == c0, 'original step1 input differs from exact initial row')
    embedding = initial['embedding']
    need(embedding['asset']['sha256'] == ASSETS['model.safetensors']
         and embedding['tensor'] == 'model.embed_tokens.weight' and embedding['stored_dtype'] == 'BF16'
         and embedding['shape'] == [VOCABULARY, DIMENSION] and embedding['row'] == 198
         and embedding['row_bytes'] == 2 * DIMENSION and type(embedding['row_byte_offset']) is int
         and embedding['row_byte_offset'] >= 8
         and embedding['exact_rational_values_sha256'] == canonical([qrecord(value) for value in c0]),
         'audited exact checkpoint row authority differs')
    need(embedding['asset']['path'] == str(CHECKPOINT_RESOLVED)
         and embedding.get('snapshot_authority') == checkpoint_definition(),
         'auditor002 exact checkpoint chain authority required')
    hash_string(embedding['row_raw_sha256'])
    images, defects = vector(step['full_image']), vector(step['defects'])
    for image, centre, defect in zip(images, c1, defects):
        need(centre == image[0] and defect[2] <= image[2] - centre and defect[3] >= image[3] - centre,
             'original midpoint selection or full signed defect containment differs')
    defect_max = max(max(abs(row[2]), abs(row[3])) for row in defects)
    _, _, lower, delta0 = bounds(step['defect_inf_upper'])
    need(lower >= 0 and delta0 >= defect_max, 'saved original upper understates complete signed defects')
    checked = report['steps'][0]
    checks = {'exact_input_chain', 'full_image_endpoints', 'exact_image_midpoints',
              'signed_defect_contains_image_minus_next', 'saved_upper_contains_all_defect_endpoints'}
    need(checked['file'] == step_pin and checked['step'] == 1 and checked['component_count'] == DIMENSION
         and set(checked['checks']) == checks and all(value is True for value in checked['checks'].values())
         and rational(checked['computed_defect_endpoint_max']) == defect_max
         and rational(checked['saved_defect_inf_upper_exact']) == delta0
         and rational(entry['delta0']) == rational(entry['e1_upper']) == delta0,
         'audit step1 exact recombination differs')
    centre_hash = canonical(step['next_centre'])
    need(checked['next_centre_canonical_sha256'] == entry['next_centre_canonical_sha256'] == centre_hash,
         'original complete next-centre hash differs')
    radius = 2 * delta0 if delta0 > 0 else F(1, 2**2048)
    need(radius > 0 and radius.denominator & (radius.denominator - 1) == 0 and delta0 <= radius,
         'positive exact dyadic radius containing the authenticated first error required')
    selection = dict(schema='codex-task33-first-tube-selection-v1', initial_file=initial_pin,
                     step1_file=step_pin, audit_result=report_pin, embedding=embedding,
                     centre_canonical_sha256=centre_hash, centre=step['next_centre'], delta0=qrecord(delta0),
                     radius=qrecord(radius), radius_rule='2*delta0 if positive; otherwise exact 2^-2048',
                     incoming_error_le_radius=True, **FLAGS)
    return step['next_centre'], c0, radius, selection


def prefix_authority(card):
    authority = card['prefix_execution']
    pc, pp, pr, pa, files, _ = original_execution(authority, True)
    need(authority['source_commit'] == PREFIX['prefix_source_commit']
         and authority['receipt']['path'] == PREFIX['prefix_receipt']
         and authority['receipt']['sha256'] == PREFIX['prefix_receipt_sha256']
         and authority['admission']['path'] == PREFIX['prefix_admission_path']
         and authority['admission']['sha256'] == PREFIX['prefix_admission_sha256']
         and authority['output'] == PREFIX['prefix_output'] and authority['lane'] == 'prefix'
         and authority['plan']['path'] == str(D / 'prefix-plan.json')
         and authority['card']['path'] == str(D / 'prefix-card.json')
         and authority['card']['sha256'] == '6c2ad90ef0332733f8f249833ec9d2adcd0799b1ec6c4dc415202df0b78e851c',
         'original Task33 prefix source/card/execution differs')
    need(authority['argv'] == [PYTHON, '-I', '-B', str(D / 'prefix_setup.py'), '--card',
         str(D / 'prefix-card.json'), '--out', PREFIX['prefix_output']]
         and (pc['alpha'], pc['log10T'], pc['precision']) == (ALPHA, LOG10_TEMPERATURE, 2048),
         'original prefix science command differs')
    blobs = {row['relative_path']: row['sha256'] for row in pa['source_blobs']}
    need(blobs.get('codex/coexistence-task33/prefix_setup.py') ==
         '72a95c042b4d5e229264b07886733b23c385f37b0dd95009dfa067935877fbc3'
         and blobs.get('codex/certify-llm/ball_qwen.py') == PREFIX['prefix_map_source_sha256'],
         'original prefix evaluated source differs')
    for name, expected in PREFIX_FILES.items():
        need(files[str(Path(PREFIX['prefix_output']) / name)]['sha256'] == expected,
             'original prefix produced bytes differ')
    report = read(Path(PREFIX['prefix_output']) / 'report.json')
    need(str(Path(PREFIX['prefix_output']) / 'report.json') in files
         and report['completed'] is True and report['precision'] == 2048
         and report['layers'] == list(range(24)), 'original prefix completion differs')
    return authority


def map_definition(prefix_ids):
    return dict(schema='codex-qwen-exact-soft-map-v1', dimension=DIMENSION, vocabulary=VOCABULARY,
                layers=LAYERS, pinned_assets=ASSETS, prefix_ids=list(prefix_ids), steering_layer=12,
                alpha_ratio=list(ALPHA.as_integer_ratio()),
                log10_temperature_ratio=list(LOG10_TEMPERATURE.as_integer_ratio()),
                temperature='exp(log(10)*exact_dyadic_log10T)', rotary='real base^(-2j/head_dim), real sin/cos',
                state='one appended full-dimensional soft embedding after the fixed prefix',
                readout='full-vocabulary expected embedding with the checkpoint embedding matrix',
                checkpoint_constants='stored binary constants interpreted as exact dyadics')


def backend_definition(flint, numpy, torch, safetensors, threadpoolctl):
    extensions = {}
    for name, module in sorted(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if name.startswith('flint.') and filename and Path(filename).suffix == '.so':
            extensions[name] = pin(Path(filename).resolve())
    need(extensions, 'actual FLINT extension binary inventory required')
    return dict(python=sys.version, executable=pin(Path(sys.executable).resolve()),
                python_flint=flint.__version__, flint=flint.__FLINT_VERSION__, numpy=numpy.__version__,
                torch=str(torch.__version__), safetensors=safetensors.__version__,
                threadpoolctl=threadpoolctl.__version__, flint_extensions=extensions,
                precision=flint.ctx.prec, threads=flint.ctx.threads)


def current_admission(card, card_pin, out, *, initial_checkpoint=True):
    need(card['schema'] == 'codex-task33-one-tube-card-v1' and card['status'] == 'reviewed_for_one_tube_execution'
         and card['repository'] == str(R) and card['output'] == str(out)
         and card['target_calls'] == 1 and card['precision'] == PRECISION and card['directions'] == 1
         and card['dimension'] == DIMENSION and card['layers'] == LAYERS
         and card['vocabulary'] == VOCABULARY and card['token_chunk'] == CHUNK,
         'new reviewed complete single-tube card required')
    false_flags(card)
    need(card.get('reachability_proved') is False and card.get('conditional_operator_bound_only') is True,
         'operator-only card flags required')
    need(card['prefix'] == PREFIX and card['regularity_contract'] == REGULARITY_CONTRACT,
         'exact original prefix and reviewed C1 interval-Jet contract required')
    admission = read(card['supervision_admission'])
    source = hash_string(admission['source_commit'], 40)
    source_pin(source, card_pin, True)
    if initial_checkpoint:
        for reference in ('HEAD', 'origin/main'):
            need(subprocess.check_output(['git', 'rev-parse', reference], cwd=R).decode().strip() == source,
                 'initial private checkpoint differs from supervisor source')
    plan_pin = pin(card['supervision_plan']); source_pin(source, plan_pin, True)
    plan = read(plan_pin['path']); lane = plan['lanes']['tube']
    command = [PYTHON, '-I', '-B', str(SELF), '--card', card_pin['path'],
               '--card-sha256', card_pin['sha256'], '--out', str(out)]
    need(plan['status'] == 'reviewed_for_execution' and plan['repository'] == str(R)
         and plan['raw_root'] == str(B) and plan['cpu_threads'] == 1 and plan['python'] == PYTHON
         and [plan['python'], '-I', '-B', lane['entry'], *lane['arguments']] == command
         and admission['argv'] == command and admission['lane'] == 'tube'
         and lane['output'] == str(out) and lane['limits'] == card['limits'] == admission['limits']
         and card['supervision_admission'] == str(Path(lane['supervision']) / 'admission.json'),
         'current exact supervisor allocation/command/limits differs')
    ap = dict(admission['plan'])
    if not Path(ap['path']).is_absolute(): ap['path'] = str(R / ap['path'])
    need(ap == plan_pin and all(os.environ.get(key) == admission['threads'].get(key) == '1' for key in THREADS),
         'current original plan/thread admission differs')
    for field, limit in (('as_bytes', resource.RLIMIT_AS), ('cpu_seconds', resource.RLIMIT_CPU),
                         ('output_bytes', resource.RLIMIT_FSIZE)):
        need(type(card['limits'][field]) is int and card['limits'][field] > 0
             and resource.getrlimit(limit) == (card['limits'][field],) * 2, 'actual worker resource bound differs')
    need(type(card['limits']['wall_seconds']) is int and card['limits']['wall_seconds'] > 0
         and os.getpriority(os.PRIO_PROCESS, 0) == 10
         and sorted(os.sched_getaffinity(0)) == plan['cpu_ids'] and len(plan['cpu_ids']) == 1,
         'bounded nice10 single-CPU worker required')
    sources = card['source_sha256']
    need(type(sources) is dict and all(sources.get(key) == value for key, value in CORE_SHA.items())
         and sources.get(str(SELF.relative_to(R))) == pin(SELF)['sha256'], 'current reviewed full source bundle differs')
    review = verify(card['regularity_review']); source_pin(source, review, True)
    need(sources.get(str(Path(review['path']).relative_to(R))) == review['sha256'], 'regularity review absent from source bundle')
    planned = by_path(plan['source_files'])
    recorded = {str(R / row['relative_path']): {key: row[key] for key in ('bytes', 'sha256')}
                for row in admission['source_blobs']}
    need(len(recorded) == len(admission['source_blobs']) and
         {key: {field: value[field] for field in ('bytes', 'sha256')} for key, value in planned.items()} == recorded
         and planned.get(card_pin['path']) == card_pin, 'complete current source/card inventory differs')
    for relative, expected in sources.items():
        record = planned.get(str(R / relative))
        need(record is not None and record['sha256'] == hash_string(expected), 'current source missing from original admission')
        source_pin(source, record, True)
    for record in planned.values(): source_pin(source, record, True)
    preflight = by_path(lane['preflight_pins'])
    required = [card['audit_result'], review]
    for execution in (card['audit_execution'], card['thin_execution'], card['prefix_execution']):
        required.extend(execution[key] for key in ('card', 'plan', 'receipt', 'admission'))
    required.extend(card['input_files'])
    for record in required:
        need(preflight.get(record['path']) == record, 'required authority/input missing from supervisor preflight')
        verify(record)
    return dict(source_commit=source, card=card_pin, plan=plan_pin,
                supervision_admission=pin(card['supervision_admission']),
                source_sha256=sources, regularity_review=review, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', required=True)
    parser.add_argument('--card-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    # The /tmp preparation cannot load evidence or a model accidentally.
    need(Path(__file__).resolve() == SELF, 'reviewed committed repository installation required')
    card_path, out = Path(args.card), Path(args.out)
    need(card_path.parent == D and out.parent == B and out.name.startswith('tube-')
         and out.resolve() == out and not out.exists() and not out.is_symlink(), 'fresh separate owned tube allocation required')
    card_pin = pin(card_path)
    need(card_pin['sha256'] == hash_string(args.card_sha256), 'explicit reviewed card hash differs')
    card = read(card_path)
    # Admission failure before allocation is preserved by the external supervisor.
    binding = current_admission(card, card_pin, out)
    out.mkdir(exist_ok=False)
    started = time.monotonic(); phase = 'saved_evidence'; calls = 0; written = {}

    def save(name, value):
        need(Path(name).name == name and name.endswith('.json'), 'one direct JSON artifact name required')
        value = dict(value)
        for key, expected in FLAGS.items():
            need(key not in value or value[key] is expected, 'artifact scope flag conflicts')
            value[key] = expected
        data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        descriptor = os.open(out / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        directory = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        written[name] = pin(out / name)

    try:
        save('admission.json', binding)
        records, initial_exact, radius, selected = candidate(card)
        save('selection.json', selected)
        authenticated_prefix = prefix_authority(card)
        phase = 'backend_admission'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['HF_HUB_OFFLINE'] = '1'
        sys.path[:0] = [str(D), str(R / 'codex/certify-llm'), str(R / 'codex/crosscheck')]
        import flint
        import numpy as np
        import torch
        import safetensors
        import threadpoolctl
        from flint import arb, ctx
        from ball_linear import DyadicWeightCache
        from cache_io import unpack
        from prefix_import import install_prefix
        from qwen_map import ManualQwenSoftMap, PREFIX_IDS, PINNED_SHA256, SNAPSHOT, VECTOR_FILE
        from streamed_head import StreamedHeadQwenJet
        from tube_receiver import evaluate_receiver
        torch.set_num_threads(1); torch.set_num_interop_threads(1); ctx.prec = 2048; ctx.threads = 1
        for name, relative in (
            ('tube_receiver', 'codex/coexistence-task33/tube_receiver.py'),
            ('tube_operator', 'codex/coexistence-task33/tube_operator.py'),
            ('prefix_import', 'codex/coexistence-task33/prefix_import.py'),
            *((name, 'codex/certify-llm/' + name + '.py') for name in
              ('streamed_head', 'ball_jet', 'ball_qwen', 'ball_linear', 'cache_io')),
            ('qwen_map', 'codex/crosscheck/qwen_map.py'),
        ):
            module = importlib.import_module(name)
            need(Path(module.__file__).resolve() == R / relative and pin(R / relative)['sha256'] == CORE_SHA[relative],
                 'loaded source module identity differs')
        backend = backend_definition(flint, np, torch, safetensors, threadpoolctl)
        need(backend == card['backend'] and canonical(backend) == card['backend_sha256'], 'exact current backend differs')
        current_map = map_definition(PREFIX_IDS)
        need(PINNED_SHA256 == ASSETS and current_map == card['map'] and canonical(current_map) == card['map_sha256'],
             'exact fixed cell/assets/prefix map differs')
        inputs = by_path(card['input_files'])
        for original_file in (selected['initial_file'], selected['step1_file']):
            need(inputs.get(original_file['path']) == original_file,
                 'original initial/step1 evidence missing from pre/post input inventory')
        checkpoint = checkpoint_binding(inputs, selected['embedding'], SNAPSHOT / 'model.safetensors')
        config = config_binding(inputs, SNAPSHOT / 'config.json')
        assets = [VECTOR_FILE if name == 'vectors.npz' else SNAPSHOT / name for name in ASSETS]
        for name, path in zip(ASSETS, assets):
            if name == 'model.safetensors':
                need(path == CHECKPOINT_SNAPSHOT and checkpoint['asset'] == selected['embedding']['asset'],
                     'actual model loader checkpoint differs')
            elif name == 'config.json':
                need(path == CONFIG_SNAPSHOT and config['asset'] == inputs.get(str(CONFIG_RESOLVED)),
                     'actual model loader configuration differs')
            else:
                need(str(path) in inputs and verify(inputs[str(path)])['sha256'] == ASSETS[name], 'full original asset pin differs')
        for path, expected in [(Path(PREFIX['prefix_receipt']), PREFIX['prefix_receipt_sha256']),
                               (Path(PREFIX['prefix_admission_path']), PREFIX['prefix_admission_sha256']),
                               *((Path(PREFIX['prefix_output']) / name, value) for name, value in PREFIX_FILES.items())]:
            need(str(path) in inputs and verify(inputs[str(path)])['sha256'] == expected, 'original prefix file pin differs')
        with threadpoolctl.threadpool_limits(limits=1):
            phase = 'model_load'
            model = ManualQwenSoftMap.from_pinned_files()
            need(model.d == DIMENSION and model.layers == LAYERS and model.vocab == VOCABULARY
                 and model.steering_layer == 12 and PREFIX_IDS[-1] == 198, 'complete exact target model required')
            weights = {name: value.numpy() for name, value in model.weights.items()}
            for value in weights.values(): value.flags.writeable = False
            steering = model.vector.numpy(); steering.flags.writeable = False
            # Rejoin the actual authenticated loader row without a model forward.
            need([F(float(value)) for value in weights['model.embed_tokens.weight'][198]] == initial_exact,
                 'loaded exact checkpoint row differs from original saved initial centre')
            ball = StreamedHeadQwenJet(model.config, weights, PREFIX_IDS, steering,
                                      direction_count=1, token_chunk=1024, alpha=ALPHA,
                                      log10_temperature=LOG10_TEMPERATURE,
                                      linear_options=dict(chunk_rows=64, chunk_cols=896,
                                                          cache=DyadicWeightCache(max_elements=262144)))
            phase = 'prefix_restore'
            install_prefix(PREFIX, ball, ASSETS, lambda path: pin(path)['sha256'], unpack)
            need(ball.cache is not None and len(ball.cache) == LAYERS and ball.precision == 2048,
                 'original full 2048-bit prefix was not installed')
            need(ball.epsilon > 0 and ball.temperature > 0 and ball.scale.is_finite()
                 and all(value.is_finite() for value in ball.frequencies), 'regularity constants not admitted')
            # Decode exact c1 only; never strip a radius from a saved image.
            centres = [arb(tuple(record['mid'])) for record in records]
            need(all(value.is_exact() and value.is_finite() for value in centres)
                 and [F(int(value.mid().man_exp()[0])) * F(2) ** int(value.mid().man_exp()[1]) for value in centres]
                 == [row[0] for row in vector(records)], 'exact c1 decoding differs')
            save('prefix-binding.json', dict(prefix=PREFIX, prefix_files=PREFIX_FILES,
                 original_prefix_execution=authenticated_prefix, checkpoint=checkpoint, config=config,
                 stored_precision=2048, receiver_precision=2048, original_radii_retained=True,
                 map=current_map, map_sha256=canonical(current_map), backend=backend,
                 backend_sha256=canonical(backend), regularity_contract=REGULARITY_CONTRACT,
                 regularity_review=card['regularity_review']))
            phase = 'one_whole_tube_call'
            need(type(ball) is StreamedHeadQwenJet and ball.forward.__func__ is StreamedHeadQwenJet.forward,
                 'actual reviewed bound forward required')
            calls += 1
            began = time.monotonic()
            result = evaluate_receiver(ball, centres, radius, save)
            forward_seconds = time.monotonic() - began
            phase = 'post_call_admission'
            need(calls == 1 and (ctx.prec, ctx.threads) == (2048, 1), 'single-call/context contract differs')
            need({'tube-inputs.json', 'actual-receiver.json', 'tube-operator.json', 'coverage.json'} <= set(written),
                 'full receiver output and final coverage artifacts required')
            saved_operator = read(out / 'tube-operator.json')
            need(all(saved_operator[key] == value for key, value in result.items()), 'saved full signed result differs')
            cover = read(out / 'coverage.json')
            expected_head = [[kind, first, min(first + CHUNK, VOCABULARY)] for kind in
                             ('head_logits', 'head_derivatives') for first in range(0, VOCABULARY, CHUNK)]
            need(cover['started_layers'] == cover['ended_layers'] == list(range(LAYERS))
                 and cover['head_passes'] == expected_head and cover['actual_receiver_equals_original'] is True
                 and cover['all_direction_cube_retained'] is True, 'final complete original coverage differs')
            need(current_admission(card, card_pin, out, initial_checkpoint=False) == binding, 'post-call source/card/admission changed')
            for record in card['input_files']: verify(record)
            need(checkpoint_binding(inputs, selected['embedding'], SNAPSHOT / 'model.safetensors') == checkpoint,
                 'post-call original checkpoint content/chain changed')
            need(config_binding(inputs, SNAPSHOT / 'config.json') == config,
                 'post-call original configuration content/chain changed')
            for record in written.values(): verify(record)
            need(backend_definition(flint, np, torch, safetensors, threadpoolctl) == backend,
                 'post-call arithmetic backend changed')
            save('status.json', dict(schema='codex-task33-one-tube-status-v1', status='operator_measurement_completed',
                 target_calls=calls, binding=binding, selection=written['selection.json'], artifacts=list(written.values()),
                 forward_seconds=forward_seconds, elapsed_seconds=time.monotonic() - started,
                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 L=result['L'], full_domain_gates_succeeded=True,
                 scope='One conditional whole-tube infinity-operator bound. External original supervision closure '
                       'and independent saved-output audit remain required; no trajectory, reachability or certificate.'))
    except BaseException as error:
        save('failure.json', dict(schema='codex-task33-one-tube-failure-v1', status='failed_or_incomplete',
             phase=phase, target_calls_started=calls, error_type=type(error).__name__, message=str(error),
             traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started,
             artifacts=list(written.values()), max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             scope='All original partial outputs retained. An operator file without successful final status and '
                   'original supervisor closure is not an accepted full-map measurement. No automatic retry.'))
        raise


if __name__ == '__main__':
    main()
