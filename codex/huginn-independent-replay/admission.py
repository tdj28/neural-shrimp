"""Local immutable-input admission for the independent Huginn replay.

The caller must externally review and SHA-pin the manifest, including the exact
receipt assertions for completion, retrieval, termination, source/invocation,
assets and ordering. This module verifies those assertions and all bytes; it
cannot authenticate a producer's statements or reconstruct historical execution.
It never extracts archives, calls a model/network/producer, or chooses evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

FROZEN = '9805ed32055e7ef790df0cc67316357c1067aef0'
ANALYSIS = 'a38586151f9acf6e60c8d355df22d44d8b8ca7aec78270a6f6e73cfc6b838dc2'
DATA_ROOT = Path('/data2/neural-shrimp-data')
OBLIGATIONS = {'complete_producer_scope', 'retrieval', 'verified_termination',
               'source_and_invocation', 'resolved_assets', 'analysis_completed'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pinned_path(pin, root):
    if not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}:
        raise ValueError('file pin requires exactly path, bytes and sha256')
    path = Path(pin['path'])
    if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root.resolve()):
        raise ValueError('input must be a canonical local file under the data root')
    if type(pin['bytes']) is not int or pin['bytes'] < 0:
        raise ValueError('invalid file byte count')
    if not isinstance(pin['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', pin['sha256']):
        raise ValueError('invalid SHA256')
    if not path.is_file() or path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
        raise ValueError('input missing or byte/hash mismatch: ' + str(path))
    return path


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('duplicate JSON object key: ' + key)
        out[key] = value
    return out


def read_json(path):
    # NaN in a frozen spectrum/result is a retained undefined observation. The
    # independent reducers/comparator handle its scientific status explicitly.
    return json.loads(path.read_text(), object_pairs_hook=_pairs)


def read_jsonl(path):
    return [json.loads(line, object_pairs_hook=_pairs) for line in path.read_text().splitlines() if line.strip()]


def _pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('assertion requires an explicit JSON pointer')
    for key in pointer[1:].split('/'):
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _same_json(left, right):
    # In particular True must not silently equal 1 in receipt assertions.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def validate_manifest(manifest, part):
    if (manifest.get('schema') != 'codex-huginn-replay-inputs-v1' or part not in ('A', 'B')
            or manifest.get('part') != part or manifest.get('frozen_commit') != FROZEN
            or manifest.get('frozen_analysis_sha256') != ANALYSIS):
        raise ValueError('wrong frozen part/source contract')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or any(not isinstance(k, str) or not k for k in files):
        raise ValueError('nonempty uniquely keyed file inventory required')
    roles = manifest.get('roles', {})
    expected = {'rows', 'producer_analysis', 'analysis_source'} | ({'spectra', 'absent_spectra'} if part == 'A' else set())
    if set(roles) != expected or not isinstance(roles['rows'], list):
        raise ValueError('wrong part role inventory')
    # Operational amendment 001: B may need completion partitions; exact
    # 200x4 scientific coverage remains the unchanged reducer's responsibility.
    row_ids = roles['rows']
    if (not row_ids or any(type(name) is not str or not name for name in row_ids)
            or len(set(row_ids)) != len(row_ids) or (part == 'A' and len(row_ids) != 1)):
        raise ValueError('one A readout or a nonempty unique B partition list required')
    role_ids = roles['rows'] + [roles['producer_analysis'], roles['analysis_source']]
    if part == 'A':
        if set(roles['spectra']) != set(map(str, range(64))):
            raise ValueError('all64 canonical spectrum dispositions required')
        missing = {q for q, value in roles['spectra'].items() if value is None}
        if set(roles['absent_spectra']) != missing:
            raise ValueError('explicit absent-path inventory required for missing spectra')
        role_ids += [value for value in roles['spectra'].values() if value is not None]
    if len(role_ids) != len(set(role_ids)) or not set(role_ids) <= set(files):
        raise ValueError('aliased or absent role files')
    if files[roles['analysis_source']]['sha256'] != ANALYSIS:
        raise ValueError('producer analysis source is not the accepted frozen bytes')
    obligations = set(OBLIGATIONS) | ({'predictor_before_readout'} if part == 'A' else set())
    refs = manifest.get('obligations', {})
    assertions = manifest.get('receipt_assertions')
    if set(refs) != obligations or not isinstance(assertions, list) or not assertions:
        raise ValueError('all externally reviewed provenance obligations required')
    used = set()
    for name, indices in refs.items():
        if not isinstance(indices, list) or not indices:
            raise ValueError('empty provenance obligation: ' + name)
        for index in indices:
            if type(index) is not int or not 0 <= index < len(assertions):
                raise ValueError('invalid assertion index')
            used.add(index)
    if used != set(range(len(assertions))):
        raise ValueError('unassigned receipt assertion')
    for assertion in assertions:
        if (set(assertion) != {'file', 'format', 'pointer', 'equals'}
                or assertion['file'] not in files or assertion['format'] not in ('json', 'jsonl')):
            raise ValueError('malformed receipt assertion')
    archives = manifest.get('archives')
    if not isinstance(archives, list) or not archives:
        raise ValueError('retrieved original archive inventory required')
    covered = []
    archive_ids = []
    member_by_file = {}
    for archive in archives:
        if set(archive) != {'file', 'members'} or archive['file'] not in files or not archive['members']:
            raise ValueError('archive/member pins required')
        archive_ids.append(archive['file'])
        for member in archive['members']:
            if set(member) != {'member', 'file'} or member['file'] not in files:
                raise ValueError('invalid selected archive member')
            path = PurePosixPath(member['member'])
            if path.is_absolute() or '..' in path.parts or str(path) != member['member']:
                raise ValueError('noncanonical archive member')
            covered.append(member['file'])
            member_by_file[member['file']] = (archive['file'], member['member'])
    if part == 'A':
        for q, name in roles['spectra'].items():
            if name is not None and (name not in member_by_file or not member_by_file[name][1].endswith(
                    '/q' + f'{int(q):04d}' + '/fixed-spectrum.json')):
                raise ValueError('spectrum question must match its original canonical archive path')
        for q, missing in roles['absent_spectra'].items():
            if (not isinstance(missing, dict) or set(missing) != {'path', 'archive', 'member'}
                    or missing['archive'] not in archive_ids
                    or not missing['member'].endswith('/q' + f'{int(q):04d}' + '/fixed-spectrum.json')):
                raise ValueError('missing spectrum needs original archive-member absence binding')
    targets = set(roles['rows']) | (set(roles['spectra'].values()) - {None} if part == 'A' else set())
    if len(archive_ids) != len(set(archive_ids)) or len(covered) != len(set(covered)) or not targets <= set(covered):
        raise ValueError('raw selected files must each have unique original archive provenance')
    return roles


def authenticate(manifest_path, expected_sha256, part, *, root=DATA_ROOT, max_input_bytes):
    """Verify all bytes and reviewed metadata before decoding scientific inputs.

    `expected_sha256` comes from a separately frozen/reviewed execution card.
    The bound includes whole archives, so the actual card must budget their
    measured byte count. File hashing and archive-member verification stream.
    """
    if digest(manifest_path) != expected_sha256:
        raise ValueError('manifest differs from external authority')
    manifest = read_json(manifest_path)
    roles = validate_manifest(manifest, part)
    if type(max_input_bytes) is not int or max_input_bytes <= 0:
        raise ValueError('measured input byte budget required')
    total = sum(pin['bytes'] for pin in manifest['files'].values())
    if total > max_input_bytes:
        raise ValueError('input inventory exceeds measured card byte bound')
    paths = {name: pinned_path(pin, root) for name, pin in manifest['files'].items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError('duplicate physical input path')
    checks = []
    for assertion in manifest['receipt_assertions']:
        path = paths[assertion['file']]
        record = read_json(path) if assertion['format'] == 'json' else read_jsonl(path)
        actual = _pointer(record, assertion['pointer'])
        if not _same_json(actual, assertion['equals']):
            raise ValueError('pinned receipt assertion failed: ' + assertion['file'] + assertion['pointer'])
        checks.append(assertion)
    archive_checks = []
    for archive in manifest['archives']:
        members = {row['member']: row['file'] for row in archive['members']}
        if len(members) != len(archive['members']):
            raise ValueError('duplicate requested archive member')
        missing_members = {record['member'] for record in roles.get('absent_spectra', {}).values()
                           if record['archive'] == archive['file']}
        spectrum_prefixes = {name.rsplit('/q', 1)[0] for name in set(members) | missing_members
                             if re.search(r'/q[0-9]{4}/fixed-spectrum\.json$', name)}
        seen = set()
        with tarfile.open(paths[archive['file']], 'r|*') as handle:
            for member in handle:
                if part == 'A':
                    for prefix in spectrum_prefixes:
                        match = re.fullmatch(re.escape(prefix) + r'/q([0-9]{4})(?:/.*)?', member.name.rstrip('/'))
                        if match and int(match[1]) not in range(64):
                            raise ValueError('extra spectrum question in original archive')
                if member.name in missing_members:
                    raise ValueError('declared missing spectrum exists in original archive')
                if member.name not in members:
                    continue
                if member.name in seen or not member.isfile():
                    raise ValueError('duplicate or nonregular original archive member')
                seen.add(member.name)
                pin = manifest['files'][members[member.name]]
                if member.size != pin['bytes']:
                    raise ValueError('original archive member size mismatch')
                stream = handle.extractfile(member); h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
                if h.hexdigest() != pin['sha256']:
                    raise ValueError('extracted scientific file differs from original archive')
                archive_checks.append({'archive': archive['file'], 'member': member.name,
                                       'file': members[member.name], 'sha256': pin['sha256']})
        if seen != set(members):
            raise ValueError('selected member absent from completed archive')
    missing = roles.get('absent_spectra', {})
    for q, record in missing.items():
        path = Path(record['path'])
        if not path.is_absolute() or path.resolve() != path or not path.is_relative_to(root.resolve()) or path.exists():
            raise ValueError('declared missing spectrum is not absent: ' + q)
    # Only after complete metadata/hash/archive admission do we decode outcomes.
    rows = [row for name in roles['rows'] for row in read_jsonl(paths[name])]
    spectra = ({int(q): None if name is None else read_json(paths[name])
                for q, name in roles['spectra'].items()} if part == 'A' else None)
    producer = read_json(paths[roles['producer_analysis']])
    recheck(manifest_path, expected_sha256, manifest, root=root)
    report = {'manifest_sha256': expected_sha256, 'part': part, 'file_count': len(paths),
              'total_input_bytes': total, 'receipt_assertions': checks, 'archive_members': archive_checks,
              'missing_spectrum_paths': missing, 'frozen_commit': FROZEN,
              'scope': 'Byte and externally reviewed local receipt authentication; historical execution and producer truth are not independently reconstructed.',
              'model_or_network_called': False, 'certificate_passed': False}
    return {'manifest': manifest, 'rows': rows, 'spectra': spectra, 'producer': producer, 'report': report}


def recheck(manifest_path, expected_sha256, manifest, *, root=DATA_ROOT):
    if digest(manifest_path) != expected_sha256:
        raise ValueError('input manifest changed during replay')
    for pin in manifest['files'].values():
        pinned_path(pin, root)
    for record in manifest['roles'].get('absent_spectra', {}).values():
        if Path(record['path']).exists():
            raise ValueError('missing spectrum appeared during replay')
