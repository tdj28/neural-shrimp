"""Small saved-file boundary; no execution, Git, scalar or scientific admission.

Pins and source blob bytes are caller-supplied claims, not original authority.
Successful reads bind the returned bytes to those pins. The future wrapper must
authenticate the pins' origin and recheck the whole inventory after its work.
Before/after metadata checks detect ordinary concurrent changes; they are not
an atomic snapshot or a guarantee of future immutability. No asset-link exception
is implemented. All imports are standard library and there is no CLI.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


# Explicit preparation ceilings, not measured target costs. A future resource
# card must confirm adequacy; exceeding a ceiling refuses rather than truncates.
MAX_FILE_BYTES = 64 * 1024**2
MAX_TOTAL_BYTES = 256 * 1024**2
MAX_PINS = 128
OUTPUT_NAMES = (
    'admission.json', 'selection.json', 'prefix-binding.json', 'tube-inputs.json',
    'actual-receiver.json', 'tube-operator.json', 'coverage.json', 'status.json',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _scope():
    return dict(scientific_verdict_admitted=False, operator_bound_admitted=False,
                conditional_operator_bound_admitted=False, trajectory_enclosure=False,
                reachability_proved=False, certificate_passed=False,
                original_execution_authenticated=False,
                original_file_authentication_performed=False,
                original_source_authentication_performed=False,
                original_git_commit_authenticated=False)


def _path(value):
    require(type(value) is str and '\0' not in value, 'literal absolute path required')
    path = Path(value)
    require(value.startswith('/') and not value.startswith('//') and
            str(path) == value and '..' not in path.parts,
            'canonical absolute path spelling required')
    return path


def validate_pin(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'exact pin schema required')
    _path(value['path'])
    require(type(value['bytes']) is int and 0 <= value['bytes'] <= MAX_FILE_BYTES,
            'bounded exact integer byte length required')
    require(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'full lowercase SHA256 required')
    return value.copy()


def _pins(values):
    require(type(values) is list and 0 < len(values) <= MAX_PINS, 'bounded nonempty pin list required')
    result = {}
    for value in values:
        pin = validate_pin(value)
        require(pin['path'] not in result, 'duplicate pin path')
        result[pin['path']] = pin
    require(sum(pin['bytes'] for pin in result.values()) <= MAX_TOTAL_BYTES, 'total byte ceiling exceeded')
    return result


def strict_json_bytes(payload):
    """Decode precisely supplied UTF-8 bytes, rejecting duplicate/nonfinite JSON."""
    require(type(payload) is bytes and len(payload) <= MAX_FILE_BYTES, 'bounded original bytes required')

    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON object key')
            out[key] = value
        return out

    def constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)

    def finite(value):
        if type(value) is float:
            require(math.isfinite(value), 'nonfinite JSON number, including exponent overflow')
        elif type(value) is list:
            for child in value:
                finite(child)
        elif type(value) is dict:
            for child in value.values():
                finite(child)

    decoded = json.loads(payload.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    finite(decoded)
    return decoded


def _version(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _regular(info):
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            'regular file without symlink or hardlink aliases required')


@contextmanager
def _directory(path):
    """Walk every ancestor using no-follow directory descriptors (Linux/POSIX)."""
    require(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY'), 'no-follow directory support required')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def _read_fd(fd, length):
    chunks, total = [], 0
    while True:
        block = os.read(fd, min(1024**2, length + 1 - total))
        if not block:
            return b''.join(chunks)
        total += len(block)
        require(total <= length, 'file grew beyond pinned length')
        chunks.append(block)


def read_pinned_bytes(record):
    """Read once, hash full returned bytes, and check file identity around I/O.

    No pathname is reopened for JSON decoding. Pin origin and later immutability
    remain external obligations. Missing, linked, changed or oversized files fail.
    """
    pin = validate_pin(record)
    path = _path(pin['path'])
    with _directory(path.parent) as parent:
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        _regular(before)
        require(before.st_size == pin['bytes'], 'pinned byte length differs')
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            opened = os.fstat(fd)
            _regular(opened)
            require(_version(opened) == _version(before), 'file identity changed before read')
            payload = _read_fd(fd, pin['bytes'])
            require(_version(os.fstat(fd)) == _version(opened), 'file changed during read')
            require(_version(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == _version(opened),
                    'file path changed during read')
        finally:
            os.close(fd)
    # Rewalk ancestors: a stable descriptor alone would not detect a moved path.
    with _directory(path.parent) as parent:
        after = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        _regular(after)
        require(_version(after) == _version(before), 'canonical path changed after read')
    require(len(payload) == pin['bytes'] and hashlib.sha256(payload).hexdigest() == pin['sha256'],
            'full original byte pin differs')
    return payload


def read_pinned_json(record):
    pin = validate_pin(record)
    payload = read_pinned_bytes(pin)
    return dict(status='pinned_json_consistent_only', pin=pin, raw_bytes=payload,
                record=strict_json_bytes(payload), **_scope())


def _snapshot(root):
    with _directory(root) as directory:
        before = _version(os.fstat(directory))
        names = []
        with os.scandir(directory) as entries:
            for entry in entries:
                require(len(names) < len(OUTPUT_NAMES), 'extra output entry')
                names.append(entry.name)
        require(set(names) == set(OUTPUT_NAMES) and len(names) == len(OUTPUT_NAMES),
                'exact eight flat successful artifacts required; partial/failure inventories refused')
        entries, identities = {}, set()
        for name in names:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            _regular(info)
            identity = (info.st_dev, info.st_ino)
            require(identity not in identities, 'duplicate file identity')
            identities.add(identity)
            entries[name] = _version(info)
        require(_version(os.fstat(directory)) == before, 'output directory changed during snapshot')
        return before, entries


def read_flat_output_inventory(root, records):
    """Require all eight direct artifacts; return same pinned bytes and decoding.

    This validates neither successful execution nor the meaning of these records.
    The full authority/scalar wrapper is still pending; a partial attempt refuses.
    """
    root = _path(root)
    pins = _pins(records)
    require(set(pins) == {str(root / name) for name in OUTPUT_NAMES}, 'exact direct output pin set required')
    before = _snapshot(root)
    raw, decoded = {}, {}
    for name in OUTPUT_NAMES:
        raw[name] = read_pinned_bytes(pins[str(root / name)])
        decoded[name] = strict_json_bytes(raw[name])
    require(_snapshot(root) == before, 'whole output inventory changed during reads')
    return dict(status='flat_file_consistency_only', root=str(root),
                pins={name: pins[str(root / name)] for name in OUTPUT_NAMES},
                raw_bytes=raw, records=decoded, **_scope())


def check_source_blobs(source_root, records, blobs):
    """Compare caller-provided original Git blob bytes with exact source pins.

    No filesystem or subprocess call occurs here. The caller must independently
    authenticate the commit and extraction; matching bytes do not prove origin.
    """
    root, pins = _path(source_root), _pins(records)
    require(type(blobs) is dict, 'relative-path to original bytes mapping required')
    expected = {}
    for name, pin in pins.items():
        relative = _path(name).relative_to(root)
        require(relative.parts, 'source file must be below source root')
        expected[relative.as_posix()] = pin
    require(set(blobs) == set(expected), 'exact complete source blob key set required')
    for relative, pin in expected.items():
        payload = blobs[relative]
        require(type(payload) is bytes and len(payload) == pin['bytes'] and
                hashlib.sha256(payload).hexdigest() == pin['sha256'], 'original source blob bytes differ')
    return dict(status='source_blob_bytes_consistent_only', source_root=str(root),
                pins_by_relative_path=expected, **_scope())
