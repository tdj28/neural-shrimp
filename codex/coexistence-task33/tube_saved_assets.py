"""Stream only the three reviewed Task33 assets; no generic symlink exception.

The no-argument public API fixes all paths, literal links, sizes and hashes. The
private definition-taking engine exists for tiny synthetic controls only. No
payload is retained or decoded, and no producer/backend/scalar code is imported.
Before/after identity checks detect ordinary mutations, not an atomic snapshot
or future immutability. Matching current bytes never attests prior target use.
The separate ~68 MiB prefix cache is NOT covered; its streamed authentication and
the complete upstream/target authority joins remain future wrapper obligations.
"""
from contextlib import contextmanager
import hashlib
import os
from pathlib import PurePosixPath
import posixpath
import re
import stat


CHUNK_BYTES = 1024**2
MAX_TOTAL_BYTES = 988113343  # Exact reviewed aggregate; no measured runtime claim.
NAMES = ('model.safetensors', 'config.json', 'vectors.npz')
HUB = '/data2/neural-shrimp-data/models/hf/hub'
MODEL = HUB + '/models--Qwen--Qwen2.5-0.5B-Instruct'
SNAPSHOT = MODEL + '/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
MODEL_HASH = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
MODEL_LEAF = HUB + '/blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7'
CONFIG_LEAF = MODEL + '/blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f'
VECTOR = '/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz'


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _target_definitions():
    return {
        'model.safetensors': dict(snapshot_path=SNAPSHOT + '/model.safetensors', links=[
            dict(path=SNAPSHOT + '/model.safetensors', link_target='../../blobs/' + MODEL_HASH),
            dict(path=MODEL + '/blobs/' + MODEL_HASH,
                 link_target='../../blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7')],
            asset=dict(path=MODEL_LEAF, bytes=988097824, sha256=MODEL_HASH)),
        'config.json': dict(snapshot_path=SNAPSHOT + '/config.json', links=[
            dict(path=SNAPSHOT + '/config.json', link_target='../../blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f')],
            asset=dict(path=CONFIG_LEAF, bytes=659, sha256='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45')),
        'vectors.npz': dict(snapshot_path=VECTOR, links=[],
            asset=dict(path=VECTOR, bytes=14860, sha256='b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41')),
    }


def _path(value):
    _require(type(value) is str and 0 < len(value) <= 4096 and '\0' not in value,
             'bounded literal path required')
    path = PurePosixPath(value)
    _require(value.startswith('/') and not value.startswith('//') and str(path) == value and
             '..' not in path.parts and len(path.parts) <= 64, 'canonical absolute path required')
    return path


def _definitions(values):
    _require(type(values) is dict and set(values) == set(NAMES), 'exact three asset definitions required')
    paths, total = set(), 0
    for name, count in zip(NAMES, (2, 1, 0)):
        value = values[name]
        _require(type(value) is dict and set(value) == {'snapshot_path', 'links', 'asset'}, 'exact asset definition fields required')
        _path(value['snapshot_path'])
        links, pin = value['links'], value['asset']
        _require(type(links) is list and len(links) == count, 'exact reviewed link count required')
        _require(type(pin) is dict and set(pin) == {'path', 'bytes', 'sha256'}, 'exact endpoint pin required')
        _path(pin['path'])
        _require(type(pin['bytes']) is int and 0 <= pin['bytes'] <= MAX_TOTAL_BYTES, 'bounded integer endpoint length required')
        _require(type(pin['sha256']) is str and re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'full lowercase SHA256 required')
        total += pin['bytes']
        _require(total <= MAX_TOTAL_BYTES, 'aggregate full-hashing byte ceiling exceeded')
        chain = []
        for row in links:
            _require(type(row) is dict and set(row) == {'path', 'link_target'}, 'exact link declaration required')
            _path(row['path']); target = row['link_target']
            _require(type(target) is str and 0 < len(target) <= 4096 and '\0' not in target and
                     not target.startswith('/') and str(PurePosixPath(target)) == target, 'literal relative link required')
            chain.append(row['path'])
        chain.append(pin['path'])
        _require(value['snapshot_path'] == chain[0], 'snapshot must start exact chain')
        for index, row in enumerate(links):
            _require(posixpath.normpath(posixpath.join(posixpath.dirname(row['path']), row['link_target'])) == chain[index + 1],
                     'literal target does not join next declared chain member')
        for path in chain:
            _require(path not in paths, 'duplicate path or asset alias')
            paths.add(path)
    return values


def _version(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _directory_identity(info):
    return [info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)]


def _regular(info):
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'canonical regular endpoint without hardlink aliases required')


@contextmanager
def _parent(path):
    _require(hasattr(os, 'O_NOFOLLOW') and hasattr(os, 'O_DIRECTORY'), 'no-follow directory support required')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags); seen = {'/': _directory_identity(os.fstat(fd))}; current = PurePosixPath('/')
    try:
        for part in path.parent.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd); fd = child; current /= part
            seen[str(current)] = _directory_identity(os.fstat(fd))
        yield fd, seen
    finally:
        os.close(fd)


def _snapshot(definitions):
    directories, leaves, inode_paths = {}, {}, {}
    for name in NAMES:
        definition = definitions[name]
        rows = [(row['path'], row['link_target']) for row in definition['links']]
        rows.append((definition['asset']['path'], None))
        for filename, target in rows:
            path = _path(filename)
            with _parent(path) as (parent, seen):
                for directory, ident in seen.items():
                    _require(directory not in directories or directories[directory] == ident, 'ancestor identity changed during snapshot')
                    directories[directory] = ident
                info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                if target is None:
                    _regular(info)
                    _require(info.st_size == definition['asset']['bytes'], 'endpoint pinned length differs')
                else:
                    _require(stat.S_ISLNK(info.st_mode) and info.st_nlink == 1, 'exact single-alias symlink required')
                    _require(os.readlink(path.name, dir_fd=parent) == target, 'literal symlink target differs')
                _require(_version(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == _version(info),
                         'leaf/link changed during snapshot')
                inode = (info.st_dev, info.st_ino)
                _require(inode not in inode_paths, 'duplicate endpoint/link inode alias')
                inode_paths[inode] = filename
                leaves[filename] = dict(version=_version(info), link_target=target)
    return dict(directories=directories, leaves=leaves)


def _hash_fd(fd, expected_bytes):
    digest, total = hashlib.sha256(), 0
    while True:
        block = os.read(fd, min(CHUNK_BYTES, expected_bytes + 1 - total))
        if not block:
            break
        total += len(block)
        _require(total <= expected_bytes, 'endpoint grew beyond pin')
        digest.update(block)
    _require(total == expected_bytes, 'full endpoint byte count differs')
    return digest.hexdigest()


def _hash_endpoint(pin, before):
    path = _path(pin['path'])
    with _parent(path) as (parent, seen):
        _require(all(before['directories'].get(path) == ident for path, ident in seen.items()), 'ancestor changed before hashing')
        info = os.stat(path.name, dir_fd=parent, follow_symlinks=False); _regular(info)
        expected = before['leaves'][pin['path']]['version']
        _require(_version(info) == expected, 'endpoint changed before hashing')
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            opened = os.fstat(fd); _regular(opened)
            _require(_version(opened) == expected, 'opened endpoint identity differs')
            digest = _hash_fd(fd, pin['bytes'])
            _require(_version(os.fstat(fd)) == expected, 'endpoint changed during hashing')
            _require(_version(os.stat(path.name, dir_fd=parent, follow_symlinks=False)) == expected, 'endpoint path changed during hashing')
        finally:
            os.close(fd)
    _require(digest == pin['sha256'], 'full endpoint SHA256 differs')


def _check_assets(definitions):
    """PRIVATE synthetic-control seam; not an arbitrary-alias public authority API."""
    definitions = _definitions(definitions)
    before = _snapshot(definitions)
    for name in NAMES:
        _hash_endpoint(definitions[name]['asset'], before)
    _require(_snapshot(definitions) == before, 'whole asset chain/ancestor inventory changed during hashing')
    assets = {}
    for name in NAMES:
        definition = definitions[name]; pin = definition['asset']
        authority = None
        if definition['links']:
            authority = dict(snapshot_path=definition['snapshot_path'], links=[row.copy() for row in definition['links']],
                             resolved_path=pin['path'], expected_content_sha256=pin['sha256'])
            if name == 'config.json':
                authority['expected_bytes'] = pin['bytes']
        assets[name] = dict(asset=pin.copy(), snapshot_authority=authority)
    return dict(status='streamed_asset_files_consistent_only', assets=assets, observations=before,
                total_hashed_bytes=sum(value['asset']['bytes'] for value in definitions.values()),
                asset_bytes_and_chains_consistent=True, original_file_authentication_performed=False,
                original_source_authentication_performed=False, original_git_commit_authenticated=False,
                original_execution_authenticated=False, original_target_asset_use_authenticated=False,
                actual_host_observation_authenticated=False, scientific_verdict_admitted=False,
                operator_bound_admitted=False, conditional_operator_bound_admitted=False,
                trajectory_enclosure=False, reachability_proved=False, certificate_passed=False,
                unchecked=['prior target/upstream loading and source/receipt/host authority',
                           'separate prefix-cache bytes, radii, precision and execution',
                           'map/backend/regularity/embedding meaning and scalar arithmetic',
                           'atomic snapshot, future immutability and whole-wrapper final recheck'])


def check_target_assets():
    """Hash exact reviewed model/config/vector only; requires separate run admission.

    The caller must budget full hashing costs and independently join authenticated
    target/upstream/card/prefix/backend records. This call grants no execution,
    source-origin, operator, trajectory or certificate authority.
    """
    return _check_assets(_target_definitions())
