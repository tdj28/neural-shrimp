"""New tiny-tree controls only; never call the public real-asset I/O path.

Public dispatch is intercepted before filesystem access. All actual I/O uses
fresh synthetic definitions and files under a TemporaryDirectory.
"""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('tube_saved_assets', Path(__file__).with_name('tube_saved_assets.py'))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def tiny_tree(root):
    hub = root / 'hub'; model = hub / 'model'; snapshot = model / 'snapshots/rev'
    endpoint = hub / 'blobs/content'; config = model / 'blobs/config'; vector = root / 'vectors.bin'
    middle = model / 'blobs/model'; model_alias = snapshot / 'model.safetensors'; config_alias = snapshot / 'config.json'
    for directory in (snapshot, endpoint.parent, config.parent): directory.mkdir(parents=True, exist_ok=True)
    payloads = {'model.safetensors': b'synthetic-model\x00' * 5, 'config.json': b'{"synthetic":true}', 'vectors.npz': b'synthetic-vectors'}
    for name, path in (('model.safetensors', endpoint), ('config.json', config), ('vectors.npz', vector)):
        path.write_bytes(payloads[name])
    model_alias.symlink_to('../../blobs/model'); middle.symlink_to('../../blobs/content'); config_alias.symlink_to('../../blobs/config')
    definitions = {}
    for name, alias, path, links in (
        ('model.safetensors', model_alias, endpoint, [(model_alias, '../../blobs/model'), (middle, '../../blobs/content')]),
        ('config.json', config_alias, config, [(config_alias, '../../blobs/config')]),
        ('vectors.npz', vector, vector, []),
    ):
        definitions[name] = dict(snapshot_path=str(alias), links=[dict(path=str(link), link_target=target) for link, target in links],
                                 asset=dict(path=str(path), bytes=len(payloads[name]), sha256=hashlib.sha256(payloads[name]).hexdigest()))
    return definitions, payloads


class AssetControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='task33-assets-synthetic-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.definitions, self.payloads = tiny_tree(self.root)

    def check(self):
        return M._check_assets(self.definitions)

    def refuse(self):
        with self.assertRaises((ValueError, OSError)):
            self.check()

    def test_01_public_dispatch_is_exact_and_does_not_accept_overrides(self):
        hub = '/data2/neural-shrimp-data/models/hf/hub'
        model = hub + '/models--Qwen--Qwen2.5-0.5B-Instruct'
        snapshot = model + '/snapshots/7ae557604adf67be50417f59c2c2f167def9a775'
        digest = 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
        expected = {
            'model.safetensors': dict(snapshot_path=snapshot + '/model.safetensors', links=[
                dict(path=snapshot + '/model.safetensors', link_target='../../blobs/' + digest),
                dict(path=model + '/blobs/' + digest, link_target='../../blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7')],
                asset=dict(path=hub + '/blobs/bb/bb5ff7e71536bbce6378f6d4bb523a77f1e9455965702d18bec33f599d5851f7', bytes=988097824, sha256=digest)),
            'config.json': dict(snapshot_path=snapshot + '/config.json', links=[dict(path=snapshot + '/config.json', link_target='../../blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f')],
                asset=dict(path=model + '/blobs/0dbb161213629a23f0fc00ef286e6b1e366d180f', bytes=659, sha256='18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45')),
            'vectors.npz': dict(snapshot_path='/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz', links=[],
                asset=dict(path='/data2/neural-shrimp-data/explore/llm-phase-map/vectors-l12/vectors.npz', bytes=14860, sha256='b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41')),
        }
        with mock.patch.object(M, '_check_assets', return_value='intercepted') as dispatch:
            self.assertEqual(M.check_target_assets(), 'intercepted')
            dispatch.assert_called_once_with(expected)
            for kwargs in ({'name': 'other'}, {'definitions': self.definitions}, {'path': '/tmp/other'}, {'sha256': '0' * 64}):
                with self.assertRaises(TypeError): M.check_target_assets(**kwargs)
            self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(M.MAX_TOTAL_BYTES, 988113343)
        self.assertEqual(M.CHUNK_BYTES, 1024**2)

    def test_02_tiny_complete_pass_flags_and_no_payload_retention(self):
        before = copy.deepcopy(self.definitions); result = self.check()
        self.assertEqual(before, self.definitions)
        self.assertEqual(result['status'], 'streamed_asset_files_consistent_only')
        self.assertEqual(result['total_hashed_bytes'], sum(len(p) for p in self.payloads.values()))
        self.assertIs(result['asset_bytes_and_chains_consistent'], True)
        for key in ('original_file_authentication_performed', 'original_source_authentication_performed',
                    'original_git_commit_authenticated', 'original_execution_authenticated', 'original_target_asset_use_authenticated',
                    'actual_host_observation_authenticated', 'scientific_verdict_admitted', 'operator_bound_admitted',
                    'conditional_operator_bound_admitted', 'trajectory_enclosure', 'reachability_proved', 'certificate_passed'):
            self.assertIs(result[key], False)
        self.assertEqual(result['assets']['config.json']['snapshot_authority']['expected_bytes'], len(self.payloads['config.json']))
        self.assertIsNone(result['assets']['vectors.npz']['snapshot_authority'])
        self.assertTrue(any('prefix-cache' in text for text in result['unchecked']))
        def no_payload(value):
            self.assertNotIsInstance(value, bytes)
            if isinstance(value, dict):
                for child in value.values(): no_payload(child)
            elif isinstance(value, list):
                for child in value: no_payload(child)
        no_payload(result)

    def test_03_multiple_stream_chunks_hash_every_byte(self):
        original_read = os.read; requested = []
        def bounded_read(fd, length):
            requested.append(length)
            return original_read(fd, length)
        with mock.patch.object(M, 'CHUNK_BYTES', 7), mock.patch.object(M.os, 'read', side_effect=bounded_read):
            self.check()
        self.assertGreater(len(requested), 10)
        self.assertTrue(all(0 < n <= 7 for n in requested))

    def test_04_missing_or_extra_asset_definition(self):
        self.definitions['extra'] = copy.deepcopy(self.definitions['vectors.npz']); self.refuse()
        del self.definitions['extra']; del self.definitions['config.json']; self.refuse()

    def test_05_pin_shape_types_hash_and_total_bound_refuse_before_io(self):
        original = copy.deepcopy(self.definitions)
        for key, value in (('bytes', True), ('bytes', -1), ('bytes', M.MAX_TOTAL_BYTES + 1), ('sha256', 'A' * 64), ('path', '/tmp/../alias')):
            with self.subTest(key=key, value=value):
                self.definitions = copy.deepcopy(original); self.definitions['vectors.npz']['asset'][key] = value
                with mock.patch.object(M, '_snapshot', side_effect=AssertionError('must refuse before I/O')): self.refuse()
        self.definitions = copy.deepcopy(original)
        self.definitions['model.safetensors']['asset']['bytes'] = M.MAX_TOTAL_BYTES
        with mock.patch.object(M, '_snapshot', side_effect=AssertionError('must refuse before I/O')): self.refuse()

    def test_06_exact_link_count_topology_and_canonical_spelling(self):
        original = copy.deepcopy(self.definitions)
        for mode in ('extra', 'missing', 'wrong_next', 'absolute_target', 'dot_target', 'duplicate', 'bad_snapshot'):
            with self.subTest(mode=mode):
                self.definitions = copy.deepcopy(original); model = self.definitions['model.safetensors']
                if mode == 'extra': model['links'].append(copy.deepcopy(model['links'][1]))
                elif mode == 'missing': model['links'].pop()
                elif mode == 'wrong_next': model['links'][0]['link_target'] = '../../blobs/unknown'
                elif mode == 'absolute_target': model['links'][0]['link_target'] = model['links'][1]['path']
                elif mode == 'dot_target': model['links'][0]['link_target'] = '../../blobs/./model'
                elif mode == 'duplicate': self.definitions['vectors.npz']['asset']['path'] = self.definitions['config.json']['asset']['path']; self.definitions['vectors.npz']['snapshot_path'] = self.definitions['config.json']['asset']['path']
                else: model['snapshot_path'] += '/'
                with mock.patch.object(M, '_snapshot', side_effect=AssertionError('must refuse before I/O')): self.refuse()

    def test_07_changed_literal_link_even_with_same_resolution(self):
        link = Path(self.definitions['model.safetensors']['links'][0]['path'])
        link.unlink(); link.symlink_to('../../blobs/./model'); self.refuse()

    def test_08_dangling_link_or_extra_endpoint_link(self):
        endpoint = Path(self.definitions['model.safetensors']['asset']['path'])
        endpoint.unlink(); self.refuse()
        substitute = self.root / 'extra-model'; substitute.write_bytes(self.payloads['model.safetensors'])
        endpoint.symlink_to(substitute); self.refuse()

    def test_09_link_replaced_by_regular_file(self):
        link = Path(self.definitions['config.json']['links'][0]['path'])
        link.unlink(); link.write_bytes(self.payloads['config.json']); self.refuse()

    def test_10_vector_cannot_use_even_an_equivalent_symlink(self):
        vector = Path(self.definitions['vectors.npz']['asset']['path']); moved = self.root / 'other-vector'
        vector.rename(moved); vector.symlink_to(moved); self.refuse()

    def test_11_ancestor_symlink_refused(self):
        original = self.root / 'hub/model'; moved = self.root / 'moved-model'
        original.rename(moved); original.symlink_to(moved, target_is_directory=True); self.refuse()

    def test_12_hardlink_aliases_refused_for_endpoint_and_link(self):
        endpoint = Path(self.definitions['vectors.npz']['asset']['path']); alias = self.root / 'hardlink'
        os.link(endpoint, alias); self.refuse(); alias.unlink()
        link = Path(self.definitions['config.json']['links'][0]['path'])
        os.link(link, alias, follow_symlinks=False); self.refuse()

    def test_13_nonregular_endpoint_refused_without_opening_fifo(self):
        vector = Path(self.definitions['vectors.npz']['asset']['path']); vector.unlink(); os.mkfifo(vector)
        self.refuse()

    def test_14_same_size_wrong_content_and_size_mismatch(self):
        path = Path(self.definitions['vectors.npz']['asset']['path'])
        path.write_bytes(b'x' * len(self.payloads['vectors.npz'])); self.refuse()
        path.write_bytes(self.payloads['vectors.npz'] + b'x'); self.refuse()

    def test_15_growth_during_stream_refuses(self):
        path = Path(self.definitions['model.safetensors']['asset']['path']); original_read = os.read; changed = False
        def mutate(fd, length):
            nonlocal changed
            block = original_read(fd, length)
            if not changed:
                changed = True
                with path.open('ab') as stream: stream.write(b'extra')
            return block
        with mock.patch.object(M, 'CHUNK_BYTES', 7), mock.patch.object(M.os, 'read', side_effect=mutate): self.refuse()

    def test_16_shrink_during_stream_refuses(self):
        path = Path(self.definitions['model.safetensors']['asset']['path']); original_read = os.read; changed = False
        def mutate(fd, length):
            nonlocal changed
            block = original_read(fd, length)
            if not changed:
                changed = True
                with path.open('r+b') as stream: stream.truncate(8)
            return block
        with mock.patch.object(M, 'CHUNK_BYTES', 7), mock.patch.object(M.os, 'read', side_effect=mutate): self.refuse()

    def test_17_leaf_replacement_after_hash_refuses(self):
        path = Path(self.definitions['model.safetensors']['asset']['path']); original_hash = M._hash_fd; changed = False
        def replace(fd, length):
            nonlocal changed
            result = original_hash(fd, length)
            if not changed:
                changed = True; path.rename(self.root / 'old-model'); path.write_bytes(self.payloads['model.safetensors'])
            return result
        with mock.patch.object(M, '_hash_fd', side_effect=replace): self.refuse()

    def test_18_ancestor_replacement_with_same_leaf_refuses(self):
        original_hash = M._hash_fd; changed = False; directory = self.root / 'hub'; moved = self.root / 'old-hub'
        def replace(fd, length):
            nonlocal changed
            result = original_hash(fd, length)
            if not changed:
                changed = True; directory.rename(moved); directory.mkdir()
                # Preserve both leaf-parent directories and every leaf inode;
                # only the higher ancestor identity is substituted.
                for name in ('blobs', 'model'): (moved / name).rename(directory / name)
            return result
        with mock.patch.object(M, '_hash_fd', side_effect=replace): self.refuse()

    def test_19_link_replacement_after_first_hash_refuses(self):
        original_hash = M._hash_fd; changed = False
        row = self.definitions['model.safetensors']['links'][0]; path = Path(row['path'])
        def replace(fd, length):
            nonlocal changed
            result = original_hash(fd, length)
            if not changed:
                changed = True; path.rename(self.root / 'old-link'); path.symlink_to(row['link_target'])
            return result
        with mock.patch.object(M, '_hash_fd', side_effect=replace): self.refuse()

    def test_20_later_asset_read_cannot_hide_earlier_leaf_mutation(self):
        original_hash = M._hash_endpoint; calls = 0
        model = Path(self.definitions['model.safetensors']['asset']['path'])
        def mutate(pin, before):
            nonlocal calls
            original_hash(pin, before); calls += 1
            if calls == 3: model.write_bytes(b'x' * len(self.payloads['model.safetensors']))
        with mock.patch.object(M, '_hash_endpoint', side_effect=mutate): self.refuse()


if __name__ == '__main__':
    unittest.main()
