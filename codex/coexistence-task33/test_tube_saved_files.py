"""NEW synthetic file-boundary controls; no scalar, model or original audit.

Prepared for Root's separate checkpoint/admission. Fixtures are tiny fresh
temporary files. Importing the helper invokes no CLI or target-file reads.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('tube_files_under_test',
                                             Path(__file__).with_name('tube_saved_files.py'))
files = importlib.util.module_from_spec(spec)
spec.loader.exec_module(files)


class FileBoundaryControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='task33-file-controls-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()

    def pin(self, path, payload):
        return dict(path=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())

    def fixture(self, name='record.json', payload=b'{ "value": 3 }\n'):
        path = self.base / name
        path.write_bytes(payload)
        return path, self.pin(path, payload), payload

    def bundle(self, name='bundle'):
        root = self.base / name
        root.mkdir()
        pins, payloads = [], {}
        for index, basename in enumerate(files.OUTPUT_NAMES):
            payload = json.dumps({'synthetic_index': index, 'not_execution_authority': True}).encode() + b'\n'
            (root / basename).write_bytes(payload)
            pins.append(self.pin(root / basename, payload))
            payloads[basename] = payload
        return root, pins, payloads

    def no_admission(self, result):
        for name in ('scientific_verdict_admitted', 'operator_bound_admitted',
                     'conditional_operator_bound_admitted', 'trajectory_enclosure',
                     'reachability_proved', 'certificate_passed',
                     'original_execution_authenticated', 'original_file_authentication_performed',
                     'original_source_authentication_performed', 'original_git_commit_authenticated'):
            self.assertIs(result[name], False, name)
        self.assertNotIn('L', result)
        self.assertNotIn('L_candidate', result)

    def test_strict_json_valid_original_utf8(self):
        self.assertEqual(files.strict_json_bytes(b'{"a":[1,1.25,true,null,{"b":"x"}]}'),
                         {'a': [1, 1.25, True, None, {'b': 'x'}]})

    def test_duplicate_keys_at_every_level(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":[{"a":1,"a":2}]}',
                    b'{"a":1,"\\u0061":2}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                files.strict_json_bytes(raw)

    def test_nonfinite_constants_and_recursive_overflow(self):
        for raw in (b'NaN', b'Infinity', b'-Infinity', b'1e999',
                    b'{"nested":[{"x":-1e999}]}', b'[0,{"a":NaN}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                files.strict_json_bytes(raw)

    def test_invalid_encoding_and_nonbytes(self):
        for raw in (b'\xff', b'\xef\xbb\xbf{}', '{}'.encode('utf-16'), b'{} {}', '{}', bytearray(b'{}')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                files.strict_json_bytes(raw)

    def test_read_full_original_bytes_and_scope(self):
        path, pin, raw = self.fixture()
        result = files.read_pinned_json(pin)
        self.assertEqual(result['raw_bytes'], raw)
        self.assertEqual(result['record'], {'value': 3})
        self.assertEqual(result['pin'], pin)
        self.assertEqual(path.read_bytes(), raw)
        self.no_admission(result)

    def test_json_decodes_same_read_bytes_without_reopening(self):
        path, pin, raw = self.fixture()
        real_read = files.read_pinned_bytes

        def mutate_after_read(record):
            content = real_read(record)
            path.write_bytes(b'{"different":true}')
            return content

        with patch.object(files, 'read_pinned_bytes', side_effect=mutate_after_read) as read:
            result = files.read_pinned_json(pin)
        read.assert_called_once_with(pin)
        self.assertEqual(result['raw_bytes'], raw)
        self.assertEqual(result['record'], {'value': 3})
        self.no_admission(result)
        # Demonstrates why later immutability still requires the wrapper's recheck.
        with self.assertRaises(ValueError):
            files.read_pinned_bytes(pin)

    def test_pin_exact_types_schema_hash_and_length(self):
        _, pin, _ = self.fixture()
        bad = [dict(pin, bytes=True), dict(pin, bytes=1.0), dict(pin, bytes=-1),
               dict(pin, sha256='A' * 64), dict(pin, sha256='0' * 63),
               dict(pin, extra=1), {k: v for k, v in pin.items() if k != 'bytes'}]
        for record in bad:
            with self.subTest(record=record), self.assertRaises(ValueError):
                files.read_pinned_bytes(record)
        for record in (dict(pin, bytes=pin['bytes'] + 1), dict(pin, sha256='0' * 64)):
            with self.assertRaises(ValueError):
                files.read_pinned_bytes(record)

    def test_path_spelling_aliases_refused(self):
        path, pin, _ = self.fixture()
        for name in ('record.json', '/' + str(path), str(path.parent) + '/./record.json',
                     str(path.parent) + '/../' + path.parent.name + '/record.json',
                     str(path) + '/', str(path) + '\0'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                files.read_pinned_bytes(dict(pin, path=name))

    def test_missing_directory_and_fifo_are_not_regular_files(self):
        path, pin, _ = self.fixture()
        with self.assertRaises(OSError):
            files.read_pinned_bytes(dict(pin, path=str(self.base / 'missing')))
        with self.assertRaises(ValueError):
            files.read_pinned_bytes(dict(pin, path=str(self.base)))
        fifo = self.base / 'fifo'
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            files.read_pinned_bytes(dict(pin, path=str(fifo)))

    def test_leaf_ancestor_and_dangling_symlinks_refused(self):
        path, pin, _ = self.fixture()
        link = self.base / 'link.json'
        link.symlink_to(path)
        dangling = self.base / 'dangling'
        dangling.symlink_to(self.base / 'absent')
        parent = self.base / 'alias-parent'
        parent.symlink_to(self.base, target_is_directory=True)
        for alias in (link, dangling, parent / path.name):
            with self.subTest(alias=alias), self.assertRaises((ValueError, OSError)):
                files.read_pinned_bytes(dict(pin, path=str(alias)))

    def test_hardlink_alias_refused(self):
        path, pin, _ = self.fixture()
        os.link(path, self.base / 'second-name')
        with self.assertRaises(ValueError):
            files.read_pinned_bytes(pin)

    def test_file_replacement_during_read_refused(self):
        path, pin, raw = self.fixture()
        replacement = self.base / 'replacement'
        replacement.write_bytes(raw)
        real_read = files._read_fd

        def replace(fd, length):
            payload = real_read(fd, length)
            os.replace(replacement, path)
            return payload

        with patch.object(files, '_read_fd', side_effect=replace), self.assertRaises(ValueError):
            files.read_pinned_bytes(pin)

    def test_file_size_change_during_read_refused(self):
        path, pin, raw = self.fixture()
        real_read = files._read_fd

        def grow(fd, length):
            payload = real_read(fd, length)
            path.write_bytes(raw + b' ')
            return payload

        with patch.object(files, '_read_fd', side_effect=grow), self.assertRaises(ValueError):
            files.read_pinned_bytes(pin)

    def test_ancestor_replacement_after_read_refused(self):
        parent = self.base / 'parent'
        parent.mkdir()
        path = parent / 'record.json'
        raw = b'{"a":1}'
        path.write_bytes(raw)
        pin = self.pin(path, raw)
        real_read = files._read_fd

        def replace_parent(fd, length):
            payload = real_read(fd, length)
            parent.rename(self.base / 'old-parent')
            parent.mkdir()
            path.write_bytes(raw)
            return payload

        with patch.object(files, '_read_fd', side_effect=replace_parent), self.assertRaises(ValueError):
            files.read_pinned_bytes(pin)

    def test_exact_flat_inventory_and_false_flags(self):
        root, pins, raw = self.bundle()
        result = files.read_flat_output_inventory(str(root), pins)
        self.assertEqual(result['raw_bytes'], raw)
        self.assertEqual(set(result['records']), set(files.OUTPUT_NAMES))
        self.assertEqual(result['pins'], {Path(p['path']).name: p for p in pins})
        self.no_admission(result)

    def test_partial_extra_directory_and_link_inventories_refused(self):
        for case in ('missing', 'extra', 'directory', 'link', 'failure'):
            root, pins, _ = self.bundle(case)
            if case == 'missing':
                (root / files.OUTPUT_NAMES[0]).unlink()
            elif case in ('extra', 'failure'):
                (root / ('failure.json' if case == 'failure' else 'extra.json')).write_bytes(b'{}')
            elif case == 'directory':
                (root / files.OUTPUT_NAMES[0]).unlink()
                (root / files.OUTPUT_NAMES[0]).mkdir()
            else:
                leaf = root / files.OUTPUT_NAMES[0]
                leaf.unlink()
                leaf.symlink_to(root / files.OUTPUT_NAMES[1])
            with self.subTest(case=case), self.assertRaises((ValueError, OSError)):
                files.read_flat_output_inventory(str(root), pins)

    def test_duplicate_missing_and_foreign_pins_refused(self):
        root, pins, _ = self.bundle()
        for records in (pins + [pins[0]], pins[:-1],
                        [dict(pins[0], path=str(self.base / 'foreign.json'))] + pins[1:]):
            with self.assertRaises(ValueError):
                files.read_flat_output_inventory(str(root), records)

    def test_flat_hardlink_alias_refused(self):
        root, pins, _ = self.bundle()
        os.link(root / files.OUTPUT_NAMES[0], self.base / 'external-hardlink')
        with self.assertRaises(ValueError):
            files.read_flat_output_inventory(str(root), pins)

    def test_flat_inventory_mutation_after_member_read_refused(self):
        root, pins, raw = self.bundle()
        real_read = files.read_pinned_bytes
        count = 0

        def mutate(record):
            nonlocal count
            payload = real_read(record)
            count += 1
            if count == len(files.OUTPUT_NAMES):
                (root / files.OUTPUT_NAMES[0]).write_bytes(raw[files.OUTPUT_NAMES[0]] + b' ')
            return payload

        with patch.object(files, 'read_pinned_bytes', side_effect=mutate), self.assertRaises(ValueError):
            files.read_flat_output_inventory(str(root), pins)

    def test_declared_size_total_and_pin_count_ceilings_refuse(self):
        _, pin, _ = self.fixture()
        with self.assertRaises(ValueError):
            files.read_pinned_bytes(dict(pin, bytes=files.MAX_FILE_BYTES + 1))
        with self.assertRaises(ValueError):
            files.check_source_blobs(str(self.base), [pin] * (files.MAX_PINS + 1), {})
        too_much = [dict(pin, path=str(self.base / str(i)), bytes=files.MAX_FILE_BYTES)
                    for i in range(files.MAX_TOTAL_BYTES // files.MAX_FILE_BYTES + 1)]
        with self.assertRaises(ValueError):
            files.check_source_blobs(str(self.base), too_much, {})

    def test_source_blob_bytes_match_without_filesystem_or_git(self):
        root = self.base / 'nonexistent-checkout'
        raw = b'# synthetic original source\n'
        pin = self.pin(root / 'pkg' / 'source.py', raw)
        with patch.object(files.os, 'open', side_effect=AssertionError('filesystem forbidden')):
            result = files.check_source_blobs(str(root), [pin], {'pkg/source.py': raw})
        self.assertEqual(result['pins_by_relative_path'], {'pkg/source.py': pin})
        self.no_admission(result)
        self.assertFalse(root.exists())

    def test_source_blob_exact_keys_types_bytes_and_membership(self):
        raw = b'first\n'
        pin = self.pin(self.base / 'pkg' / 'source.py', raw)
        for blobs in ({}, {'pkg/source.py': raw, 'extra.py': b''},
                      {'pkg/./source.py': raw}, {'pkg/source.py': 'first\n'},
                      {'pkg/source.py': bytearray(raw)}, {'pkg/source.py': b'other\n'},
                      {'pkg/source.py': raw + b' '}):
            with self.subTest(blobs=blobs), self.assertRaises(ValueError):
                files.check_source_blobs(str(self.base), [pin], blobs)
        with self.assertRaises(ValueError):
            files.check_source_blobs(str(self.base), [pin, pin], {'pkg/source.py': raw})
        outside = dict(pin, path=str(self.base.parent / 'outside.py'))
        with self.assertRaises(ValueError):
            files.check_source_blobs(str(self.base), [outside], {'outside.py': raw})


if __name__ == '__main__':
    unittest.main(verbosity=2)
