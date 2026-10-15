"""Focused controls for the supervisor's separately appended output manifest."""
import json
from pathlib import Path
import tempfile
import unittest
import audit


def pin(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=audit.sha(path))


def fixture(root, edit=None):
    path = root / 'report.json'
    path.write_text('{"synthetic":true}\n')
    ref = pin(path)
    row = dict(path=path.name, bytes=ref['bytes'], sha256=ref['sha256'])
    data = dict(files=[row])
    if edit:
        edit(data)
    manifest = root / 'final-manifest.json'
    manifest.write_text(json.dumps(data))
    return {path.name: ref}, pin(manifest)


class ManifestControls(unittest.TestCase):
    def test_valid_appended_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs, manifest = fixture(root)
            self.assertEqual(audit.verify_output_inventory(root, outputs, manifest, {'report.json'}), outputs['report.json']['bytes'])

    def test_mutated_manifest_rows_refuse(self):
        for field, value in [('path', 'other.json'), ('bytes', 99), ('sha256', '0' * 64)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                outputs, manifest = fixture(root, lambda d: d['files'][0].__setitem__(field, value))
                with self.assertRaisesRegex(ValueError, 'manifest path/bytes/hash'):
                    audit.verify_output_inventory(root, outputs, manifest, {'report.json'})

    def test_duplicate_missing_manifest_rows_refuse(self):
        for edit in [lambda d: d['files'].append(dict(d['files'][0])), lambda d: d['files'].clear()]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                outputs, manifest = fixture(root, edit)
                with self.assertRaisesRegex(ValueError, 'manifest file count'):
                    audit.verify_output_inventory(root, outputs, manifest, {'report.json'})

    def test_wrong_manifest_path_and_extra_file_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outputs, manifest = fixture(root)
            other = root / 'other.json'; other.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest path'):
                audit.verify_output_inventory(root, outputs, pin(other), {'report.json'})
            with self.assertRaisesRegex(ValueError, 'sealed original inventory'):
                audit.verify_output_inventory(root, outputs, manifest, {'report.json'})


if __name__ == '__main__':
    unittest.main()
