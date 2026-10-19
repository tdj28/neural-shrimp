"""Eight NEW stdlib-only mocked controls. Prepared, not run by this agent.

Fixtures replace authority constants only inside tests, under TemporaryDirectory.
runpy.run_path is always mocked; the numerical checker is never imported/run.
No original output, NumPy, SciPy, torch, model or producer module is used.
"""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).with_name('run_saved_numerics.py')
spec = importlib.util.spec_from_file_location('prepared_saved_wrapper', source)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


class WrapperControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='task37-wrapper-control-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / 'r-pair-001'
        self.root.mkdir()
        self.out = self.base / 'r-pair-001-audit-001'
        self.checker = self.base / 'checker.py'
        self.checker.write_bytes(b'# synthetic inert checker bytes; never executed\n')

    def authority(self):
        return patch.multiple(wrapper, INPUT_ROOT=self.root, OUTPUT_ROOT=self.out,
                              CHECKER=self.checker,
                              CHECKER_SHA256=hashlib.sha256(self.checker.read_bytes()).hexdigest())

    def args(self):
        return ['--root', str(self.root), '--out', str(self.out)]

    def test_01_fixed_paths_and_no_checker_override(self):
        self.assertEqual(wrapper.CHECKER, Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/presubmit-task37/audit_saved_numerics.py'))
        self.assertEqual(wrapper.INPUT_ROOT, Path('/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001'))
        self.assertEqual(wrapper.OUTPUT_ROOT, Path('/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001-audit-001'))
        self.assertEqual(wrapper.REPORT_NAME, 'result.json')
        self.assertEqual(wrapper.CHECKER_SHA256, 'e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215')
        with self.authority(), patch.object(wrapper.runpy, 'run_path') as run:
            for args in [self.args() + ['--checker', '/arbitrary.py'],
                         ['--root', str(self.root) + '/.', '--out', str(self.out)],
                         ['--root', 'r-pair-001', '--out', str(self.out)],
                         ['--root', str(self.root), '--out', str(self.out) + '-retry']]:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises((SystemExit, ValueError)):
                        wrapper.main(args)
            run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_02_exact_hash_before_allocation(self):
        wrapper.require_checker_hash(wrapper.CHECKER_SHA256)
        with self.assertRaises(ValueError):
            wrapper.require_checker_hash('0' * 64)
        with self.authority(), patch.object(wrapper.runpy, 'run_path') as run:
            self.checker.write_bytes(b'# changed synthetic checker\n')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                wrapper.main(self.args())
            run.assert_not_called()
        self.assertFalse(self.out.exists())

    def test_03_missing_and_symlink_paths_refused(self):
        link = self.base / 'alias'
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            wrapper.plain_path(link, directory=True)
        with self.assertRaises(FileNotFoundError):
            wrapper.plain_path(self.base / 'missing', directory=True)
        with self.assertRaises(ValueError):
            wrapper.plain_path(self.root, directory=False)
        source_link = self.base / 'source-alias.py'
        source_link.symlink_to(self.checker)
        with self.assertRaises(ValueError):
            wrapper.plain_path(source_link, directory=False)
        with self.authority(), patch.object(wrapper, 'CHECKER', source_link):
            with self.assertRaises(ValueError):
                wrapper.allocate(str(self.root), str(self.out))
        self.assertFalse(self.out.exists())

    def test_04_output_is_created_once_without_parent_creation(self):
        with self.authority():
            root, report = wrapper.allocate(str(self.root), str(self.out))
            self.assertEqual((root, report), (self.root, self.out / 'result.json'))
            self.assertTrue(self.out.is_dir())
            self.assertFalse(report.exists())
            with self.assertRaises(FileExistsError):
                wrapper.allocate(str(self.root), str(self.out))
            regular = self.base / 'existing-file'
            regular.write_bytes(b'preserve existing allocation')
            link = self.base / 'existing-link'
            link.symlink_to(self.out, target_is_directory=True)
            dangling = self.base / 'dangling-link'
            dangling.symlink_to(self.base / 'absent-target')
            for occupied in (regular, link, dangling):
                with self.subTest(occupied=occupied), patch.object(wrapper, 'OUTPUT_ROOT', occupied):
                    with self.assertRaises(FileExistsError):
                        wrapper.allocate(str(self.root), str(occupied))
            self.assertEqual(regular.read_bytes(), b'preserve existing allocation')
            self.assertTrue(link.is_symlink())
            self.assertTrue(dangling.is_symlink())
            absent = self.base / 'missing-parent' / 'audit'
            with patch.object(wrapper, 'OUTPUT_ROOT', absent):
                with self.assertRaises(FileNotFoundError):
                    wrapper.allocate(str(self.root), str(absent))
            self.assertFalse(absent.parent.exists())

    def check_exit(self, code):
        original_argv = sys.argv
        expected = b'{"saved_numerical_checks_passed": false, "scientific_verdict_admitted": false, "certificate_passed": false}\n'
        failure = SystemExit(code)
        def fake_run(path, run_name):
            self.assertEqual((path, run_name), (str(self.checker), '__main__'))
            self.assertEqual(sys.argv, [str(self.checker), '--root', str(self.root), '--report', str(self.out / 'result.json')])
            (self.out / 'result.json').write_bytes(expected)
            raise failure
        with self.authority(), patch.object(wrapper.runpy, 'run_path', side_effect=fake_run) as run:
            with self.assertRaises(SystemExit) as caught:
                wrapper.main(self.args())
            self.assertIs(caught.exception, failure)
            self.assertEqual(caught.exception.code, code)
            run.assert_called_once()
        self.assertIs(sys.argv, original_argv)
        self.assertEqual((self.out / 'result.json').read_bytes(), expected)

    def test_05_success_exit_keeps_scientific_false_flags(self):
        self.check_exit(0)

    def test_06_failure_exit_and_report_are_unchanged(self):
        self.check_exit(1)

    def test_07_uncaught_error_and_partial_report_survive(self):
        original_argv = sys.argv
        failure = OSError('synthetic checker failure')
        def fake_run(*args, **kwargs):
            (self.out / 'result.json').write_bytes(b'{partial')
            raise failure
        with self.authority(), patch.object(wrapper.runpy, 'run_path', side_effect=fake_run):
            with self.assertRaises(OSError) as caught:
                wrapper.main(self.args())
            self.assertIs(caught.exception, failure)
        self.assertIs(sys.argv, original_argv)
        self.assertTrue(self.out.is_dir())
        self.assertEqual((self.out / 'result.json').read_bytes(), b'{partial')

    def test_08_unexpected_normal_return_is_not_success(self):
        original_argv = sys.argv
        with self.authority(), patch.object(wrapper.runpy, 'run_path', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'without SystemExit'):
                wrapper.main(self.args())
        self.assertIs(sys.argv, original_argv)
        self.assertTrue(self.out.is_dir())
        self.assertFalse((self.out / 'result.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
