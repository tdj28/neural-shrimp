"""Allocate the sole Task37 saved-audit output, then run the frozen checker.

This wrapper does not admit source provenance or science. Root must separately
bind original execution/source/card/assets/full inventory and host closure.
No checker result, scientific false flag, SystemExit or exception is rewritten.
The external supervisor retains stdout/stderr and the original worker exit.
"""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys


CHECKER = Path('/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/presubmit-task37/audit_saved_numerics.py')
CHECKER_SHA256 = 'e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215'
INPUT_ROOT = Path('/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001')
OUTPUT_ROOT = Path('/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001-audit-001')
REPORT_NAME = 'result.json'


def fixed_path(value, expected):
    """Literal authority check: aliases, relative paths and other roots fail."""
    if value != str(expected):
        raise ValueError('Only the fixed Task37 path is allowed: ' + str(expected))
    return expected


def require_checker_hash(actual):
    if actual != CHECKER_SHA256:
        raise ValueError('Frozen saved checker SHA256 mismatch')


def plain_path(path, directory):
    """Reject missing paths, symlink resolution and wrong filesystem kinds."""
    if path.resolve(strict=True) != path:
        raise ValueError('Symlink or noncanonical path refused: ' + str(path))
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Wrong filesystem kind: ' + str(path))


def allocate(root_arg, out_arg):
    root = fixed_path(root_arg, INPUT_ROOT)
    out = fixed_path(out_arg, OUTPUT_ROOT)
    plain_path(root, directory=True)
    plain_path(out.parent, directory=True)
    plain_path(CHECKER, directory=False)
    require_checker_hash(hashlib.sha256(CHECKER.read_bytes()).hexdigest())
    # No parents=True or exist_ok=True: parent admission is external, and any
    # existing directory/file/symlink, including an incomplete attempt, refuses.
    out.mkdir()
    return root, out / REPORT_NAME


def invoke_checker(root, report):
    original_argv = sys.argv
    sys.argv = [str(CHECKER), '--root', str(root), '--report', str(report)]
    try:
        runpy.run_path(str(CHECKER), run_name='__main__')
        # The pinned checker always exits explicitly; an unexpected return
        # must not acquire an implicit success exit from this wrapper.
        raise RuntimeError('Pinned checker returned without SystemExit')
    finally:
        sys.argv = original_argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    root, report = allocate(args.root, args.out)
    invoke_checker(root, report)


if __name__ == '__main__':
    main()
