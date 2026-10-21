"""BSC3 archive/runner compatibility gates; no provider calls or rented work."""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import runpod_scan_group as group
import runpod_soft_float64 as uploader


def command():
    return ("bsc3 120 baby_shrimp_saddle_memory.py "
            "--out llm-soft-loop/pod64/bsc3-memory/continuation-01 "
            "--plan docs/baby-shrimp-zoom/saddle-memory-plan.json "
            "--source-commit " + "a" * 40 + " --mode continue --betas 0 1e-7 -1e-7 --anchor-saddle")


def test_archive_covers_saddle_worker_local_imports_and_plans():
    uploaded = set(uploader.UPLOAD_PATHS)
    required = {"tools/baby_shrimp_saddle_memory.py", "tools/baby_shrimp_memory.py",
                "tools/trace_saddle_continuation.py", "tools/trace_pair_sensitivity.py",
                "docs/baby-shrimp-zoom/saddle-memory-plan.json",
                "docs/baby-shrimp-zoom/crossing-receipt.json"}
    assert required <= uploaded
    for path in sorted(required):
        assert uploader.UPLOAD_PATHS.count(path) == 1
        if not path.endswith(".py"):
            continue
        tree = ast.parse((ROOT / path).read_text())
        for node in ast.walk(tree):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for module in modules:
                dependency = "tools/" + module.split(".")[0] + ".py"
                if (ROOT / dependency).is_file():
                    assert dependency in uploaded, f"unpackaged dependency of {path}: {dependency}"


def test_saddle_worker_has_plan_anchor_instead_of_unused_input():
    group.validate_orbit_job(command(), source_commit="a" * 40)


@pytest.mark.parametrize("old,new,match", [
    ("bsc3 120", "bsc3 121", "lifetime"),
    ("--source-commit " + "a" * 40, "--source-commit " + "b" * 40, "source commit"),
    ("docs/baby-shrimp-zoom/saddle-memory-plan.json", "docs/baby-shrimp-zoom/../other.json", "plan"),
    ("llm-soft-loop/pod64/bsc3-memory/continuation-01", "/tmp/bsc3", "retrieved tree"),
    ("--mode continue", "--mode unsupported", "mode"),
])
def test_saddle_worker_rejects_unbounded_or_mismatched_packets(old, new, match):
    with pytest.raises(ValueError, match=match):
        group.validate_orbit_job(command().replace(old, new), source_commit="a" * 40)


def test_saddle_packet_cannot_omit_source_commit():
    with pytest.raises(ValueError, match="source-commit"):
        group.validate_orbit_job(command().replace("--source-commit " + "a" * 40, ""))
