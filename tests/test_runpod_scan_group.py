"""Synthetic-unit budget and resource-identity gates; no provider calls."""
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
SPEC = importlib.util.spec_from_file_location("runpod_scan_group", Path(__file__).resolve().parents[1] /
                                             "tools/runpod_scan_group.py")
group = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(group)


def config():
    c = dict(names=["neural-shrimp-test"], source_commit="a" * 40, approval="test-approval",
             max_rate=1, disk_hourly_rate=0.1, ceiling=5, hard_minutes=120,
             retrieval_minutes=20, no_progress_minutes=15, started_epoch=100, preexisting_ids=["old"])
    c["commands"] = [["python", "tools/runpod_soft_float64.py", "--name", c["names"][0],
                      "--commit", c["source_commit"], "--approval", c["approval"],
                      "--jobset", "scanband", "--max-rate", "1", "--max-cost", "5"]]
    return c


def creation(**changes):
    return dict(event="created", pod_id="new", name="neural-shrimp-test", commit="a" * 40,
                approval="test-approval", time="1970-01-01T00:02:00+00:00", **changes)


def test_budget_includes_storage_and_every_pod():
    c = config()
    assert group.validate(c) == pytest.approx(2.2)
    c["ceiling"] = 2
    with pytest.raises(ValueError, match="lifetime"):
        group.validate(c)


@pytest.mark.parametrize("field,value", [("pod_id", "old"), ("commit", "b" * 40),
                                        ("approval", "other"), ("name", "neural-shrimp-other"),
                                        ("time", "1970-01-01T00:01:00+00:00")])
def test_excludes_resources_without_exact_new_ownership(field, value):
    e = creation()
    e[field] = value
    assert group.owned(config(), [e]) == {}


def test_only_matching_termination_marks_the_owned_resource():
    e = creation()
    rows = [e, dict(event="terminated_verified", pod_id="old")]
    assert group.owned(config(), rows)["new"]["terminated"] is False
    rows.append(dict(event="terminated_verified", pod_id="new"))
    assert group.owned(config(), rows)["new"]["terminated"] is True


def test_rejects_extra_gpu_and_command_identity_changes():
    c = config()
    c["commands"][0] += ["--gpu-count", "2"]
    with pytest.raises(ValueError, match="one GPU"):
        group.validate(c)
    c = config()
    c["commands"][0][3] = "neural-shrimp-other"
    with pytest.raises(ValueError, match="mismatch"):
        group.validate(c)


def test_numerical_progress_ignores_repeated_wall_clock_chatter(tmp_path):
    p = tmp_path / "workers/neural-shrimp-test/attempt"
    p.mkdir(parents=True)
    f = p / "runner-stdout.txt"
    value = '{"chunk_start": 0, "phase": "transient", "step": 100}'
    f.write_text("00:01 progress " + value + "\n00:02 progress " + value + "\n")
    assert group.progress(tmp_path, "neural-shrimp-test") == (value, False)
    with f.open("a") as h:
        h.write("end scanband-test status 0\n")
    assert group.progress(tmp_path, "neural-shrimp-test") == (value, True)
