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
    assert group.progress(tmp_path, "neural-shrimp-test") == (value, False)
    with f.open("a") as h:
        h.write("f" * 64 + "  /root/results.tgz\n")
    assert group.progress(tmp_path, "neural-shrimp-test") == (value, True)


def test_bounded_orbit_job_is_allowed():
    group.validate_orbit_job("probe 70 baby_shrimp_orbits.py --input llm-soft-loop/pod64/seeds "
                             "--out llm-soft-loop/pod64/probes")


def test_geometry_worker_has_its_own_bounded_allowance():
    args = "geometry 195 baby_shrimp_geometry.py --input llm-soft-loop/pod64/seeds --out llm-soft-loop/pod64/geometry"
    group.validate_orbit_job(args)
    with pytest.raises(ValueError, match="lifetime"):
        group.validate_orbit_job(args.replace("195", "196"))
    with pytest.raises(ValueError, match="lifetime"):
        group.validate_orbit_job(args.replace("baby_shrimp_geometry.py", "baby_shrimp_orbits.py"))


@pytest.mark.parametrize("command", [
    "probe 70 arbitrary.py --input llm-soft-loop/pod64/seeds --out llm-soft-loop/pod64/probes",
    "probe 0 baby_shrimp_orbits.py --input llm-soft-loop/pod64/seeds --out llm-soft-loop/pod64/probes",
    "probe 70 baby_shrimp_orbits.py --input llm-soft-loop/pod64/seeds --out elsewhere/probes",
    "probe 70 baby_shrimp_orbits.py --input llm-soft-loop/pod64/seeds --out llm-soft-loop/pod64/../probes",
])
def test_orbit_job_rejects_unbounded_or_unretrieved_work(command):
    with pytest.raises(ValueError):
        group.validate_orbit_job(command)


def test_orbit_numerical_progress_is_monitored(tmp_path):
    p = tmp_path / "workers/neural-shrimp-test/attempt"
    p.mkdir(parents=True)
    value = '{"cell_index": 0, "stage": "jacobian", "phase": 2}'
    (p / "runner-stdout.txt").write_text("progress probe " + value + "\n")
    assert group.progress(tmp_path, "neural-shrimp-test") == (value, False)


def test_memory_image_worker_requires_bound_plan_source_and_scan_mode():
    source='a'*40
    command=(f'image 75 baby_shrimp_memory_scan.py --plan docs/baby-shrimp-zoom/memory-junction-scan-plan.json '
             f'--mode scan --source-commit {source} --out llm-soft-loop/pod64/bsmz1/connection-01')
    group.validate_orbit_job(command,source_commit=source)
    for bad in [command.replace('image 75','image 76'),command.replace('--mode scan','--mode benchmark'),
                command.replace(source,'b'*40),command.replace('--out llm-soft-loop/pod64/','--out elsewhere/')]:
        with pytest.raises(ValueError): group.validate_orbit_job(bad,source_commit=source)


@pytest.mark.parametrize('plan_name', ['map', 'zoom', 'tight-zoom'])
def test_huginn_map_bound_source_plan_and_lifetime(plan_name):
    source='a'*40
    plan=f'docs/huginn-chaos-{plan_name}-plan.json'
    command=f'hcm1 720 huginn_chaos_map.py --plan {plan} --mode scan --source-commit {source} --out llm-soft-loop/pod64/hcm1/map'
    group.validate_orbit_job(command,source_commit=source)
    for bad in [command.replace('720','721'),command.replace(source,'b'*40),command.replace(plan,'docs/other.json')]:
        with pytest.raises(ValueError):group.validate_orbit_job(bad,source_commit=source)
