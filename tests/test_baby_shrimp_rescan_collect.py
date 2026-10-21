import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_rescan_collect import ready_workers


def setup(tmp_path):
    name = "neural-shrimp-test"
    config = {"names": [name], "source_commit": "a"*40, "approval": "approved-test",
              "started_epoch": 100, "preexisting_ids": []}
    created = {"event": "created", "pod_id": "owned", "name": name, "commit": "a"*40,
               "approval": "approved-test", "time": "1970-01-01T00:02:00+00:00"}
    worker = tmp_path / "workers" / name / "attempt"
    worker.mkdir(parents=True)
    receipt = {"pod_id": "owned", "commit": "a"*40, "hash_verified": True, "runner_returncode": 0}
    (worker / "result.json").write_text(json.dumps(receipt))
    return name, config, created, worker, receipt


def test_collection_waits_for_exact_verified_termination(tmp_path):
    name, config, created, worker, receipt = setup(tmp_path)
    assert ready_workers(tmp_path, [name], config, [created]) is None
    assert ready_workers(tmp_path, [name], config, [created, {"event": "terminated_verified", "pod_id": "foreign"}]) is None
    ledger = [created, {"event": "terminated_verified", "pod_id": "owned"}]
    assert ready_workers(tmp_path, [name], config, ledger) == [worker]
    receipt["hash_verified"] = False
    (worker / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="retrieval/source"):
        ready_workers(tmp_path, [name], config, ledger)


def test_collection_rejects_ambiguous_ownership_or_receipt(tmp_path):
    name, config, created, worker, receipt = setup(tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        ready_workers(tmp_path, [name], config, [created, {**created, "pod_id": "second"}])
    (worker / "result.json").unlink()
    with pytest.raises(ValueError, match="unique retrieval"):
        ready_workers(tmp_path, [name], config, [created, {"event": "terminated_verified", "pod_id": "owned"}])
