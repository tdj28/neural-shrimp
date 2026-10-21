import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_geometry_finish as finish


def test_empty_geometry_has_no_default_seed():
    fields,report=finish.track({},(3,3),np.arange(3),np.arange(3),[1,1])
    assert fields is None and report["tracked"]==0
    assert "no admissible" in report["status"]


def test_complex_eigenvectors_must_remain_conjugate_pair():
    lam=np.array([.01+.02j,.01-.02j,-.03])
    vec=np.array([[1,1,0],[1j,-1j,0],[0,0,1]],complex)
    pairs=finish.candidate_spans(lam,vec,3)
    assert [p for p,q in pairs]==[(0,1)]
    assert np.allclose(pairs[0][1].T@pairs[0][1],np.eye(2))


def test_nearly_parallel_vectors_do_not_define_resolved_plane():
    lam=np.array([.1,.01])
    vec=np.array([[1,1],[0,1e-14]])
    assert not finish.candidate_spans(lam,vec,2)


def test_missing_cell_cannot_bridge_tracked_geometry():
    arr={"multipliers":np.array([.2,-.02]),"vectors":np.eye(2),"points":np.array([[0,0],[1,1]])}
    fields,report=finish.track({(0,0):arr,(0,2):arr},(3,3),np.arange(3),np.arange(3),[0,0])
    assert report["tracked"]==1
    assert np.isnan(fields["tau"][0,2])
def test_operator_interruption_is_partial_and_requires_exact_checkpoint(tmp_path):
    import json, hashlib
    from baby_shrimp_geometry_finish import completion_evidence
    grid = {"attempts": 2, "solved": 1, "orbit_solved": 2, "planned_cells": 10,
            "remaining_frontier": 3, "rows": [{}, {}]}
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(grid))
    result = {"pod_id": "owned", "commit": "abc"}
    note = {"pod_id": "owned", "source_commit": "abc", "signal": "SIGTERM", "saved_grid_attempts": 2,
            "grid_sha256_before_signal": hashlib.sha256(path.read_bytes()).hexdigest()}
    marker = tmp_path / "operator-stop.json"
    marker.write_text(json.dumps(note))
    status, kept = completion_evidence(tmp_path, grid, result)
    assert not status["complete"] and status["solved"] == 1 and kept == note
    note["pod_id"] = "foreign"
    marker.write_text(json.dumps(note))
    import pytest
    with pytest.raises(ValueError, match="identity"):
        completion_evidence(tmp_path, grid, result)
