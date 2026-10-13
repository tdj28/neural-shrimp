import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_partA_merge as hm  # noqa: E402


def make_run(root, stamp, pod, files, verified=True):
    """A run folder with results.tgz holding the given {member name: bytes} and a matching result.json."""
    run = root / stamp
    run.mkdir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    (run / "results.tgz").write_bytes(buf.getvalue())
    digest = hashlib.sha256(buf.getvalue()).hexdigest()
    (run / "result.json").write_text(json.dumps({"pod_id": pod, "hash_verified": verified,
                                                 "local_sha256": digest if verified else "0" * 64}))
    return run


P1 = "explore/llm-soft-loop/pod64/huginn/partA-spectra/"
P2 = "explore/llm-soft-loop/pod64/huginn/partA-spectra-2/"


def test_earliest_complete_copy_wins_and_incomplete_folders_are_never_used(tmp_path):
    soft = tmp_path / "soft64"
    soft.mkdir()
    make_run(soft, "20261013T151945Z", "first", {P1 + "q0000/fixed-spectrum.json": b"a0",
                                                 P1 + "q0000/fixed-state.npy": b"s0",
                                                 P1 + "q0001/fixed-state.npy": b"cut"})      # q1 incomplete
    make_run(soft, "20261013T175506Z", "slow", {P2 + "q0001/fixed-state.npy": b"cut-again",
                                                P2 + "q0002/fixed-spectrum.json": b"slow2"})
    make_run(soft, "20261013T184600Z", "backup", {P2 + "q0001/fixed-spectrum.json": b"b1",
                                                  P2 + "q0002/fixed-spectrum.json": b"backup2",
                                                  P2 + "logs/x.log": b"ignored"})
    out = tmp_path / "explore" / "merged"
    m = hm.merge(soft, ["20261013T184600Z", "20261013T151945Z", "20261013T175506Z"], out, range(4))
    assert m["questions"]["q0000"]["run"] == "20261013T151945Z"
    assert (out / "q0000" / "fixed-state.npy").read_bytes() == b"s0"
    assert m["questions"]["q0001"]["run"] == "20261013T184600Z"                      # the only complete copy
    assert not (out / "q0001" / "fixed-state.npy").exists()                           # nothing from cut-off runs
    assert m["questions"]["q0002"]["run"] == "20261013T175506Z"                      # launched before the backup
    assert (out / "q0002" / "fixed-spectrum.json").read_bytes() == b"slow2"
    assert m["questions"]["q0002"]["unused_complete_copies"] == ["20261013T184600Z"]
    assert m["missing"] == [3]
    assert m["questions"]["q0000"]["files"]["fixed-spectrum.json"] == hashlib.sha256(b"a0").hexdigest()
    with pytest.raises(SystemExit):
        hm.merge(soft, ["20261013T151945Z"], out, range(4))                        # never overwritten


def test_unverified_archive_is_refused(tmp_path):
    soft = tmp_path / "soft64"
    soft.mkdir()
    make_run(soft, "20261013T151945Z", "p", {P1 + "q0000/fixed-spectrum.json": b"a"}, verified=False)
    with pytest.raises(SystemExit):
        hm.merge(soft, ["20261013T151945Z"], tmp_path / "explore" / "m", range(1))
