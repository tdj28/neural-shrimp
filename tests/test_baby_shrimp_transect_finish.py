import io
import hashlib
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_transect_finish as finish


def test_ambiguous_side_is_never_called_superstable(monkeypatch):
    monkeypatch.setattr(finish.sd, "cell_word", lambda *a: ("RLC", "ok", {"usable_phases": 6}))
    row = finish.word_record(np.ones(3000), 18, True)
    assert row["eligible"] and row["mss_catalogue_through_7_match"]
    assert row["last_symbol"] == "C"
    assert "not a proven" in row["last_symbol_scope"]
    assert "superstable" not in row


def test_inconsistent_tail_windows_fail(monkeypatch):
    words = iter(("RLL", "RRL", "RLL"))
    monkeypatch.setattr(finish.sd, "cell_word", lambda *a: (next(words), "ok", {}))
    assert not finish.word_record(np.ones(3000), 18, True)["eligible"]


def test_missing_domain_and_empty_order_are_explicit():
    assert not finish.word_record(np.ones(3000), 18, False)["eligible"]
    assert finish.rank_order([])["increasing"] is None
    assert finish.rank_order([8, 1, 7])["increasing"] is False
    assert finish.rank_order([8, 1, 7])["decreasing"] is False


def test_stream_hash_covers_all_content():
    data = b"evidence" * 2000000
    assert finish.stream_digest(io.BytesIO(data)) == hashlib.sha256(data).hexdigest()
