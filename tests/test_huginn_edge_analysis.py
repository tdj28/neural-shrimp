"""The development analysis separates problems by whether an arm lies inside or beyond their own edge."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_edge_analysis as hea  # noqa: E402


def test_inside_beyond_split(tmp_path, monkeypatch):
    monkeypatch.setattr(hea, "DATA", tmp_path)
    (tmp_path / "e").mkdir()
    (tmp_path / "r").mkdir()
    # problems 0-9 have their edge at 1.12, problems 10-19 at 1.07; at lam 1.10 the first group is inside, the second beyond
    edges = [{"index": q, "outcome": "edge located", "lam_edge": 1.12 if q < 10 else 1.07} for q in range(20)]
    (tmp_path / "e" / "edges.jsonl").write_text("\n".join(json.dumps(e) for e in edges))
    rows = []
    for q in range(20):
        rows.append({"index": q, "lam": 1.0, "correct": False})
        rows.append({"index": q, "lam": 1.10, "correct": q < 10})       # gains inside the edge only
    (tmp_path / "r" / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--edges", "e", "--rows", "r", "--out-json", str(out)])
    hea.main()
    res = json.loads(out.read_text())["within_arm"]["1.1"]
    assert res["inside"] == 10 and res["beyond"] == 10
    assert res["mean_delta_inside"] == 1.0 and res["mean_delta_beyond"] == 0.0 and res["p_permutation"] < 0.01
