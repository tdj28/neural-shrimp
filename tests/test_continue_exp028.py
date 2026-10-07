import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module(name)


def test_continuation_reuses_verified_chunks_and_recomputes_missing_ones_identically(tmp_path):
    runner = _tool("run_exp028")
    cont = _tool("continue_exp028")
    plan = _tool("test_run_exp028")._synthetic_plan(steps=3000)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = runner.sha256_bytes(plan_path.read_bytes())
    complete = tmp_path / "complete"
    assert runner.execute(plan, plan_sha, complete, workers=2) == "complete"
    zoom_spec = json.loads((complete / "stage1" / "S2-99991-zoom.json").read_text())
    assert zoom_spec["window"] is not None, "fixture expected to select a zoom window"

    # Simulate an interruption during stage 2: drop one zoom chunk, the zoom map and stage 3.
    parent = tmp_path / "interrupted"
    shutil.copytree(complete, parent)
    dropped = "stage2/S2-99991-zoom/chunk-0001.npz"
    (parent / dropped).unlink()
    (parent / "stage2" / "S2-99991-zoom.npz").unlink()
    shutil.rmtree(parent / "stage3")
    receipt = json.loads((parent / "receipt.json").read_text())
    receipt["status"] = "failed"
    (parent / "receipt.json").write_text(json.dumps(receipt))

    out = tmp_path / "continued"
    assert cont.continue_attempt(plan, plan_sha, parent, out, workers=2) == "complete"
    end = json.loads((out / "runtime-end.json").read_text())
    assert end["recomputed"] == [dropped]
    assert "stage1/S2-99991.npz" in end["reused"]
    # Deterministic recomputation reproduces the original chunk and zoom map bytes.
    assert runner.sha256_file(out / dropped) == runner.sha256_file(complete / dropped)
    assert (runner.sha256_file(out / "stage2" / "S2-99991-zoom.npz")
            == runner.sha256_file(complete / "stage2" / "S2-99991-zoom.npz"))

    def strip(records):
        return [{k: v for k, v in r.items() if k not in ("pid", "seconds")} for r in records]

    for name in ("certificates.json", "ring-confirmation.json", "chaos-diagnostics.json"):
        a = json.loads((out / "stage3" / name).read_text())
        b = json.loads((complete / "stage3" / name).read_text())
        assert strip(a) == strip(b)
    continued_receipt = json.loads((out / "receipt.json").read_text())
    assert continued_receipt["continuation_of"]["receipt_sha256"] == runner.sha256_file(parent / "receipt.json")
