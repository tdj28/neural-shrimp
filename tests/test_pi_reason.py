import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pi_reason as pr  # noqa: E402

LLAMA = Path("/data2/neural-shrimp-data/models/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/"
             "6a6f4aa4197940add57724a7707d069478df56b1")


def test_entropy_matches_definition():
    torch = pytest.importorskip("torch")
    p = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    e = pr.entropy(torch, p)
    assert abs(float(e[0])) < 1e-9
    assert abs(float(e[1]) - 0.6931471805599453) < 1e-6


def test_extract_boxed_and_markers():
    assert pr.extract("so \\boxed{1,234}") == "1234"
    assert pr.extract("ĠtheĠanswerĠisĠ\\boxed{18}Ċ") == "18"


@pytest.mark.skipif(not (LLAMA / "tokenizer.json").exists(), reason="R1-Distill-Llama-8B tokenizer not cached")
def test_checked_tokenizer_keeps_spaces_for_r1_llama():
    legacy = pr.load_tokenizer(LLAMA, "legacy")
    checked = pr.load_tokenizer(LLAMA, "checked")
    assert not pr.roundtrip_ok(legacy)          # the PM2 defect: spaces dropped
    assert pr.roundtrip_ok(checked)
    text = checked.decode(pr.prompt_ids(checked, "Natalia sold 48 clips."))
    assert "Natalia sold 48 clips.\nPlease reason step by step" in text
