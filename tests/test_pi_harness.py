import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pi_harness as ph  # noqa: E402


def test_extract_answer():
    assert ph.extract_answer("... so 3 + 4 = 7. The answer is 7.") == "7"
    assert ph.extract_answer("The answer is $1,250.") == "1250"
    assert ph.extract_answer("The answer is -3.5 dollars") == "-3.5"
    assert ph.extract_answer("We get 12 and then 18") == "18"
    assert ph.extract_answer("no digits here") is None
    assert ph.extract_answer("The answer is 18.0") == "18"


class FakeTok:
    def apply_chat_template(self, msgs, add_generation_prompt, return_dict):
        n = len(msgs[0]["content"].split())
        return {"input_ids": list(range(n))}


def test_batches_have_equal_lengths_and_mark_fillers():
    test = [{"index": i, "question": "w " * (3 + i % 2), "gold": "1"} for i in range(5)]
    train = [{"index": i, "question": "w " * (3 + i % 2), "gold": "1"} for i in range(10)]
    for size in (2, 4):
        batches = ph.make_batches(test, train, FakeTok(), size, 0)
        scored = sorted(r["index"] for b in batches for r in b if r["scored"])
        assert scored == list(range(5))                          # every test problem scored exactly once
        for b in batches:
            assert len(b) == size
            assert len({len(r["ids"]) for r in b}) == 1           # one prompt length per batch: no padding
            assert all(r["index"] < 0 for r in b if not r["scored"])


def test_gain_proxy_row_invariant_ignores_batch_position():
    import torch
    g = torch.Generator().manual_seed(1)
    E = torch.randn(200, 16, generator=g)
    p = torch.softmax(torch.randn(3, 200, generator=g) * 3, dim=-1)
    full = ph.gain_proxy(torch, p, E, 0.6, row_invariant=True)
    for r in range(3):
        alone = ph.gain_proxy(torch, p[r:r + 1], E, 0.6, row_invariant=True)
        assert torch.allclose(full[r], alone[0], rtol=1e-5)


def test_vote_plurality_ties_and_missing():
    import pi_e2e3_analyze as pe
    assert pe.vote(["3", "4", "4", None]) == "4"
    assert pe.vote(["3", "4", "4", "3"]) == "4"        # 4 reached two first
    assert pe.vote([None, None]) is None
    assert pe.vote(["5"]) == "5"
