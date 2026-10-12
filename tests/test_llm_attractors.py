import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_llm_attractors as ela  # noqa: E402


def roles(messages):
    return [m["role"] for m in messages]


def test_window_roles_are_relative_to_the_speaker():
    transcript = ["Hello!", "a1", "b1", "a2", "b2", "a3", "b3", "a4", "b4"]   # agent 0 says a*, agent 1 says b*
    m0 = ela.window_messages("sys", transcript, 0, 7)       # agent 0 speaks next (after b4)
    m1 = ela.window_messages("sys", transcript[:-1], 1, 7)  # agent 1 speaks next (after a4)
    assert roles(m0) == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert roles(m1) == roles(m0)
    assert [m["content"] for m in m0[1:]] == transcript[-7:]
    assert m0[-1]["content"] == "b4" and m1[-1]["content"] == "a4"


def test_short_transcript_keeps_the_opener():
    m1 = ela.window_messages("sys", ["Hello!", "a1"], 1, 7)
    assert roles(m1) == ["system", "assistant", "user"]
    m0 = ela.window_messages("sys", ["Hello!"], 0, 7)
    assert roles(m0) == ["system", "user"]


def test_find_cycle():
    states = [("a",), ("b",), ("c",)]
    assert ela.find_cycle(states, ("b",)) == 1
    assert ela.find_cycle(states, ("d",)) is None


def test_first_run():
    assert ela.first_run([False, True, True, True, False], 3) == 1
    assert ela.first_run([True, True, False, True], 3) is None
