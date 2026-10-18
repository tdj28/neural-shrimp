"""Multi-plan packet helpers of tools/cert_remote.py (Task 33): attempt selection and the command check refuse rather
than adapt."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import cert_remote as cr  # noqa: E402


def row(name, plan=None, cmd=None):
    plan = plan or f"codex/coexistence-task33/{name}-plan.json"
    cmd = cmd or (f"python -I -B -u codex/coexistence-task33/launch.py --attempt {name} --driver derivative_shard "
                  f"--plan {Path(plan).name} --threads 1 --seconds 3600")
    return {"attempt": name, "plan": plan, "plan_sha256": "0" * 64, "command": cmd}


PACKET = {"plans": [row("a-001"), row("b-001"), row("c-001")]}


def test_plan_rows_selects_in_given_order():
    assert [r["attempt"] for r in cr.plan_rows(PACKET, "c-001,a-001")] == ["c-001", "a-001"]


@pytest.mark.parametrize("names", ["", "a-001,a-001", "a-001,zz-001", None])
def test_plan_rows_refuses_unknown_repeated_or_empty(names):
    with pytest.raises(SystemExit):
        cr.plan_rows(PACKET, names)


def test_plan_rows_refuses_packet_listing_an_attempt_twice():
    with pytest.raises(SystemExit):
        cr.plan_rows({"plans": [row("a-001"), row("a-001")]}, "a-001")


def test_command_check_accepts_the_packet_form():
    assert cr.command_problems(row("a-001")) == []


@pytest.mark.parametrize("cmd", [
    "python -I -B -u codex/certify-llm/launch.py --attempt a-001 --plan a-001-plan.json",          # other launcher
    "python -I -B -u codex/coexistence-task33/launch.py --attempt b-001 --plan a-001-plan.json",   # other attempt
    "python -I -B -u codex/coexistence-task33/launch.py --attempt a-001 --plan b-001-plan.json",   # other plan
    "python -I -B -u codex/coexistence-task33/launch.py --attempt a-001",                          # no plan
    "bash -c true",
])
def test_command_check_refuses_mismatches(cmd):
    assert cr.command_problems(row("a-001", cmd=cmd))


def test_attempt_roots():
    assert cr.attempt_root(PACKET) == "codex/coexistence"
    assert cr.attempt_root({"attempt": "x", "plan": "p"}) == "codex/llm-p8-certificate"
