import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import doubly_superstable as dss  # noqa: E402


def run(shape, start, attempts, solve):
    """Drive a Frontier with solve(cell, parent) -> solved; returns it and the attempts made, in order."""
    front = dss.Frontier(shape, start, attempts)
    log = []
    while front.queue:
        cell, parent = front.pop()
        ok = solve(cell, parent)
        log.append((cell, parent, ok))
        front.done(cell, ok)
    return front, log


def test_failed_cells_do_not_seed_their_neighbours():
    # column 2 always fails (a band where Newton slips onto the half-period orbit); beyond it, nothing is attempted
    front, log = run((3, 5), (1, 0), 3, lambda c, p: c[1] != 2)
    assert not any(c[1] > 2 for c, _, _ in log)
    assert front.solved == {(i, j) for i in range(3) for j in range(2)}


def test_failed_cell_is_retried_from_another_solved_neighbour():
    # (1, 1) fails when continued from (1, 0) and succeeds from (0, 1)
    front, log = run((2, 3), (0, 0), 3, lambda c, p: not (c == (1, 1) and p == (1, 0)))
    tries = [(p, ok) for c, p, ok in log if c == (1, 1)]
    assert tries == [((1, 0), False), ((0, 1), True)]
    assert front.solved == {(i, j) for i in range(2) for j in range(3)}


def test_attempts_are_capped_and_parents_released():
    front, log = run((3, 3), (0, 0), 2, lambda c, p: c != (1, 1))
    assert sum(c == (1, 1) for c, _, _ in log) == 2
    assert len({p for c, p, _ in log if c == (1, 1)}) == 2          # from two distinct parents
    assert len(front.solved) == 8
    assert front.parents_still_useful() == set()                    # nothing left to continue from


def test_every_cell_is_attempted_once_when_all_succeed():
    front, log = run((4, 5), (2, 3), 3, lambda c, p: True)
    cells = [c for c, _, _ in log]
    assert len(cells) == len(set(cells)) == 20
    assert log[0] == ((2, 3), None, True)
