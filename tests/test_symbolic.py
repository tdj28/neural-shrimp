import numpy as np

from neural_shrimp.symbolic import mss_words, orbit_word


def test_mss_words_match_the_classical_table():
    # Metropolis-Stein-Stein: period 3 -> RL; period 4 -> RLR (doubling of 2) and RLL;
    # period 5 -> RLRR, RLLR, RLLL in order of increasing parameter.
    assert [w for _, w in mss_words(3)] == ["RL"]
    assert [w for _, w in mss_words(4)] == ["RLR", "RLL"]
    assert [w for _, w in mss_words(5)] == ["RLRR", "RLLR", "RLLL"]
    six = [w for _, w in mss_words(6)]
    assert six == ["RLRRR", "RLLRL", "RLLRR", "RLLLR", "RLLLL"]


def test_orbit_word_recovers_logistic_itinerary_in_one_dimension():
    # Logistic map near the superstable period-5 orbit with word RLLR.
    r, word = mss_words(5)[1]
    r += 1e-7
    x = 0.5
    for _ in range(20000):
        x = r * x * (1 - x)
    orbit = [np.array([x])]
    for _ in range(4):
        orbit.append(np.array([r * orbit[-1][0] * (1 - orbit[-1][0])]))
    jac = lambda p: np.array([[r * (1 - 2 * p[0])]])
    computed, lam, negatives, c_index = orbit_word(orbit, jac)
    assert computed == word
    assert min(abs(v) for v in lam) == abs(lam[c_index])


def test_combinatorial_admissibility_matches_numerical_mss_words():
    import itertools
    from neural_shrimp.symbolic import is_mss_admissible
    for n in range(2, 9):
        numeric = sorted(w for _, w in mss_words(n))
        admissible = sorted("".join(s) for s in itertools.product("LR", repeat=n - 1)
                            if is_mss_admissible("".join(s)))
        assert admissible == numeric, n
