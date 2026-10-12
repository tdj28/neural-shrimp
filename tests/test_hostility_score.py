import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hostility_score as hs  # noqa: E402


def test_lexicon_fraction_counts_words_and_phrases():
    assert hs.lexicon_fraction("") == 0.0
    assert hs.lexicon_fraction("The train leaves at nine.") == 0.0
    assert hs.lexicon_fraction("You are an idiot.") == 0.25                    # 1 of 4 words
    assert abs(hs.lexicon_fraction("Shut up, you stupid fool!") - 3 / 5) < 1e-12   # phrase + 2 words, 5 words
