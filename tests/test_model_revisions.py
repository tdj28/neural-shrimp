"""The model loader's pinned Hugging Face revisions agree with docs/model-revisions.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import explore_llm_attractors as ela  # noqa: E402


def test_every_model_is_pinned_to_the_listed_revision():
    listed = {r["model"]: r["revision"] for r in json.loads((ROOT / "docs" / "model-revisions.json").read_text())["models"]}
    assert set(ela.MODELS.values()) <= set(ela.REVISIONS)
    for repo in ela.MODELS.values():
        assert ela.REVISIONS[repo] == listed[repo]
        assert len(ela.REVISIONS[repo]) == 40 and int(ela.REVISIONS[repo], 16) >= 0
