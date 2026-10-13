"""Tests for tools/steering_controls.py: the frozen SAE feature rule, norm-matched rescaling and the build step."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import steering_controls as sc  # noqa: E402

CATS = ["gratitude", "hostility", "kindness", "philosophy", "cooking", "neutral", "politeness", "sadness", "swearing"]


def profile(**vals):
    return {c: float(vals.get(c, 0.0)) for c in CATS}


@pytest.fixture
def profiles(tmp_path):
    table = {
        "3": profile(hostility=0.5, swearing=0.9),           # specific to hostility once swearing is left out
        "5": profile(hostility=0.6, kindness=0.4),           # larger on hostility but not specific
        "8": profile(philosophy=0.3),
        "9": profile(philosophy=0.3),                        # tie with 8: the lower ID wins
        "11": profile(cooking=0.2),
        "12": profile(),                                     # dead on these texts: never the random live feature
    }
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"categories": CATS, "profiles": {"toy": table}}))
    return path


def test_select_rule(profiles):
    sel = sc.select(profiles, "toy")
    assert sel["features"]["hostility"]["feature"] == 3
    assert sel["features"]["philosophy"]["feature"] == 8
    assert sel["features"]["neighbour"]["feature"] == 4
    live = sel["features"]["random_live"]
    assert live["pool"] == 5 and live["feature"] in (3, 5, 8, 9, 11)
    assert sc.select(profiles, "toy")["features"]["random_live"] == live        # reproducible


def test_rescale_and_build(tmp_path, monkeypatch, profiles):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(sc, "explore_dir", lambda: tmp_path)
    import explore_llm_phase_map as elpm
    rng = np.random.default_rng(1)
    d, F = 8, 16
    concepts = {k: rng.normal(size=d) * s for k, s in (("hostility-vs-kindness", 3.0), ("philosophy", 2.0),
                                                      ("random1", 5.0), ("random2", 5.0))}
    (tmp_path / "vec").mkdir()
    np.savez(tmp_path / "vec" / "vectors.npz", **concepts)
    monkeypatch.setattr(elpm, "load_vectors", lambda path: dict(np.load(tmp_path / path / "vectors.npz")))
    sc.rescale("vec", "hostility-vs-kindness", "resc", ["random1", "random2"])
    out = np.load(tmp_path / "resc" / "vectors.npz")
    for k in ("random1", "random2"):
        assert np.isclose(np.linalg.norm(out[k]), np.linalg.norm(concepts["hostility-vs-kindness"]))
        assert np.isclose(out[k] @ concepts[k] / (np.linalg.norm(out[k]) * np.linalg.norm(concepts[k])), 1.0)
    with pytest.raises(SystemExit):
        sc.rescale("vec", "hostility-vs-kindness", "resc", ["random1"])        # never overwritten
    sd = {"encoder_linear.weight": torch.randn(F, d), "encoder_linear.bias": torch.zeros(F),
          "decoder_linear.weight": torch.randn(d, F), "decoder_linear.bias": torch.zeros(d)}
    torch.save(sd, tmp_path / "sae.pth")
    sel = sc.select(profiles, "toy")
    (tmp_path / "sel.json").write_text(json.dumps(sel))
    args = type("A", (), {"concept_vectors": "vec", "sae_kind": "goodfire", "sae_path": str(tmp_path / "sae.pth"),
                          "layer": 3, "selection": str(tmp_path / "sel.json"), "out": "built"})()
    sc.build(args)
    built = np.load(tmp_path / "built" / "vectors.npz")
    info = json.loads((tmp_path / "built" / "vectors.json").read_text())
    W = sd["decoder_linear.weight"].T.double().numpy()
    h = sel["features"]["hostility"]["feature"]
    v = built[f"sae-hostility-{h}"]
    assert np.isclose(np.linalg.norm(v), np.linalg.norm(concepts["hostility-vs-kindness"]))
    assert np.isclose(v @ W[h] / (np.linalg.norm(v) * np.linalg.norm(W[h])), 1.0)
    p = sel["features"]["philosophy"]["feature"]
    assert np.isclose(np.linalg.norm(built[f"sae-philosophy-{p}"]), np.linalg.norm(concepts["philosophy"]))
    assert np.isclose(np.linalg.norm(built["random1"]), np.linalg.norm(concepts["hostility-vs-kindness"]))
    assert info["sae_index"] == 4 and info["layer"] == 3
