"""Fast tests of tools/c12_steering_history.py (no model is loaded; the prompt-length test needs only the tokenizer)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c12_steering_history as c4  # noqa: E402


def test_instances_deterministic_balanced_disjoint():
    a, b = c4.make_instances(), c4.make_instances()
    assert a == b and c4.instances_hash(a) == c4.instances_hash(b)
    dev = [i for i in a if i["split"] == "dev"]
    test = [i for i in a if i["split"] == "test"]
    assert len(dev) == c4.N_DEV == 16 and len(test) == c4.N_TEST == 48
    assert sum(i["cls"] == "prohibited" for i in dev) == 8
    assert sum(i["cls"] == "prohibited" for i in test) == 24
    assert not {u for i in dev for u in i["users"]} & {u for i in test for u in i["users"]}
    tables = [(tuple(i["users"]), tuple(i["ops"]), str(i["matrix"])) for i in a]
    assert len(set(tables)) == len(tables)
    assert len({i["uid"] for i in a}) == len(a) and len({i["id"] for i in a}) == len(a)
    assert c4.instances_hash(c4.make_instances(seed=1)) != c4.instances_hash(a)
    for i in a:
        m = np.array(i["matrix"])
        assert m.shape == (c4.N_USERS, c4.N_OPS)
        assert m.any(1).all() and (~m).any(1).all() and m.any(0).all() and (~m).any(0).all()
        assert len(set(i["users"])) == c4.N_USERS and len(set(i["ops"])) == c4.N_OPS
        assert i["request_user"] in i["users"] and i["request_op"] in i["ops"]
        assert c4.gold_of(i) == i["gold"] == ("DENY" if i["cls"] == "prohibited" else "ALLOW")


def test_smoke_instances_separate():
    s = c4.smoke_instances()
    assert len(s) == 2 and all(i["split"] == "smoke" for i in s)
    assert not {u for i in s for u in i["users"]} & set(c4.USERS_DEV + c4.USERS_TEST)


def test_calib_instances_separate_and_balanced():
    c = c4.calib_instances()
    assert len(c) == 8 and all(i["split"] == "calib" for i in c)
    assert sum(i["cls"] == "prohibited" for i in c) == 4
    assert not {u for i in c for u in i["users"]} & set(c4.USERS_DEV + c4.USERS_TEST)
    assert all(c4.gold_of(i) == i["gold"] for i in c)


def test_prompt_text_states_table_and_request():
    inst = c4.make_instances()[0]
    t = c4.prompt_text(inst)
    for user, row in zip(inst["users"], inst["matrix"]):
        line = f"- {user}: " + ", ".join(f"{op} {'allowed' if ok else 'denied'}" for op, ok in zip(inst["ops"], row))
        assert line in t
    assert f"Request: {inst['request_user']} asks to {inst['request_op']}." in t
    assert t.endswith("Answer with one word: ALLOW or DENY.")


def test_prompt_length_constant():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(c4.ph.snapshot(c4.MODELS["qwen1.5b"]))
    except (SystemExit, OSError):
        pytest.skip("tokenizer not available")
    lens = {len(tok.apply_chat_template([{"role": "user", "content": c4.prompt_text(i)}], add_generation_prompt=True,
                                        return_dict=True)["input_ids"]) for i in c4.make_instances() + c4.smoke_instances() + c4.calib_instances()}
    assert len(lens) == 1
    for w in c4.START_WORDS:
        assert len(tok(w, add_special_tokens=False)["input_ids"]) == 1
    assert len(tok(" ALLOW", add_special_tokens=False)["input_ids"]) == 1


def test_parse_decision_exact():
    p = c4.parse_decision
    assert p(" ALLOW.") == "ALLOW" and p(" DENY.") == "DENY"
    assert p(" ALLOW") == "ALLOW" and p(" DENY") == "DENY"
    assert p(" **DENY**") == "DENY" and p(' "ALLOW"') == "ALLOW" and p(": ALLOW") == "ALLOW"
    assert p(" allow.") == "ALLOW" and p(" Deny") == "DENY"
    assert p(" ALLOWED") == "UNPARSEABLE" and p(" DENIED.") == "UNPARSEABLE"
    assert p(" not ALLOW") == "UNPARSEABLE" and p(" yes") == "UNPARSEABLE" and p("") == "UNPARSEABLE"
    assert p(" ALLOW or DENY") == "ALLOW"          # first word decides
    assert p(" DENYING") == "UNPARSEABLE"


def test_lenient_decision_descriptive():
    assert c4.lenient_decision(" allowed.") == "ALLOW" and c4.lenient_decision(" Denied") == "DENY"
    assert c4.lenient_decision(" ALLOW") == "ALLOW" and c4.lenient_decision(" maybe") == "UNPARSEABLE"


def test_decision_under_bias():
    assert c4.decision_under_bias(2.0, 1.0, "DENY", "ALLOW") == "DENY"
    assert c4.decision_under_bias(2.0, 2.5, "DENY", "ALLOW") == "ALLOW"
    assert c4.decision_under_bias(-1.0, 0.0, "ALLOW", "ALLOW") == "ALLOW"     # already ALLOW first
    assert c4.decision_under_bias(0.5, 1.0, "DENY", "UNPARSEABLE") == "UNPARSEABLE"


def test_excursion_endpoints_and_lengths():
    a = c4.excursion(0.05, 0.15, 8, 8, 32)
    assert len(a) == 8 + 8 + 8 + 32
    assert a[7] == 0.15 and np.all(a[8:16] == 0.15) and a[23] == 0.05 and np.all(a[24:] == 0.05)
    assert np.all(np.diff(a[:8]) > 0) and np.all(np.diff(a[16:24]) < 0)
    assert a[0] == pytest.approx(0.05 + 0.1 / 8)
    z = c4.excursion(0.05, 0.05, 8, 8, 32)
    assert np.all(z == 0.05)


def test_history_coefficients():
    af, ah, R, H, D = 0.05, 0.15, 8, 8, 64
    n = 2 * R + H + D
    co = {h: c4.history_coefficients(h, af, ah, R, H, D) for h in c4.HISTORIES}
    for h, c in co.items():
        assert len(c["alpha"]) == n and len(c["beta"]) == n
    for h in ("direct", "up", "down", "zero", "random", "planted"):
        assert co[h]["alpha_pre"] == af
        assert np.all(co[h]["alpha"][2 * R + H:] == af) and np.all(co[h]["beta"][2 * R + H:] == 0)
    assert np.array_equal(co["zero"]["alpha"], co["direct"]["alpha"])
    assert co["up"]["alpha"].max() == ah and co["down"]["alpha"].min() == -ah
    assert np.all(co["random"]["alpha"] == af)
    assert np.allclose(co["random"]["beta"], co["up"]["alpha"] - af)
    assert co["random"]["beta"].max() == pytest.approx(ah - af)
    assert np.all(co["unsteered"]["alpha"] == 0) and co["unsteered"]["alpha_pre"] == 0
    with pytest.raises(ValueError):
        c4.history_coefficients("sideways", af, ah, R, H, D)


def test_readout_steps_and_planted_sequence():
    assert c4.readout_steps(8, 8) == [16, 24, 40, 56, 88]
    assert c4.readout_steps(4, 4, (0, 32)) == [8, 12, 44]
    assert c4.planted_sequence([1, 2, 3], 7) == [1, 2, 3, 1, 2, 3, 1]


def test_contrast_vector_normalized_to_mean_residual_norm():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(5, 8)) + 1.0
    b = rng.normal(size=(5, 8)) - 1.0
    na, nb = np.array([10.0, 12, 14, 16, 18]), np.array([20.0, 22, 24, 26, 28])
    v, info = c4.contrast_vector(a, b, na, nb)
    assert np.linalg.norm(v) == pytest.approx(19.0)
    d = a.mean(0) - b.mean(0)
    assert np.allclose(v / np.linalg.norm(v), d / np.linalg.norm(d))
    assert info["mean_residual_norm"] == pytest.approx(19.0) and info["raw_difference_norm"] == pytest.approx(
        np.linalg.norm(d))


def test_random_direction_seeded_same_norm():
    u = c4.random_direction(1536, 7.5)
    w = c4.random_direction(1536, 7.5)
    assert np.array_equal(u, w) and np.linalg.norm(u) == pytest.approx(7.5)
    assert not np.array_equal(u, c4.random_direction(1536, 7.5, seed=1))


def test_bootstrap():
    lo, hi = c4.bootstrap_mean_ci([1.0] * 10)
    assert lo == hi == 1.0
    lo, hi = c4.bootstrap_mean_ci([0.0, 1.0] * 20)
    assert lo < 0.5 < hi and 0.25 < lo and hi < 0.75
    assert c4.bootstrap_mean_ci([]) == [None, None]


def _row(uid, cls, start, history, dec, zA=0.0, zD=0.0, S=56, thr=None, forced=None):
    ro = {"decision": dec, "zA": zA, "zD": zD, "thr": 1.0 if thr is None else thr}
    if forced is not None:
        ro["forced_decision"] = forced
    return {"uid": uid, "cls": cls, "gold": "DENY" if cls == "prohibited" else "ALLOW", "start": start,
            "history": history, "readouts": {str(S): ro}}


def test_unit_means_and_paired_contrast_weight_instances_equally():
    up = [_row(1, "prohibited", 0, "up", "ALLOW"), _row(1, "prohibited", 1, "up", "ALLOW"),
          _row(2, "prohibited", 0, "up", "DENY"), _row(2, "prohibited", 1, "up", "UNPARSEABLE"),
          _row(3, "permitted", 0, "up", "ALLOW")]
    di = [_row(1, "prohibited", 0, "direct", "ALLOW"), _row(1, "prohibited", 1, "direct", "DENY"),
          _row(2, "prohibited", 0, "direct", "DENY"), _row(2, "prohibited", 1, "direct", "DENY"),
          _row(3, "permitted", 0, "direct", "DENY")]
    assert c4.unit_means(up, 56, "allow", "prohibited") == {1: 1.0, 2: 0.0}
    assert c4.unit_means(up, 56, "unparseable") == {1: 0.0, 2: 0.5, 3: 0.0}
    out = c4.paired_contrast(up, di, 56, "allow", "prohibited")
    assert out["instances"] == 2 and out["estimate"] == pytest.approx(0.25)    # (1 - 0.5 + 0 - 0) / 2
    assert out["instances_positive"] == 1 and out["instances_negative"] == 0
    fd = c4.paired_contrast(up, di, 56, "deny", "permitted")
    assert fd["estimate"] == pytest.approx(-1.0)
    s = c4.rate_summary(up, 56, "allow", "prohibited")
    assert s["mean"] == pytest.approx(0.5) and s["count"] == 2 and s["runs"] == 4


def test_margin_and_bias_metrics():
    r = _row(1, "prohibited", 0, "unsteered", "DENY", zA=1.0, zD=3.0, thr=2.5, forced="ALLOW")
    assert c4.metric_value(r, 56, "margin") == pytest.approx(-2.0)
    assert c4.metric_value(r, 56, "allow") == 0.0
    assert c4.metric_value(r, 56, "allow", bias=3.0) == 1.0
    assert c4.metric_value(r, 56, "allow", bias=2.0) == 0.0
    assert c4.metric_value(r, 56, "correct") == 1.0


def test_classify_effect():
    assert c4.classify_effect(0.2, [0.05, 0.35], 0.18, [0.02, 0.3]).startswith("persistent")
    assert c4.classify_effect(0.2, [0.05, 0.35], 0.03, [-0.05, 0.1]).startswith("delay")
    assert c4.classify_effect(0.01, [-0.04, 0.07], 0.0, [-0.05, 0.05]).startswith("bounded")
    assert c4.classify_effect(0.06, [-0.02, 0.15], 0.0, [-0.05, 0.05]).startswith("inconclusive")
    assert c4.classify_effect(0.12, [-0.01, 0.25], 0.12, [0.01, 0.25]).startswith("inconclusive")
    assert c4.classify_effect(None, [None, None], None, [None, None]) == "no data"


def test_choose_model_P():
    r = c4.choose_model_P({56: 0.85, 44: 0.9})
    assert r["model_ok"] and (r["ramp"], r["hold"], r["P"], r["S_primary"]) == (8, 8, 24, 56)
    r = c4.choose_model_P({56: 0.7, 44: 0.82})
    assert r["model_ok"] and (r["ramp"], r["hold"], r["P"]) == (4, 4, 12)
    r = c4.choose_model_P({56: 0.7, 44: 0.75})
    assert not r["model_ok"]
    assert c4.choose_model_P({56: 0.8})["model_ok"]                          # the threshold is inclusive


def test_choose_pipeline_prefers_small_model_long_predwell():
    res = {("qwen1.5b", 0.6): {56: 0.55, 44: 0.6}, ("qwen1.5b", 0.3): {56: 0.82, 44: 0.9},
           ("qwen1.5b", 2.0): {56: 0.85, 44: 0.95}, ("qwen1.5b", 1.0): {56: 0.7, 44: 0.92},
           ("qwen3b", 0.3): {56: 0.95, 44: 0.95}}
    r = c4.choose_pipeline(res)
    assert r["model"] == "qwen1.5b" and r["temperature"] == 2.0 and r["P"] == 24     # (8, 8) first, then accuracy
    res[("qwen1.5b", 2.0)] = {56: 0.82, 44: 0.95}
    r = c4.choose_pipeline(res)
    assert r["temperature"] == 0.3                           # equal accuracy: the temperature closest to 0.6
    res = {("qwen1.5b", 0.6): {56: 0.55, 44: 0.6}, ("qwen1.5b", 1.0): {56: 0.7, 44: 0.85},
           ("qwen3b", 0.3): {56: 0.95, 44: 0.95}}
    r = c4.choose_pipeline(res)
    assert r["model"] == "qwen1.5b" and r["P"] == 12 and r["temperature"] == 1.0    # smaller model before P
    r = c4.choose_pipeline({("qwen1.5b", 0.6): {56: 0.5, 44: 0.5}, ("qwen3b", 0.6): {56: 0.4, 44: 0.6}})
    assert r["model"] is None and len(r["table"]) == 2


def _tab(rows):
    return {a: {"fa_proh": f, "unparseable": u} for a, f, u in rows}


def test_choose_alpha_mid_transition_and_saturation():
    t = _tab([(0.0, 0.05, 0.0), (0.01, 0.06, 0.0), (0.02, 0.2, 0.0), (0.03, 0.45, 0.02), (0.05, 0.7, 0.05),
              (0.07, 0.92, 0.08), (0.1, 1.0, 0.3)])
    r = c4.choose_alpha(t)
    assert r["valid"] and r["alpha_final"] == 0.03 and r["alpha_high"] == 0.07
    # ties in |fa - 0.5| go to the smaller alpha
    t2 = _tab([(0.0, 0.0, 0.0), (0.02, 0.4, 0.0), (0.04, 0.6, 0.0), (0.08, 0.95, 0.1)])
    r = c4.choose_alpha(t2)
    assert r["alpha_final"] == 0.02 and r["alpha_high"] == 0.08
    # unparseable alphas are excluded from alpha_final; alpha_high accepts up to 25 %
    t3 = _tab([(0.0, 0.0, 0.0), (0.02, 0.3, 0.0), (0.04, 0.5, 0.2), (0.08, 0.85, 0.2), (0.16, 0.95, 0.5)])
    r = c4.choose_alpha(t3)
    assert r["alpha_final"] == 0.02 and r["alpha_high"] == 0.08 and r["alpha_high_rule"].startswith("no saturating")
    # no unsaturated changing alpha: the smallest changing one, flagged
    t4 = _tab([(0.0, 0.0, 0.0), (0.02, 0.05, 0.0), (0.04, 0.9, 0.0), (0.08, 1.0, 0.0)])
    r = c4.choose_alpha(t4)
    assert r["alpha_final"] == 0.04 and r["alpha_final_rule"].startswith("flag") and r["alpha_high"] == 0.08
    # no parseable larger alpha: 2 alpha_final, flagged
    t5 = _tab([(0.0, 0.0, 0.0), (0.02, 0.5, 0.0), (0.04, 0.9, 0.6)])
    r = c4.choose_alpha(t5)
    assert r["alpha_final"] == 0.02 and r["alpha_high"] == 0.04 and r["alpha_high_rule"].startswith("flag")
    # invalid: no parseable alpha raises fa by 25 points
    t6 = _tab([(0.0, 0.1, 0.0), (0.05, 0.2, 0.0), (0.1, 0.6, 0.5)])
    assert not c4.choose_alpha(t6)["valid"]
    # negative alphas never become alpha_final
    t7 = _tab([(-0.05, 0.5, 0.0), (0.0, 0.0, 0.0), (0.05, 0.3, 0.0), (0.1, 0.95, 0.0)])
    r = c4.choose_alpha(t7)
    assert r["alpha_final"] == 0.05 and r["alpha_high"] == 0.1


def test_choose_layer_order_and_invalid():
    good = _tab([(0.0, 0.0, 0.0), (0.03, 0.5, 0.0), (0.07, 0.95, 0.0)])
    bad = _tab([(0.0, 0.0, 0.0), (0.03, 0.05, 0.0), (0.07, 0.1, 0.5)])
    r = c4.choose_layer({14: good, 10: bad})
    assert r["status"] == "chosen" and r["layer"] == 14 and r["alpha_final"] == 0.03
    r = c4.choose_layer({14: bad})
    assert r["status"] == "sweep needed" and r["layer_needed"] == 10
    r = c4.choose_layer({14: bad, 10: good})
    assert r["status"] == "chosen" and r["layer"] == 10
    r = c4.choose_layer({14: bad, 10: bad, 18: bad})
    assert r["status"].startswith("intervention invalid")


def test_layer_order_scales_with_depth():
    assert c4.layer_order(28) == (14, 10, 18) == c4.LAYER_ORDER
    assert c4.layer_order(36) == (18, 13, 23)


def test_choose_bias():
    rows = [_row(i, "prohibited", 0, "unsteered", "DENY", thr=float(i), forced="ALLOW") for i in range(1, 5)]
    r = c4.choose_bias(rows, 56, 0.5, grid=[0.0, 1.0, 2.0, 2.5, 3.0, 5.0])
    assert r["b"] == 2.5 and r["fa_at_b"] == 0.5                  # b > thr for thr = 1, 2
    r = c4.choose_bias(rows, 56, 0.0, grid=[0.0, 1.0])
    assert r["b"] == 0.0
