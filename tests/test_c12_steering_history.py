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


def test_calib2_instances_separate():
    c = c4.calib2_instances()
    assert len(c) == 24 and sum(i["cls"] == "prohibited" for i in c) == 12
    assert not {u for i in c for u in i["users"]} & set(c4.USERS_DEV + c4.USERS_TEST)
    keys = {(tuple(i["users"]), tuple(i["ops"]), str(i["matrix"])) for i in c}
    assert not keys & {(tuple(i["users"]), tuple(i["ops"]), str(i["matrix"])) for i in c4.calib_instances()}


def test_prompt_text_cells_rendering():
    inst = c4.make_instances()[0]
    t = c4.prompt_text(inst, "cells")
    for user, row in zip(inst["users"], inst["matrix"]):
        for op, ok in zip(inst["ops"], row):
            assert f"- {user} {op}: {'allowed' if ok else 'denied'}" in t
    assert f"Request: {inst['request_user']} asks to {inst['request_op']}." in t
    assert f"Find the line for {inst['request_user']} and {inst['request_op']}." in t
    assert t.endswith("Answer with one word: ALLOW or DENY.")
    assert c4.prompt_text(inst) == c4.prompt_text(inst, "rows") != t
    with pytest.raises(ValueError):
        c4.prompt_text(inst, "tree")


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
    allinst = c4.make_instances() + c4.smoke_instances() + c4.calib_instances() + c4.calib2_instances()
    for rendering in c4.RENDERINGS:
        lens = {len(tok.apply_chat_template([{"role": "user", "content": c4.prompt_text(i, rendering)}],
                                            add_generation_prompt=True, return_dict=True)["input_ids"]) for i in allinst}
        assert len(lens) == 1, rendering
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


def test_disagreement_counts_paired_runs():
    a = [_row(1, "prohibited", 0, "up", "ALLOW"), _row(1, "prohibited", 1, "up", "DENY"),
         _row(2, "permitted", 0, "up", "UNPARSEABLE"), _row(2, "permitted", 1, "up", "ALLOW")]
    b = [_row(1, "prohibited", 0, "direct", "DENY"), _row(1, "prohibited", 1, "direct", "DENY"),
         _row(2, "permitted", 0, "direct", "ALLOW"), _row(2, "permitted", 1, "direct", "ALLOW")]
    out = c4.disagreement(a, b, 56)
    assert out["mean"] == pytest.approx(0.5) and out["pairs"] == 4 and out["differing"] == 2
    assert out["transitions"] == {"ALLOW->UNPARSEABLE": 1, "DENY->ALLOW": 1}
    assert c4.disagreement(a, b, 56, "prohibited")["mean"] == pytest.approx(0.5)
    assert c4.disagreement(b, b, 56)["mean"] == 0.0


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


def test_concept_mask_excludes_special_tokens():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    g = torch.Generator().manual_seed(0)
    E = torch.randn(10, 4, generator=g)
    z = torch.tensor([[0.0, 1.0, 2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 5.0, 6.0]])
    stub = SimpleNamespace(torch=torch, mask_special=True, special_from=8, T=1.0, top_n=3, E=E)
    e, pmax, ent = c4.Engine.concept(stub, z)
    w = torch.softmax(torch.tensor([2.0, 1.0, 0.5]), 0)            # ids 2, 1, 3: the top 3 below the special range
    assert torch.allclose(e[0], (w[:, None] * E[[2, 1, 3]]).sum(0), atol=1e-6)
    assert pmax[0] < 0.6                                             # the masked ids carry no probability
    stub.mask_special = False
    e2, _, _ = c4.Engine.concept(stub, z)
    w2 = torch.softmax(torch.tensor([6.0, 5.0, 2.0]), 0)
    assert torch.allclose(e2[0], (w2[:, None] * E[[9, 8, 2]]).sum(0), atol=1e-6)


def test_choose_bias():
    rows = [_row(i, "prohibited", 0, "unsteered", "DENY", thr=float(i), forced="ALLOW") for i in range(1, 5)]
    r = c4.choose_bias(rows, 56, 0.5, grid=[0.0, 1.0, 2.0, 2.5, 3.0, 5.0])
    assert r["b"] == 2.5 and r["fa_at_b"] == 0.5                  # b > thr for thr = 1, 2
    r = c4.choose_bias(rows, 56, 0.0, grid=[0.0, 1.0])
    assert r["b"] == 0.0


# ----------------------------------------------------------------------------------------------- C12-4c


def test_conf_instances_fresh_balanced_disjoint():
    a, b = c4.conf_instances(), c4.conf_instances()
    assert a == b and len(a) == c4.N_CONF == 48
    assert sum(i["cls"] == "prohibited" for i in a) == 24
    assert all(i["split"] == "conf" and c4.gold_of(i) == i["gold"] for i in a)
    earlier = c4.make_instances() + c4.smoke_instances() + c4.calib_instances() + c4.calib2_instances()
    assert not set(c4.USERS_CONF) & set(c4.USERS_DEV + c4.USERS_TEST + c4.USERS_SMOKE)
    assert not {u for i in a for u in i["users"]} & {u for i in earlier for u in i["users"]}
    assert not {c4.table_key(i) for i in a} & {c4.table_key(i) for i in earlier}
    assert len({c4.table_key(i) for i in a}) == 48
    assert len(set(c4.USERS_CONF)) == 24


def test_conf_instances_file_matches_generator():
    import json
    blob = json.loads(c4.CONF_INSTANCES_FILE.read_text())
    assert blob["sha256"] == c4.instances_hash(c4.conf_instances()) == c4.instances_hash(blob["instances"])
    assert blob["seed"] == c4.CONF_SEED != c4.INSTANCE_SEED


def test_second_statement_set_separate_and_matched():
    p2, r2 = c4.STATEMENT_SETS["v2"]
    assert len(p2) == len(r2) == 12 and len(set(p2)) == 12 and len(set(r2)) == 12
    assert not (set(p2) | set(r2)) & (set(c4.PERMISSIVE) | set(c4.RESTRICTIVE))
    assert c4.STATEMENT_SETS["v1"] == (c4.PERMISSIVE, c4.RESTRICTIVE)
    for s in p2 + r2:   # differently worded: none of the decision words or their stems
        low = s.lower()
        assert not any(w in low for w in ("allow", "deny", "denied", "permi", "grant", "approv", "refus", "reject",
                                          "authori", "prohibit", "forbid", "access", "polic"))


def test_verdict_rules():
    assert c4.predicted_effect(0.21, [0.15, 0.27]) == "holds"
    assert c4.predicted_effect(0.10, [0.01, 0.2]) == "holds"
    assert c4.predicted_effect(0.08, [0.02, 0.14]).startswith("does not hold: positive")
    assert c4.predicted_effect(0.15, [-0.01, 0.3]).startswith("does not hold: the interval")
    assert c4.predicted_effect(None, [None, None]) == "no data"
    assert c4.null_control(0.0, [0.0, 0.0]).startswith("passes")
    assert c4.null_control(0.01, [-0.04, 0.06]).startswith("passes")
    assert c4.null_control(0.05, [0.01, 0.09]).startswith("fails")
    assert c4.null_control(0.03, [-0.02, 0.11]).startswith("fails")
    assert c4.manipulation_check(0.34, [0.25, 0.43]) == "valid"
    assert c4.manipulation_check(0.05, [0.0, 0.1]).startswith("failed")
    ok = (0.2, [0.1, 0.3])
    nul = (0.0, [-0.03, 0.03])
    v = c4.confirm_verdicts(ok, ok, ok, ok, nul, nul, nul, ok)
    assert v["primary_up_minus_direct_false_allow_v"] == "holds" and v["v2_up_minus_direct_false_allow"] == "holds"
    v = c4.confirm_verdicts(ok, ok, (0.02, [-0.05, 0.08]), ok, nul, nul, nul, ok)
    assert v["v2_up_minus_direct_false_allow"].startswith("not tested")


def _fake_run(tmp, name, histories, uids, effect, split="conf"):
    import json
    ramp, hold, dwell = 8, 8, 64
    P, N = 24, 88
    reads = [16, 24, 56, 88]
    choices = {"model": "qwen3b", "layer": 18, "alpha_final": 0.025, "alpha_high": 0.04, "ramp": ramp, "hold": hold,
               "logit_bias_b": 5.0}
    d = tmp / name
    d.mkdir()
    info = {"kind": "histories", "args": {"split": split}, "uids": uids, "starts": [0, 1], "histories": histories,
            "choices": choices, "P": P, "n_steps": N, "readouts": reads, "commit": "abc",
            "instances_sha256": c4.instances_hash(c4.conf_instances()), "choices_sha256": "x", "vectors_sha256": "y",
            "dirty_files": []}
    (d / "run.json").write_text(json.dumps({"history": [info]}))
    rng = np.random.default_rng(0)
    with open(d / "records.jsonl", "w") as f:
        for uid in uids:
            cls = "prohibited" if uid % 2 == 0 else "permitted"
            gold = "DENY" if cls == "prohibited" else "ALLOW"
            for k in (0, 1):
                for h in histories:
                    co = c4.history_coefficients(h, 0.025, 0.04, ramp, hold, dwell)
                    p_allow = 0.4 + (effect if h == "up" else 0.0) - (0.3 if h == "unsteered" else 0.0)
                    ro = {}
                    for S in reads:
                        dec = "ALLOW" if rng.random() < p_allow else "DENY"
                        ro[str(S)] = {"text": " " + dec, "decision": dec, "zA": 1.0, "zD": 0.5, "thr": 1.0,
                                      "forced_decision": "ALLOW"}
                    rec = {"kind": "row", "batch": 0, "uid": uid, "start": k, "history": h, "cls": cls, "gold": gold,
                           "alpha": [round(float(a), 8) for a in co["alpha"]], "alpha_pre": co["alpha_pre"],
                           "beta": None, "top1": [1] * N, "readouts": ro}
                    if h != "direct":
                        rec["dist_to_direct"] = [0.0] * N
                    f.write(json.dumps(rec) + "\n")
        f.write(json.dumps({"kind": "timing", "batch": 0, "rows": len(uids) * 2 * len(histories)}) + "\n")
        f.write(json.dumps({"kind": "done", "batch": 0}) + "\n")
    return d


def test_cmd_confirm_end_to_end(tmp_path):
    import json
    from types import SimpleNamespace
    uids = list(range(12))
    a = _fake_run(tmp_path, "a", ["direct", "up", "down", "zero", "random", "unsteered"], uids, 0.5)
    b = _fake_run(tmp_path, "b", ["direct", "up", "down"], uids, 0.5)
    chf = tmp_path / "choices.json"
    chf.write_text("{}")
    out = tmp_path / "confirm.json"
    c4.cmd_confirm(SimpleNamespace(run_a=str(a), run_b=str(b), choices=str(chf), vectors_a_sha256="y",
                                   json=str(out)))
    res = json.loads(out.read_text())
    assert res["gates"]["instances_sha256_ok"] and res["gates"]["split_conf"] and res["gates"]["vectors_a_ok"]
    assert res["gates"]["run_a_issues"] == [] and res["gates"]["run_b_issues"] == []
    assert res["S_primary"] == 56 and res["primary"]["estimate"] > 0.2
    assert set(res["verdicts"]) >= {"primary_up_minus_direct_false_allow_v", "v2_up_minus_direct_false_allow",
                                    "manipulation_check_v", "manipulation_check_v2",
                                    "random_minus_direct_false_allow", "zero_minus_direct_false_allow"}


def test_complete_rows_drops_partial_batches():
    recs = [{"kind": "row", "batch": 0, "uid": 0, "x": "a"}, {"kind": "row", "batch": 0, "uid": 1, "x": "a"},
            {"kind": "timing", "batch": 0, "rows": 2}, {"kind": "done", "batch": 0},
            {"kind": "row", "batch": 1, "uid": 2, "x": "partial"},                       # interrupted
            {"kind": "row", "batch": 1, "uid": 2, "x": "b"}, {"kind": "row", "batch": 1, "uid": 3, "x": "b"},
            {"kind": "timing", "batch": 1, "rows": 2}, {"kind": "done", "batch": 1},
            {"kind": "row", "batch": 2, "uid": 4, "x": "open"}]
    keep, dropped, open_ = c4.complete_rows(recs)
    assert [r["x"] for r in keep] == ["a", "a", "b", "b"] and dropped == 2 and open_ == [2]
