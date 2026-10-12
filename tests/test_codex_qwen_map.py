"""Synthetic tests only: no pretrained model loading or inference."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "codex/crosscheck/qwen_map.py"
SPEC = importlib.util.spec_from_file_location("codex_independent_qwen_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ManualQwenSoftMap = MODULE.ManualQwenSoftMap


@pytest.fixture
def tiny():
    generator = torch.Generator(device="cpu").manual_seed(107)
    config = dict(hidden_size=8, num_hidden_layers=4, num_attention_heads=2,
                  num_key_value_heads=1, intermediate_size=12, vocab_size=19,
                  rms_norm_eps=1e-6, rope_theta=10000., hidden_act="silu",
                  tie_word_embeddings=True, attention_dropout=0., use_sliding_window=False)
    sample = lambda *shape: torch.randn(shape, generator=generator, dtype=torch.float64)
    weights = {"model.embed_tokens.weight": .2 * sample(19, 8), "model.norm.weight": 1 + .03 * sample(8)}
    for i in range(4):
        name = f"model.layers.{i}."
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            weights[name + norm + ".weight"] = 1 + .03 * sample(8)
        for key, rows in [("q", 8), ("k", 4), ("v", 4)]:
            weights[name + f"self_attn.{key}_proj.weight"] = .12 * sample(rows, 8)
            weights[name + f"self_attn.{key}_proj.bias"] = .03 * sample(rows)
        weights[name + "self_attn.o_proj.weight"] = .12 * sample(8, 8)
        for key in ["gate", "up"]:
            weights[name + f"mlp.{key}_proj.weight"] = .12 * sample(12, 8)
        weights[name + "mlp.down_proj.weight"] = .12 * sample(8, 12)
    vector = .2 * sample(8)
    return ManualQwenSoftMap(config, weights, (1, 3, 5), vector, steering_layer=1)


def full_sequence_reference(model, state, alpha, log_t):
    """Separate scalar-query full-prefill equations; no cache/block helper use."""
    weights = model.weights
    hidden = torch.cat([weights["model.embed_tokens.weight"][model.prefix_ids], state[None]], dim=0)
    length = len(hidden)
    heads, dh = model.heads, model.head_dim
    for i in range(model.layers):
        name = f"model.layers.{i}."
        normalized = hidden / torch.sqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + model.epsilon)
        normalized = normalized * weights[name + "input_layernorm.weight"]
        query, key, value = [(normalized @ weights[name + f"self_attn.{p}_proj.weight"].T
                              + weights[name + f"self_attn.{p}_proj.bias"]).reshape(length, -1, dh)
                             for p in ["q", "k", "v"]]
        angles = torch.arange(length, dtype=torch.float64)[:, None] / (model.config["rope_theta"] ** (torch.arange(0, dh, 2, dtype=torch.float64) / dh))[None]
        cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
        def rotate(v):
            left, right = v[..., :dh // 2], v[..., dh // 2:]
            return torch.cat([left * cos - right * sin, right * cos + left * sin], dim=-1)
        query, key = rotate(query), rotate(key)
        outputs = []
        for t in range(length):
            pieces = []
            for h in range(heads):
                kh = h // model.groups
                scores = (key[:t + 1, kh] @ query[t, h]) / (dh ** .5)
                pieces.append(torch.softmax(scores, dim=0) @ value[:t + 1, kh])
            outputs.append(torch.cat(pieces))
        hidden = hidden + torch.stack(outputs) @ weights[name + "self_attn.o_proj.weight"].T
        normalized = hidden / torch.sqrt(torch.mean(hidden * hidden, dim=-1, keepdim=True) + model.epsilon)
        normalized = normalized * weights[name + "post_attention_layernorm.weight"]
        gate = normalized @ weights[name + "mlp.gate_proj.weight"].T
        up = normalized @ weights[name + "mlp.up_proj.weight"].T
        hidden = hidden + ((gate * torch.sigmoid(gate)) * up) @ weights[name + "mlp.down_proj.weight"].T
        if i == model.steering_layer:
            hidden = hidden + alpha * model.vector
    last = hidden[-1] / torch.sqrt(torch.mean(hidden[-1] ** 2) + model.epsilon)
    last = last * weights["model.norm.weight"]
    logits = weights["model.embed_tokens.weight"] @ last
    return torch.softmax(logits / (10. ** log_t), dim=0) @ weights["model.embed_tokens.weight"]


def test_cached_step_matches_independent_causal_full_prefill(tiny):
    x = torch.linspace(-.2, .3, tiny.d, dtype=torch.float64)
    for alpha, log_t in [(.0, -.12), (.07, .1), (-.12, -.3)]:
        got = tiny(x, alpha, log_t)
        expected = full_sequence_reference(tiny, x, alpha, log_t)
        torch.testing.assert_close(got, expected, rtol=2e-13, atol=2e-14)
    torch.testing.assert_close(tiny.initial_state(), tiny.embedding[5], rtol=0, atol=0)


def test_cache_is_immutable_batching_is_rowwise_and_positions_do_not_advance(tiny):
    x = tiny.initial_state()
    cache = tiny.prefix_cache(.07)
    before = [(a.clone(), b.clone()) for a, b in cache.layers]
    batch = torch.stack([x, x + .02, x - .03])
    expected = torch.stack([tiny(v, .07, -.12) for v in batch])
    torch.testing.assert_close(tiny(batch, .07, -.12, cache), expected, rtol=1e-13, atol=1e-14)
    twice = tiny(tiny(x, .07, -.12, cache), .07, -.12, cache)
    torch.testing.assert_close(tiny.period_map(x, .07, -.12, 2, cache), twice, rtol=0, atol=0)
    for pair, saved in zip(cache.layers, before, strict=True):
        for value, original in zip(pair, saved, strict=True):
            assert value.dtype == torch.float64 and value.shape == (1, 1, 3, 4)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="steering parameter"):
        tiny(x, .08, -.12, cache)


def test_prefix_steering_starts_in_next_layer_cache_and_alpha_derivative_includes_it(tiny):
    zero = tiny.prefix_cache(0.)
    cache = tiny.prefix_cache(.07)
    for i in range(tiny.steering_layer + 1):
        for a, b in zip(zero.layers[i], cache.layers[i], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.linalg.vector_norm(zero.layers[2][0] - cache.layers[2][0]) > 1e-6
    x = tiny.initial_state()
    alpha = torch.tensor(.07, dtype=torch.float64)
    full = lambda a: tiny(x, a, -.12)
    tangent = torch.func.jvp(full, (alpha,), (torch.ones_like(alpha),))[1]
    h = 1e-5
    finite = (full(alpha + h) - full(alpha - h)) / (2 * h)
    torch.testing.assert_close(tangent, finite, rtol=2e-7, atol=2e-10)
    frozen = lambda a: torch.softmax(tiny._logits_with_cache(x, a, cache) / (10. ** -.12), dim=-1) @ tiny.embedding
    omitted = torch.func.jvp(frozen, (alpha,), (torch.ones_like(alpha),))[1]
    assert torch.linalg.vector_norm(tangent - omitted) > 1e-5
    with pytest.raises(ValueError, match="parameter derivatives"):
        torch.func.jvp(lambda a: tiny(x, a, -.12, cache), (alpha,), (torch.ones_like(alpha),))


def test_state_and_temperature_jvps_have_central_difference_convergence(tiny):
    x = torch.linspace(-.2, .3, tiny.d, dtype=torch.float64)
    direction = torch.linspace(.1, -.2, tiny.d, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    alpha = .07
    for period in [1, 2]:
        function = lambda state: tiny.period_map(state, alpha, -.12, period)
        tangent = tiny.jvp(x, direction, alpha, -.12, period)
        errors = []
        for h in [1e-3, 5e-4, 2.5e-4]:
            difference = (function(x + h * direction) - function(x - h * direction)) / (2 * h)
            errors.append(float(torch.linalg.vector_norm(difference - tangent)))
        assert 3.2 < errors[0] / errors[1] < 4.8
        assert 3.2 < errors[1] / errors[2] < 4.8
    log_t = torch.tensor(-.12, dtype=torch.float64)
    function = lambda t: tiny.period_map(x, alpha, t, 2)
    tangent = torch.func.jvp(function, (log_t,), (torch.ones_like(log_t),))[1]
    difference = (function(log_t + 1e-5) - function(log_t - 1e-5)) / 2e-5
    torch.testing.assert_close(tangent, difference, rtol=3e-7, atol=2e-10)


def test_chunked_dense_jacobian_has_columns_in_state_coordinate_order(tiny):
    x = tiny.initial_state()
    matrix = tiny.jacobian(x, .07, -.12, period=2, chunk_size=3)
    expected = torch.func.jacfwd(lambda state: tiny.period_map(state, .07, -.12, 2))(x)
    torch.testing.assert_close(matrix, expected, rtol=2e-12, atol=2e-13)
    direction = torch.arange(1, tiny.d + 1, dtype=torch.float64) / tiny.d
    torch.testing.assert_close(matrix @ direction, tiny.jvp(x, direction, .07, -.12, 2), rtol=2e-12, atol=2e-13)


def test_precision_and_model_scope_fail_loudly(tiny):
    with pytest.raises(ValueError, match="float64"):
        tiny(tiny.initial_state().float(), .07, -.12)
    with pytest.raises(ValueError, match="scalar float64"):
        tiny(tiny.initial_state(), torch.tensor(.07, dtype=torch.float32), -.12)
    with pytest.raises(ValueError, match="positive integer"):
        tiny.period_map(tiny.initial_state(), .07, -.12, 0)
    config = dict(tiny.config, use_sliding_window=True)
    with pytest.raises(ValueError, match="full attention"):
        ManualQwenSoftMap(config, tiny.weights, tiny.prefix_ids, tiny.vector, 1)
    weights = dict(tiny.weights)
    del weights["model.layers.0.self_attn.q_proj.bias"]
    with pytest.raises(ValueError, match="tensor inventory"):
        ManualQwenSoftMap(tiny.config, weights, tiny.prefix_ids, tiny.vector, 1)
