"""Manufactured, source-only preparation. No target paths or producer imports."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("own_reduce", Path(__file__).with_name("reduce.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    H = [0, .3, 1, 3, 6, 11]
    g = [0, -1, -2, -3, -4, -5]
    cells = [{"log10_temperature": i/10, "mean_entropy": h, "mean_log_g_net": x,
              "mean_stack": 2., "mean_dilution": 1., "mean_final_norm": x-1.,
              "mean_log_g_read": 0., "boundary_term": .25, "lyapunov": x+.25,
              "period": 0, "chaotic": bool(x+.25 > .001), "mean_c_x": 1., "max_closure": 0.}
             for i, (h, x) in enumerate(zip(H, g))]
    arrays = {"log10_temperature": np.array([c["log10_temperature"] for c in cells]),
              "mean_A": np.tile(np.arange(4)[:, None], (1, 6)), "mean_N": np.zeros((4, 6)),
              "mean_c_attn": np.zeros((4, 6)), "mean_c_mlp": np.zeros((4, 6))}
    doc = {"args": {"record": 10}, "layers": 4, "hidden": 7, "cells": cells,
           "knockouts": {"cells": [1, 3], "steps": 2, "delta_attn": [[-.44001, 1.], [0, 0], [0, 0], [0, 0]],
                         "delta_mlp": [[2., 0], [0, 0], [0, 0], [0, 0]], "plain_twin_check": [0., 1e-9]}}
    return doc, arrays


def check_bins():
    d, a = fixture(); out = r.damping(d, a, "qwen-test")
    assert out["main_bin"]["indices"] == [1, 2, 3]
    assert out["main_bin"]["literal_median"] == -2
    assert out["projection"]["bin_medians"] == {"0-1": -.5, "1-3": -2., "3-6": -3.}
    assert out["above_10"]["literal_median"] == -5


def check_depth_and_olmo():
    d, a = fixture(); out = r.damping(d, a, "olmo2-1b")
    assert out["profile"]["literal_plot_final_relative_change"] == 3
    assert out["profile"]["true_final_residual_relative_change"] == 1
    assert not any(x["sublayer_quantities_qualified"] for x in out["rows"])
    assert all(x["sublayer_quantities_qualified"] for x in r.damping(d, a, "olmoe-1b-7b")["rows"])


def check_knockout():
    d, a = fixture(); out = r.damping(d, a, "qwen-test")
    assert out["nearest_one_nat_knockout_row"] == 0
    assert not out["nearest_grid_cell_has_knockout"]
    assert out["knockouts"][0]["maximum_factor_rise_on_removal"] == .44001
    assert out["knockouts"][0]["maximum_factor_drop_on_removal"] == 2.


def check_routing():
    d, a = fixture(); d["routers"] = 2
    for i, c in enumerate(d["cells"]):
        c.update(margin_median=.001*(i+1), margin_min_over_steps=.0001*(i+1),
                 mean_switches=0., mean_log_g_net_frozen=c["mean_log_g_net"])
    out = r.damping(d, a, "moe-test")["routing"]
    assert out["median_gap_range"] == [.001, .006]
    assert out["all_zero_means"] and out["max_frozen_minus_free"] == 0


def check_invalid_scan():
    out = r.scan({"lyapunov": np.array([.001, .002, np.inf, -.1, .2, .3]),
                  "period": np.array([0, 0, 0, 0, 2, np.nan])}, 64)
    assert out["literal_count"] == 2 and out["qualified_count"] == 1
    assert out["invalid_period"] == 1 and out["nonfinite"] == 2
    assert out["period_positive_exponent"] == 1 and out["no_period_not_above_threshold"] == 2


def text_fixture():
    steps = [{"top": [{"token": 1, "text": "a", "p": .3}, {"token": 2, "text": "b", "p": .25},
                       {"token": 3, "text": "c", "p": .1}], "twin": t, "log10_distance": x}
             for t, x in zip(("a", "a", "b"), (-10., -2.01, -1.99))]
    return {"record": 3}, {"alpha": .1, "log10T": .2, "period": 0, "lyapunov": [.2, .2],
                           "chaotic": True, "split_step": 3, "text": "aaa", "twin_text": "aab", "steps": steps}


def check_text():
    a, c = text_fixture(); out = r.text_cell(a, c)
    assert out["first_saved_text_difference"] == 3 and out["split_row"]["twin_matches_runner_up_text"]
    assert out["first_rounded_distance_above_0.01"] == 3
    assert out["steps_split_after_rounded_0.01_crossing"] == 0
    assert out["base_concatenation_matches"] and out["twin_concatenation_matches"]
    assert not out["native_twin_token_ids_saved"]
    c["split_step"] = 2
    assert not r.text_cell(a, c)["saved_split_equals_text_split"]


def check_rounded_bounds():
    x = r.distance_interval(.24)
    assert x["possible_distance_lower"] < 10**.24 < x["possible_distance_upper"]
    assert x["display_value"] > 1.7
    assert r.distance_interval(-300)["possible_distance_lower"] == 0


def check_thread_ids():
    _, a = text_fixture(); b = copy.deepcopy(a)
    b["steps"][1]["top"][0]["token"] = 9
    out = r.thread_comparison(a, b)
    assert out["first_token_difference"] == 2 and out["first_text_difference"] is None


def check_refusals():
    d, a = fixture(); a["log10_temperature"][0] = 99
    try:
        r.damping(d, a, "qwen-test")
    except ValueError:
        pass
    else:
        raise AssertionError("wrong temperature association admitted")
    d, a = fixture(); d["knockouts"]["cells"] = [1, 1]
    try:
        r.damping(d, a, "qwen-test")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate knockout association admitted")


def main():
    names = ["bins", "depth_and_olmo", "knockout", "routing", "invalid_scan", "text", "rounded_bounds", "thread_ids", "refusals"]
    for name in names:
        globals()["check_"+name]()
    return {"manufactured_controls": names, "passed": len(names), "failed": 0}


if __name__ == "__main__":
    print(json.dumps(main()))
