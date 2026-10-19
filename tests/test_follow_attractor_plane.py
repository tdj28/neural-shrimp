"""tools/follow_attractor.py: argument parsing and grids of both planes, and whole runs with a stub loop in place of
the model (CPU, no model files, a few seconds). The alpha-temperature plane gives every cell (and its twin) the
temperature of its row and has no second steering block; the two-block plane builds the grids of the runs before the
plane option and writes the same outputs, byte for byte, as the tool before it (from git history, when available)."""
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import follow_attractor as fa  # noqa: E402

BEFORE = "e168359331f62de5fbe32e21713377d7511a7c5c"      # the last commit of tools/follow_attractor.py before --plane
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))   # as the tools
FA_RUNS = DATA / "explore" / "llm-soft-loop" / "pod64" / "fa"
OLD_KEYS = ["model", "block1", "vectors1", "concept1", "block2", "vectors2", "concept2", "log_temperature", "alpha1",
            "alpha2", "step", "seed", "seed_period", "seed_transient", "seed_state", "reference_word", "transient",
            "record", "token_steps", "max_slow_exponent", "max_period", "tol", "eps", "min_match", "max_attempts",
            "max_batch", "max_minutes", "threads", "device", "out"]
FA100 = ["--log-temperature", "-0.1", "--seed", "-0.118730", "0.100693", "--alpha1", "-0.1320", "-0.1040",
         "--alpha2", "0.0860", "0.1225", "--step", "1.25e-4", "--device", "cuda", "--out", "llm-soft-loop/pod64/fa/t"]
P9S = ["--plane", "alpha-temperature", "--block1", "12", "--vectors1", "llm-phase-map/vectors-l12", "--concept1",
       "philosophy", "--alpha1", "0.07430", "0.07450", "--log-temperature-range", "-0.15380", "-0.15340",
       "--step", "2.93e-5", "--step2", "8.8e-5", "--seed", "0.074388", "-0.15362", "--seed-period", "9",
       "--out", "llm-soft-loop/follow-test/p9-smoke"]


def without(argv, option, values):
    """argv with option and its values removed."""
    k = argv.index(option)
    return argv[:k] + argv[k + 1 + values:]


def grid(lo, hi, step):
    return np.arange(lo, hi + step / 2, step)          # the expression of the tool before --plane


def test_two_block_arguments_and_grid_as_before():
    args = fa.parse_args(FA100)
    assert args.plane == "two-block" and args.step2 is None and args.log_temperature_range is None
    assert (args.block2, args.vectors2, args.concept2) == (16, "llm-phase-map/vectors-l16", "hostility")
    rec = fa.recorded_args(args)
    assert list(rec) == OLD_KEYS                       # summary.json's args: the keys of the tool before, in order
    assert rec["log_temperature"] == -0.1 and rec["alpha2"] == [0.086, 0.1225] and rec["step"] == 1.25e-4
    a1, a2 = fa.build_grid(args)
    assert a1.tobytes() == grid(-0.1320, -0.1040, 1.25e-4).tobytes()
    assert a2.tobytes() == grid(0.0860, 0.1225, 1.25e-4).tobytes()
    assert (len(a2), len(a1)) == (293, 225)            # FA-100b's grid


@pytest.mark.parametrize("name", ["t-0.100", "t-0.105", "p11-zh2"])
def test_two_block_grid_reproduces_earlier_runs(name):
    """The grids saved by FA-100b (-0.100, -0.105) and FA-P11, rebuilt from their recorded options, bit for bit."""
    run = FA_RUNS / name
    if not (run / "summary.json").exists():
        pytest.skip(f"{run} not present")
    a = json.loads((run / "summary.json").read_text())["args"]
    argv = ["--log-temperature", repr(a["log_temperature"]), "--alpha1", *map(repr, a["alpha1"]),
            "--alpha2", *map(repr, a["alpha2"]), "--step", repr(a["step"]), "--seed", *map(repr, a["seed"]),
            "--out", a["out"]]
    a1, a2 = fa.build_grid(fa.parse_args(argv))
    saved = np.load(run / "follow.npz")
    assert a1.tobytes() == saved["alpha1"].tobytes() and a2.tobytes() == saved["alpha2"].tobytes()


def test_two_block_step2_spaces_the_rows_only():
    args = fa.parse_args(FA100 + ["--step2", "2.5e-4"])
    a1, a2 = fa.build_grid(args)
    assert a1.tobytes() == grid(-0.1320, -0.1040, 1.25e-4).tobytes()
    assert a2.tobytes() == grid(0.0860, 0.1225, 2.5e-4).tobytes()
    rec = fa.recorded_args(args)
    assert "plane" not in rec and rec["step2"] == 2.5e-4      # a given option is recorded


def test_alpha_temperature_arguments_and_grid():
    args = fa.parse_args(P9S)
    assert args.plane == "alpha-temperature"
    assert (args.alpha2, args.log_temperature, args.block2, args.vectors2, args.concept2) == (None,) * 5
    a1, a2 = fa.build_grid(args)
    assert a1.tobytes() == grid(0.07430, 0.07450, 2.93e-5).tobytes()
    assert a2.tobytes() == grid(-0.15380, -0.15340, 8.8e-5).tobytes()     # log10 T, with its own spacing
    assert (len(a2), len(a1)) == (6, 8)
    assert np.min(np.abs(a1 - 0.074388)) < 2.93e-5 / 2 and np.min(np.abs(a2 + 0.15362)) < 8.8e-5 / 2
    rec = fa.recorded_args(args)
    assert rec["plane"] == "alpha-temperature" and rec["step2"] == 8.8e-5
    assert rec["log_temperature_range"] == [-0.1538, -0.1534] and rec["alpha2"] is None
    args = fa.parse_args(without(P9S, "--step2", 1))           # the rows default to --step
    assert fa.build_grid(args)[1].tobytes() == grid(-0.15380, -0.15340, 2.93e-5).tobytes()
    assert fa.recorded_args(args)["step2"] == 2.93e-5
    full = P9S[:P9S.index("--alpha1")] + ["--alpha1", "0.0700", "0.0770", "--log-temperature-range", "-0.1660",
                                          "-0.1440"] + P9S[P9S.index("--step"):]
    a1, a2 = fa.build_grid(fa.parse_args(full))                # the full P9S-follow window
    assert (len(a2), len(a1)) == (251, 240)


@pytest.mark.parametrize("argv", [
    P9S + ["--log-temperature", "-0.15"],
    P9S + ["--alpha2", "0.0", "0.1"],
    P9S + ["--block2", "16"],
    P9S + ["--vectors2", "llm-phase-map/vectors-l16"],
    P9S + ["--concept2", "hostility"],
    without(P9S, "--log-temperature-range", 2),
    without(P9S, "--log-temperature-range", 2) + ["--log-temperature-range", "-0.1534", "-0.1538"],
    P9S + ["--step2", "0"],
    FA100 + ["--log-temperature-range", "-0.2", "-0.1"],
    FA100 + ["--step2", "-1e-4"],
    without(FA100, "--log-temperature", 1),
    without(FA100, "--alpha2", 2),
    FA100 + ["--plane", "temperature"],
])
def test_refused(argv, capsys):
    with pytest.raises(SystemExit) as err:
        fa.parse_args(argv)
    assert err.value.code == 2


# ---- whole runs with a stub loop -------------------------------------------------------------------------------------

P = 3                                                   # the stub's period inside its region
X0 = np.array([3.0, 2.0, 1.0, 0.5, 0.25])              # the stub's fixed start


class StubLoop:
    """The part of SoftLoop that follow_attractor uses. Its map cycles the first P components of the state (period P)
    in the cells where region(alpha1, alpha2 or None, log10 T) holds, and the first P + 1 (period P + 1) elsewhere;
    a step's token is the largest component. It records the layers, rows and temperatures it is given."""

    def __init__(self, region, log):
        import torch
        self.torch, self.device, self.state_dtype = torch, torch.device("cpu"), torch.float64
        self.region, self.log = region, log
        self.extra = self.extra_rows = self.steer_rows = None
        self.sm = types.SimpleNamespace(add_steer_layer=log["layers"].append,
                                        tok=types.SimpleNamespace(decode=lambda ids: "".join(f"<{t}>" for t in ids)))

    def prepare(self, steer_rows):
        steer_rows = np.asarray(steer_rows)
        B = len(steer_rows)
        self.steer_rows = steer_rows[:, 0].copy()           # the vectors are e_0: component 0 is the strength
        self.extra_rows = None
        if self.extra is not None:                          # as SoftLoop.prepare: the cells' rows, repeated
            rows2 = np.asarray(self.extra[1])
            assert B % len(rows2) == 0
            self.extra_rows = np.concatenate([rows2] * (B // len(rows2)))[:, 0]
            self.log["extra_layers"].add(self.extra[0])
        return self.torch.as_tensor(np.tile(X0, (B, 1, 1)), dtype=self.state_dtype)

    def step(self, X, temps):
        torch = self.torch
        temps = np.asarray(temps, dtype=np.float64)
        B = X.shape[0]
        assert temps.shape == (B,) and len(self.steer_rows) == B and B % 2 == 0
        n = B // 2
        assert np.array_equal(temps[:n], temps[n:])         # every twin at its cell's temperature
        a2 = None if self.extra_rows is None else self.extra_rows
        lt = np.log10(temps)
        for b in range(n):
            self.log["cells"].add((round(float(self.steer_rows[b]), 12),
                                   None if a2 is None else round(float(a2[b]), 12), round(float(lt[b]), 12)))
        inside = torch.as_tensor(self.region(self.steer_rows, a2, lt))[:, None]
        x = X[:, -1, :]
        short = torch.cat([x[:, P - 1:P], x[:, :P - 1], x[:, P:]], dim=1)
        long_ = torch.cat([x[:, P:P + 1], x[:, :P], x[:, P + 1:]], dim=1)
        new = torch.where(inside, short, long_)
        return new[:, None, :], torch.softmax(new, dim=-1)


@pytest.fixture
def stub(monkeypatch):
    """install(region, data_root): stub modules in place of the model's, so that main() runs the stub loop."""
    pytest.importorskip("torch")

    def install(region, root):
        log = {"loops": [], "layers": [], "vectors": [], "cells": set(), "extra_layers": set()}
        e0 = np.eye(len(X0))[0]
        ela = types.ModuleType("explore_llm_attractors")
        ela.OPENER = "Hello!"
        elpm = types.ModuleType("explore_llm_phase_map")

        def load_vectors(path):
            log["vectors"].append(path)
            return {"gratitude": e0, "hostility": e0, "philosophy": e0}

        elpm.load_vectors = load_vectors
        esl = types.ModuleType("explore_llm_soft_loop")
        esl.DATA = Path(root)

        def soft_loop(*a):
            log["loops"].append(a)
            return StubLoop(region, log)

        esl.SoftLoop = soft_loop
        for mod in (ela, elpm, esl):
            monkeypatch.setitem(sys.modules, mod.__name__, mod)
        return log

    return install


SMALL = ["--seed-period", str(P), "--min-match", str(P), "--max-period", "8", "--record", "20", "--token-steps", "12",
         "--transient", "4", "--seed-transient", "6", "--out", "run"]


def expected_labels(inside):
    """Members where the region holds (it is connected here), non-members next to them, the rest not computed."""
    near = np.zeros_like(inside)
    n2, n1 = inside.shape
    for i in range(n2):
        for j in range(n1):
            near[i, j] = inside[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].any()
    return np.where(inside, 1, np.where(near, 0, -1))


def test_alpha_temperature_run_with_stub_loop(stub, tmp_path):
    region = lambda a1, a2, lt: a1 + 2 * lt < 0.15                 # noqa: E731  (depends on the temperature)
    log = stub(region, tmp_path)
    fa.main(["--plane", "alpha-temperature", "--block1", "12", "--vectors1", "v12", "--concept1", "philosophy",
             "--alpha1", "0.0", "0.5", "--step", "0.1", "--log-temperature-range", "-0.2", "0.1", "--step2", "0.05",
             "--seed", "0.0", "-0.2", *SMALL])
    out = tmp_path / "explore" / "run"
    a1, a2 = grid(0.0, 0.5, 0.1), grid(-0.2, 0.1, 0.05)
    assert log["loops"] == [("qwen0.5b", 12, 8, "open", "Hello!", 1, "float64", "cpu")]
    assert log["layers"] == [] and log["extra_layers"] == set()     # no second steering block
    assert log["vectors"] == ["v12"]
    f = np.load(out / "follow.npz")
    assert str(f["plane"]) == "alpha-temperature" and "alpha2" not in f.files
    assert f["alpha1"].tobytes() == a1.tobytes() and f["log10_temperature"].tobytes() == a2.tobytes()
    inside = region(a1[None, :], None, a2[:, None])
    assert np.array_equal(f["label"], expected_labels(inside))     # membership follows each row's temperature
    assert np.all(f["period"][inside] == P) and np.all(f["period"][f["label"] == 0] == P + 1)
    computed = {(round(float(a1[j]), 12), None, round(float(a2[i]), 12)) for i, j in zip(*np.nonzero(f["label"] >= 0))}
    assert log["cells"] == computed                                # every cell ran at T = 10 ** (its row value)
    s = json.loads((out / "summary.json").read_text())
    assert s["plane"] == "alpha-temperature" and s["members"] == int(inside.sum()) == 21
    assert s["args"]["step2"] == 0.05 and s["args"]["log_temperature_range"] == [-0.2, 0.1]
    assert s["reference_text"] == [f"<{t}>" for t in s["reference_word"]]
    recs = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert recs[0]["plane"] == "alpha-temperature" and recs[0]["seed"] == [0.0, -0.2] and recs[0]["period"] == P
    assert {"stop": "no frontier", "wave": s["waves"]} in recs
    st = np.load(out / "states.npz")
    assert [tuple(r) for r in st["index"]] == sorted(zip(*map(list, np.nonzero(inside))))


def test_two_block_run_with_stub_loop(stub, tmp_path):
    region = lambda a1, a2, lt: a1 + 2 * a2 < 0.45                 # noqa: E731
    log = stub(region, tmp_path)
    fa.main(["--log-temperature", "-0.1", "--alpha1", "0.0", "0.5", "--alpha2", "0.0", "0.3", "--step", "0.1",
             "--seed", "0.0", "0.0", *SMALL])
    out = tmp_path / "explore" / "run"
    a1, a2 = grid(0.0, 0.5, 0.1), grid(0.0, 0.3, 0.1)
    assert log["layers"] == [16] and log["extra_layers"] == {16} and log["vectors"] == ["llm-phase-map/vectors-l4",
                                                                                       "llm-phase-map/vectors-l16"]
    f = np.load(out / "follow.npz")
    assert f.files == ["alpha1", "alpha2", "log10_temperature", "label", "lyapunov", "period", "words", "agreement",
                       "parent", "attempts", "wave", "reference_word"]
    assert float(f["log10_temperature"]) == -0.1 and f["alpha2"].tobytes() == a2.tobytes()
    inside = region(a1[None, :], a2[:, None], None)
    assert np.array_equal(f["label"], expected_labels(inside))
    computed = {(round(float(a1[j]), 12), round(float(a2[i]), 12), -0.1) for i, j in zip(*np.nonzero(f["label"] >= 0))}
    assert log["cells"] == computed                                # one temperature everywhere
    s = json.loads((out / "summary.json").read_text())
    assert "plane" not in s and list(s["args"]) == OLD_KEYS
    assert "plane" not in json.loads((out / "progress.jsonl").read_text().splitlines()[0])


def test_two_block_outputs_byte_identical_to_the_tool_before(stub, tmp_path, monkeypatch):
    """The same stub run with the tool of commit BEFORE and with this one: follow.npz and states.npz are equal byte
    for byte, summary.json and progress.jsonl apart from their timings."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{BEFORE}:tools/follow_attractor.py"],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git not available")
    if source.returncode != 0:
        pytest.skip(f"commit {BEFORE} not available")
    path = tmp_path / "follow_attractor_before.py"
    path.write_text(source.stdout)
    monkeypatch.setattr(sys, "path", list(sys.path))               # the old module inserts its own tools path
    spec = importlib.util.spec_from_file_location("follow_attractor_before", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    region = lambda a1, a2, lt: a1 + 2 * a2 < 0.45                 # noqa: E731
    argv = ["--log-temperature", "-0.1", "--alpha1", "0.0", "0.5", "--alpha2", "0.0", "0.3", "--step", "0.1",
            "--seed", "0.0", "0.0", "--seed-state", "start.npy:1", "--reference-word", "2,0,1", *SMALL]
    for name in ("old", "new"):                                    # a saved seed state and a given reference word
        (tmp_path / name / "explore").mkdir(parents=True)
        np.save(tmp_path / name / "explore" / "start.npy", np.stack([X0, X0[[1, 2, 0, 3, 4]]]))
    stub(region, tmp_path / "old")
    monkeypatch.setattr(sys, "argv", ["follow_attractor.py", *argv])
    old.main()
    stub(region, tmp_path / "new")
    fa.main(argv)
    a, b = tmp_path / "old" / "explore" / "run", tmp_path / "new" / "explore" / "run"
    for name in ("follow.npz", "states.npz"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    timing = re.compile(r'"seconds": [0-9.e+-]+')
    for name in ("summary.json", "progress.jsonl"):
        assert timing.sub("", (a / name).read_text()) == timing.sub("", (b / name).read_text()), name
    assert json.loads((b / "summary.json").read_text())["members"] > 1
