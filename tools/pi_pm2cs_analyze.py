"""Analysis of PM2CS (docs/pi/pm2cs-design.md): PM2's endpoints with Cold Stop and a 4096-step thinking budget.

Per cell: the endpoints of tools/pi_matrix2_analyze.py, plus how thinking ended per arm and layout (the model's own
end-of-thinking token, Cold Stop, or the budget) and the median switch step. The same endpoints are also computed for
the PM2 cell restricted to the same problems (the PM2 rows are subset by index).
Usage: python tools/pi_pm2cs_analyze.py --cell NAME:SOFT:PLAIN:PM2_SOFT:PM2_PLAIN ... --json OUT
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def ends(run_dir):
    out = {}
    for line in open(Path(run_dir) / "records.jsonl"):
        r = json.loads(line)
        if r["index"] < 0:
            continue
        d = out.setdefault(str(r["layout"]), {"rows": 0, "own": 0, "cold_stop": 0, "budget": 0, "switch": []})
        d["rows"] += 1
        if r.get("cold_stop"):
            d["cold_stop"] += 1
        elif r["forced_switch"]:
            d["budget"] += 1
        else:
            d["own"] += 1
        if r["switch_step"] is not None:
            d["switch"].append(r["switch_step"])
    for d in out.values():
        s = d.pop("switch")
        d["median_switch"] = float(np.median(s)) if s else None
    return out


def subset(run_dir, keep, dest):
    dest.mkdir(parents=True)
    (dest / "run.json").write_text((Path(run_dir) / "run.json").read_text())
    with open(dest / "records.jsonl", "w") as f:
        for line in open(Path(run_dir) / "records.jsonl"):
            if json.loads(line)["index"] in keep:
                f.write(line)


def analyze(cells):
    with tempfile.NamedTemporaryFile(suffix=".json") as tf:
        cmd = [sys.executable, str(HERE / "pi_matrix2_analyze.py")]
        for c in cells:
            cmd += ["--cell", c]
        subprocess.run(cmd + ["--json", tf.name], check=True, capture_output=True)
        return json.loads(Path(tf.name).read_text())["cells"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for spec in args.cell:
            name, soft, plain, psoft, pplain = spec.split(":")
            keep = {json.loads(line)["index"] for line in open(Path(soft) / "records.jsonl")}
            ps, pp = Path(tmp) / name / "soft", Path(tmp) / name / "plain"
            subset(psoft, keep, ps)
            subset(pplain, keep, pp)
            res = analyze([f"cs:exploratory:{soft}:{plain}", f"pm2:exploratory:{ps}:{pp}"])
            out[name] = {"cold_stop": res["cs"], "pm2_same_problems": res["pm2"],
                         "ends": {"soft": ends(soft), "plain": ends(plain), "pm2_soft": ends(ps), "pm2_plain": ends(pp)},
                         "dirs": [soft, plain, psoft, pplain]}
    print(json.dumps(out, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
