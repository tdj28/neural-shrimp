"""Systematic exploratory A2 survey over all audit-confirmed EXP-028 windows.

Subcommands (run in order; each writes under --out):
  rays     CPU ray continuation (64 rays) for every confirmed window
  maps     GPU local period maps (256 x 256 by default) around every window
  analyze  boundary-type counts, flip-to-2p check, ray/map edge agreement,
           draft A2 criteria, and an overlay figure per window

Confirmed windows are those qualifying in both the EXP-028 frozen analysis and
its independent audit. Exploratory under the adaptive research protocol; the
draft criteria are those in docs/track-a-next-design-2026-10-06.md, scaled to
64 rays (fold and flip each on at least 8 rays).
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import explore_a2_rays as rays_tool  # noqa: E402
from neural_shrimp.continuation import ray_directions  # noqa: E402

SUMMARY = ROOT / "docs" / "EXP-028-summary.json"
AUDIT = ROOT / "../neural-shrimp-data/audits/EXP-028/validation-001/audit.json"


def confirmed_windows():
    summary = json.loads(SUMMARY.read_text())
    audit = json.loads(AUDIT.read_text())
    producer = {(q["map_id"], q["component"]) for q in summary["gate"]["qualifying"]}
    audited = {(q["map_id"], q["component"]) for q in audit["audit_gate"]["qualifying"]}
    return sorted(producer & audited)


def window_records():
    certificates = json.loads((rays_tool.RAW / "stage3" / "certificates.json").read_text())
    out = []
    for map_id, component in confirmed_windows():
        record = next(r for r in certificates if r["map_id"] == map_id and r["component"] == component)
        saved = np.load(rays_tool.map_path(map_id))
        scale, _ = rays_tool.component_geometry(saved, record)
        out.append({"key": f"{map_id}:{component}", "map_id": map_id, "component": component,
                    "system": map_id.split("-")[0], "seed": int(map_id.split("-")[1]),
                    "period": record["period"], "orbit0": record["certificate"]["orbit"][0],
                    "center": np.log10([record["lr_hidden"], record["lr_output"]]).tolist(),
                    "scale": scale.tolist()})
    return out


def cmd_rays(args, out):
    windows = window_records()
    tasks, index = [], []
    for w in windows:
        for d in ray_directions(args.rays):
            tasks.append((w["system"], w["seed"], np.array(w["center"]), np.array(w["scale"]), d,
                          w["orbit0"], w["period"]))
            index.append((w["key"], d.tolist()))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(rays_tool.one_ray, tasks, chunksize=1))
    report = {w["key"]: {**{k: v for k, v in w.items() if k != "orbit0"}, "rays": []} for w in windows}
    for (key, direction), result in zip(index, results):
        report[key]["rays"].append({"direction": direction, **result})
    (out / "rays.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"rays for {len(windows)} windows written")


def cmd_maps(args, out):
    for w in window_records():
        name = w["key"].replace(":", "-c")
        target = out / "maps" / name
        if (target / "map.npz").exists():
            continue
        center, scale = np.array(w["center"]), np.array(w["scale"])
        hidden = [center[0] - scale[0], center[0] + scale[0]]
        output = [center[1] - scale[1], center[1] + scale[1]]
        command = [sys.executable, str(ROOT / "tools" / "explore_gpu_map.py"), "--name",
                   f"a2-survey/maps/{name}", "--system", w["system"], "--seed", str(w["seed"]),
                   "--hidden", *map(str, hidden), "--output", *map(str, output),
                   "--resolution", str(args.resolution), "--library", args.library]
        subprocess.run(command, check=True, cwd=ROOT)
        print("map", name, flush=True)


def edge_metrics(window, rays, saved):
    period, status = saved["period"], saved["status"]
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    dh, do = lh[1] - lh[0], lo[1] - lo[0]
    center, scale, p = np.array(window["center"]), np.array(window["scale"]), window["period"]
    agree, flip_2p, flip_total, types = 0, 0, 0, {}
    compared = 0
    for ray in rays:
        types[ray["type"]] = types.get(ray["type"], 0) + 1
        if ray["s_boundary"] is None:
            continue
        d = np.array(ray["direction"]) * scale
        edge, first = None, None
        for s in np.linspace(0, 1, 4000):
            point = center + s * d
            r, c = int(round((point[0] - lh[0]) / dh)), int(round((point[1] - lo[0]) / do))
            if not (0 <= r < len(lh) and 0 <= c < len(lo)):
                break
            if not (status[r, c] == 2 and period[r, c] == p):
                edge, first = s, (int(status[r, c]), int(period[r, c]))
                break
        if edge is None:
            continue
        compared += 1
        agree += abs(edge - ray["s_boundary"]) * np.hypot(d[0] / dh, d[1] / do) <= 2.0
        if ray["type"] == "flip":
            flip_total += 1
            flip_2p += first == (2, 2 * p)
    return {"types": types, "edge_agreement": [int(agree), compared],
            "flip_to_2p": [int(flip_2p), flip_total]}


def cmd_analyze(args, out):
    report = json.loads((out / "rays.json").read_text())
    rows = []
    for key, window in report.items():
        name = key.replace(":", "-c")
        map_file = ROOT / "../neural-shrimp-data/explore/a2-survey/maps" / name / "map.npz"
        metrics = edge_metrics(window, window["rays"], np.load(map_file)) if map_file.exists() else None
        n = len(window["rays"])
        types = metrics["types"] if metrics else {}
        verdict = None
        if metrics:
            a, c = metrics["edge_agreement"]
            f2, ft = metrics["flip_to_2p"]
            verdict = bool(types.get("fold", 0) >= n / 8 and types.get("flip", 0) >= n / 8
                           and ft > 0 and f2 >= ft / 2 and c > 0 and a >= 0.75 * c)
        rows.append({"key": key, "period": window["period"], "system": window["system"],
                     "metrics": metrics, "shrimp_consistent_draft": verdict})
    (out / "survey.json").write_text(json.dumps(rows, indent=1, default=float))
    passed = [r for r in rows if r["shrimp_consistent_draft"]]
    print(f"windows: {len(rows)}; shrimp-consistent (draft criteria): {len(passed)}")
    for r in rows:
        m = r["metrics"] or {}
        print(r["key"], r["period"], m.get("types"), "edge", m.get("edge_agreement"),
              "flip->2p", m.get("flip_to_2p"), "PASS" if r["shrimp_consistent_draft"] else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["rays", "maps", "analyze"])
    parser.add_argument("--out", default=str(ROOT / "../neural-shrimp-data/explore/a2-survey"))
    parser.add_argument("--rays", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--library", default=str(ROOT / "../neural-shrimp-data/builds/gdmap-cuda-64af0462/gdmap_cuda.so"))
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    {"rays": cmd_rays, "maps": cmd_maps, "analyze": cmd_analyze}[args.command](args, out)


if __name__ == "__main__":
    main()
