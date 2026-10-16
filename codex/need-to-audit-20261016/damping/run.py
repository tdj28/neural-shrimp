"""Hash-bound reader for the exact saved items 3–5; --card PATH --out FRESH_DIR."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("own_reducer", HERE / "reduce.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
D = [("qwen0.5b", "Qwen2.5-0.5B", "qwen", .49), ("qwen1.5b", "Qwen2.5-1.5B", "qwen", 1.54),
     ("qwen3b", "Qwen2.5-3B", "qwen", 3.09), ("qwen7b", "Qwen2.5-7B", "qwen", 7.62),
     ("qwen14b", "Qwen2.5-14B", "qwen", 14.7), ("qwen32b", "Qwen2.5-32B", "qwen", 32.8),
     ("qwen72b", "Qwen2.5-72B", "qwen", 72.7), ("llama3.1-8b", "Llama-3.1-8B", "llama", 8.03),
     ("llama3.1-70b", "Llama-3.1-70B", "llama", 70.6), ("llama3.3-70b", "Llama-3.3-70B", "llama", 70.6),
     ("olmo2-1b", "OLMo-2-1B", "olmo", 1.48), ("olmo2-7b", "OLMo-2-7B", "olmo", 7.30),
     ("olmo2-13b", "OLMo-2-13B", "olmo", 13.7), ("olmo2-32b", "OLMo-2-32B", "olmo", 32.2)]
MOE = ["qwen3.5-4b", "qwen3.5-9b", "qwen3.5-35b-a3b", "olmo2-1b", "olmo2-7b", "olmoe-1b-7b"]
PUB = {"damping": "paper/llm/figures/fig-damping.json", "moe": "paper/llm/figures/fig-moe.json",
       "texts": "paper/llm/figures/fig-chaos-twins.json"}


def verify(pin):
    p = Path(pin["path"])
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    if p.stat().st_size != pin["bytes"] or h.hexdigest() != pin["sha256"]:
        raise ValueError("input pin mismatch: " + str(p))


def load(pin):
    return json.loads(Path(pin["path"]).read_text())


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [safe(v) for v in x]
    if isinstance(x, np.generic): return safe(x.item())
    if isinstance(x, float) and not math.isfinite(x): return str(x)
    return x


def write(out, name, value):
    p = out / name
    with p.open("x") as f: json.dump(safe(value), f, ensure_ascii=False, allow_nan=False, indent=1)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def npz(row, key):
    with np.load(row[key]["path"], allow_pickle=False) as z:
        expected = {x["name"][:-4]: x for x in row["npz_members"]}
        if set(z.files) != set(expected): raise ValueError("NPZ field coverage")
        arrays = {k: z[k] for k in z.files}
        for k, a in arrays.items():
            e = expected[k]
            if list(a.shape) != e["shape"] or a.dtype.str != e["descr"]: raise ValueError("NPZ schema: " + k)
    return arrays


def input_union(card, meta, extra, source):
    pins = [card[k] for k in ("metadata", "supplement", "source_manifest", "pt128_manifest")]
    pins += card["authority_files"]
    for row in meta["damping"]: pins += [row["json"], row["layers"]]
    for row in meta["scans"] + extra["scans"]: pins += [row["npz"], row["metadata"]]
    pins += [row["json"] for row in meta["chaos_texts"]] + [meta["vocabulary"]["json"]]
    pubs = {}
    for k, name in PUB.items():
        found = [x for x in source["files"] if x["source_path"] == name]
        if len(found) != 1: raise ValueError("publication pin uniqueness")
        pubs[k] = found[0]; pins += found
    unique = {}
    for pin in pins:
        p = str(Path(pin["path"]).resolve())
        if p in unique and (pin["sha256"], pin["bytes"]) != (unique[p]["sha256"], unique[p]["bytes"]):
            raise ValueError("conflicting input pins")
        unique[p] = {k: pin[k] for k in ("path", "bytes", "sha256")}
    return list(unique.values()), pubs


def execute(card, out):
    tic = time.perf_counter()
    if card["status"] != "reviewed_for_execution": raise ValueError("prepared card is unlaunchable")
    for name, sha in card["source_sha256"].items():
        if hashlib.sha256((HERE/(name+".py")).read_bytes()).hexdigest() != sha: raise ValueError("source hash")
    if set(card["source_sha256"]) != {"run", "reduce", "controls"}: raise ValueError("source inventory")
    cs=importlib.util.spec_from_file_location("own_controls",HERE/"controls.py")
    controls=importlib.util.module_from_spec(cs);cs.loader.exec_module(controls)
    control_result=controls.main()
    for k in ("metadata", "supplement", "source_manifest", "pt128_manifest"): verify(card[k])
    meta, extra, source = load(card["metadata"]), load(card["supplement"]), load(card["source_manifest"])
    pins, pubs = input_union(card, meta, extra, source)
    if sum(p["bytes"] for p in pins) > card["input_bytes_cap"]: raise ValueError("input cap")
    if card["expected_input_count"] != len(pins): raise ValueError("input union count")
    for p in pins: verify(p)
    dr = meta["damping"]; sr = meta["scans"]+extra["scans"]; tr = meta["chaos_texts"]
    expected_models = {x[0] for x in D} | set(MOE)
    if len(dr) != 18 or {x["model"] for x in dr} != expected_models: raise ValueError("18-model coverage")
    if len(sr) != 10 or len({x["id"] for x in sr}) != 10: raise ValueError("10-scan coverage")
    expected_scan_ids={"olmoe-1b-7b-alpha0-wide"}|{p+q for p in ("qwen3.5-4b-","qwen3.5-9b-","m1t-") for q in ("alpha0","host","phil")}
    if {x["id"] for x in sr} != expected_scan_ids: raise ValueError("scan identities")
    if len(tr) != 4 or len({x["id"] for x in tr}) != 4: raise ValueError("4-text coverage")
    damping = {}
    for row in dr:
        doc = load(row["json"])
        if doc["args"] != row["args"] or len(doc["cells"]) != 24: raise ValueError("damping args/count")
        damping[row["model"]] = r.damping(doc, npz(row, "layers"), row["model"])
    scans = {}
    for row in sr:
        doc = load(row["metadata"])
        if doc["args"] != row["args"]: raise ValueError("scan args")
        arrays = npz(row, "npz")
        if list(arrays["lyapunov"].shape) != row["args"]["resolution"]: raise ValueError("scan complete grid")
        scans[row["id"]] = r.scan(arrays, row["args"]["max_period"])
        scans[row["id"]]["args"] = row["args"]
        scans[row["id"]]["axes"] = {k: arrays[k].tolist() for k in ("alpha", "log10_temperature")}
    docs, texts = {}, {}
    for row in tr:
        doc = load(row["json"])
        if doc["args"] != row["args"] or len(doc["cells"]) != len(row["cells"]): raise ValueError("text args/count")
        for c, e in zip(doc["cells"], row["cells"]):
            if (c["alpha"], c["log10T"], len(c["steps"])) != (e["alpha"], e["log10T"], e["step_count"]): raise ValueError("text cell association")
        docs[row["id"]] = doc
        texts[row["id"]] = [r.text_cell(doc["args"], c) for c in doc["cells"]]
    chosen, projection = [], []
    if len(meta["text_selections"]) != 7: raise ValueError("seven figure selections")
    for e in meta["text_selections"]:
        key = Path(e["receipt"]).stem; doc = docs[key]
        found = [i for i, c in enumerate(doc["cells"]) if (c["alpha"], c["log10T"]) == (e["alpha"], e["log10T"])]
        if len(found) != 1 or doc["args"]["model"] != e["model"]: raise ValueError("text selection uniqueness")
        q = texts[key][found[0]]; lam = q["mean_lyapunov"]; split = q["saved_id_based_split"]
        chosen.append({"receipt": e["receipt"], "index": found[0], "result": q})
        logs = [x["distance"]["rounded_log10"] for x in q["all_steps"]]
        projection.append(dict(e, lyapunov=q["lyapunov"], split_step=split,
            order_one_steps_predicted=math.log(1/doc["args"]["eps_text"])/lam if math.isfinite(lam) and lam != 0 else None,
            lyapunov_times_at_split=lam*split if split else None,
            log10_distance_at_split=logs[split-1] if split else None,
            **{"first_step_distance_above_0.1":next((i+1 for i,v in enumerate(logs) if v > -1), None)}))
    if len({(x["receipt"],x["index"]) for x in chosen}) != 7: raise ValueError("duplicate figure selection")
    left, right = docs["qwen0.5b-philosophy"], docs["qwen0.5b-philosophy-long"]
    if (left["args"]["threads"],right["args"]["threads"]) != (6,4) or (left["cells"][0]["alpha"],left["cells"][0]["log10T"]) != (right["cells"][0]["alpha"],right["cells"][0]["log10T"]): raise ValueError("thread association")
    thread = r.thread_comparison(left["cells"][0], right["cells"][0])
    vd = load(meta["vocabulary"]["json"])
    if vd["args"] != meta["vocabulary"]["args"] or len(vd["cells"]) != 24: raise ValueError("vocabulary args/count")
    vocab = r.vocabulary(vd)
    dp = {"models": {k: dict(damping[k]["projection"], label=label,family=fam,params_b=p) for k,label,fam,p in D}}
    mp = {"damping":{},"routing":{},"chaotic":{}}
    for k in MOE:
        a=damping[k]; mp["damping"][k]={"median_g_net_0.3_6":a["main_bin"]["literal_median"],"cells_0.3_6":a["main_bin"]["count"],"max_closure":a["projection"]["max_closure"]}
        if a["routing"]: mp["routing"][k]={f:a["routing"][f] for f in ("routers","median_gap_range","min_gap_range","max_frozen_minus_free","total_mean_switches")}
    for row in sr:
        if row["args"]["model"] == "olmoe-1b-7b": continue
        label = "unsteered" if "alpha0" in row["id"] else "philosophy" if row["id"].endswith("phil") else "hostility"
        a=scans[row["id"]]; mp["chaotic"].setdefault(row["args"]["model"],{})[label]={"chaotic":a["literal_count"],"cells":a["total"],"largest_exponent":a["literal_max_exponent"]}
    compare = {k:r.comparisons(own,load(pubs[k])) for k,own in (("damping",dp),("moe",mp),("texts",projection))}
    near_ko=[{"model":k,"row":damping[k]["knockouts"][damping[k]["nearest_one_nat_knockout_row"]]} for k,*_ in D if damping[k]["nearest_one_nat_knockout_row"] is not None]
    qualified_ko=len(near_ko)==14 and all(x["row"]["qualified"] for x in near_ko)
    maxrise=max(x["row"]["maximum_factor_rise_on_removal"] for x in near_ko) if qualified_ko else None
    split_rows=[x["result"]["split_row"] for x in chosen]
    all_split=all(x is not None and x["probabilities_qualified"] and x["distance"]["qualified"] for x in split_rows)
    claims={"near_one_nat_knockouts":near_ko,"all14_knockouts_qualified":qualified_ko,"maximum_factor_rise_on_removal":maxrise,
            "literal_at_most_0.44":maxrise<=.44 if maxrise is not None else None,
            "split_summary_qualified":all_split,
            "maximum_top_probability_at_split":max(x["leading_probability"] for x in split_rows) if all_split else None,
            "maximum_top_two_lead_at_split":max(x["top_two_lead"] for x in split_rows) if all_split else None,
            "runner_up_text_matches_at_split":sum(x["twin_matches_runner_up_text"] for x in split_rows) if all_split else None,
            "split_distance_display_range":[min(x["distance"]["display_value"] for x in split_rows),max(x["distance"]["display_value"] for x in split_rows)] if all_split else None,
            "split_distance_rounding_envelope":[min(x["distance"]["possible_distance_lower"] for x in split_rows),max(x["distance"]["possible_distance_upper"] for x in split_rows)] if all_split else None}
    outputs=[write(out,"damping.json",damping),write(out,"scans.json",scans),write(out,"texts.json",{"all_records":texts,"selected":chosen,"thread_comparison":thread}),write(out,"vocabulary.json",vocab),write(out,"comparisons.json",compare)]
    elapsed=time.perf_counter()-tic; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for p in pins: verify(p)
    report={"schema":"need-to-audit-items3-5-v1","computational_status":"complete","source_sha256":card["source_sha256"],"input_pins":pins,"outputs":outputs,"controls":control_result,"claim_summaries":claims,
            "coverage":{"damping_models":len(damping),"damping_cells":sum(len(x["rows"]) for x in damping.values()),"scans":len(scans),"scan_cells":sum(x["total"] for x in scans.values()),"text_cells":sum(len(x) for x in texts.values()),"selected_text_cells":len(chosen),"vocabulary_rows":len(vocab["rows"])},
            "published_comparison_counts":{k:{"leaves":len(v),"nonexact":sum(x["status"]!="exact" for x in v)} for k,v in compare.items()},
            "elapsed_before_final_rehash_report":elapsed,"observed_peak_rss_kib_before_final_rehash_report":rss,
            "scope":"saved arithmetic only; no native/model/process replay; exact scientific failures and differences retained; direct lifecycle observation belongs to coordinator"}
    write(out,"report.json",report)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--card",required=True);ap.add_argument("--out",required=True);a=ap.parse_args()
    out=Path(a.out);created=False
    try:
        card=json.loads(Path(a.card).read_text());out.mkdir(parents=True,exist_ok=False);created=True;execute(card,out)
    except Exception as e:
        if created: write(out,"failure.json",{"type":type(e).__name__,"message":str(e)})
        raise


if __name__ == "__main__": main()
