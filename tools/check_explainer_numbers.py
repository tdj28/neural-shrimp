"""Check every number the explainer speaks or shows against the current Chaos article (or a named raw file).

Collects the display text of every narration line used by the timeline, every card text of the cards the v1 shots
use, every caption, lower third and text layer of the rendered shot specs, the numbers the data clips record in their
manifest, and the on-screen numbers of the existing videos (listed below from their frames). Each number (integers,
decimals, percentages, powers such as 10⁻¹³ or 2⁻⁶⁴⁰) must occur in the article's main.tex (normalized: LaTeX spacing,
math signs and thousands separators removed), or be listed in RAW with the raw file that holds it, or belong to a
classical-model illustration listed in CLASSICAL. Figure, table and section references in on-screen text are
checked against the labels in main.aux. Prints a report and exits 1 if anything is unsupported.

Usage: python tools/check_explainer_numbers.py [--timeline ../neural-shrimp-data/videos/explainer/explainer-v1.timeline.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT.parent / "chaos-journal-neural-shrimp"
WORK = ROOT.parent / "neural-shrimp-data" / "render-tmp" / "explainer-v1"
CARDS = ROOT.parent / "neural-shrimp-data" / "videos" / "explainer" / "assets" / "cards" / "cards-receipt.json"

# numbers that are not in the article but in a raw file named in the manifests or the timeline (value: source)
RAW = {
    "0.303": "docs/chaos-texts/qwen0.5b-hostility.json steps[44].top[0].p (0.30263)",
    "0.298": "docs/chaos-texts/qwen0.5b-hostility.json steps[44].top[1].p (0.29796)",
    "1,000": "docs/chaos-texts/*.json args.transient",
    "102": "docs/repro-summary.json trajectories.'qwen0.5b steered (REPRO1)'.batch64.chaotic.median_step; h2cells/repro.npz",
    "0.33": "h2cells/repro.npz + docs/h2-horizon-cells.json: map exponent of the chaotic cell shown (0.3337)",
    "77": "h2cells/repro.npz: step at which the chaotic cell shown passes 1e-2",
    "84": "h2cells/repro.npz: ln(1e-2/d1)/lambda of that cell (84.3)",
    "81": "h2cells/repro.npz batch64_agree: first step with a different most probable token",
    "16": "docs/h2-horizon-cells.json: period of the periodic cell shown",
    "2.3": "h2cells/repro.npz: largest distance of the periodic cell shown (2.28e-12)",
    "86": "docs/pi/pm2-summary.json cells.*-gsm8k.soft.accuracy (min 0.8633)",
    "90": "docs/pi/pm2-summary.json cells.*-gsm8k.soft.accuracy (max 0.8984)",
    "128": "docs/rr/rr2-h05-heldout-analysis.json bf16.problems",
    "509": "portraits/p8-approach-f32/portrait.json census.orbits[0].starts",
    "0.0734": "portraits/p8-approach-f32/portrait.json args.alpha",
    "400": "pod64/portraits/f64-strong/portrait.json args.transient",
    "298,800": "pod64/portraits/f64-strong/portrait.json: 498 aperiodic starts x 600 recorded states (also article Fig. caption)",
    "40": "the clip's own step count (N3a shows steps 1 to 40)",
    "56": "the clip's own step count (N3b shows steps 1 to 56); also article Sec. IV (56 %)",
    "80": "attractor-trace-spoken-v2.mp4.json steps (80)",
    "81,920": "portraits/p8-approach-f32/portrait.json: 512 starts x 160 recorded states",
    "160": "portraits/p8-approach-f32/portrait.json args.record",
    "0.07248": "article Fig. (attractor) caption",
    "0.188": "docs/chaos-texts/qwen0.5b-philosophy.json steps[27].top[0].p",
    "0.186": "docs/chaos-texts/qwen0.5b-philosophy.json steps[27].top[1].p",
    "0.197": "docs/chaos-texts/qwen0.5b-philosophy-long.json steps[27].top[0].p",
    "0.171": "docs/chaos-texts/qwen0.5b-philosophy-long.json steps[27].top[1].p",
    "-1.47": "docs/chaos-texts/qwen0.5b-hostility.json steps[44].log10_distance",
    "0.3518": "docs/chaos-texts/qwen0.5b-hostility.json lyapunov[0]",
    "14.2": "morph-video/morph-shares.json: chaotic share of the frame shown (within the article's 13 to 24 %)",
    "22.9": "morph-video/morph-shares.json: chaotic share of the frame shown (within the article's 13 to 24 %)",
    "89.9": "morph-video/frames.txt: last sampled angle of the 0 to 90 degree rotation (article: 0 to 90 degrees)",
    "-0.49": "article Sec. III: steering only the network response 'lowers it by 0.49' ([-0.63,-0.35])",
    "2.3 × 10⁻¹²": "h2cells/repro.npz: largest distance of the periodic cell shown (2.28e-12)",
}
# numbers inside the classical-model illustrations and the clips' own counters (the renderers compute them)
CLASSICAL = {"bifurcation-primer.mp4": "tools/render_bifurcation_primer.py (logistic and quadratic maps)",
             "cusp-data-examples.mp4": "tools/render_data_examples.py (slides of the article's Fig. 16 and the plan)"}
# references that must point at these labels (on-screen 'Fig. N' etc. are resolved from main.aux at render time;
# hard-coded section and table references are checked here)
SECTIONS = {"Sec. III": "sec:theory", "Sec. IV": "sec:chaos", "Sec. V": "sec:adding", "Sec. VI": "sec:cascade",
            "Sec. VII": "sec:shrimp", "Sec. VIII": "sec:discussion", "Table I": "tab:chaostexts",
            "Table II": "tab:orbits", "Table III": "tab:prospective"}
NUM = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?(?:\s?[×x]\s?10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?|(?:10|2)[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+")
SUP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


def normalize_article(tex):
    t = tex.replace("\\,", "").replace("{,}", ",").replace("$", "").replace("~", " ").replace("\\%", "%")
    t = t.replace("--", "-").replace("\\times", "×")
    t = re.sub(r"\^\{(-?\d+)\}", r"^\1", t)
    t = re.sub(r"\\textbackslash n", "⏎", t)
    t = re.sub(r"\s+", " ", t)
    return t


def variants(num):
    """Forms a number may take in the article's LaTeX source."""
    n = num.replace("−", "-").strip()
    out = {n}
    m = re.match(r"^(-?[\d.]+)\s?[×x]\s?10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", n)
    if m:
        e = m.group(2).translate(SUP)
        out |= {f"{m.group(1)}×10^{e}", f"{m.group(1)}\\times10^{e}"}
    m = re.match(r"^(10|2)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$", n)
    if m:
        e = m.group(2).translate(SUP)
        out |= {f"{m.group(1)}^{e}"}
    if "," in n:
        out.add(n.replace(",", ""))
    return out


def in_article(num, art):
    for v in variants(num):
        if re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", art):
            return True
    return False


def collect(timeline):
    items = []                                  # (where, text)
    narr = {ln["id"]: ln for ln in json.loads((ROOT / "docs/explainer/narration-v1.json").read_text())["lines"]}
    tl = json.loads(Path(timeline).read_text())
    for ln in tl["lines"]:
        items.append((f"spoken {ln['id']}", narr[ln["id"]]["display"]))
    cards = json.loads(CARDS.read_text())["cards"]
    used_cards = set()
    for clip in tl["clips"]:
        for f in [clip.get("source") or ""] + clip.get("stage_images", []) + clip.get("overlays", []):
            for name, c in cards.items():
                if any(Path(x["file"]).name == Path(str(f)).name for x in c["files"]):
                    used_cards.add(name)
    used_cards.add("title-overlay")             # composited inside the title shot
    for name in sorted(used_cards):
        for b in cards[name]["content"]:
            if b.get("text"):
                items.append((f"card {name}", b["text"]))
    shots = json.loads((WORK / "v1-shots.json").read_text())
    for s in shots:
        for k in ("caption",):
            if s.get(k):
                items.append((f"shot {s['name']} {k}", s[k]))
        for e in s.get("lower_third", []) + s.get("texts", []):
            items.append((f"shot {s['name']} text", e["text"]))
    data = json.loads((WORK / "MANIFEST-v1-data.json").read_text())
    for c in data["clips"]:
        ns = c["numbers_shown"]
        if c["clip"] == "N5":                   # only what the chart prints (the receipts' raw shares are not shown)
            shown = [v["shown"] for v in ns.values() if isinstance(v, dict) and "shown" in v]
            shown += [v["count"] for v in ns.values() if isinstance(v, dict) and "count" in v]
            items.append(("data clip N5", " · ".join(shown) + " · " + c.get("reasoning_model_accuracy_shown", "")))
        elif c["clip"] == "N4":
            g = c["gates"]
            items.append(("data clip N4", f"{g['chaotic_pass_1e-2']} of {g['chaotic_cells']} · median step "
                                          f"{g['median_first_step']:.0f} · {g['periodic_stay_below_1e-2']} of "
                                          f"{g['periodic_cells']} · exactly 0 in all {g['cells_all']} cells, "
                                          f"{g['steps']} steps · λ = {ns['chaotic_cell']['lambda_shown']} · step "
                                          f"{ns['chaotic_cell']['crossing_step']} · step {ns['chaotic_cell']['predicted_step_shown']} · "
                                          f"step {ns['chaotic_cell']['first_token_difference_step']} · period "
                                          f"{ns['periodic_cell']['period']} · 192 chaotic, 192 periodic, 64 other"))
        else:
            items.append((f"data clip {c['clip']}", json.dumps(ns, ensure_ascii=False)))
    # existing videos: on-screen numbers read from their frames (titles, subtitles, footers, counters)
    items += [
        ("video attractor-tour-f64.mp4", "steering α = 0.07248, log₁₀T = −0.148, float64 · 298,800 states of 498 "
                                         "trajectories, each after 400 steps; principal coordinates 1, 2, 3"),
        ("video attractor-trace-spoken-v2.mp4", "steering α = 0.07248, log₁₀T = −0.148, float64 · step 1 of 80"),
        ("video attractor-tour-p8-fade.mp4", "steering α = 0.0734, log₁₀T = −0.148 (the period-8 window), float32 · "
                                             "81,920 states of 512 trajectories · the 8 states of the loop"),
        ("video morph-steering-rotation.mp4", "float32 maps, 296 x 512 cells · chaotic cells 14.2 to 22.9 % on the frames shown · of 89.9 degrees"),
        ("video cusp-data-examples.mp4", "at 13 points fixed in advance the period-5 orbit counts matched the "
                                         "swallowtail's prediction at 9"),
    ]
    return items, tl


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeline", default=str(ROOT.parent / "neural-shrimp-data/videos/explainer/explainer-v1.timeline.json"))
    a = ap.parse_args()
    tex = (ART / "main.tex").read_text()
    art = normalize_article(tex)
    labels = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", (ART / "main.aux").read_text())}
    items, tl = collect(a.timeline)
    bad, raw_used, checked = [], {}, 0
    for lab, num in (tl.get("article_labels") or {}).items():   # renumbered since the video was built?
        if lab.split(":")[0] in ("fig", "tab", "sec") and labels.get(lab) != num:
            bad.append(("article labels", lab, f"built with {num}, the article now has {labels.get(lab)}"))
    for where, text in items:
        t = text.replace("\u00a0", " ")
        # references
        for m in re.finditer(r"(Sec\.|Table) ([IVX]+)", t):
            ref = f"{m.group(1)} {m.group(2)}"
            want = SECTIONS.get(ref)
            if want is None or labels.get(want) != m.group(2):
                bad.append((where, ref, f"reference does not match main.aux ({want} = {labels.get(want)})"))
        for m in re.finditer(r"Fig\. (\d+)", t):
            if m.group(1) not in labels.values():
                bad.append((where, f"Fig. {m.group(1)}", "no such figure in main.aux"))
        if where.startswith("video ") and any(where.endswith(k) for k in CLASSICAL):
            continue
        for m in NUM.finditer(t):
            num = m.group(0).strip().rstrip(",")
            if not num or num in ("0", "1", "2", "3"):
                continue
            ctx = t[max(0, m.start() - 12):m.end() + 12]
            if re.match(r"^(?:\d+)$", num) and re.search(r"(Sec\.|Table|Fig\.|P\d|step\s)\s?$", t[max(0, m.start() - 6):m.start()]):
                pass
            checked += 1
            if in_article(num, art):
                continue
            if num.replace("−", "-") in RAW:
                raw_used[num] = RAW[num.replace("−", "-")]
                continue
            bad.append((where, num, ctx))
    print(json.dumps({"article_commit": __import__("subprocess").run(["git", "-C", str(ART), "rev-parse", "HEAD"],
                                                                  capture_output=True, text=True).stdout.strip(),
                      "numbers_checked": checked, "raw_sources_used": raw_used, "unsupported": bad},
                     indent=1, ensure_ascii=False))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
