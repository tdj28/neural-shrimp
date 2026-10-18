"""Answers changed by the batch layout alone (explainer chart N5): horizontal bars read from the receipts of the
article's answer-level tests (Sec. IV "Answers of a task" and the abstract), each next to its comparison arm, revealed
group by group over 20 s.

  group 1  a short Soft Thinking procedure on 96 two-digit additions (bfloat16; docs/repro-summary.json, runs REPRO-T
           and REPRO-T7): soft feedback against greedy decoding;
  group 2  Soft Thinking with its published settings, Qwen2.5-7B, 256 GSM8K problems (docs/pi/pm-summary.json, run PM):
           against replaying the fed-back tokens;
  group 3  reasoning models, 256 GSM8K problems each (docs/pi/pm2-summary.json, run PM2): Soft Thinking against discrete
           sampled thinking, the matched control.
Footnote: the exploratory reset test (docs/rr/rr2-*-heldout-analysis.json, run RR2).

No value is typed in: each is read from its receipt and must round to the number printed in the article, and the
article's sentence must still contain it; otherwise nothing is rendered. Gold marks soft-token feedback, violet the
comparison arm (validated for color-vision deficiency on the #0B0B0F background; values in text colors). Frames go to
--frames-dir (on disk), then ffmpeg (H.264, yuv420p, CRF 16, 30 fps); sources and numbers go into --manifest.

Usage: python tools/plot_explainer_answers.py --out ../neural-shrimp-data/videos/explainer/parts/C7-03-N5-answers.mp4
       (add --still 19 to write one PNG at that time instead of the clip)
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import multiprocessing as mp
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = ROOT.parent / "chaos-journal-neural-shrimp" / "main.tex"
W, H, DPI, FPS, SECONDS = 1920, 1080, 72, 30, 20.0
BG, WHITE, BODY, MUTED, GRID = "#0B0B0F", "#FFFFFF", "#C8C8D0", "#9A9AA6", "#23232B"
SOFT, COMP = "#FFD93F", "#9085E9"
SANS = "DejaVu Sans"

REPRO, PM, PM2 = "docs/repro-summary.json", "docs/pi/pm-summary.json", "docs/pi/pm2-summary.json"
RR05, RR7 = "docs/rr/rr2-h05-heldout-analysis.json", "docs/rr/rr2-q7-heldout-analysis.json"
# key: (receipt, path to a share in [0, 1], the number printed in the article)
VALUES = {
    "t05_soft": (REPRO, ["answers", "qwen0.5b (REPRO-T)", "conditions", "bf16 soft 0.0", "answers_differ"], "56"),
    "t05_greedy": (REPRO, ["answers", "qwen0.5b (REPRO-T)", "conditions", "bf16 greedy None", "answers_differ"], "0"),
    "t7_soft": (REPRO, ["answers", "qwen7b (REPRO-T7)", "conditions", "bf16 soft 0.2", "answers_differ"], "55"),
    "t7_greedy": (REPRO, ["answers", "qwen7b (REPRO-T7)", "conditions", "bf16 greedy None", "answers_differ"], "10"),
    "pm_soft": (PM, ["arms", "top15-s64", "flip_rate"], "21"),
    "pm_replay": (PM, ["arms", "replay", "flip_rate"], "6"),
    "r1_soft": (PM2, ["cells", "A-gsm8k", "soft", "flip_rate"], "14.5"),
    "r1_plain": (PM2, ["cells", "A-gsm8k", "plain", "flip_rate"], "10.2"),
    "qwq_soft": (PM2, ["cells", "C-gsm8k", "soft", "flip_rate"], "19.5"),
    "qwq_plain": (PM2, ["cells", "C-gsm8k", "plain", "flip_rate"], "19.9"),
    "rr05_free": (RR05, ["bf16", "disagreement", "free"], "66"),
    "rr05_reset": (RR05, ["bf16", "disagreement", "reset"], "1.6"),
    "rr7_free": (RR7, ["bf16", "disagreement", "free"], "64"),
    "rr7_reset": (RR7, ["bf16", "disagreement", "reset"], "1.6"),
}
# sentences of the article (whitespace-normalized) that must still carry the printed numbers
SENTENCES = [
    r"changed the answer in 56\\,\\% of the problems for Qwen2\.5-0\.5B \(at \$T=1\$\) and in 55\\,\\% for Qwen2\.5-7B",
    r"greedy decoding changed none of the 0\.5B model's answers and 10 of the 96 of the 7B model's",
    r"changed the answers to 21\\,\\% of 256 GSM8K problems",
    r"replaying the fed-back tokens cut this to 6\\,\\%",
    r"changed 14\.5\\,\\% of the Soft Thinking answers of DeepSeek-R1-Distill-Qwen-7B and 19\.5\\,\\% of those of QwQ-32B",
    r"discrete sampled thinking, the matched control, changed 10\.2 and 19\.9\\,\\%",
    r"from 66 to 1\.6\\,\\% in Qwen2\.5-0\.5B and from 64 to 1\.6\\,\\% in Qwen2\.5-7B",
    r"reasoning models that answer 86 to 90\\,\\% of GSM8K problems",
    r"Under a design fixed in advance, we posed 96 two-digit additions",
]


def fail(msg):
    raise SystemExit(f"plot_explainer_answers: gate failed, nothing rendered: {msg}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dig(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            fail(f"missing field {' -> '.join(path)}")
        d = d[k]
    return d


def pct(share, printed):
    """The share as a percentage with the article's number of decimals."""
    dec = len(printed.split(".")[1]) if "." in printed else 0
    return f"{100 * share:.{dec}f}"


def prepare():
    if not ARTICLE.exists():
        fail(f"article not found at {ARTICLE}")
    art = re.sub(r"\s+", " ", ARTICLE.read_text())
    for pat in SENTENCES:
        if not re.search(pat, art):
            fail(f"the article no longer says: {pat}")
    cache, V = {}, {}
    for key, (rec, path, printed) in VALUES.items():
        cache.setdefault(rec, json.loads((ROOT / rec).read_text()))
        share = float(dig(cache[rec], path))
        if pct(share, printed) != printed:
            fail(f"{key}: {rec} gives {100 * share:.4f} %, the article prints {printed}")
        V[key] = {"share": share, "shown": printed + " %", "receipt": rec, "field": " -> ".join(path)}
    # counts behind "none of 96" and "10 of 96" (greedy decoding, group 1)
    for key, name in (("t05_greedy", "qwen0.5b (REPRO-T)"), ("t7_greedy", "qwen7b (REPRO-T7)")):
        c = cache[REPRO]["answers"][name]["conditions"]["bf16 greedy None"]
        V[key]["count"] = round(c["answers_differ"] * c["problems"])
        V[key]["problems"] = c["problems"]
    if (V["t05_greedy"]["count"], V["t7_greedy"]["count"], V["t7_greedy"]["problems"]) != (0, 10, 96):
        fail("greedy counts are not 0 and 10 of 96")
    for key in ("t05_soft", "t7_soft"):
        name = "qwen0.5b (REPRO-T)" if key == "t05_soft" else "qwen7b (REPRO-T7)"
        if cache[REPRO]["answers"][name]["conditions"][VALUES[key][1][3]]["problems"] != 96:
            fail(f"{key}: not 96 problems")
    for key in ("pm_soft", "pm_replay"):
        if cache[PM]["arms"][VALUES[key][1][1]]["problems"] != 256:
            fail(f"{key}: not 256 problems")
    acc = [v for c in ("A-gsm8k", "C-gsm8k") for v in cache[PM2]["cells"][c]["soft"]["accuracy"].values()]
    for c in ("A-gsm8k", "C-gsm8k"):
        for arm in ("soft", "plain"):
            if cache[PM2]["cells"][c][arm]["problems"] != 256:
                fail(f"{c} {arm}: not 256 problems")
    acc_range = (round(100 * min(acc)), round(100 * max(acc)))
    if acc_range != (86, 90):
        fail(f"Soft Thinking accuracy of the reasoning models runs {acc_range}, the article says 86 to 90")
    for rec in (RR05, RR7):
        if cache[rec]["bf16"]["problems"] != 128:
            fail(f"{rec}: not 128 problems")
    groups = [
        dict(t=0.6, title="A short Soft Thinking procedure · 96 two-digit additions", rows=[
            ("Qwen2.5-0.5B", "t05_soft", "t05_greedy", "greedy decoding (none of 96)"),
            ("Qwen2.5-7B", "t7_soft", "t7_greedy", "greedy decoding (10 of 96)")]),
        dict(t=5.0, title="Soft Thinking with its published settings · 256 GSM8K problems", rows=[
            ("Qwen2.5-7B", "pm_soft", "pm_replay", "replaying the fed-back tokens")]),
        dict(t=8.5, title=f"Reasoning models that answer {acc_range[0]} to {acc_range[1]} % correctly · "
                          "256 GSM8K problems", rows=[
            ("DeepSeek-R1-Distill-Qwen-7B", "r1_soft", "r1_plain", "discrete sampled thinking, the matched control"),
            ("QwQ-32B", "qwq_soft", "qwq_plain", "discrete sampled thinking")]),
    ]
    return dict(V=V, groups=groups, acc_range=acc_range,
                sources=sorted({rec for rec, _, _ in VALUES.values()}))


# ----------------------------------------------------------------------------------------------- drawing
def ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def ramp(t, start, dur=0.4):
    return min(1.0, max(0.0, (t - start) / dur))


def text(ax, x, y, s, size, color=BODY, weight="normal", alpha=1.0, ha="left"):
    if alpha > 0:
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=SANS, alpha=alpha, ha=ha,
                va="baseline")


def bar(ax, x, y, w, h, color, alpha):
    """A bar from the baseline x: square at the baseline, 4 px rounded data end; a 3 px stub for zero."""
    if alpha <= 0:
        return
    if w < 8:
        ax.add_patch(Rectangle((x, y), max(w, 3), h, color=color, alpha=alpha, lw=0))
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", color=color, alpha=alpha,
                                lw=0))
    ax.add_patch(Rectangle((x, y), 6, h, color=color, alpha=alpha, lw=0))


def frame(S, t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    V = S["V"]
    text(ax, 80, 100, "Answers changed by the batch layout alone", 52, WHITE, "bold")
    text(ax, 80, 152, "Share of problems whose answer differs between two batch layouts · tests fixed in advance · "
                      "bfloat16", 30, BODY)
    # legend
    ax.add_patch(Rectangle((80, 196), 26, 26, color=SOFT, lw=0))
    text(ax, 120, 218, "soft-token feedback (Soft Thinking)", 28, BODY)
    ax.add_patch(Rectangle((680, 196), 26, 26, color=COMP, lw=0))
    text(ax, 720, 218, "comparison run, named at each bar", 28, BODY)
    bx, unit = 640, 15.5                                     # bars start at bx; one percentage point = 15.5 px
    top, bh, gap = 270, 24, 2
    y = top
    layout, blocks = [], []
    for gi, g in enumerate(S["groups"]):
        y += 36 if gi == 0 else 84
        layout.append(("header", g, y))
        y += 20
        y_block = y
        for r, row in enumerate(g["rows"]):
            y += 10 if r == 0 else 32
            layout.append(("row", (g, row), y))
            y += 2 * bh + gap
        blocks.append((y_block, y + 10, g["t"]))
    y_axis = y + 46
    for v in (0, 20, 40, 60):                                # recessive scale, broken at the group headers
        x = bx + v * unit
        for b0, b1, tg in blocks:
            ax.plot([x, x], [b0, b1], color=GRID, lw=1, zorder=0, alpha=ramp(t, tg - 0.3, 0.4))
        text(ax, x, y_axis, f"{v} %", 24, MUTED, ha="center")
    for kind, obj, yy in layout:
        if kind == "header":
            text(ax, 80, yy, obj["title"], 30, WHITE, "bold", alpha=ramp(t, obj["t"] - 0.3, 0.4))
            continue
        g, (name, soft, comp, comp_name) = obj
        t0 = g["t"] + 0.25 * g["rows"].index(obj[1])
        a = ramp(t, t0 - 0.2, 0.3)
        text(ax, 80, yy + bh + 10, name, 28, BODY, alpha=a)
        for j, (key, color) in enumerate(((soft, SOFT), (comp, COMP))):
            p = ease((t - t0 - 0.2 * j) / 0.8)
            w = V[key]["share"] * 100 * unit * p
            yb = yy + j * (bh + gap)
            bar(ax, bx, yb, w, bh, color, a)
            la = ramp(t, t0 + 0.2 * j + 0.7, 0.3)
            label = V[key]["shown"] if j == 0 else f"{V[key]['shown']} · {comp_name}"
            x_lab = bx + max(V[key]["share"] * 100 * unit, 3) + 14
            text(ax, x_lab, yb + bh - 3, label, 26, WHITE if j == 0 else BODY, "bold" if j == 0 else "normal",
                 alpha=la)
    fa = ramp(t, 3.0, 0.6)
    text(ax, 80, 996, f"Exploratory, 128 further problems: resetting the fed-back state before every step lowered the "
                      f"share from {V['rr05_free']['shown'][:-2]} to {V['rr05_reset']['shown']} (Qwen2.5-0.5B)", 24,
         MUTED, alpha=fa)
    text(ax, 80, 1030, f"and from {V['rr7_free']['shown'][:-2]} to {V['rr7_reset']['shown']} (Qwen2.5-7B) · "
                       "article Sec. IV and abstract", 24, MUTED, alpha=fa)
    return fig


def _worker(job):
    S, frames, outdir = job
    for i in frames:
        fig = frame(S, i / FPS)
        fig.savefig(outdir / f"{i:05d}.png", dpi=DPI, facecolor=BG)
        plt.close(fig)
    return len(frames)


# ----------------------------------------------------------------------------------------------- output
def upsert(manifest, entry):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(manifest.read_text()) if manifest.exists() else {"clips": []}
        data["clips"] = [c for c in data["clips"] if c["file"] != entry["file"]] + [entry]
        data["clips"].sort(key=lambda c: c["file"])
        manifest.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def entry(S, out, seconds):
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--",
                                 "tools/plot_explainer_answers.py"], capture_output=True, text=True).stdout.strip())
    V = S["V"]
    shown = {k: {"shown": v["shown"], "value": round(100 * v["share"], 4), "receipt": v["receipt"],
                 "field": v["field"], **({"count": f"{v['count']} of {v['problems']}"} if "count" in v else {})}
             for k, v in V.items()}
    return {"file": out.name, "clip": "N5", "chapter": "C7", "shot": "7.3", "seconds": seconds,
            "format": "1920x1080, 30 fps, H.264 High, yuv420p, CRF 16, no audio",
            "tool": "tools/plot_explainer_answers.py", "commit": head, "tool_uncommitted": dirty, "sha256": sha256(out),
            "data_sources": [{"path": p, "sha256": sha256(ROOT / p)} for p in S["sources"]],
            "numbers_shown": shown,
            "reasoning_model_accuracy_shown": f"{S['acc_range'][0]} to {S['acc_range'][1]} % "
                                              "(Soft Thinking arms, both layouts, pm2-summary cells A-gsm8k and C-gsm8k)",
            "colors": {"soft-token feedback": SOFT, "comparison arm": COMP},
            "article": "abstract and Sec. IV 'Answers of a task' (every bar value; the matched control 10.2 and 19.9 %; "
                       "the exploratory reset test 66 -> 1.6 % and 64 -> 1.6 %)",
            "rendered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, default=ROOT.parent / "neural-shrimp-data/render-tmp/explainer-data")
    ap.add_argument("--manifest", type=Path, default=None, help="default: MANIFEST-data-clips.json beside --out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--still", type=float, default=None, help="write one PNG at this time (s) and stop")
    a = ap.parse_args()
    S = prepare()
    print(json.dumps({k: v["shown"] for k, v in S["V"].items()}, ensure_ascii=False))
    if a.still is not None:
        out = a.out.with_name(a.out.stem + f"-still-{a.still:05.2f}.png")
        fig = frame(S, a.still)
        fig.savefig(out, dpi=DPI, facecolor=BG)
        print(f"wrote {out}")
        return
    frames = a.frames_dir / a.out.stem
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    n = int(round(SECONDS * FPS))
    chunks = [list(range(i, n, a.workers)) for i in range(a.workers)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        done = sum(pool.map(_worker, [(S, c, frames) for c in chunks]))
    if done != n:
        fail(f"rendered {done} of {n} frames")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "%05d.png"),
                    "-vf", "setsar=1,format=yuv420p", "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS), "-g", "60", "-movflags", "+faststart", "-an",
                    str(a.out)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(a.out)], capture_output=True, text=True, check=True).stdout)
    upsert(a.manifest or a.out.parent / "MANIFEST-data-clips.json", entry(S, a.out, round(dur, 3)))
    shutil.rmtree(frames)
    print(f"wrote {a.out} ({dur:.2f} s, {n} frames)")


if __name__ == "__main__":
    sys.exit(main())
