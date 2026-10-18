"""Two runs of the soft-token loop that write the same tokens and then part (explainer clips N3a and N3b).

Every word on screen is the most probable next token of a step; the loop itself feeds back the probability-weighted
average embedding, never that token. All tokens and probabilities are read from saved receipts of
tools/chaos_texts.py (Qwen2.5-0.5B-Instruct, float64, recorded after a 1000-step transient):

  --mode pair  two recordings of one cell, made with six and with four CPU threads (the thread count only reorders
               floating-point sums): docs/chaos-texts/qwen0.5b-philosophy.json and qwen0.5b-philosophy-long.json,
               alpha 0.0652, log10 T -0.156. The two texts are drawn as two columns of paragraphs (each blank-line
               token, shown as a return sign, ends a line). Gate: the first step whose most probable tokens differ
               is step 28, a near tie of " question" and " existence" in both runs.
  --mode twin  one loop and its free twin started at relative distance 1e-10 (docs/chaos-texts/qwen0.5b-hostility.json,
               alpha 0.08946, log10 T -0.05566): two token rows and the relative distance against the step, with the
               growth set by the twin exponent (lyapunov[0]). Gates: first difference at step 45, where the loop has
               "Yes" 0.303 and "No" 0.298 and the copy writes "No"; exponent 0.35 to two decimals.

Both modes also check the article's own statements (27 equal steps; split at step 45; lambda 0.35) and refuse to
render when any gate fails. Tokens after the first difference are red (as in the article's Table I), tokens that
differ from the other run are underlined, and only the two most probable tokens of a step are shown (a second token
below 0.01 is summarized as "others < 0.01", which also keeps rare profanity of the hostility cell off screen).
Frames go to --frames-dir (on disk, not /tmp), then ffmpeg (H.264, yuv420p, CRF 16, 30 fps); the clip's sources,
gates and every number shown are upserted into --manifest.

Usage: python tools/render_text_split.py --mode pair --out ../neural-shrimp-data/videos/explainer/parts/C0-01-N3a-thread-split.mp4
       python tools/render_text_split.py --mode twin --out ../neural-shrimp-data/videos/explainer/parts/C2-03-N3b-twin-split.mp4
       (add --still 17.5 to write one PNG at that time instead of the clip)
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H, DPI = 1920, 1080, 72                                   # at 72 dpi a font size in points is one in pixels
BG, WHITE, BODY, MUTED, GRID = "#0B0B0F", "#FFFFFF", "#C8C8D0", "#9A9AA6", "#23232B"
RED, ORANGE = "#FF6B4D", "#FF8C1A"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
MONO_ADV = 1233 / 2048                                       # advance of every DejaVu Sans Mono glyph (em), incl. the return sign
BLOCKED = ("fuck", "shit", "cunt", "bitch", "nigg", "fag", "whore", "slut", "dick", "piss", "bastard")
ARTICLE = ROOT.parent / "chaos-journal-neural-shrimp" / "main.tex"

MODES = {
    "pair": dict(a="docs/chaos-texts/qwen0.5b-philosophy.json", b="docs/chaos-texts/qwen0.5b-philosophy-long.json",
                 steps=40, split=28, threads=(6, 4), cell=(0.0652, -0.156), seconds=27.0,
                 title="The same loop, run twice",
                 subtitle="Only the number of CPU threads differs, which reorders floating-point sums",
                 rows=("6 CPU threads", "4 CPU threads"),
                 footer=("Qwen2.5-0.5B-Instruct · soft-token loop · philosophy steering α = 0.0652, "
                         "log₁₀T = −0.156 · float64",
                         "recorded after 1,000 warm-up steps · shown: the most probable next token at each step")),
    "twin": dict(a="docs/chaos-texts/qwen0.5b-hostility.json", steps=56, split=45, cell=(0.08946, -0.05566),
                 seconds=18.0, title="A loop and a copy started 10⁻¹⁰ apart",
                 subtitle="The copy's state starts displaced by one part in 10¹⁰; nothing else differs",
                 rows=("loop", "copy, started 10⁻¹⁰ away"),
                 footer=("Qwen2.5-0.5B-Instruct · hostility steering at layer 12 · α = 0.08946, log₁₀T = −0.05566 · "
                         "λ = 0.35 · float64 · article Table I",
                         "recorded after 1,000 warm-up steps · shown: the most probable next token at each step")),
}


def fail(msg):
    raise SystemExit(f"render_text_split: gate failed, nothing rendered: {msg}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def show(tok):
    """Display form of a token: line breaks as return signs."""
    return tok.replace("\n", "⏎")


def label(tok):
    """A token in the probability panel: return signs, no leading space, in quotes."""
    return "'" + show(tok).strip(" ") + "'"


def load(path):
    d = json.loads((ROOT / path).read_text())
    if not d.get("cells"):
        fail(f"{path} has no cells")
    return d["args"], d["cells"][0]


def schedule(mode, n, split):
    """Time at which each step (1-based) appears; the step of the first difference is held 3 s."""
    if mode == "pair":
        t0, slow, fast = 2.0, 0.55, 0.35
    else:
        t0, slow, fast = 1.5, 0.22, 0.20
    appear, t = [], t0
    for k in range(1, n + 1):
        appear.append(t)
        t += 3.0 if k == split else (slow if k < split else fast)
    return appear


def prepare(mode):
    """Read the receipts, run the gates and return everything a frame needs."""
    spec = MODES[mode]
    article = ARTICLE.read_text() if ARTICLE.exists() else fail(f"article not found at {ARTICLE}")
    n, split = spec["steps"], spec["split"]
    argsA, A = load(spec["a"])
    if (round(A["alpha"], 6), round(A["log10T"], 6)) != spec["cell"] or argsA["transient"] != 1000:
        fail(f"{spec['a']}: cell or transient differs from {spec['cell']}, 1000")
    S = dict(mode=mode, spec=spec, n=n, split=split, sources=[spec["a"]])
    if mode == "pair":
        argsB, B = load(spec["b"])
        if (round(B["alpha"], 6), round(B["log10T"], 6)) != spec["cell"] or argsB["transient"] != 1000:
            fail(f"{spec['b']}: cell or transient differs")
        if (argsA["threads"], argsB["threads"]) != spec["threads"]:
            fail(f"thread counts {argsA['threads']}, {argsB['threads']} are not {spec['threads']}")
        if len(A["steps"]) < n or len(B["steps"]) < n:
            fail("fewer recorded steps than shown")
        rows = [[s["top"][0]["text"] for s in A["steps"][:n]], [s["top"][0]["text"] for s in B["steps"][:n]]]
        tops = [[s["top"][:2] for s in A["steps"][:n]], [s["top"][:2] for s in B["steps"][:n]]]
        S["sources"].append(spec["b"])
        if not re.search(r"wrote the same text for\s+27 steps", article):
            fail("the article no longer says 'the same text for 27 steps'")
    else:
        if len(A["steps"]) < n:
            fail("fewer recorded steps than shown")
        rows = [[s["top"][0]["text"] for s in A["steps"][:n]], [s["twin"] for s in A["steps"][:n]]]
        tops = [[s["top"][:2] for s in A["steps"][:n]], None]
        S["dist"] = [s["log10_distance"] for s in A["steps"][:n]]
        S["lam"] = float(A["lyapunov"][0])
        if round(S["lam"], 2) != 0.35 or round(S["lam"], 3) != 0.352:
            fail(f"twin exponent {S['lam']} does not round to 0.35 (article) and 0.352 (plan)")
        if A.get("split_step") != split or "texts split at step 45" not in article or r"\lambda=0.35" not in article:
            fail("split step 45 or lambda 0.35 not confirmed by the receipt and the article")
    diff = [k for k in range(n) if rows[0][k] != rows[1][k]]
    if not diff or diff[0] + 1 != split:
        fail(f"first differing step is {diff[0] + 1 if diff else None}, expected {split}")
    k = split - 1
    if mode == "pair":
        pa, pb = tops[0][k], tops[1][k]
        got = [(t["text"], round(t["p"], 3)) for t in pa + pb]
        if got != [(" question", 0.188), (" existence", 0.186), (" existence", 0.197), (" question", 0.171)]:
            fail(f"top two at step {split} are {got}")
    else:
        got = [(t["text"], round(t["p"], 3)) for t in tops[0][k]] + [rows[1][k]]
        if got != [("Yes", 0.303), ("No", 0.298), "No"]:
            fail(f"step {split}: loop top two and copy are {got}")
        if round(S["dist"][k], 2) != -1.47:
            fail(f"log10 distance at step {split} is {S['dist'][k]}")
    shown = [t for r in rows for t in r]
    for run in tops:
        for pair in run or []:
            shown += [pair[0]["text"]] + ([pair[1]["text"]] if pair[1]["p"] >= 0.01 else [])
    bad = sorted({t for t in shown if any(b in t.lower() for b in BLOCKED)})
    if bad:
        fail(f"a displayed token is on the blocklist: {bad}")
    S.update(rows=rows, tops=tops, diff=set(diff), appear=schedule(mode, n, split))
    return S


# ----------------------------------------------------------------------------------------------- drawing helpers
def canvas():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size, color=BODY, weight="normal", family=SANS, alpha=1.0, ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, alpha=alpha, ha=ha,
            va="baseline")


def ramp(t, start, dur=0.3):
    return min(1.0, max(0.0, (t - start) / dur))


def layout(tokens, wrap, break_newline, prefix=0):
    """(line, column) of each token in characters; a line ends after a newline token or before overflowing wrap."""
    pos, line, col = [], 0, prefix
    for tok in tokens:
        w = len(show(tok))
        if wrap and col + w > wrap and col > 0:
            line, col = line + 1, 0
        pos.append((line, col))
        col += w
        if break_newline and "\n" in tok:
            line, col = line + 1, 0
    return pos


def draw_tokens(ax, S, t, r, x0, y0, size, lead, pos):
    """Tokens of run r shown so far: older ones in body color, the current one bold white, red from the split."""
    adv = MONO_ADV * size
    shown = sum(1 for a in S["appear"] if a <= t)
    for k in range(shown):
        line, col = pos[k]
        x, y = x0 + col * adv, y0 + line * lead
        alpha = ramp(t, S["appear"][k], 0.12)
        cur = k == shown - 1
        col_ = RED if k >= S["split"] - 1 else (WHITE if cur else BODY)
        tok = show(S["rows"][r][k])
        text(ax, x, y, tok, size, col_, "bold" if cur else "normal", MONO, alpha)
        if k in S["diff"]:
            off = len(tok) - len(tok.lstrip(" "))
            ax.plot([x + off * adv, x + len(tok) * adv], [y + 9, y + 9], color=RED, lw=3, alpha=alpha,
                    solid_capstyle="butt")
    return shown


def chrome(ax, S, shown):
    spec = S["spec"]
    text(ax, 80, 100, spec["title"], 52, WHITE, "bold")
    text(ax, 80, 152, spec["subtitle"], 30, BODY)
    if shown:
        text(ax, W - 80, 100, f"step {shown}", 52, WHITE, "bold", ha="right")
    for i, line in enumerate(spec["footer"]):
        text(ax, 80, 1000 + 34 * i, line, 26, MUTED)


def frame_pair(S, t):
    fig, ax = canvas()
    size, lead = 36, 47
    xs = (140, 620)
    shown = 0
    for r in (0, 1):
        text(ax, xs[r], 228, S["spec"]["rows"][r], 32, BODY, "bold")
        pos = layout(S["rows"][r], 0, True, prefix=1)
        text(ax, xs[r], 290, "…", size, MUTED, family=MONO, alpha=ramp(t, S["appear"][0], 0.12))
        shown = draw_tokens(ax, S, t, r, xs[r], 290, size, lead, pos)
    chrome(ax, S, shown)
    # the two most probable next tokens of the current step, per run
    px, bx = 1130, 1330
    bw = 300 / max(tp["p"] for run in S["tops"] for step in run for tp in step)   # largest bar: 300 px
    vx = bx + 320
    if shown:
        k = shown - 1
        text(ax, px, 300, "two most probable next tokens", 28, MUTED)
        for r in (0, 1):
            y = 372 + 170 * r
            text(ax, px, y, S["spec"]["rows"][r], 28, BODY, "bold")
            for j, tp in enumerate(S["tops"][r][k]):
                yy = y + 50 + 46 * j
                written = j == 0
                tcol = (RED if k >= S["split"] - 1 else WHITE) if written else BODY
                text(ax, px, yy, label(tp["text"]), 28, tcol, "bold" if written else "normal", MONO)
                ax.add_patch(plt.Rectangle((bx, yy - 22), bw * tp["p"], 24, color=BODY if written else MUTED,
                                           alpha=0.9 if written else 0.6, lw=0))
                text(ax, vx, yy, f"{tp['p']:.3f}", 28, tcol, family=MONO)
        if shown >= S["split"]:
            a = ramp(t, S["appear"][S["split"] - 1], 0.3)
            ax.add_patch(plt.Rectangle((px, 752), 8, 58, color=RED, alpha=a, lw=0))
            text(ax, px + 28, 796, f"step {S['split']}: different words", 40, WHITE, "bold", alpha=a)
            text(ax, px + 28, 846, "the two most probable tokens swap places", 28, BODY, alpha=a)
            text(ax, 140, 948, "red: from the first difference on · underlined: a token that differs between the runs",
                 24, MUTED, alpha=a)
    return fig


def frame_twin(S, t):
    fig, ax = canvas()
    size, lead, wrap = 34, 46, 80
    shown = 0
    for r in (0, 1):
        y0 = 222 + 158 * r
        text(ax, 120, y0, S["spec"]["rows"][r], 28, MUTED, "bold")
        pos = layout(S["rows"][r], wrap, False, prefix=1)
        text(ax, 120, y0 + 50, "…", size, MUTED, family=MONO, alpha=ramp(t, S["appear"][0], 0.12))
        shown = draw_tokens(ax, S, t, r, 120, y0 + 50, size, lead, pos)
    chrome(ax, S, shown)
    k = shown - 1
    if shown and shown != S["split"]:                        # during the hold at the split the callout replaces it
        p = S["tops"][0][k]
        second = f"{label(p[1]['text'])} {p[1]['p']:.3f}" if p[1]["p"] >= 0.01 else "others < 0.01"
        text(ax, 120, 562, f"loop, two most probable next tokens:  {label(p[0]['text'])} {p[0]['p']:.3f}  ·  {second}",
             28, MUTED, family=SANS)
    if shown >= S["split"]:
        a = ramp(t, S["appear"][S["split"] - 1], 0.3) * (1.0 if shown == S["split"] else 0.0)
        if a > 0:
            p = S["tops"][0][S["split"] - 1]
            ax.add_patch(plt.Rectangle((120, 530), 8, 46, color=RED, alpha=a, lw=0))
            text(ax, 148, 565, f"step {S['split']}, loop:  {label(p[0]['text'])} {p[0]['p']:.3f}  ·  "
                 f"{label(p[1]['text'])} {p[1]['p']:.3f};  the copy writes {label(S['rows'][1][S['split'] - 1])}",
                 34, WHITE, "bold", alpha=a)
            text(ax, 1500, 565, "(probabilities: full report)", 24, MUTED, alpha=a)
    # relative distance between loop and copy
    x0, x1, y0, y1 = 210, 1840, 628, 930
    pax = fig.add_axes([x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H], facecolor=BG)
    n = S["n"]
    pax.set_xlim(0, n + 1)
    pax.set_ylim(-10.6, 0.7)
    for s in pax.spines.values():
        s.set_visible(False)
    pax.axhline(-10.6, color=MUTED, lw=1)
    for yv in range(-10, 1, 2):
        pax.axhline(yv, color=GRID, lw=1, zorder=0)
    pax.set_yticks(range(-10, 1, 2))
    pax.set_yticklabels(["10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "1"])
    pax.set_xticks(range(0, n + 1, 10))
    pax.tick_params(colors=MUTED, labelsize=24, length=0, pad=10)
    pax.text(0.6, 1.2, "distance between loop and copy, relative to the size of the state", fontsize=26,
             color=BODY, va="center")
    pax.text(n + 1, -10.95, "step", fontsize=24, color=MUTED, ha="right", va="top")
    lam = S["lam"]
    xs = [0, n]
    pax.plot(xs, [-10 + lam * x / math.log(10) for x in xs], color=BODY, lw=2, ls=(0, (1.2, 2.2)), zorder=2)
    pax.plot([1, 4], [-1.3, -1.3], color=ORANGE, lw=3)
    pax.text(4.8, -1.3, "measured", fontsize=24, color=BODY, va="center")
    pax.plot([1, 4], [-2.6, -2.6], color=BODY, lw=2, ls=(0, (1.2, 2.2)))
    pax.text(4.8, -2.6, f"growth at the Lyapunov exponent, λ = {lam:.2f} per step", fontsize=24, color=BODY,
             va="center")
    if shown:
        steps = list(range(1, shown + 1))
        pax.plot(steps, S["dist"][:shown], color=ORANGE, lw=3, solid_joinstyle="round", zorder=3)
        if shown != S["split"]:
            pax.plot([shown], [S["dist"][shown - 1]], "o", ms=9, color=WHITE, mec=BG, mew=2, zorder=5)
    if shown >= S["split"]:
        ks = S["split"]
        pax.plot([ks], [S["dist"][ks - 1]], "o", ms=14, color=RED, mec=BG, mew=2, zorder=4)
    return fig


def render(S, t):
    return frame_pair(S, t) if S["mode"] == "pair" else frame_twin(S, t)


def _worker(job):
    S, frames, outdir = job
    for i in frames:
        fig = render(S, i / 30.0)
        fig.savefig(outdir / f"{i:05d}.png", dpi=DPI, facecolor=BG)
        plt.close(fig)
    return len(frames)


# ----------------------------------------------------------------------------------------------- output
def git_state():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tools/render_text_split.py"],
                           capture_output=True, text=True).stdout.strip()
    return head, bool(dirty)


def upsert(manifest, entry):
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(manifest.read_text()) if manifest.exists() else {"clips": []}
        data["clips"] = [c for c in data["clips"] if c["file"] != entry["file"]] + [entry]
        data["clips"].sort(key=lambda c: c["file"])
        manifest.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def manifest_entry(S, out, seconds):
    spec, k = S["spec"], S["split"] - 1
    head, dirty = git_state()
    nums = {"steps_shown": [1, S["n"]], "first_different_step": S["split"]}
    if S["mode"] == "pair":
        nums.update(alpha=0.0652, log10T=-0.156, threads=list(spec["threads"]), warmup_steps=1000,
                    top_two_at_split={r: [[t["text"], round(t["p"], 3)] for t in S["tops"][i][k]]
                                      for i, r in enumerate(spec["rows"])},
                    probabilities="every step's two most probable tokens of both runs, 3 decimals, from steps[k].top")
        clip, chapter, shot = "N3a", "C0", "0.1"
    else:
        nums.update(alpha=0.08946, log10T=-0.05566, lyapunov=round(S["lam"], 4), lyapunov_shown=round(S["lam"], 2),
                    copy_offset=1e-10, warmup_steps=1000,
                    at_split={"loop": [[t["text"], round(t["p"], 3)] for t in S["tops"][0][k]],
                              "copy": S["rows"][1][k], "log10_distance": S["dist"][k]},
                    distance="steps[k].log10_distance, relative |copy - loop| / |loop|",
                    probabilities="loop's two most probable tokens per step, 3 decimals; a second below 0.01 shown "
                                  "as 'others < 0.01'")
        clip, chapter, shot = "N3b", "C2", "2.3"
    return {"file": out.name, "clip": clip, "chapter": chapter, "shot": shot, "seconds": seconds,
            "format": "1920x1080, 30 fps, H.264 High, yuv420p, CRF 16, no audio",
            "tool": "tools/render_text_split.py", "mode": S["mode"], "commit": head, "tool_uncommitted": dirty,
            "sha256": sha256(out),
            "data_sources": [{"path": p, "sha256": sha256(ROOT / p)} for p in S["sources"]],
            "fields": "steps[k].top[0].text (most probable token), steps[k].top[:2] (probabilities)"
                      + (", steps[k].twin, steps[k].log10_distance, lyapunov[0]" if S["mode"] == "twin" else ""),
            "numbers_shown": nums,
            "article": "Sec. V and Table I (27 equal steps with six and four threads; hostility split at step 45, "
                       "lambda 0.35); the probabilities are in the full report and the receipts",
            "rendered_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames-dir", type=Path, default=ROOT.parent / "neural-shrimp-data/render-tmp/explainer-data")
    ap.add_argument("--manifest", type=Path, default=None, help="default: MANIFEST-data-clips.json beside --out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--still", type=float, default=None, help="write one PNG at this time (s) and stop")
    a = ap.parse_args()
    S = prepare(a.mode)
    seconds = MODES[a.mode]["seconds"]
    print(json.dumps({"mode": a.mode, "first_different_step": S["split"], "steps": S["n"], "seconds": seconds,
                      "step_times": [round(x, 2) for x in S["appear"]][S["split"] - 2:S["split"] + 1]}))
    if a.still is not None:
        out = a.out.with_name(a.out.stem + f"-still-{a.still:05.2f}.png")
        fig = render(S, a.still)
        fig.savefig(out, dpi=DPI, facecolor=BG)
        print(f"wrote {out}")
        return
    frames = a.frames_dir / a.out.stem
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    n = int(round(seconds * 30))
    chunks = [list(range(i, n, a.workers)) for i in range(a.workers)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        done = sum(pool.map(_worker, [(S, c, frames) for c in chunks]))
    if done != n:
        fail(f"rendered {done} of {n} frames")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", str(frames / "%05d.png"),
                    "-vf", "setsar=1,format=yuv420p", "-c:v", "libx264", "-profile:v", "high", "-preset", "slow",
                    "-crf", "16", "-pix_fmt", "yuv420p", "-r", "30", "-g", "60", "-movflags", "+faststart", "-an",
                    str(a.out)], check=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(a.out)], capture_output=True, text=True, check=True).stdout)
    upsert(a.manifest or a.out.parent / "MANIFEST-data-clips.json", manifest_entry(S, a.out, round(dur, 3)))
    shutil.rmtree(frames)
    print(f"wrote {a.out} ({dur:.2f} s, {n} frames)")


if __name__ == "__main__":
    sys.exit(main())
