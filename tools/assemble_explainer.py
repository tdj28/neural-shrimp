"""Assemble the explainer video (docs/explainer/plan.md, section 5) from its narration and edit decision list.

Steps (each a subcommand; all CPU, intermediates under ../neural-shrimp-data/render-tmp/explainer-v1):

  tts      synthesize every narration line of docs/explainer/narration-v1.json with OpenAI speech (key from .env,
           never printed; cached in ../neural-shrimp-data/videos/narration-cache by the hash that
           tools/narrate_video.py uses) and measure each clip: length, speech onset and end, level.
  plan     lay the measured lines on the shots of the edit list (docs/explainer/edit-v1.json): every shot lasts as
           long as its lines plus a short lead and tail, so the picture is fitted to the voice; write the timeline.
  shots    render every shot of the timeline at its fitted length (figure shots and cards with
           tools/render_figure_shot.py, data clips with their own tools and a time map, existing videos cut and
           re-timed with ffmpeg and dressed with render_figure_shot.py).
  assemble join the shots into chapters, the chapters with crossfades, lay the narration and the spoken trace's
           own audio (ducked under the voice) on one track, normalize loudness, write the master, the preview and
           the timeline JSON.
  qa       measure every narration line in the final mix, find black and frozen stretches, extract frames.

The tool refuses to continue when a line, a clip or a measured value is missing; it never substitutes a default.

Rebuild after a change of the article (figure numbers are resolved from its main.aux at render time):
  uv run --no-sync python ../neural-shrimp-data/videos/explainer/parts/specs/make_specs.py   # cards and shot specs
  uv run --no-sync python tools/render_cards.py ../neural-shrimp-data/videos/explainer/parts/specs/cards.json \
      --out ../neural-shrimp-data/videos/explainer/assets/cards
  uv run --no-sync python tools/assemble_explainer.py tts | plan | shots | assemble | qa   (in this order)
  uv run --no-sync python tools/check_explainer_numbers.py                                  # numbers and labels
The narration is synthesized once per text (cache); "tempo" in the edit list time-stretches every line (ffmpeg
atempo, pitch kept), because the voice reads at about 120 words per minute.
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"
WORK = DATA / "render-tmp" / "explainer-v1"
VIDEOS = DATA / "videos"
EXPL = VIDEOS / "explainer"
CACHE = VIDEOS / "narration-cache"
NARRATION = ROOT / "docs" / "explainer" / "narration-v1.json"


def fail(msg):
    raise SystemExit(f"assemble_explainer: refused: {msg}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        fail(f"command failed ({r.returncode}): {' '.join(map(str, cmd))[:400]}\n{r.stderr[-1500:]}")
    return r


def ffprobe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    return float(r.stdout.strip())


# ----------------------------------------------------------------------------------------------- speech


def cache_path(text, model, voice, instructions):
    """The cache file tools/narrate_video.py uses for this line and these settings."""
    h = hashlib.sha256(json.dumps([text, model, voice, instructions]).encode()).hexdigest()[:16]
    return CACHE / f"{h}.wav"


def read_wav(path):
    """Samples (float, -1..1) and rate of a 16-bit PCM WAV, tolerant of streamed headers with unknown sizes."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        fail(f"{path} is not a WAV file")
    pos, rate, bits, chans = 12, None, None, None
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            _, chans, rate, _, _, bits = struct.unpack("<HHIIHH", raw[pos + 8:pos + 24])
        if cid == b"data":
            data = raw[pos + 8:] if size in (0, 0xFFFFFFFF) or pos + 8 + size > len(raw) else raw[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)
    else:
        fail(f"{path}: no data chunk")
    if bits != 16:
        fail(f"{path}: {bits}-bit samples, expected 16")
    import numpy as np
    x = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float64) / 32768.0
    if chans and chans > 1:
        x = x[: len(x) // chans * chans].reshape(-1, chans).mean(axis=1)
    return x, rate


def speech_bounds(x, rate, floor_db=-45.0, win=0.02):
    """Onset and end of speech: first and last 20 ms window whose RMS is within 45 dB of the clip's loudest window."""
    import numpy as np
    n = max(1, int(win * rate))
    m = len(x) // n
    if m == 0:
        fail("empty speech clip")
    rms = np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-20)
    db = 20 * np.log10(rms)
    loud = np.nonzero(db > db.max() + floor_db)[0]
    return loud[0] * n / rate, (loud[-1] + 1) * n / rate, float(db.max())


def cmd_tts(args):
    sc = json.loads(NARRATION.read_text())
    model, voice, instr = sc["model"], sc["voice"], sc["instructions"]
    ids = [ln["id"] for ln in sc["lines"]]
    if len(set(ids)) != len(ids):
        fail("duplicate line ids in the narration")
    lines = [ln for ln in sc["lines"] if not args.ids or ln["id"] in args.ids]
    sys.path.insert(0, str(ROOT / "tools"))
    import narrate_video as nv                              # same endpoint, cache and key handling
    key = None
    todo = [ln for ln in lines if not cache_path(ln["text"], model, voice, instr).exists()]
    if todo:
        key = nv.api_key()

    def synth(ln):
        p = nv.synthesize(ln["text"], model, voice, instr, key)
        if p != cache_path(ln["text"], model, voice, instr):
            fail(f"cache path mismatch for {ln['id']}")
        return ln["id"], p

    paths = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for lid, p in ex.map(synth, lines):
            paths[lid] = p
    out = WORK / "tts.json"
    old = json.loads(out.read_text()) if out.exists() else {"lines": {}}
    for ln in lines:
        p = paths[ln["id"]]
        x, rate = read_wav(p)
        on, end, peak_win = speech_bounds(x, rate)
        old["lines"][ln["id"]] = {"wav": str(p), "sha256": sha256(p), "rate": rate, "seconds": round(len(x) / rate, 3),
                                  "speech_start": round(on, 3), "speech_end": round(end, 3),
                                  "loudest_window_dbfs": round(peak_win, 1), "text": ln["text"],
                                  "chapter": ln["chapter"], "optional": bool(ln.get("optional"))}
    old.update(model=model, voice=voice, instructions=instr, narration=str(NARRATION),
               narration_sha256=sha256(NARRATION))
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    tot = {"core": 0.0, "optional": 0.0}
    for lid, m in old["lines"].items():
        tot["optional" if m["optional"] else "core"] += m["speech_end"] - m["speech_start"]
    print(json.dumps({"lines": len(old["lines"]), "synthesized_now": len(todo),
                      "speech_seconds": {k: round(v, 1) for k, v in tot.items()}}))


FPS = 30
EDIT = ROOT / "docs" / "explainer" / "edit-v1.json"
SHOTS_SPEC = EXPL / "parts" / "specs" / "shots.json"
OUT = EXPL / "explainer-v1.mp4"


def rel(path):
    """Paths in the edit list are relative to the repository root unless absolute."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def frames(sec):
    return int(round(sec * FPS))


AUX = ROOT.parent / "chaos-journal-neural-shrimp" / "main.aux"


def article_labels():
    import re
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", AUX.read_text())}


def resolve_refs(obj, labels):
    """Replace {fig:x} by 'Fig. N', {tab:x} by 'Table N' and {sec:x} by 'Sec. N' from the article's main.aux, so that
    on-screen references follow a renumbered article; an unknown label is refused."""
    import re
    if isinstance(obj, str):
        def sub(m):
            lab = m.group(1)
            if not labels.get(lab):
                fail(f"label {lab} not in {AUX}")
            kind = {"fig": "Fig.", "tab": "Table", "sec": "Sec."}[lab.split(":")[0]]
            return f"{kind} {labels[lab]}"
        return re.sub(r"\{((?:fig|tab|sec):[^}]+)\}", sub, obj)
    if isinstance(obj, list):
        return [resolve_refs(v, labels) for v in obj]
    if isinstance(obj, dict):
        return {k: resolve_refs(v, labels) for k, v in obj.items()}
    return obj


# ----------------------------------------------------------------------------------------------- plan


def paragraph(lid):
    """Lines P2.1a, P2.1b, ... belong to paragraph P2.1; P2.3opt to P2.3."""
    base = lid.replace("opt", "")
    return base[:-1] if base[-1].isalpha() else base


def line_lengths(edit, tts):
    g = edit["gaps"]
    out = {}
    for lid, m in tts["lines"].items():
        a = max(0.0, m["speech_start"] - g["trim_before"])
        b = min(m["seconds"], m["speech_end"] + g["trim_after"])
        out[lid] = {"src_in": round(a, 4), "src_out": round(b, 4), "seconds": (b - a) / edit["tempo"]}
    return out


def evaluate(expr, shot, lines_at):
    """A cue: number, 'END', or 'LINE@F[+S|-S]' relative to the shot's start."""
    if isinstance(expr, (int, float)):
        return float(expr)
    e = expr.strip()
    if e == "END":
        return shot["frames"] / FPS
    if e.startswith("END-"):
        return shot["frames"] / FPS - float(e[4:])
    if e.startswith("END*"):
        return round(shot["frames"] / FPS * float(e[4:]), 3)
    lid, _, rest = e.partition("@")
    if lid not in lines_at or not rest:
        fail(f"cue {expr!r}: line {lid!r} is not spoken in this chapter")
    off = 0.0
    for sign in ("+", "-"):
        if sign in rest[1:]:
            i = rest.index(sign, 1)
            rest, off = rest[:i], float(rest[i:])
            break
    frac = float(rest)
    ln = lines_at[lid]
    t = ln["start"] - shot["start"] + frac * ln["seconds"] + off
    if not -1e-6 <= t <= shot["frames"] / FPS + 1e-6:
        fail(f"cue {expr!r} of shot {shot['id']} falls at {t:.2f} s, outside the shot (0 to {shot['seconds']:.2f} s)")
    return round(t, 3)


def cmd_plan(args):
    edit = json.loads(EDIT.read_text())
    tts = json.loads((WORK / "tts.json").read_text())
    narr = {ln["id"]: ln for ln in json.loads(NARRATION.read_text())["lines"]}
    if tts.get("narration_sha256") != sha256(NARRATION):
        fail("tts.json was measured from another narration file; run the tts step again")
    L = line_lengths(edit, tts)
    g = edit["gaps"]
    chapters, t_abs, used = [], 0.0, set()
    for ci, ch in enumerate(edit["chapters"]):
        shots, lines_at, t, prev = [], {}, None, None
        cut = 0.0
        for si, sh in enumerate(ch["shots"]):
            if not sh.get("lines"):
                fail(f"shot {sh['id']} has no lines")
            pre = sh.get("pre", g["chapter_lead"] if si == 0 else g["cut_pre"])
            if si == 0:
                t = pre
            else:
                last = shots[-1]
                if last.get("post") is not None:
                    cut = last["_end"] + last["post"]
                    t = cut + pre
                else:
                    gap = g["sentence"] if paragraph(prev) == paragraph(sh["lines"][0]) else g["paragraph"]
                    t = max(last["_end"] + gap, last["_end"] + pre + 0.05)
                    cut = t - pre
                cut = round(cut * FPS) / FPS
                shots[-1]["seconds"] = cut - shots[-1]["start"]
            rec = {"id": sh["id"], "kind": sh["kind"], "start": cut if si else 0.0, "lines": [],
                   "post": sh.get("post")}
            for j, lid in enumerate(sh["lines"]):
                if lid not in L or lid not in narr:
                    fail(f"line {lid} of shot {sh['id']} is not synthesized or not in the narration")
                if lid in used:
                    fail(f"line {lid} used twice")
                used.add(lid)
                if j:
                    t += g["sentence"] if paragraph(prev) == paragraph(lid) else g["paragraph"]
                ln = {"id": lid, "start": round(t, 4), "seconds": round(L[lid]["seconds"], 4)}
                rec["lines"].append(ln)
                lines_at[lid] = ln
                t += L[lid]["seconds"]
                prev = lid
            rec["_end"] = t
            shots.append(rec)
        last = shots[-1]
        end = last["_end"] + (last["post"] if last.get("post") is not None else g["chapter_tail"])
        end = round(end * FPS) / FPS
        last["seconds"] = end - last["start"]
        for sh in shots:
            sh["seconds"] = round(sh["seconds"], 4)
            sh["frames"] = frames(sh["seconds"])
            sh.pop("_end")
        if sum(s["frames"] for s in shots) != frames(end):
            fail(f"{ch['id']}: shot frames {sum(s['frames'] for s in shots)} != chapter frames {frames(end)}")
        chapters.append({"id": ch["id"], "title": ch["title"], "plan": ch["plan"], "start": round(t_abs, 4),
                         "seconds": round(end, 4), "frames": frames(end), "shots": shots})
        t_abs += end - g["xfade"]
    total = chapters[-1]["start"] + chapters[-1]["seconds"]
    missing = [lid for lid, ln in narr.items() if not ln.get("optional") and lid not in used]
    if missing:
        fail(f"core lines not placed in any shot: {missing}")
    tl = {"edit": str(EDIT), "edit_sha256": sha256(EDIT), "narration": str(NARRATION),
          "narration_sha256": sha256(NARRATION), "tempo": edit["tempo"], "gaps": g, "seconds": round(total, 3),
          "line_audio": L, "chapters": chapters}
    (WORK / "timeline-plan.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    for ch in chapters:
        print(f"{ch['id']} {ch['start']:7.2f} +{ch['seconds']:6.2f}  plan {ch['plan'][1] - ch['plan'][0]:4d} s  "
              + " ".join(f"{s['id']}:{s['seconds']:.1f}" for s in ch["shots"]))
    print(f"total {total:.1f} s = {int(total // 60)}:{total % 60:04.1f}  (tempo {edit['tempo']})")


# ----------------------------------------------------------------------------------------------- shots


def set_path(obj, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]
    last = keys[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        if last not in obj and not isinstance(obj, dict):
            fail(f"cannot set {path}")
        obj[last] = value


def eval_events(value, shot, lines_at):
    """Copy of a list of event dicts with cue expressions in 't' and 'until' evaluated."""
    out = json.loads(json.dumps(value))
    for e in out if isinstance(out, list) else []:
        if isinstance(e, dict):
            for k in ("t", "until"):
                if k in e and isinstance(e[k], str):
                    e[k] = evaluate(e[k], shot, lines_at)
    return out


def scale_events(spec, factor):
    for key in ("keys", "reveal", "stages", "overlays", "texts", "lower_third", "dim"):
        for e in spec.get(key, []):
            for tk in ("t", "until"):
                if isinstance(e.get(tk), (int, float)) and e[tk] > 0:
                    e[tk] = round(e[tk] * factor, 3)


def figshot_spec(sh, shot, lines_at, base_specs):
    if "spec_inline" in sh:
        spec = json.loads(json.dumps(sh["spec_inline"]))
    else:
        if sh["spec"] not in base_specs:
            fail(f"no shot spec named {sh['spec']} in {SHOTS_SPEC}")
        spec = json.loads(json.dumps(base_specs[sh["spec"]]))
        if sh.get("retime"):
            scale_events(spec, shot["seconds"] / float(spec["seconds"]))
    for k, v in sh.get("replace", {}).items():
        spec[k] = eval_events(v, shot, lines_at)
    for path, expr in sh.get("set", {}).items():
        set_path(spec, path, evaluate(expr, shot, lines_at))
    for key in ("stages", "overlays", "texts", "lower_third", "keys"):
        spec[key] = eval_events(spec.get(key, []), shot, lines_at)
    spec.update(name=f"v1-{shot['id']}", seconds=shot["frames"] / FPS, out=str(WORK / "shots" / f"{shot['id']}.mp4"),
                work=str(WORK / "fs-work"), still=[])
    spec.pop("variant", None)
    return spec


def precut(sh, shot, lines_at):
    """Cut and re-time an existing video into WORK/precut/<id>.mp4: exact frames at 30 fps, segment by segment."""
    src = rel(sh["src"])
    if not src.exists():
        fail(f"source video {src} not found")
    sdur = ffprobe_duration(src)
    D = shot["frames"] / FPS
    segs, c0, parts, chain = sh["segments"], 0.0, [], []
    for i, sg in enumerate(segs):
        c1 = D if i == len(segs) - 1 else evaluate(sg["until"], shot, lines_at)
        if c1 <= c0:
            fail(f"shot {shot['id']}: segment {i} has no length")
        a = float(sg["in"])
        if "out" in sg:
            b = float(sg["out"])
        else:
            b = a + (c1 - c0) * float(sg.get("speed", 1.0))
            if b > sdur:
                b = sdur
        if b > sdur + 1e-6 or b <= a:
            fail(f"shot {shot['id']}: source range {a}-{b} s outside {src.name} ({sdur:.2f} s)")
        speed = (b - a) / (c1 - c0)
        parts.append({"src_in": round(a, 3), "src_out": round(b, 3), "clip_from": round(c0, 3), "clip_to": round(c1, 3),
                      "speed": round(speed, 4)})
        chain.append(f"[0:v]trim=start={a:.4f}:end={b:.4f},setpts=(PTS-STARTPTS)/{speed:.6f},fps={FPS}[s{i}]")
        c0 = c1
    n = shot["frames"]
    fc = ";".join(chain) + ";" + "".join(f"[s{i}]" for i in range(len(segs))) + \
        f"concat=n={len(segs)}:v=1:a=0,tpad=stop_mode=clone:stop_duration=2,trim=end_frame={n},setsar=1,format=yuv420p[v]"
    out = WORK / "precut" / f"{shot['id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-filter_complex", fc, "-map", "[v]", "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "12", "-r", FPS, out])
    return out, parts


def count_frames(path):
    r = run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate", "-of", "json", path])
    return json.loads(r.stdout)["streams"][0]


def cmd_shots(args):
    edit = json.loads(EDIT.read_text())
    tl = json.loads((WORK / "timeline-plan.json").read_text())
    if tl["edit_sha256"] != sha256(EDIT):
        fail("the edit list changed after planning; run plan again")
    base_specs = {s["name"]: s for s in json.loads(SHOTS_SPEC.read_text())}
    labels = article_labels()
    (WORK / "shots").mkdir(parents=True, exist_ok=True)
    batch, tools_jobs, simple, record = [], [], [], {}
    only = set(args.only or [])
    for ch_edit, ch in zip(edit["chapters"], tl["chapters"], strict=True):
        lines_at = {ln["id"]: ln for s in ch["shots"] for ln in s["lines"]}
        for sh, shot in zip(ch_edit["shots"], ch["shots"], strict=True):
            if only and shot["id"] not in only:
                continue
            rec = {"id": shot["id"], "kind": sh["kind"], "frames": shot["frames"], "out": str(WORK / "shots" / f"{shot['id']}.mp4")}
            if sh["kind"] == "figshot":
                spec = resolve_refs(figshot_spec(sh, shot, lines_at, base_specs), labels)
                batch.append(spec)
                rec["spec"] = spec
                rec["source_spec"] = sh.get("spec", "inline")
            elif sh["kind"] == "video":
                pc, parts = precut(sh, shot, lines_at)
                spec = {"name": f"v1-{shot['id']}", "video": str(pc), "video_ss": 0.0, "seconds": shot["frames"] / FPS,
                        "out": rec["out"], "work": str(WORK / "fs-work"), "still": []}
                for k, v in sh.get("dress", {}).items():
                    spec[k] = eval_events(v, shot, lines_at) if isinstance(v, list) else v
                spec = resolve_refs(spec, labels)
                batch.append(spec)
                rec.update(src=str(rel(sh["src"])), segments=parts, spec=spec)
            elif sh["kind"] in ("textsplit", "batchdiv", "answers"):
                knots = [[evaluate(a, shot, lines_at), float(b)] for a, b in sh["knots"]]
                tool = {"textsplit": "render_text_split.py", "batchdiv": "render_batch_divergence.py",
                        "answers": "plot_explainer_answers.py"}[sh["kind"]]
                cmd = ["uv", "run", "--no-sync", "python", f"tools/{tool}"]
                if sh["kind"] == "textsplit":
                    cmd += ["--mode", sh["mode"]]
                cmd += ["--out", rec["out"], "--seconds", f"{shot['frames'] / FPS:.6f}", "--time-map", json.dumps(knots),
                        "--push", str(sh.get("push", 1.0)), "--workers", "6", "--frames-dir", str(WORK / "frames"),
                        "--manifest", str(WORK / "MANIFEST-v1-data.json")]
                tools_jobs.append((shot["id"], cmd))
                rec.update(tool=f"tools/{tool}", knots=knots, push=sh.get("push", 1.0))
            elif sh["kind"] == "trim":
                simple.append((shot, sh))
                rec["src"] = str(rel(sh["src"]))
            elif sh["kind"] == "trace":
                simple.append((shot, sh))
                rec["src"] = str(rel(sh["src"]))
            else:
                fail(f"unknown shot kind {sh['kind']}")
            record[shot["id"]] = rec
    # simple cuts: trim and trace (the trace keeps its audio, written separately for the mix)
    for shot, sh in simple:
        src = rel(sh["src"])
        n = shot["frames"]
        ss = 0.0
        if sh["kind"] == "trace":
            side = json.loads(Path(str(src) + ".json").read_text())
            ss = round(side["steps_detail"][0]["start"] - sh.get("pre", 0.3), 3)
            wav = WORK / "audio" / f"trace-{shot['id']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            run(["ffmpeg", "-y", "-v", "error", "-ss", ss, "-i", src, "-t", f"{n / FPS:.4f}", "-vn", "-ac", "1",
                 "-ar", "48000", "-c:a", "pcm_f32le", wav])
            record[shot["id"]].update(src_in=ss, audio=str(wav))
        if ffprobe_duration(src) < ss + n / FPS - 1e-3:
            fail(f"{src.name} is shorter than shot {shot['id']} needs")
        run(["ffmpeg", "-y", "-v", "error", "-ss", ss, "-i", src, "-frames:v", n, "-vf", f"fps={FPS},setsar=1,format=yuv420p",
             "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-r", FPS, "-g", 2 * FPS, record[shot["id"]]["out"]])
        record[shot["id"]]["src_in"] = ss
    procs = []
    for sid, cmd in tools_jobs:
        log = open(WORK / f"tool-{sid}.log", "w")
        procs.append((sid, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
    if batch:
        bpath = WORK / "v1-shots.json"
        bpath.write_text(json.dumps(batch, indent=1, ensure_ascii=False))
        r = subprocess.run(["uv", "run", "--no-sync", "python", "tools/render_figure_shot.py", "--batch", str(bpath),
                            "--jobs", str(args.jobs)], cwd=ROOT, capture_output=True, text=True)
        (WORK / "figure-shots.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            fail(f"render_figure_shot.py failed: {r.stderr[-1500:]}")
    for sid, pr in procs:
        if pr.wait() != 0:
            fail(f"data clip {sid} failed; see {WORK / f'tool-{sid}.log'}")
    for sid, rec in record.items():
        st = count_frames(rec["out"])
        if (int(st["nb_read_frames"]), st["width"], st["height"], st["pix_fmt"], st["r_frame_rate"]) != \
                (rec["frames"], 1920, 1080, "yuv420p", f"{FPS}/1"):
            fail(f"shot {sid}: {rec['out']} has {st}, expected {rec['frames']} frames 1920x1080 yuv420p {FPS} fps")
        rec["sha256"] = sha256(rec["out"])
    path = WORK / "shots.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(record)
    path.write_text(json.dumps(old, indent=1, ensure_ascii=False))
    print(f"{len(record)} shots rendered: {', '.join(sorted(record))}")


# ----------------------------------------------------------------------------------------------- audio


def load_mono(path):
    """Float samples of a WAV written by ffmpeg as pcm_f32le, 48 kHz mono."""
    import numpy as np
    raw = Path(path).read_bytes()
    pos = 12
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt, chans, rate = struct.unpack("<HHI", raw[pos + 8:pos + 16])
            bits = struct.unpack("<H", raw[pos + 22:pos + 24])[0]
        if cid == b"data":
            data = raw[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)
    if not (fmt == 3 and bits == 32 and chans == 1 and rate == 48000):
        fail(f"{path}: expected 48 kHz mono float32, got format {fmt}, {bits} bits, {chans} ch, {rate} Hz")
    return np.frombuffer(data, dtype="<f4").astype(np.float64)


def write_wav(path, x, rate=48000, chans=1):
    import numpy as np
    data = np.asarray(x, dtype="<f4").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack(
        "<IHHIIHH", 16, 3, chans, rate, rate * 4 * chans, 4 * chans, 32) + b"data" + struct.pack("<I", len(data))
    Path(path).write_bytes(hdr + data)


def gated_rms_db(x, rate=48000, win=0.02, rel=-25.0):
    """RMS (dBFS) over the 20 ms windows within 25 dB of the loudest one: the level of the speech itself."""
    import numpy as np
    n = int(win * rate)
    m = len(x) // n
    w = x[: m * n].reshape(m, n)
    rms = np.sqrt((w ** 2).mean(axis=1) + 1e-20)
    db = 20 * np.log10(rms)
    keep = db > db.max() + rel
    return float(10 * np.log10((w[keep] ** 2).mean() + 1e-20))


def ebur128(path):
    """Integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) measured by ffmpeg."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr[r.stderr.rfind("Summary:"):]
    import re

    def get(pat):
        m = re.search(pat, txt)
        if not m:
            fail(f"ebur128 summary of {path} lacks {pat}")
        return float(m.group(1))
    return {"I": get(r"I:\s+(-?[\d.]+) LUFS"), "LRA": get(r"LRA:\s+(-?[\d.]+) LU"), "TP": get(r"Peak:\s+(-?[\d.]+) dBFS")}


def build_audio(tl, edit, tts, shots_rec):
    """One narration track: every line trimmed, re-timed and leveled at its place; the trace's own token audio
    under it, ducked while the narrator speaks."""
    import numpy as np
    rate, total = 48000, tl["seconds"]
    mix = np.zeros(int(math.ceil(total * rate)) + rate, dtype=np.float64)
    lines_dir = WORK / "audio" / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    placed, spans = [], []
    target = -22.0
    for ch in tl["chapters"]:
        for sh in ch["shots"]:
            for ln in sh["lines"]:
                la = tl["line_audio"][ln["id"]]
                src = tts["lines"][ln["id"]]["wav"]
                out = lines_dir / f"{ln['id']}.wav"
                run(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
                     f"atrim=start={la['src_in']}:end={la['src_out']},asetpts=N/SR/TB,atempo={tl['tempo']},aresample=48000",
                     "-ac", "1", "-c:a", "pcm_f32le", out])
                x = load_mono(out)
                lvl = gated_rms_db(x)
                x = x * 10 ** ((target - lvl) / 20)
                n = 240                                            # 5 ms fades against clicks at the trim points
                ramp = np.linspace(0, 1, n)
                x[:n] *= ramp
                x[-n:] *= ramp[::-1]
                t0 = ch["start"] + ln["start"]
                i0 = int(round(t0 * rate))
                mix[i0:i0 + len(x)] += x
                placed.append({"id": ln["id"], "start": round(t0, 3), "end": round(t0 + len(x) / rate, 3),
                               "level_in_dbfs": round(lvl, 1), "gain_db": round(target - lvl, 1)})
                spans.append((t0, t0 + len(x) / rate))
    trace = []
    for ch in tl["chapters"]:
        for sh in ch["shots"]:
            if sh["kind"] != "trace":
                continue
            rec = shots_rec[sh["id"]]
            x = load_mono(rec["audio"])
            x = x[: sh["frames"] * rate // FPS]
            lvl = gated_rms_db(x)
            x = x * 10 ** ((target - 1.0 - lvl) / 20)              # tokens 1 dB under the narration
            t0 = ch["start"] + sh["start"]
            tt = t0 + np.arange(len(x)) / rate
            duck = np.ones(len(x))
            g = 10 ** (edit["duck_db"] / 20)
            for a, b in spans:
                if b < t0 or a > t0 + len(x) / rate:
                    continue
                lo, hi = a - 0.15, b + 0.35
                env = np.clip(np.minimum((tt - (lo - 0.15)) / 0.15, ((hi + 0.35) - tt) / 0.35), 0, 1)
                duck = np.minimum(duck, 1 - (1 - g) * env)
            x = x * duck
            n = 480
            x[:n] *= np.linspace(0, 1, n)
            x[-n:] *= np.linspace(1, 0, n)
            i0 = int(round(t0 * rate))
            mix[i0:i0 + len(x)] += x
            trace.append({"shot": sh["id"], "start": round(t0, 3), "end": round(t0 + len(x) / rate, 3),
                          "level_in_dbfs": round(lvl, 1), "duck_db": edit["duck_db"]})
    mix = mix[: int(round(total * rate))]
    raw = WORK / "audio" / "mix-raw.wav"
    write_wav(raw, mix)
    m = ebur128(raw)
    gain = -16.0 - m["I"]
    final = WORK / "audio" / "mix-final.wav"
    lim = 10 ** (-1.8 / 20)                    # speech peaks limited (4x oversampled) under -1.5 dBTP, then the level
    passes = []                                # is re-measured and the gain corrected until it sits at -16 LUFS
    for _ in range(4):
        run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af",
             f"volume={gain:.3f}dB,aresample=192000,alimiter=limit={lim:.4f}:attack=0.5:release=40:level=0:latency=1,"
             f"aresample=48000,aformat=channel_layouts=stereo", "-c:a", "pcm_f32le", final])
        m2 = ebur128(final)
        passes.append({"gain_db": round(gain, 3), **m2})
        if abs(m2["I"] + 16) <= 0.1 and m2["TP"] <= -1.5:
            break
        gain += -16.0 - m2["I"]
    else:
        fail(f"loudness did not settle at -16 LUFS / -1.5 dBTP: {passes}")
    return final, placed, trace, {"raw": m, "final": m2, "gain_db": round(gain, 2), "passes": passes}


# ----------------------------------------------------------------------------------------------- assemble


def cmd_assemble(args):
    edit = json.loads(EDIT.read_text())
    tl = json.loads((WORK / "timeline-plan.json").read_text())
    tts = json.loads((WORK / "tts.json").read_text())
    shots_rec = json.loads((WORK / "shots.json").read_text())
    narr = {ln["id"]: ln for ln in json.loads(NARRATION.read_text())["lines"]}
    if tl["edit_sha256"] != sha256(EDIT):
        fail("the edit list changed after planning")
    chap_dir = WORK / "chapters"
    chap_dir.mkdir(parents=True, exist_ok=True)
    chap_files = []
    for ch in tl["chapters"]:
        files = []
        for sh in ch["shots"]:
            rec = shots_rec.get(sh["id"])
            if not rec or rec["frames"] != sh["frames"] or not Path(rec["out"]).exists() or sha256(rec["out"]) != rec["sha256"]:
                fail(f"shot {sh['id']} is missing or does not match the plan; run the shots step")
            files.append(rec["out"])
        lst = chap_dir / f"{ch['id']}.txt"
        lst.write_text("".join(f"file '{f}'\n" for f in files))
        out = chap_dir / f"{ch['id']}.mp4"
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-vf", "setsar=1,format=yuv420p",
             "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "14", "-r", FPS, "-g", 2 * FPS, out])
        n = int(count_frames(out)["nb_read_frames"])
        if n != ch["frames"]:
            fail(f"{ch['id']}: {n} frames joined, {ch['frames']} planned")
        chap_files.append(out)
    audio, placed, trace, loud = build_audio(tl, edit, tts, shots_rec)
    x = edit["gaps"]["xfade"]
    fc, v = [], "[0:v]"
    for i in range(len(chap_files)):
        fc.append(f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={FPS},format=yuv420p[c{i}]")
    v = "[c0]"
    acc = tl["chapters"][0]["frames"] / FPS
    for i in range(1, len(chap_files)):
        off = acc - x
        fc.append(f"{v}[c{i}]xfade=transition=fade:duration={x}:offset={off:.4f}[x{i}]")
        v = f"[x{i}]"
        acc = off + tl["chapters"][i]["frames"] / FPS
    if abs(acc - tl["seconds"]) > 0.02:
        fail(f"crossfaded length {acc:.3f} s differs from the plan {tl['seconds']:.3f} s")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for f in chap_files:
        cmd += ["-i", f]
    cmd += ["-i", audio, "-filter_complex", ";".join(fc), "-map", v, "-map", f"{len(chap_files)}:a",
            "-c:v", "libx264", "-profile:v", "high", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-r", FPS,
            "-g", 2 * FPS, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-t", f"{tl['seconds']:.3f}",
            "-movflags", "+faststart", OUT]
    run(cmd)
    # preview under 30 MB
    prev = OUT.with_name("explainer-v1-preview.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", OUT, "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264", "-profile:v",
         "high", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags",
         "+faststart", prev])
    if prev.stat().st_size >= 30e6:                  # two-pass to a 27 MB budget: 96 kb/s audio, the rest video
        vbr = int(27e6 * 8 / tl["seconds"] / 1000) - 96
        base = ["ffmpeg", "-y", "-v", "error", "-i", OUT, "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264",
                "-profile:v", "high", "-preset", "slow", "-b:v", f"{vbr}k", "-pix_fmt", "yuv420p",
                "-passlogfile", WORK / "preview-2pass"]
        run(base + ["-pass", "1", "-an", "-f", "mp4", "/dev/null"])
        run(base + ["-pass", "2", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", prev])
    if prev.stat().st_size >= 30e6:
        fail(f"preview is {prev.stat().st_size / 1e6:.1f} MB, not under 30 MB")
    # timeline
    chapters_out, lines_out, clips_out = [], [], []
    pl = {p["id"]: p for p in placed}
    for ch in tl["chapters"]:
        chapters_out.append({"id": ch["id"], "title": ch["title"], "start": round(ch["start"], 3),
                             "end": round(ch["start"] + ch["seconds"], 3), "seconds": round(ch["seconds"], 3),
                             "plan_start": ch["plan"][0], "plan_end": ch["plan"][1],
                             "plan_seconds": ch["plan"][1] - ch["plan"][0]})
        for sh in ch["shots"]:
            rec = shots_rec[sh["id"]]
            a0 = ch["start"] + sh["start"]
            clip = {"shot": sh["id"], "chapter": ch["id"], "start": round(a0, 3), "end": round(a0 + sh["frames"] / FPS, 3),
                    "kind": rec["kind"], "rendered": rec["out"], "rendered_sha256": rec["sha256"]}
            if rec["kind"] == "figshot":
                spec = rec["spec"]
                src = spec.get("image") or spec.get("video")
                clip["source"] = src
                clip["in"], clip["out"] = 0.0, round(sh["frames"] / FPS, 3)
                clip["source_spec"] = rec.get("source_spec")
                clip["stage_images"] = [st.get("image") for st in spec.get("stages", []) if st.get("image")]
                clip["overlays"] = [o["image"] for o in spec.get("overlays", [])]
            elif rec["kind"] == "video":
                clip["source"] = rec["src"]
                clip["segments"] = rec["segments"]
                clip["in"], clip["out"] = rec["segments"][0]["src_in"], rec["segments"][-1]["src_out"]
            elif rec["kind"] in ("trim", "trace"):
                clip["source"] = rec["src"]
                clip["in"] = rec["src_in"]
                clip["out"] = round(rec["src_in"] + sh["frames"] / FPS, 3)
                if rec["kind"] == "trace":
                    clip["audio"] = "the clip's own spoken tokens, ducked by %d dB under the narration" % edit["duck_db"]
            else:
                clip["source"] = rec["tool"]
                clip["time_map"] = rec["knots"]
                clip["push_in"] = rec["push"]
                clip["in"], clip["out"] = rec["knots"][0][1], rec["knots"][-1][1]
            ed = next(s for c in edit["chapters"] for s in c["shots"] if s["id"] == sh["id"])
            if ed.get("sources"):
                clip["data_sources"] = [str(rel(x)) for x in ed["sources"]]
            clips_out.append(clip)
            for ln in sh["lines"]:
                p = pl[ln["id"]]
                lines_out.append({"id": ln["id"], "chapter": ch["id"], "shot": sh["id"], "start": p["start"],
                                  "end": p["end"], "text": narr[ln["id"]]["display"], "tts_text": narr[ln["id"]]["text"]})
    meta = {"video": str(OUT), "sha256": sha256(OUT), "seconds": tl["seconds"], "preview": str(prev),
            "preview_sha256": sha256(prev), "preview_mb": round(prev.stat().st_size / 1e6, 2),
            "format": "1920x1080, 30 fps, H.264 High CRF 18 yuv420p, AAC 192 kb/s 48 kHz stereo, faststart",
            "tts": {"model": tts["model"], "voice": tts["voice"], "instructions": tts["instructions"],
                    "tempo": tl["tempo"], "tempo_note": "each line time-stretched by this factor (ffmpeg atempo, pitch kept)"},
            "loudness": loud, "trace_audio": trace,
            "edit": str(EDIT), "edit_sha256": sha256(EDIT), "narration": str(NARRATION), "narration_sha256": sha256(NARRATION),
            "article": str(ROOT.parent / "chaos-journal-neural-shrimp" / "main.tex"),
            "article_commit": run(["git", "-C", ROOT.parent / "chaos-journal-neural-shrimp", "rev-parse", "HEAD"]).stdout.strip(),
            "tool_commit": run(["git", "-C", ROOT, "rev-parse", "HEAD"]).stdout.strip(),
            "article_labels": article_labels(),
            "chapters": chapters_out, "lines": lines_out, "clips": clips_out}
    OUT.with_suffix(".timeline.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({"video": str(OUT), "seconds": tl["seconds"], "loudness": loud["final"],
                      "preview_mb": meta["preview_mb"]}))


# ----------------------------------------------------------------------------------------------- qa


def cmd_qa(args):
    import numpy as np
    meta = json.loads(OUT.with_suffix(".timeline.json").read_text())
    qa = WORK / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    wav = qa / "final-mono.wav"
    run(["ffmpeg", "-y", "-v", "error", "-i", OUT, "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_f32le", wav])
    x = load_mono(wav)
    rate = 48000
    levels, low = [], []
    for ln in meta["lines"]:
        seg = x[int(ln["start"] * rate):int(ln["end"] * rate)]
        rms = float(20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-20))
        gated = gated_rms_db(seg)
        levels.append({"id": ln["id"], "start": ln["start"], "rms_dbfs": round(rms, 1), "speech_dbfs": round(gated, 1)})
        if rms < -40:
            low.append(ln["id"])
    loud = ebur128(OUT)
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(OUT), "-vf", "blackdetect=d=0.1:pic_th=0.98", "-an", "-f",
                        "null", "-"], capture_output=True, text=True)
    import re
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", r.stderr)]
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(OUT), "-vf", "freezedetect=n=-60dB:d=0.5", "-an", "-f",
                        "null", "-"], capture_output=True, text=True)
    fs = [float(v) for v in re.findall(r"freeze_start: ([\d.]+)", r.stderr)]
    fd = [float(v) for v in re.findall(r"freeze_duration: ([\d.]+)", r.stderr)]
    freezes = [(round(a, 2), round(d, 2)) for a, d in zip(fs, fd)]
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(OUT), "-af", "silencedetect=n=-45dB:d=1.2", "-vn", "-f",
                        "null", "-"], capture_output=True, text=True)
    s0 = [float(v) for v in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    s1 = [float(v) for v in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    silences = [(round(a, 2), round(b, 2), round(b - a, 2)) for a, b in zip(s0, s1)]
    if len(s0) > len(s1):
        silences.append((round(s0[-1], 2), None, None))
    # frames at chapter boundaries and inside each chapter
    times = []
    for i, ch in enumerate(meta["chapters"]):
        if i:
            times.append((f"{ch['id']}-boundary", ch["start"] + 0.25))
        for f in (0.12, 0.4, 0.65, 0.9):
            times.append((f"{ch['id']}-{int(f * 100)}", ch["start"] + f * ch["seconds"]))
    fdir = qa / "frames"
    fdir.mkdir(exist_ok=True)
    for name, t in times:
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", OUT, "-frames:v", "1", fdir / f"{name}.png"])
    probe = json.loads(run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration,size:stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,sample_rate,channels,bit_rate",
                            "-of", "json", OUT]).stdout)
    rep = {"loudness": loud, "lines_below_-40_dbfs": low, "line_levels": levels, "black": black,
           "freezes_over_0.5s": freezes, "silences_over_1.2s": silences, "frames": [n for n, _ in times], "probe": probe}
    (qa / "qa.json").write_text(json.dumps(rep, indent=1))
    worst = sorted(levels, key=lambda d: d["rms_dbfs"])[:3]
    print(json.dumps({"loudness": loud, "low_lines": low, "quietest": worst, "black": black,
                      "freezes": [f for f in freezes if f[1] >= 1.0], "silences": silences,
                      "duration": probe["format"]["duration"]}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tts")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_tts)
    p = sub.add_parser("plan")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("shots")
    p.add_argument("--only", nargs="*")
    p.add_argument("--jobs", type=int, default=10)
    p.set_defaults(func=cmd_shots)
    p = sub.add_parser("assemble")
    p.set_defaults(func=cmd_assemble)
    p = sub.add_parser("qa")
    p.set_defaults(func=cmd_qa)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
