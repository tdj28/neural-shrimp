"""Add a spoken narration to a video (human, 18 Oct 2026: "use my openai token to create a tts narration of these
videos").

A script (JSON) lists lines, each with the source-video time at which it should start and, optionally, a hold: seconds
for which the picture freezes at that moment while the line is spoken ("line": as long as the line itself). Each line is synthesized separately by the
OpenAI speech endpoint (key OPENAI_API_KEY from the repository's .env; never printed) and cached by a hash of its text
and voice settings. When a line would still be playing when the next one is due, the picture freezes at the next
line's moment until it ends (plus a short gap), so speech and picture never drift apart. The narrated video has the
original frames with these freezes inserted, and the lines placed on one soundtrack.

Script: {"video": SRC, "out": OUT, "model": "gpt-4o-mini-tts", "voice": "sage", "instructions": "...",
         "gap": 0.35, "tail": 1.0, "pack": false, "lines": [{"at": 0.0, "hold": 4.0, "text": "..."}, ...]}
With "pack": true the lines play back to back (gap seconds apart, from the first line's time) and the narrated video
ends tail seconds after the last line; the picture does not freeze while the narration waits.
Paths are relative to the repository root. Output: OUT and OUT with suffix .timeline.json (where each line starts in
the narrated video, every freeze, and the synthesis settings).

Usage: python tools/narrate_video.py docs/narration/cusp-sweep.json
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "neural-shrimp-data" / "videos" / "narration-cache"


def api_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY not found in .env")


def synthesize(text, model, voice, instructions, key):
    h = hashlib.sha256(json.dumps([text, model, voice, instructions]).encode()).hexdigest()[:16]
    path = CACHE / f"{h}.wav"
    if path.exists():
        return path
    body = {"model": model, "voice": voice, "input": text, "response_format": "wav"}
    if instructions:
        body["instructions"] = instructions
    req = urllib.request.Request("https://api.openai.com/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"speech request failed: HTTP {e.code}: {e.read()[:300]!r}") from None
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return path


def probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0" if "r_frame_rate" in entries else "a:0",
                          "-show_entries", entries, "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("script", type=Path)
    args = ap.parse_args()
    sc = json.loads(args.script.read_text())
    src, out = ROOT / sc["video"], ROOT / sc["out"]
    model, voice, instr = sc.get("model", "gpt-4o-mini-tts"), sc.get("voice", "sage"), sc.get("instructions", "")
    gap, tail = float(sc.get("gap", 0.35)), float(sc.get("tail", 1.0))
    st = probe(src, "stream=r_frame_rate")["streams"][0]["r_frame_rate"].split("/")
    fps = float(st[0]) / float(st[1])
    vdur = duration(src)
    key = api_key()
    lines = sorted(sc["lines"], key=lambda ln: ln["at"])
    wavs = [synthesize(ln["text"], model, voice, instr, key) for ln in lines]
    durs = [duration(w) for w in wavs]
    if sc.get("pack"):                              # back to back: each line starts gap seconds after the previous one
        t = lines[0]["at"]
        for ln, d in zip(lines, durs):
            ln["at"] = round(t, 3)
            t += d + gap
        if t - gap > vdur:
            raise SystemExit(f"packed narration ends at {t - gap:.1f} s, after the video ({vdur:.1f} s): use a longer video")
    H, prev_end, holds, starts = 0.0, 0.0, {}, []
    for ln, d in zip(lines, durs):
        if not 0.0 <= ln["at"] <= vdur:
            raise SystemExit(f"line at {ln['at']} s lies outside the video (0 to {vdur:.2f} s)")
        out_at = ln["at"] + H
        wait = max(0.0, prev_end + gap - out_at)
        start = out_at + wait
        hold = ln.get("hold", 0.0)
        freeze = wait + (d if hold == "line" else float(hold))        # "line": freeze while this line is spoken
        if freeze > 0:
            f = min(int(round(ln["at"] * fps)), int(vdur * fps) - 1)
            holds[f] = holds.get(f, 0.0) + freeze
        H += freeze
        starts.append(start)
        prev_end = start + d
    total = vdur + H
    end_hold = max(0.0, prev_end + tail - total)
    total += end_hold
    if sc.get("pack"):                              # cut the video shortly after the last line
        total = min(total, prev_end + tail)
    chain = []
    for f in sorted(holds, reverse=True):                 # later frames first, so earlier indices do not shift
        chain.append(f"loop=loop={int(round(holds[f] * fps))}:size=1:start={f}")
    vf = ",".join(chain + [f"setpts=N/({fps}*TB)"] + ([f"tpad=stop_mode=clone:stop_duration={end_hold:.3f}"]
                                                     if end_hold > 0 else []))
    af = "".join(f"[{i + 1}:a]adelay=delays={int(round(s * 1000))}:all=1[a{i}];" for i, s in enumerate(starts))
    af += "".join(f"[a{i}]" for i in range(len(starts))) + f"amix=inputs={len(starts)}:duration=longest:normalize=0,"
    af += f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={total:.3f},atrim=0:{total:.3f}[aout]"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:            # same filesystem as the output
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
        for w in wavs:
            cmd += ["-i", str(w)]
        cmd += ["-filter_complex", f"[0:v]{vf}[vout];{af}", "-map", "[vout]", "-map", "[aout]", "-c:v", "libx264",
                "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", "-r", f"{fps}", "-c:a", "aac", "-b:a", "192k",
                "-t", f"{total:.3f}", "-movflags", "+faststart", str(Path(tmp) / "out.mp4")]
        subprocess.run(cmd, check=True)
        Path(tmp, "out.mp4").replace(out)
    timeline = {"video": sc["video"], "out": sc["out"], "model": model, "voice": voice, "instructions": instr,
                "fps": fps, "source_seconds": vdur, "narrated_seconds": round(total, 3),
                "freezes": [{"source_frame": f, "source_time": round(f / fps, 3), "seconds": round(holds[f], 3)}
                            for f in sorted(holds)],
                "lines": [{"at_source": ln["at"], "starts": round(s, 3), "seconds": round(d, 3), "text": ln["text"]}
                          for ln, s, d in zip(lines, starts, durs)]}
    Path(str(out) + ".timeline.json").write_text(json.dumps(timeline, indent=1))
    print(json.dumps({k: timeline[k] for k in ("out", "source_seconds", "narrated_seconds")} |
                     {"freezes": len(holds), "lines": len(lines)}))


if __name__ == "__main__":
    main()
