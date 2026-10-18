"""Movie: one chaotic trajectory traced through the strange attractor, with every word it writes shown and spoken
(human, 18 Oct 2026: "a shiny dot very clear to see that traces a chaotic trajectory, and the tts reads out the words
from the output, also shown below").

The attractor of a portrait run (tools/attractor_portrait.py; aperiodic trajectories only) is drawn dimmed, as in
tools/render_attractor_tour.py, on its first three principal coordinates, while the camera turns slowly. A bright dot
follows one trajectory: at each step of the loop it hops to the next state (a short glide, so the eye can follow the
jump; the map itself is discrete) and leaves a fading trail. Below, the text written so far and, large, the token
written at this step. Each token is spoken when the dot lands (OpenAI speech, one clip per distinct token, cached;
key from .env, never printed); newlines and punctuation are shown, not spoken. A step lasts as long as its word.

Usage: python tools/render_attractor_trace.py --run llm-soft-loop/pod64/portraits/f64-strong --steps 80 --out trace.mp4
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import narrate_video as nv  # noqa: E402
import render_attractor_tour as rt  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
SILENT = 0.30            # seconds for a token that is shown but not spoken


def project(P, az, el, W, H, scale, dist=3.0):
    Q = P @ rt.rotation(az, el).T
    f = dist / (dist + Q[:, 1])
    return Q[:, 0] * f * scale + W / 2, -Q[:, 2] * f * scale + H / 2


def trimmed(path, floor_dbfs=-50.0, lead=0.04, tail=0.08):
    """The clip without its leading and trailing background: 10 ms frames quieter than floor_dbfs are cut at both ends,
    keeping lead and tail seconds, with 10 ms fades; cached next to the clip. In these speech clips the words run from
    about -24 to -49 dBFS and the background lies at -55 dBFS and below. A cut at -40 dBFS removed soft onsets (the
    "Th" of "This" at -43 to -45 dBFS, so it sounded like "is"; human, 18 Oct: "the tts misses some words"). Every word
    is also checked in the final mix."""
    import warnings
    from scipy.io import wavfile
    out = path.with_suffix(".trim5.wav")
    if not out.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # streamed WAV: the header gives no length
            rate, x = wavfile.read(path)
        y = x.astype(np.float64)
        y = y.mean(axis=1) if y.ndim > 1 else y
        full = float(np.iinfo(x.dtype).max) if np.issubdtype(x.dtype, np.integer) else 1.0
        n = max(1, int(0.01 * rate))
        frames = len(y) // n
        db = 20 * np.log10(np.sqrt(np.mean(y[:frames * n].reshape(frames, n) ** 2, axis=1)) / full + 1e-12)
        loud = np.nonzero(db > floor_dbfs)[0]
        a = max(0, loud[0] * n - int(lead * rate))
        b = min(len(y), (loud[-1] + 1) * n + int(tail * rate))
        seg = x[a:b].astype(np.float64)
        f = max(1, int(0.01 * rate))
        ramp = np.linspace(0.0, 1.0, f)
        seg[:f] = (seg[:f].T * ramp).T
        seg[-f:] = (seg[-f:].T * ramp[::-1]).T
        wavfile.write(out, rate, seg.astype(x.dtype))
    return out


def spoken(tok_text):
    w = tok_text.strip()
    return w if any(ch.isalnum() for ch in w) else ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trajectory", type=int, default=0, help="index among the aperiodic trajectories")
    ap.add_argument("--first", type=int, default=0, help="first recorded step")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--turn", type=float, default=100.0, help="degrees the camera turns over the movie")
    ap.add_argument("--voice", default="sage")
    ap.add_argument("--model", default="gpt-4o-mini-tts")
    ap.add_argument("--intro", default="Follow one trajectory of the loop. At every step, the model writes the word "
                                      "you hear, and the dot jumps to its next state.")
    ap.add_argument("--glide", default=0.14, type=lambda v: v if v == "step" else float(v), help="seconds of each hop spent gliding to the next state "
                    "(a guide for the eye: the map visits only the states); 'step' uses 65 percent of each step")
    ap.add_argument("--all-trajectories", action="store_true", help="include trajectories that reach a periodic orbit "
                    "(for periodic cells); otherwise only the aperiodic ones")
    ap.add_argument("--title", default="One chaotic trajectory through the strange attractor")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    z = np.load(DATA / "explore" / args.run / "portrait.npz", allow_pickle=True)
    rec = json.loads((DATA / "explore" / args.run / "portrait.json").read_text())
    aper = np.arange(len(z["periods"])) if args.all_trajectories else np.nonzero(z["periods"] == 0)[0]
    C, T = z["coords"][aper][:, :, :3].astype(np.float64), z["tokens"][aper]
    P = C.reshape(-1, 3)
    med = np.median(P, axis=0)
    sc = np.percentile(np.linalg.norm(P - med, axis=1), 99.0)
    P = (P - med) / sc
    traj = (C[args.trajectory, args.first:args.first + args.steps] - med) / sc
    ttok = T[args.trajectory, args.first:args.first + args.steps]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    toks_all = T.reshape(-1)
    uniq, counts = np.unique(toks_all, return_counts=True)
    top = uniq[np.argsort(-counts)][:len(rt.PALETTE)]
    pal = {int(t): rt.PALETTE[k] for k, t in enumerate(top)}
    col = np.tile(np.array(rt.OTHER), (len(toks_all), 1))
    for k, t in enumerate(top):
        col[toks_all == t] = rt.PALETTE[k]
    texts = [tok.decode([int(t)]) for t in ttok]
    # audio: one clip per distinct spoken token, and the intro line
    key = nv.api_key()
    instr = "Read the single word clearly and evenly, as one item of a list."
    clip = {}
    for w in sorted({spoken(x) for x in texts} - {""}):
        clip[w] = trimmed(nv.synthesize(w, args.model, args.voice, instr, key))
    intro = nv.synthesize(args.intro, args.model, args.voice,
                          "Calm, warm science-documentary narrator. Measured pace.", key) if args.intro else None
    t0 = (nv.duration(intro) + 0.6) if intro else 0.5
    starts, durs = [], []
    t = t0
    for x in texts:
        w = spoken(x)
        d = max(nv.duration(clip[w]) + 0.10, 0.45) if w else SILENT
        starts.append(t)
        durs.append(d)
        t += d
    total = t + 1.5
    nF = int(np.ceil(total * args.fps))
    W, H = args.width, args.height
    fs = W / 1920
    scale = 0.62 * min(W, H)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="k")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im = ax.imshow(np.zeros((H, W, 3)), interpolation="nearest", extent=[0, W, H, 0])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    trail = [ax.plot([], [], "-", color="w", lw=2.2 * fs, alpha=a, solid_capstyle="round")[0]
             for a in np.linspace(0.55, 0.06, 8)]
    halo2 = ax.scatter([0], [0], s=420 * fs, alpha=0.20, linewidths=0)
    halo1 = ax.scatter([0], [0], s=130 * fs, alpha=0.50, linewidths=0)
    core = ax.scatter([0], [0], s=38 * fs, color="w", linewidths=0)
    a = rec["args"]
    fig.text(0.02, 0.955, args.title, color="w", fontsize=24 * fs)
    fig.text(0.02, 0.922, f"Qwen2.5-0.5B-Instruct, soft-token loop, steering $\\alpha={a['alpha']}$, "
             f"$\\log_{{10}}T={a['log_t']}$, {a['dtype']}; each jump of the dot is one step of the loop", color="0.8",
             fontsize=15 * fs)
    if args.glide == "step" or (isinstance(args.glide, float) and args.glide > 0.2):
        fig.text(0.02, 0.89, "the loop is a map: its state jumps once per step; the dot glides between states only "
                 "to guide the eye", color="0.6", fontsize=12 * fs)
    for k, t in enumerate(top):                       # the colors: the token the loop writes next
        fig.text(0.02, 0.84 - 0.034 * k, "\u25cf", color=rt.PALETTE[k], fontsize=17 * fs)
        fig.text(0.04, 0.842 - 0.034 * k, repr(tok.decode([int(t)])), color="0.9", fontsize=14 * fs, family="monospace")
    fig.text(0.02, 0.84 - 0.034 * len(top), "\u25cf", color=rt.OTHER, fontsize=17 * fs)
    fig.text(0.04, 0.842 - 0.034 * len(top), "other tokens", color="0.9", fontsize=14 * fs)
    fig.text(0.02, 0.875, "color: the token written next", color="0.75", fontsize=13 * fs)
    fig.text(0.5, 0.165, "\u23ce marks a line break that the loop writes", color="0.55", fontsize=12 * fs, ha="center")
    t_hist = fig.text(0.5, 0.115, "", color="0.75", fontsize=19 * fs, family="monospace", ha="center")
    t_word = fig.text(0.5, 0.04, "", color="w", fontsize=40 * fs, ha="center", weight="bold")
    t_step = fig.text(0.98, 0.955, "", color="0.6", fontsize=14 * fs, ha="right")
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    for i in range(nF):
        tt = i / args.fps
        u = tt / total
        az = np.radians(30.0 + args.turn * u)
        el = np.radians(22.0 + 8.0 * np.sin(2 * np.pi * u))
        im.set_data(np.clip(rt.render(P, col, az, el, W, H, scale, 3.0, 0.8), 0, 1))
        k = int(np.searchsorted(starts, tt, side="right")) - 1          # current step (-1: intro)
        X, Y = project(traj, az, el, W, H, scale)
        if k < 0:
            px, py = X[0], Y[0]
        else:
            g = 0.65 * durs[k] if args.glide == "step" else float(args.glide)
            frac = min(1.0, (tt - starts[k]) / g)
            frac = frac * frac * (3 - 2 * frac)                         # smooth glide from the previous state
            j0 = max(k - 1, 0)
            px, py = X[j0] + (X[k] - X[j0]) * frac, Y[j0] + (Y[k] - Y[j0]) * frac
        kk = max(k, 0)
        for m, ln in enumerate(trail):
            j1, j2 = kk - m, kk - m - 1
            if j2 >= 0:
                xs = [px if m == 0 else X[j1], X[j2]]
                ys = [py if m == 0 else Y[j1], Y[j2]]
                ln.set_data(xs, ys)
            else:
                ln.set_data([], [])
        c = pal.get(int(ttok[kk]), rt.OTHER)
        for art in (halo2, halo1, core):
            art.set_offsets([[px, py]])
        halo2.set_color([c])
        halo1.set_color([c])
        if k < 0:
            t_word.set_text("")
            t_hist.set_text("")
            t_step.set_text("")
        else:
            hist = "".join(texts[max(0, k - 24):k]).replace("\n", "⏎")
            t_hist.set_text("…" + hist[-70:] if hist else "")
            cur = texts[k].replace("\n", "⏎")
            t_word.set_text(f"“{cur.strip() or cur}”")
            t_word.set_color(c)
            t_step.set_text(f"step {k + 1} of {len(texts)}")
        fig.savefig(tmp / f"f{i:05d}.png", facecolor="k")
    plt.close(fig)
    clips = ([(0.3, intro)] if intro else []) + [(s, clip[spoken(x)]) for s, x in zip(starts, texts) if spoken(x)]
    af = "".join(f"[{n + 1}:a]adelay=delays={int(round(s * 1000))}:all=1[a{n}];" for n, (s, _) in enumerate(clips))
    af += "".join(f"[a{n}]" for n in range(len(clips))) + f"amix=inputs={len(clips)}:duration=longest:normalize=0,"
    af += f"loudnorm=I=-16:TP=-1.5:LRA=11,apad=whole_dur={nF / args.fps:.3f},atrim=0:{nF / args.fps:.3f}[aout]"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", str(tmp / "f%05d.png")]
    for _, w in clips:
        cmd += ["-i", str(w)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part.mp4")
    cmd += ["-filter_complex", af, "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-preset", "slow", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(part)]
    subprocess.run(cmd, check=True)
    part.replace(out)
    shutil.rmtree(tmp)
    # check: every spoken word must be audible in the final soundtrack (RMS in its window, dBFS)
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a16 = np.frombuffer(pcm, dtype=np.int16).astype(np.float64) / 32768.0
    levels = []
    for st, d, x in zip(starts, durs, texts):
        if spoken(x):
            seg = a16[int(st * 16000):int((st + d) * 16000)]
            levels.append((x.strip(), round(st, 2), round(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12), 1)))
    quiet = [lv for lv in levels if lv[2] < -40.0]
    info = {"out": str(out), "seconds": round(nF / args.fps, 2), "steps": len(texts), "trajectory": int(aper[args.trajectory]),
            "first_step": args.first, "text": "".join(texts), "distinct_spoken_tokens": sorted(clip),
            "steps_detail": [{"text": x, "start": round(st, 3), "seconds": round(d, 3)} for st, d, x in zip(starts, durs, texts)],
            "word_levels_dbfs": {"min": min(lv[2] for lv in levels), "median": float(np.median([lv[2] for lv in levels])),
                                 "below_minus_40": quiet}}
    Path(str(out) + ".json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("out", "seconds", "steps", "distinct_spoken_tokens", "word_levels_dbfs")}))


if __name__ == "__main__":
    main()
