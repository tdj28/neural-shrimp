"""Explainer figure shots (docs/explainer/plan.md, section 3.2): slow push-ins, panel reveals and card stages on the
dark 1920x1080 frame, encoded as H.264 (CRF 16, yuv420p, 30 fps) from RGB frames piped to ffmpeg.

Sources: a PNG figure, an SVG (rendered first with inkscape at --svg-width pixels), several PNGs joined side by side
at a common height (--image a.png b.png), a full-frame card PNG from render_cards.py (--plate full), or a video
(--video, decoded by ffmpeg at the output rate from --video-ss; --video-vf adds filters, e.g. eq=brightness=-0.12).

Geometry: a figure sits on a plate (default 1760x990 at 80/45 px; "WxH+X+Y" elsewhere; "full" for the whole frame)
filled with --plate-bg (white; "auto" samples the source's corner, for the beige blog SVGs). The camera follows
--keys, boxes [x0, y0, x1, y1] in fractions of the uncropped source, each expanded about its center to the plate's
aspect and padded with the plate color where it leaves the image, eased with smoothstep (center linear, width
geometric) and held after the last key; without keys, --drift D pushes in by the factor 1 + D over the shot. --crop
restricts the source first (boxes stay in uncropped fractions). --reveal covers each box with the plate color until
its time t and fades it in over "fade" seconds (0.4); --hide covers boxes for the whole shot. --stages swaps the
source at time t with a crossfade: {"t", "image"}, or for an SVG {"t", "svg_drop": [[from, to], ...]}, which removes
the SVG text from each "from" marker up to (not including) the next "to" marker; --svg-drop does the same to the
first image. --dim [{"t", "level", "dur"}] darkens the plate toward the background (level 1: full brightness).

Layers on top: --overlays (RGBA PNGs, e.g. transparent cards: {"t", "until", "image", "xy", "fade", "fade_out",
"slide_from": [dx, dy], "slide_dur"}), --texts ({"t", "until", "text", "xy", "anchor", "font", "size", "color", "w",
"align", "lh", "bg", "bg_alpha", "pad", "fade"}), --caption (26 px #9A9AA6 under the plate, or at --caption-y; the
default plate shrinks when it needs a second line), --lower-third ([{"t", "until", "text", "small", "size_as"}]:
30 px white on black at 55 % (--lower-third-alpha), 24 px padding, bottom edge 90 px above the frame's), --tag
(chapter tag, 34 px bold on a dark box, top right, first 3.5 s), --fade-in and --fade-out (from and to black).

Gates: the tool refuses to render when a source, stage or overlay file is missing, a box is malformed, an SVG marker
is not found, inkscape does not produce the image, or ffmpeg fails; it never substitutes another image or a default.

Usage (shot 2.1 of the plan):
  uv run --no-sync python tools/render_figure_shot.py \\
    --image ../chaos-journal-neural-shrimp/figures/fig-cross512-lyapunov.png --seconds 30 \\
    --keys '[{"t":0,"box":[0,0,1,1]},{"t":4,"box":[0,0,1,1]},{"t":24,"box":[0.22,0.18,0.86,0.66]}]' \\
    --caption "512 × 512 settings · ... · article Fig. 1" --tag "2  Repeating text and chaos" --out C2-01.mp4
  uv run --no-sync python tools/render_figure_shot.py --batch SHOTS.json [--only NAME ...] [--jobs 4] \\
    [--manifest MANIFEST.json] [--stills-only] [--retime NAME=SECONDS ...]
--retime renders the named shots at a new length with every event time (keys, reveals, stages, overlays, texts,
lower thirds, dims) scaled in proportion, for fitting picture to measured narration; fades keep their lengths.
A batch entry has the keys of the options (underscores or dashes) plus "name" and free fields for the manifest
("chapter", "shot", "narration", "notes", "sources_note", "variant"); an entry with "concat": [clips] joins rendered
clips of the same format into "out" after the other entries. Relative paths resolve against the repository root.
--still T writes the frame at time T as a PNG in WORK/stills (with --stills-only, no video).
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
WORK = ROOT.parent / "neural-shrimp-data" / "render-tmp" / "figure-shot"
FONT_FILES = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf",
              "monob": "DejaVuSansMono-Bold.ttf"}
DEFAULTS = {"name": None, "out": None, "image": None, "svg_drop": [], "video": None, "video_ss": 0.0,
            "video_vf": None, "svg_width": 3200, "seconds": None, "fps": 30, "crop": None, "plate": "1760x990",
            "plate_bg": "#ffffff", "bg": "#0b0b0f", "keys": None, "drift": 0.0, "reveal": [], "hide": [],
            "stages": [], "stage_fade": 0.4, "dim": [], "overlays": [], "texts": [], "caption": None,
            "caption_y": None, "lower_third": [], "lower_third_alpha": 0.55, "tag": None, "tag_alpha": 0.88,
            "fade_in": 0.0, "fade_out": 0.0, "still": [], "stills_only": False, "crf": 16, "preset": "slow",
            "work": None, "concat": None}
META = ("chapter", "shot", "narration", "notes", "sources_note", "variant")


def font(family, size):
    import matplotlib
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / FONT_FILES[family]),
                              int(round(size)))


def rgb(c):
    return ImageColor.getrgb(c if c.startswith("#") else "#" + c)[:3]


def smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def need(path, what):
    if not path.exists():
        sys.exit(f"refused: {what} {path} not found")
    return path


def check_box(box, what):
    if not (isinstance(box, (list, tuple)) and len(box) == 4 and box[2] > box[0] and box[3] > box[1]):
        sys.exit(f"refused: malformed {what} box {box!r}")
    return box


# ----------------------------------------------------------------------------------------------- sources


def render_svg(svg, width, work, text=None):
    """Render an SVG (or a modified SVG text) to PNG with inkscape; cached by content hash."""
    data = text.encode() if text is not None else svg.read_bytes()
    key = hashlib.sha256(data + str(width).encode()).hexdigest()[:12]
    png = work / f"{svg.stem}-{key}.png"
    if not png.exists():
        src = svg
        if text is not None:
            src = work / f"{svg.stem}-{key}.svg"
            src.write_text(text)
        r = subprocess.run(["inkscape", str(src), "--export-type=png", f"--export-width={width}",
                            f"--export-filename={png}"], capture_output=True, text=True)
        if r.returncode != 0 or not png.exists():
            sys.exit(f"refused: inkscape failed on {src}: {r.stderr[-400:]}")
    im = Image.open(png)
    if im.width != width:
        sys.exit(f"refused: {png} is {im.width} px wide, not {width}")
    return png


def flatten(im):
    im = im.convert("RGBA")
    base = Image.new("RGBA", im.size, (255, 255, 255, 255))
    base.alpha_composite(im)
    return base.convert("RGB")


def load_still(spec, image, work):
    """The source image of a spec (or of a stage): PNG, SVG, or a list joined side by side."""
    paths = image if isinstance(image, list) else [image]
    ims, used = [], []
    for p in paths:
        p = need(resolve(p), "source image")
        if p.suffix.lower() == ".svg":
            png = render_svg(p, spec["svg_width"], work)
            used.append(str(p))
            ims.append(flatten(Image.open(png)))
        else:
            used.append(str(p))
            ims.append(flatten(Image.open(p)))
    if len(ims) > 1:
        hh = max(im.height for im in ims)
        ims = [im.resize((round(im.width * hh / im.height), hh), Image.LANCZOS) for im in ims]
        joined = Image.new("RGB", (sum(im.width for im in ims), hh), (255, 255, 255))
        x = 0
        for im in ims:
            joined.paste(im, (x, 0))
            x += im.width
        ims = [joined]
    return ims[0], used


def svg_stage(spec, svg_path, drops, work):
    text = need(resolve(svg_path), "SVG").read_text()
    for a, b in drops:
        i = text.find(a)
        j = text.find(b, i + len(a)) if i >= 0 else -1
        if i < 0 or j < 0:
            sys.exit(f"refused: SVG marker pair {a!r} .. {b!r} not found in {svg_path}")
        text = text[:i] + text[j:]
    return flatten(Image.open(render_svg(resolve(svg_path), spec["svg_width"], work, text=text)))


class Padded:
    """A still source padded with the plate color, so camera boxes may leave the image."""

    def __init__(self, im, crop, plate_rgb, pad):
        self.w, self.h, self.pad = im.width, im.height, pad
        self.im = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad), plate_rgb)
        if crop:
            x0, y0, x1, y1 = (round(crop[0] * im.width), round(crop[1] * im.height), round(crop[2] * im.width),
                              round(crop[3] * im.height))
            self.im.paste(im.crop((x0, y0, x1, y1)), (x0 + pad, y0 + pad))
        else:
            self.im.paste(im, (pad, pad))

    def view(self, box, size):
        p = self.pad
        return self.im.resize(size, Image.LANCZOS, box=(box[0] + p, box[1] + p, box[2] + p, box[3] + p))


class Video:
    """Frames of a video decoded by ffmpeg at the output rate; the last frame is held past the end."""

    def __init__(self, spec):
        path = need(resolve(spec["video"]), "video")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        if probe.returncode != 0:
            sys.exit(f"refused: ffprobe failed on {path}")
        self.w, self.h = (int(v) for v in probe.stdout.strip().split(",")[:2])
        vf = f"fps={spec['fps']}" + (f",{spec['video_vf']}" if spec["video_vf"] else "")
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", str(spec["video_ss"]), "-i", str(path), "-t",
                                      str(spec["seconds"] + 1), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.last, self.frames = None, 0
        self.path = path

    def next(self):
        n = self.w * self.h * 3
        buf = self.proc.stdout.read(n)
        if len(buf) == n:
            self.last = Image.frombytes("RGB", (self.w, self.h), buf)
            self.frames += 1
        elif self.last is None:
            sys.exit(f"refused: no frame decoded from {self.path} at {self.proc.args}")
        return self.last

    def view(self, frame, box, size):
        return frame.resize(size, Image.LANCZOS, box=box)


# ----------------------------------------------------------------------------------------------- text layers


def wrap(text, f, width):
    out = []
    for para in text.split("\n"):
        line = ""
        for word in para.split(" "):
            trial = word if not line else line + " " + word
            if line and f.getlength(trial) > width:
                out.append(line)
                line = word
            else:
                line = trial
        out.append(line)
    return out


def text_layer(lines_spec, pad=0, bg=None, bg_alpha=0.0, align="left", min_w=0, radius=10):
    """RGBA image of stacked lines; lines_spec = [(text, font, color, line_height)]."""
    widths = [f.getlength(t) for t, f, c, lh in lines_spec]
    w = int(math.ceil(max(max(widths), min_w))) + 2 * pad
    h = int(math.ceil(sum(lh for *_, lh in lines_spec))) + 2 * pad
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if bg:
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=rgb(bg) + (int(255 * bg_alpha),))
    y = pad
    for (t, f, c, lh), tw in zip(lines_spec, widths, strict=True):
        asc, desc = f.getmetrics()
        x = pad + ((w - 2 * pad - tw) / 2 if align == "center" else 0)
        d.text((x, y + (lh - asc - desc) / 2 + asc), t, font=f, fill=rgb(c) + (255,), anchor="ls")
        y += lh
    return im


def make_text(e):
    f = font(e.get("font", "sans"), e.get("size", 26))
    lh = e.get("lh", 1.3) * e.get("size", 26)
    lines = wrap(e["text"], f, e["w"]) if e.get("w") else e["text"].split("\n")
    pad = e.get("pad", 12 if e.get("bg") else 0)
    return text_layer([(ln, f, e.get("color", "#c8c8d0"), lh) for ln in lines], pad=pad, bg=e.get("bg"),
                      bg_alpha=e.get("bg_alpha", 0.8), align=e.get("align", "left"))


def anchored(xy, size, anchor):
    """Top-left corner of a layer of the given size placed at xy with a two-letter anchor (l/m/r, t/m/b)."""
    x, y = xy
    ax, ay = anchor[0], anchor[1]
    x -= {"l": 0, "m": size[0] / 2, "r": size[0]}[ax]
    y -= {"t": 0, "m": size[1] / 2, "b": size[1]}[ay]
    return int(round(x)), int(round(y))


def lower_third_layer(e, alpha):
    f, fs = font("sans", 30), font("sans", 24)
    maxw = 1560
    lines = [(ln, f, "#ffffff", 40) for ln in wrap(e["text"], f, maxw)]
    if e.get("small"):
        lines += [(ln, fs, "#c8c8d0", 32) for ln in wrap(e["small"], fs, maxw)]
    min_w = 0
    if e.get("size_as"):
        min_w = max(f.getlength(ln) for ln in wrap(e["size_as"], f, maxw))
        if e.get("small"):
            min_w = max([min_w] + [fs.getlength(ln) for ln in wrap(e["small"], fs, maxw)])
    return text_layer(lines, pad=24, bg="#000000", bg_alpha=alpha, min_w=min_w, radius=8)


def tag_layer(text, alpha, bg):
    return text_layer([(text, font("bold", 34), "#ffffff", 44)], pad=14, bg=bg, bg_alpha=alpha, radius=6)


class Layer:
    """An RGBA layer split once into RGB and alpha, pasted with an opacity factor."""

    def __init__(self, im):
        self.rgb, self.a = im.convert("RGB"), im.getchannel("A")
        self.size, self.width, self.height = im.size, im.width, im.height


def paste_alpha(frame, layer, xy, a):
    if a <= 0:
        return
    mask = layer.a if a >= 1 else layer.a.point(lambda v: int(v * a))
    frame.paste(layer.rgb, xy, mask)


# ----------------------------------------------------------------------------------------------- geometry


def plate_rect(spec, n_caption):
    p = spec["plate"]
    if p == "full":
        return 0, 0, W, H
    size, _, pos = p.partition("+")
    pw, ph = (int(v) for v in size.split("x"))
    if pos:
        px, py = (int(v) for v in pos.split("+"))
        return px, py, pw, ph
    if p == "1760x990" and n_caption > 1:
        ph = H - 45 - (16 + 34 * n_caption)
        pw = round(ph * 16 / 9)
    return (W - pw) // 2, 45, pw, ph


def expand(box, sw, sh, aspect):
    x0, y0, x1, y1 = box[0] * sw, box[1] * sh, box[2] * sw, box[3] * sh
    bw, bh = x1 - x0, y1 - y0
    if bw / bh < aspect:
        bw = bh * aspect
    return [(x0 + x1) / 2, (y0 + y1) / 2, bw]


def cam_box(cam, aspect):
    cx, cy, bw = cam
    bh = bw / aspect
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def camera(t, keys):
    if t <= keys[0][0]:
        return keys[0][1]
    for (ta, a), (tb, b) in zip(keys, keys[1:], strict=False):
        if t < tb:
            u = smooth((t - ta) / (tb - ta))
            return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] * (b[2] / a[2]) ** u]
    return keys[-1][1]


def level_at(t, dims):
    lvl = 1.0
    for d in sorted(dims, key=lambda e: e["t"]):
        if t <= d["t"]:
            break
        u = smooth((t - d["t"]) / max(d.get("dur", 0.6), 1e-6))
        lvl = lvl + (d["level"] - lvl) * u
    return lvl


def ramp(t, t0, until, fade, fade_out):
    a = 1.0 if fade <= 0 else smooth((t - t0) / fade)
    if t < t0:
        return 0.0
    if until is not None:
        if t >= until:
            return 0.0
        if fade_out > 0:
            a = min(a, smooth((until - t) / fade_out))
    return a


# ----------------------------------------------------------------------------------------------- one shot


def normalize(raw):
    spec = dict(DEFAULTS)
    for k, v in raw.items():
        k = k.replace("-", "_")
        if k not in DEFAULTS and k not in META:
            sys.exit(f"refused: unknown key {k!r} in shot {raw.get('name')}")
        spec[k] = v
    if spec["concat"]:
        return spec
    if spec["seconds"] is None or spec["out"] is None:
        sys.exit(f"refused: shot {spec['name']} needs seconds and out")
    if (spec["image"] is None) == (spec["video"] is None):
        sys.exit(f"refused: shot {spec['name']} needs exactly one of image and video")
    return spec


def retime(raw, seconds):
    """A copy of a shot spec lasting `seconds`, with every event time scaled in proportion (fade lengths, the tag's
    3.5 s and the global fades are kept), for fitting a shot to the measured narration."""
    old = float(raw["seconds"])
    if seconds <= 0 or raw.get("concat"):
        sys.exit(f"refused: cannot retime {raw.get('name')} to {seconds} s")
    f = seconds / old
    out = json.loads(json.dumps(raw))
    out["seconds"] = seconds
    for key in ("keys", "reveal", "stages", "overlays", "texts", "lower_third", "lower-third", "dim"):
        for e in out.get(key, []):
            for tk in ("t", "until"):
                if isinstance(e.get(tk), (int, float)) and e[tk] > 0:
                    e[tk] = round(e[tk] * f, 3)
    if out.get("still"):
        out["still"] = sorted({round(min(t * f, seconds - 0.3), 2) for t in out["still"]})
    return out


def probe_clip(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_read_frames,width,height,pix_fmt,r_frame_rate", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"refused: ffprobe failed on {path}")
    return json.loads(r.stdout)["streams"][0]


def concat_shot(raw):
    """Join rendered clips (same format) into one, re-encoded; the parts must exist and match the frame format."""
    spec = normalize(raw)
    out = resolve(spec["out"])
    parts = [need(resolve(p), "concat part") for p in spec["concat"]]
    frames = 0
    for p in parts:
        st = probe_clip(p)
        if (st["width"], st["height"], st["pix_fmt"], st["r_frame_rate"]) != (W, H, "yuv420p", f"{spec['fps']}/1"):
            sys.exit(f"refused: concat part {p} has {st}")
        frames += int(st["nb_read_frames"])
    lst = out.with_suffix(".concat.txt")
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v",
                        "libx264", "-preset", spec["preset"], "-crf", str(spec["crf"]), "-pix_fmt", "yuv420p", "-r",
                        str(spec["fps"]), "-g", str(2 * spec["fps"]), "-movflags", "+faststart", "-an", str(out)])
    lst.unlink()
    st = probe_clip(out)
    if r.returncode != 0 or int(st["nb_read_frames"]) != frames:
        sys.exit(f"refused: concat of {parts} gave {st}, expected {frames} frames")
    print(f"{spec['name']}: {frames} frames joined from {len(parts)} parts", flush=True)
    return {"name": spec["name"], "file": str(out), "seconds": frames / spec["fps"], "frames": frames,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "sources": [str(p) for p in parts],
            "stills": [], "on_screen": {"parts": [p.name for p in parts]}, "spec": dict(raw)}


def render_shot(raw):
    t_start = time.time()
    spec = normalize(raw)
    fps, seconds = spec["fps"], float(spec["seconds"])
    n_frames = int(round(seconds * fps))
    out = resolve(spec["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    work = resolve(spec["work"]) if spec["work"] else WORK
    work.mkdir(parents=True, exist_ok=True)
    bg = rgb(spec["bg"])
    sources_used = []

    # caption lines decide the plate size
    cap_font = font("sans", 26)
    cap_lines = wrap(spec["caption"], cap_font, 1800) if spec["caption"] else []
    px, py, pw, ph = plate_rect(spec, len(cap_lines))
    aspect = pw / ph

    video = Video(spec) if spec["video"] else None
    if video:
        sw, sh = video.w, video.h
        sources_used.append(str(video.path))
        plate_rgb = rgb(spec["plate_bg"]) if spec["plate_bg"] != "auto" else (255, 255, 255)
        stills = []
    else:
        if spec["svg_drop"]:
            if not (isinstance(spec["image"], str) and spec["image"].endswith(".svg")):
                sys.exit("refused: svg_drop needs an SVG image")
            base, used = svg_stage(spec, spec["image"], spec["svg_drop"], work), [str(resolve(spec["image"]))]
        else:
            base, used = load_still(spec, spec["image"], work)
        sources_used += used
        sw, sh = base.size
        plate_rgb = base.getpixel((2, 2)) if spec["plate_bg"] == "auto" else rgb(spec["plate_bg"])
        stills = [base]
        for st in sorted(spec["stages"], key=lambda e: e["t"]):
            if "image" in st:
                im, used = load_still(spec, st["image"], work)
                sources_used += used
            else:
                svg = spec["image"] if isinstance(spec["image"], str) else None
                if not svg or not svg.endswith(".svg"):
                    sys.exit("refused: svg_drop stages need an SVG image")
                im = svg_stage(spec, svg, st["svg_drop"], work)
            if im.size != base.size:
                sys.exit(f"refused: stage at t={st['t']} is {im.size}, the source {base.size}")
            stills.append(im)

    # camera keys (center x, center y, width in source pixels)
    default_box = spec["crop"] or [0, 0, 1, 1]
    if spec["keys"]:
        keys = [(k["t"], expand(check_box(k["box"], "key"), sw, sh, aspect)) for k in spec["keys"]]
    else:
        b0 = expand(check_box(default_box, "crop"), sw, sh, aspect)
        keys = [(0.0, b0)]
        if spec["drift"]:
            keys.append((seconds, [b0[0], b0[1], b0[2] / (1 + spec["drift"])]))
    keys.sort(key=lambda k: k[0])
    over = 0.0
    for _, c in keys:
        x0, y0, x1, y1 = cam_box(c, aspect)
        over = max(over, -x0, -y0, x1 - sw, y1 - sh)
    pad = int(math.ceil(over)) + 4
    padded = [Padded(im, spec["crop"], plate_rgb, pad) for im in stills]

    masks = [{"t": r["t"], "fade": r.get("fade", 0.4), "box": check_box(r["box"], "reveal")} for r in spec["reveal"]]
    masks += [{"t": math.inf, "fade": 1.0, "box": check_box(r["box"] if isinstance(r, dict) else r, "hide")}
              for r in spec["hide"]]
    stage_times = sorted(st["t"] for st in spec["stages"])
    sfade = spec["stage_fade"]

    # cached layers
    def load_overlay(e):
        p = need(resolve(e["image"]), "overlay")
        sources_used.append(str(p))
        im = Image.open(p).convert("RGBA")
        if e.get("scale"):
            im = im.resize((round(im.width * e["scale"]), round(im.height * e["scale"])), Image.LANCZOS)
        return im

    overlays = [(e, Layer(load_overlay(e))) for e in spec["overlays"]]
    texts = [(e, Layer(make_text(e))) for e in spec["texts"]]
    cap_layer = None
    if cap_lines:
        cap_layer = Layer(text_layer([(ln, cap_font, "#9a9aa6", 34) for ln in cap_lines], align="center"))
        bottom = py + ph if spec["plate"] != "full" else H - 8 - cap_layer.height
        cap_xy = ((W - cap_layer.width) // 2,
                  int(bottom + (H - bottom - cap_layer.height) / 2) if spec["plate"] != "full" else bottom)
        if spec["caption_y"] is not None:
            cap_xy = (cap_xy[0], int(spec["caption_y"]))
    lts = [(e, Layer(lower_third_layer(e, spec["lower_third_alpha"]))) for e in spec["lower_third"]]
    tag = Layer(tag_layer(spec["tag"], spec["tag_alpha"], spec["bg"])) if spec["tag"] else None
    solid_plate = Image.new("RGB", (pw, ph), plate_rgb)
    solid_bg = Image.new("RGB", (pw, ph), bg)
    frame_bg = Image.new("RGB", (W, H), bg)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    cache = {"key": None, "img": None}

    def main_layer(t, vframe):
        box = cam_box(camera(t, keys), aspect)
        si, sj, f = 0, 0, 0.0
        for k, ts in enumerate(stage_times):
            if t >= ts + sfade:
                si = sj = k + 1
            elif t >= ts:
                si, sj, f = k, k + 1, (t - ts) / sfade
        active = []
        for m in masks:
            a = 1.0 if m["t"] == math.inf else (0.0 if m["t"] <= 0 else 1 - smooth((t - m["t"]) / m["fade"]))
            if a > 0:
                active.append((round(a, 3), m["box"]))
        lvl = level_at(t, spec["dim"])
        key = (tuple(round(v, 3) for v in box), si, sj, round(f, 3), tuple(active), round(lvl, 3))
        if vframe is None and key == cache["key"]:
            return cache["img"]
        if vframe is not None:
            img = video.view(vframe, box, (pw, ph))
        else:
            img = padded[si].view(box, (pw, ph))
            if f > 0:
                img = Image.blend(img, padded[sj].view(box, (pw, ph)), smooth(f))
        if active:
            mask = Image.new("L", (pw, ph), 0)
            d = ImageDraw.Draw(mask)
            sx, sy = pw / (box[2] - box[0]), ph / (box[3] - box[1])
            for a, (x0, y0, x1, y1) in sorted(active):
                r = ((x0 * sw - box[0]) * sx, (y0 * sh - box[1]) * sy, (x1 * sw - box[0]) * sx,
                     (y1 * sh - box[1]) * sy)
                d.rectangle([round(r[0]), round(r[1]), round(r[2]) - 1, round(r[3]) - 1], fill=int(255 * a))
            img = Image.composite(solid_plate, img, mask)
        if lvl < 0.999:
            img = Image.blend(solid_bg, img, lvl)
        cache["key"], cache["img"] = key, img
        return img

    def compose(t, vframe):
        frame = frame_bg.copy()
        frame.paste(main_layer(t, vframe), (px, py))
        for e, im in overlays:
            a = ramp(t, e.get("t", 0), e.get("until"), e.get("fade", 0.4), e.get("fade_out", 0.4))
            if a > 0:
                x, y = e.get("xy", [0, 0])
                if e.get("slide_from"):
                    u = 1 - smooth((t - e.get("t", 0)) / e.get("slide_dur", 0.8))
                    x, y = x + e["slide_from"][0] * u, y + e["slide_from"][1] * u
                paste_alpha(frame, im, (int(round(x)), int(round(y))), a)
        for e, im in texts:
            a = ramp(t, e.get("t", 0), e.get("until"), e.get("fade", 0.4), e.get("fade_out", 0.3))
            paste_alpha(frame, im, anchored(e["xy"], im.size, e.get("anchor", "lt")), a)
        if cap_layer is not None:
            paste_alpha(frame, cap_layer, cap_xy, 1.0)
        for e, im in lts:
            a = ramp(t, e["t"], e.get("until"), 0.3, 0.3)
            paste_alpha(frame, im, ((W - im.width) // 2, H - 90 - im.height), a)
        if tag is not None:
            a = t / 0.4 if t < 0.4 else (1.0 if t < 3.1 else max(0.0, (3.5 - t) / 0.4))
            paste_alpha(frame, tag, (W - 60 - tag.width, 26), a)
        g = 1.0
        if spec["fade_in"] > 0:
            g = min(g, t / spec["fade_in"])
        if spec["fade_out"] > 0:
            g = min(g, (seconds - t - 1.0 / fps) / spec["fade_out"])
        if g < 1:
            frame = Image.blend(black, frame, max(0.0, g))
        return frame

    still_times = [float(s) for s in (spec["still"] if isinstance(spec["still"], list) else [spec["still"]])]
    still_files = []
    if spec["stills_only"] or still_times:
        for ts in still_times:
            vframe = None
            if video:
                v2 = Video(dict(spec, video_ss=float(spec["video_ss"]) + ts))
                vframe = v2.next()
                v2.proc.kill()
            cache["key"] = None
            (work / "stills").mkdir(exist_ok=True)
            p = work / "stills" / f"{out.stem}-t{ts:05.2f}.png"
            compose(ts, vframe).save(p)
            still_files.append(str(p))
        if video:
            video.proc.kill()
            video.proc.wait()
            video = None if spec["stills_only"] else Video(spec)
        if spec["stills_only"]:
            return {"name": spec["name"], "stills": still_files}

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r",
           str(fps), "-i", "-", "-c:v", "libx264", "-preset", spec["preset"], "-crf", str(spec["crf"]), "-pix_fmt",
           "yuv420p", "-g", str(2 * fps), "-movflags", "+faststart", "-an", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cache["key"] = None
    try:
        for i in range(n_frames):
            t = i / fps
            enc.stdin.write(compose(t, video.next() if video else None).tobytes())
    finally:
        enc.stdin.close()
        rc = enc.wait()
        if video:
            video.proc.kill()
    if rc != 0:
        sys.exit(f"refused: ffmpeg exited with {rc} for {out}")
    st = probe_clip(out)
    if int(st["nb_read_frames"]) != n_frames or (st["width"], st["height"]) != (W, H) or st["pix_fmt"] != "yuv420p":
        sys.exit(f"refused: {out} has {st}, expected {n_frames} frames of {W}x{H} yuv420p")
    rec = {"name": spec["name"], "file": str(out), "seconds": n_frames / fps, "frames": n_frames,
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "render_s": round(time.time() - t_start, 1),
           "sources": sorted(set(sources_used)), "stills": still_files,
           "on_screen": {"caption": spec["caption"], "tag": spec["tag"],
                         "lower_thirds": [{k: e[k] for k in ("t", "until", "text", "small") if k in e}
                                          for e in spec["lower_third"]],
                         "texts": [{"t": e.get("t", 0), "text": e["text"]} for e in spec["texts"]]},
           "spec": {k: v for k, v in raw.items()}}
    print(f"{spec['name'] or out.name}: {n_frames} frames, {rec['render_s']} s", flush=True)
    return rec


# ----------------------------------------------------------------------------------------------- CLI


def card_contents(paths):
    """On-screen text of card PNGs used as sources, from the cards-receipt.json next to them."""
    found = []
    for p in paths:
        rp = Path(p).parent / "cards-receipt.json"
        if not rp.exists():
            continue
        cards = json.loads(rp.read_text())["cards"]
        for name, c in cards.items():
            if any(f["file"] == p for f in c["files"]) and name not in [x["card"] for x in found]:
                found.append({"card": name, "content": c["content"]})
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", help="JSON list of shot specs")
    ap.add_argument("--only", nargs="*", help="batch: render only these names")
    ap.add_argument("--jobs", type=int, default=1, help="batch: shots rendered in parallel")
    ap.add_argument("--manifest", help="batch: write the manifest of the rendered clips here")
    ap.add_argument("--retime", nargs="*", metavar="NAME=SECONDS",
                    help="batch: render these shots at a new length, event times scaled in proportion")
    ap.add_argument("--stills-only", action="store_true", help="write the --still frames only")
    ap.add_argument("--image", nargs="+")
    ap.add_argument("--video")
    ap.add_argument("--video-ss", type=float)
    ap.add_argument("--video-vf")
    ap.add_argument("--svg-width", type=int)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--plate")
    ap.add_argument("--plate-bg")
    ap.add_argument("--bg")
    ap.add_argument("--drift", type=float)
    ap.add_argument("--caption")
    ap.add_argument("--caption-y", type=int)
    ap.add_argument("--lower-third-alpha", type=float)
    ap.add_argument("--tag-alpha", type=float)
    ap.add_argument("--tag")
    ap.add_argument("--fade-in", type=float)
    ap.add_argument("--fade-out", type=float)
    ap.add_argument("--crf", type=int)
    ap.add_argument("--preset")
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--work")
    for opt in ("crop", "keys", "reveal", "hide", "stages", "svg-drop", "dim", "overlays", "texts", "lower-third"):
        ap.add_argument(f"--{opt}", type=json.loads, help="JSON")
    args = ap.parse_args()

    if args.batch:
        bpath = Path(args.batch)
        shots = json.loads(bpath.read_text())
        names = [s.get("name") for s in shots]
        if None in names or len(set(names)) != len(names):
            sys.exit("refused: every batch entry needs a unique name")
        new_len = {}
        for item in args.retime or []:
            name, _, sec = item.partition("=")
            if name not in names or not sec:
                sys.exit(f"refused: --retime {item!r} names no shot of the batch or no length")
            new_len[name] = float(sec)
        only = args.only or (sorted(new_len) if new_len else None)
        if only:
            missing = sorted(set(only) - set(names))
            if missing:
                sys.exit(f"refused: no shot named {missing}")
            shots = [s for s in shots if s["name"] in only]
        shots = [retime(s, new_len[s["name"]]) if s["name"] in new_len else s for s in shots]
        joins = [s for s in shots if s.get("concat")]
        shots = [s for s in shots if not s.get("concat")]
        if args.stills_only:
            shots, joins = [dict(s, stills_only=True) for s in shots], []
        if args.jobs > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                recs = list(ex.map(render_shot, shots))
        else:
            recs = [render_shot(s) for s in shots]
        recs += [concat_shot(s) for s in joins]
        shots += joins
        if args.manifest and not args.stills_only:
            mpath = Path(args.manifest)
            old = {e["name"]: e for e in json.loads(mpath.read_text())["clips"]} if mpath.exists() else {}
            for raw, rec in zip(shots, recs, strict=True):
                entry = {"name": raw["name"], "file": Path(rec["file"]).name, "path": rec["file"],
                         "seconds": rec["seconds"], "frames": rec["frames"], "sha256": rec["sha256"]}
                entry.update({k: raw[k] for k in META if k in raw})
                entry["sources"] = rec["sources"]
                entry["on_screen"] = rec["on_screen"]
                cards = card_contents(rec["sources"])
                if cards:
                    entry["on_screen"]["cards"] = cards
                entry["spec"] = rec["spec"]
                old[raw["name"]] = entry
            order = [s["name"] for s in json.loads(bpath.read_text())]
            clips = sorted(old.values(), key=lambda e: order.index(e["name"]) if e["name"] in order else 999)
            mpath.write_text(json.dumps({"tool": "tools/render_figure_shot.py", "batch": str(bpath.resolve()),
                                         "batch_sha256": hashlib.sha256(bpath.read_bytes()).hexdigest(),
                                         "format": "1920x1080, 30 fps, H.264 CRF 16, yuv420p, no audio",
                                         "clips": clips}, indent=1, ensure_ascii=False))
        return

    raw = {k: v for k, v in vars(args).items() if v is not None and k not in ("batch", "only", "jobs", "manifest")}
    if "image" in raw and len(raw["image"]) == 1:
        raw["image"] = raw["image"][0]
    raw["stills_only"] = args.stills_only
    render_shot(raw)


if __name__ == "__main__":
    main()
