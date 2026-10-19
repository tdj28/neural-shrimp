"""Zoom video from the wide Lyapunov map of the soft-token loop down to the hook inside the kite of the period-6 shrimp
(human, 19 Oct 2026: "Can we do a video zoom in of the chaos from the larger map all the way down to the hook, being
clever about interpolations so it came out smooth? Also with narration please").

Maps. Six float64 Lyapunov maps of one system (Qwen2.5-0.5B-Instruct soft-token loop, philosophy direction at layer 12),
from the wide map MAP-STRIPES-256 through CROSS-512, HR9, HR10 (the period-6 shrimp) and HR11 (the kite) to HR12 (the
hook), are colored once on their own grids with one normalization for all (the papers' convention: lambda <= 1e-3 gray,
clip(lambda, -0.6, 0) from black to white; chaotic cells, lambda > 1e-3, in plasma over 0 to 0.45; rows upward in
log10 T). Every frame samples each map that meets the view window from its RGB image with a separable kernel chosen per
axis by how many pixels a cell spans there: bilinear up to 3 pixels per cell (the same numbers as
scipy.ndimage.map_coordinates with order=1; --check-sampling compares them), the cubic B-spline from 8 pixels per cell
(smooth and never overshooting, so strongly magnified cells do not show the grid), and a linear blend of the two weight
sets between. The maps are composited from fine to coarse. A finer map fades in with a smootherstep weight while the
window shrinks from FADE_FROM to 1.0 times its extent (log size, worse axis): 2.0 for most maps, 3.5 for HR11 (its cells
are 7.5 times finer in alpha than HR10's) and 1.9 for HR12 (so the kite's rest shows HR11 alone). Its border is feathered
over 10 % of its extent, the ramp narrowing to the visible margin where the border meets the frame edge, so no edge ever
shows and the last frames are pure HR12.

Camera. Each axis is mapped to the frame independently (16:9, like the paper figures) and shrinks geometrically: the
alpha width and the log10 T height follow s(phi) = s0 (sE / s0)^phi from the full wide map (phi = 0) to the HR12 extent
(phi = 1), so the window keeps one data aspect and shapes never stretch during the zoom. The camera travels at constant
speed in log size (default 9 s per e-fold) between rests: the shrimp (window inside HR10), the kite (inside HR11) and the
end (the HR12 extent). Each travel is an exact zoom toward a fixed point, window min = P - r * size with r and P fixed
for that travel (the brief's formula), and starts and ends with smootherstep speed ramps, so the camera eases into
each rest and out of it. A single fixed point from the wide map would keep the hook 12 % above the bottom of the frame
the whole way and cut off the shrimp's legs and the kite's lower half when they are labeled; with --single-point it is
rendered for comparison. Windows are nested (0 <= r <= 1, checked).

Text (small, unobtrusive): a zoom factor in the top right corner (geometric mean of the two axes), a caption at the
start, and labels that fade in near the end of each approach.

Frames go to a scratch folder as PNG, then ffmpeg (H.264 High, yuv420p BT.709, CRF 18, 30 fps, 1920x1080, at most 8
threads). A contact sheet of 12 frames and a JSON receipt (timeline, keyframes, label times, hashes) go next to the
video. The narration is added by tools/narrate_video.py (script docs/narration/zoom-to-hook.json).

Usage:
  python tools/render_zoom_to_hook.py --out ../neural-shrimp-data/videos/zoom-to-hook.mp4 \
      --frames ../neural-shrimp-data/videos/cache/zoom-to-hook-frames --workers 8
  python tools/render_zoom_to_hook.py --timeline-only            # print the timeline (rests, labels) and exit
  python tools/render_zoom_to_hook.py --still-time 30 --still-out test.png
  python tools/render_zoom_to_hook.py --check-sampling
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data" / "explore" / "llm-soft-loop" / "pod64"
LEVELS = ["map-stripes-256", "cross-512", "hr9", "hr10", "hr11", "hr12"]       # coarse to fine

W, H, FPS = 1920, 1080, 30
GRAY_LO, CHAOS_LEVEL, CHAOS_HI = -0.6, 1e-3, 0.45
# A finer map fades in while the window shrinks from FADE_FROM to 1.0 times its extent (log size, worse axis). HR11
# starts early: its cells are 7.5 times finer in alpha than HR10's, which would otherwise be magnified to about 36 pixels
# per cell before HR11 covers the frame. HR12 starts below the window/extent ratio at the kite's rest (1.94), so that
# rest shows HR11 alone.
FADE_FROM = {"cross-512": 2.0, "hr9": 2.0, "hr10": 2.0, "hr11": 3.5, "hr12": 1.9}
FADE_TO = 1.0
FEATHER = 0.10                          # border feather, fraction of the map's extent
KERNEL_MAG = (3.0, 8.0)                 # pixels per cell: bilinear below, cubic B-spline above, blended between

# Rests of the camera: (name, alpha center, log10 T center, log10 T height). The alpha width follows from the
# geometric schedule of the window. "start" is the full wide map, "end" the HR12 extent.
RESTS = [("shrimp", 0.07310, -0.15300, 0.00780),     # inside HR10 (alpha 0.0718..0.0746, log10 T -0.157..-0.149)
         ("kite", 0.07296, -0.15370, 0.00136)]       # inside HR11 (alpha 0.0728..0.0731, log10 T -0.1544..-0.1530)
START_HOLD, REST_HOLD, END_HOLD = 3.0, 2.5, 3.0      # seconds
RAMP = 2.5                                           # seconds of speed ramp at each end of a travel
FADE = 1.5                                           # label fade, seconds

# Labels: (text, anchor, fade-in start in s (or (rest, offset): relative to the arrival at a rest), fade-out start)
CAPTION = "steering strength (horizontal), temperature (vertical)"
LABELS = [("chaos (colors) and periodic windows (gray)", 0.6, 21.5),
          ("the period-6 shrimp", ("shrimp", -3.0), ("shrimp", 9.0)),
          ("the kite", ("kite", -7.0), ("kite", 5.0)),
          ("the hook", ("end", -4.5), None)]

# --------------------------------------------------------------------------------------------------------------------
# maps


def level_rgb(lam):
    """RGB (rows = log10 T ascending, columns = alpha ascending) in the papers' convention, one normalization."""
    import matplotlib
    g = ((np.clip(lam, GRAY_LO, 0.0) - GRAY_LO) / -GRAY_LO).astype(np.float32)
    rgb = np.repeat(g[..., None], 3, axis=-1)
    chaos = lam > CHAOS_LEVEL
    rgb[chaos] = matplotlib.colormaps["plasma"](np.clip(lam[chaos] / CHAOS_HI, 0.0, 1.0))[:, :3].astype(np.float32)
    return np.ascontiguousarray(rgb)


def load_levels(with_rgb=True):
    levels = []
    for name in LEVELS:
        path = DATA / name / "scan.npz"
        d = np.load(path)
        a, t, lam = d["alpha"], d["log10_temperature"], d["lyapunov"]
        if lam.shape != (t.size, a.size):
            raise SystemExit(f"{name}: lyapunov shape {lam.shape} does not match ({t.size}, {a.size})")
        da, dt = np.diff(a), np.diff(t)
        if not (np.allclose(da, da[0], rtol=1e-6) and np.allclose(dt, dt[0], rtol=1e-6)) or da[0] <= 0 or dt[0] <= 0:
            raise SystemExit(f"{name}: grid is not uniform and ascending")
        if not np.isfinite(lam).all():
            raise SystemExit(f"{name}: non-finite exponents")
        lv = {"name": name, "a0": float(a[0]), "a1": float(a[-1]), "t0": float(t[0]), "t1": float(t[-1]),
              "na": int(a.size), "nt": int(t.size), "chaotic_share": float((lam > CHAOS_LEVEL).mean()),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if with_rgb:
            lv["rgb"] = level_rgb(lam)
        levels.append(lv)
    return levels


# --------------------------------------------------------------------------------------------------------------------
# sampling and compositing


def smootherstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def axis_coords(w0, w1, n, flip=False):
    """Pixel-center coordinates of a frame axis (flip: first pixel at w1, i.e. image rows from the top)."""
    u = (np.arange(n) + 0.5) / n
    return w1 - u * (w1 - w0) if flip else w0 + u * (w1 - w0)


def feather_1d(x, lo, hi, wlo, whi):
    """Feather of a map's extent [lo, hi] along one axis, for the frame axis [wlo, whi]: on each side a smootherstep
    ramp of FEATHER times the extent, narrowed to the visible margin when that border nears the frame edge."""
    fmax = FEATHER * (hi - lo)
    m = np.ones_like(x)
    for edge, margin, sign in ((lo, lo - wlo, 1.0), (hi, whi - hi, -1.0)):
        f = min(fmax, max(margin, 0.0))
        d = sign * (x - edge)                   # distance inside the map from this border
        m *= smootherstep(d / f) if f > 0 else (d >= 0)
    m[(x < lo) | (x > hi)] = 0.0
    return m


def axis_taps(u, n, mag):
    """Taps and weights along one axis at fractional indices u (inside [0, n-1]): bilinear while a cell spans at most
    KERNEL_MAG[0] pixels, the cubic B-spline (smooth, never overshooting) once it spans KERNEL_MAG[1] or more, and a
    linear blend of the two weight sets between."""
    i = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
    f = (u - i).astype(np.float32)
    beta = float(np.clip((mag - KERNEL_MAG[0]) / (KERNEL_MAG[1] - KERNEL_MAG[0]), 0.0, 1.0))
    if beta == 0.0:
        return [i, i + 1], [1.0 - f, f]
    f2, f3 = f * f, f * f * f
    bs = [(1.0 - f) ** 3 / 6.0, (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0, (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0, f3 / 6.0]
    lin = [0.0, 1.0 - f, f, 0.0]
    w = [(1.0 - beta) * l + beta * b for l, b in zip(lin, bs)]
    return [np.clip(i + k, 0, n - 1) for k in (-1, 0, 1, 2)], w


def sample_separable(img, rows_f, cols_f, mag_rows=0.0, mag_cols=0.0):
    """Separable interpolation of img (M, N, 3) on the tensor grid rows_f x cols_f (fractional indices inside the grid);
    mag_*: pixels per cell along each axis, which sets the kernel (axis_taps)."""
    M, N = img.shape[:2]
    ct, cw = axis_taps(cols_f, N, mag_cols)
    rt, rw = axis_taps(rows_f, M, mag_rows)
    need = np.unique(np.concatenate(rt))
    sub = img[need]                                         # only the source rows this frame touches
    tmp = sum(sub[:, t, :] * np.asarray(w, np.float32)[None, :, None] for t, w in zip(ct, cw))
    return sum(tmp[np.searchsorted(need, t)] * np.asarray(w, np.float32)[:, None, None] for t, w in zip(rt, rw))


def bilinear_separable(img, rows_f, cols_f):
    """Bilinear interpolation on a tensor grid: the same numbers as map_coordinates(order=1) per channel."""
    return sample_separable(img, rows_f, cols_f, 0.0, 0.0)


def fade_weight(lv, win):
    a0, a1, t0, t1 = win
    rho = max((a1 - a0) / (lv["a1"] - lv["a0"]), (t1 - t0) / (lv["t1"] - lv["t0"]))
    f0 = FADE_FROM[lv["name"]]
    x = (math.log(f0) - math.log(rho)) / (math.log(f0) - math.log(FADE_TO))
    return float(smootherstep(x))


def render_map(levels, win, w=W, h=H, stats=None):
    """Composite the maps for the window (a0, a1, t0, t1); float32 (h, w, 3) in [0, 1]."""
    a0, a1, t0, t1 = win
    xs = axis_coords(a0, a1, w)
    ys = axis_coords(t0, t1, h, flip=True)                  # image rows from the top: log10 T descending
    out = np.zeros((h, w, 3), np.float32)
    trans = np.ones((h, w), np.float32)                     # front to back: transmittance left for coarser maps
    used = []
    for k in range(len(levels) - 1, -1, -1):
        lv = levels[k]
        wk = 1.0 if k == 0 else fade_weight(lv, win)
        if wk <= 0.0:
            continue
        ci = np.nonzero((xs >= lv["a0"]) & (xs <= lv["a1"]))[0]
        ri = np.nonzero((ys >= lv["t0"]) & (ys <= lv["t1"]))[0]
        if ci.size == 0 or ri.size == 0:
            continue
        cs, ce, rs, re = ci[0], ci[-1] + 1, ri[0], ri[-1] + 1
        if trans[rs:re, cs:ce].max() <= 0.0:
            continue
        cols_f = (xs[cs:ce] - lv["a0"]) / (lv["a1"] - lv["a0"]) * (lv["na"] - 1)
        rows_f = (ys[rs:re] - lv["t0"]) / (lv["t1"] - lv["t0"]) * (lv["nt"] - 1)
        mag_c = 1.0 / abs(cols_f[1] - cols_f[0]) if cols_f.size > 1 else 1.0
        mag_r = 1.0 / abs(rows_f[1] - rows_f[0]) if rows_f.size > 1 else 1.0
        rgb = sample_separable(lv["rgb"], rows_f, cols_f, mag_r, mag_c)
        if k == 0:
            alpha = np.ones((re - rs, ce - cs), np.float32)
        else:
            mx = feather_1d(xs[cs:ce], lv["a0"], lv["a1"], a0, a1)
            my = feather_1d(ys[rs:re], lv["t0"], lv["t1"], t0, t1)
            alpha = (wk * my[:, None] * mx[None, :]).astype(np.float32)
        tr = trans[rs:re, cs:ce]
        out[rs:re, cs:ce] += (tr * alpha)[..., None] * rgb
        trans[rs:re, cs:ce] = tr * (1.0 - alpha)
        used.append((lv["name"], round(wk, 4)))
        if trans.max() <= 0.0:
            break
    uncovered = float(trans.max())
    if uncovered > 1e-6:                                    # outside every map: neutral dark background
        out += trans[..., None] * np.float32(0.08)
    if stats is not None:
        stats["used"] = used
        stats["uncovered"] = uncovered
    return out


# --------------------------------------------------------------------------------------------------------------------
# camera


class Camera:
    """Window as a function of time: geometric shrink of each axis, constant speed in log size between rests, an exact
    zoom toward a fixed point in each travel, smootherstep speed ramps into and out of every rest."""

    def __init__(self, levels, seconds_per_efold=9.0, single_point=False):
        L0, LE = levels[0], levels[-1]
        self.s0 = np.array([L0["a1"] - L0["a0"], L0["t1"] - L0["t0"]])
        self.sE = np.array([LE["a1"] - LE["a0"], LE["t1"] - LE["t0"]])
        self.Z = np.log(self.s0 / self.sE)                  # total e-folds per axis
        self.Zm = float(self.Z.mean())
        self.v = 1.0 / seconds_per_efold                    # cruise speed, e-folds (mean of the axes) per second
        keys = [("start", 0.0, np.array([L0["a0"], L0["t0"]]))]
        if not single_point:
            for name, ac, tc, ht in RESTS:
                phi = math.log(self.s0[1] / ht) / self.Z[1]
                s = self.size(phi)
                keys.append((name, phi, np.array([ac, tc]) - s / 2))
        keys.append(("end", 1.0, np.array([LE["a0"], LE["t0"]])))
        self.keys = keys
        self.segs = []                                       # per travel: phi range, r and P per axis
        for (n0, p0, A0), (n1, p1, A1) in zip(keys[:-1], keys[1:]):
            s0, s1 = self.size(p0), self.size(p1)
            r = (A1 - A0) / (s0 - s1)
            if np.any(r < 0) or np.any(r > 1):
                raise SystemExit(f"travel {n0} -> {n1}: fixed point outside the window (r = {r}); windows not nested")
            self.segs.append({"from": n0, "to": n1, "phi0": p0, "phi1": p1, "r": r, "P": A0 + r * s0})
        # timeline: hold, travel, rest, travel, rest, travel, hold
        self.phases, t = [], 0.0
        self.phases.append(("hold", t, t + START_HOLD, 0.0, 0.0))
        t += START_HOLD
        for i, sg in enumerate(self.segs):
            dz = (sg["phi1"] - sg["phi0"]) * self.Zm
            T = dz / self.v + RAMP
            if dz / self.v < RAMP:
                raise SystemExit(f"travel to {sg['to']} too short for the speed ramps")
            self.phases.append(("travel", t, t + T, sg["phi0"], sg["phi1"]))
            t += T
            hold = END_HOLD if i == len(self.segs) - 1 else REST_HOLD
            self.phases.append(("hold", t, t + hold, sg["phi1"], sg["phi1"]))
            t += hold
        self.duration = t
        self.arrive = {"start": START_HOLD}
        for ph, sg in zip([p for p in self.phases if p[0] == "travel"], self.segs):
            self.arrive[sg["to"]] = ph[2]                   # time at which the camera comes to rest there

    def size(self, phi):
        return self.s0 * np.exp(-phi * self.Z)

    def phi(self, t):
        """Progress phi(t): constant in holds; in a travel, the log size runs at constant speed between smootherstep
        speed ramps (the distance covered in a ramp is half of RAMP at cruise speed)."""
        integ = lambda x: x ** 6 - 3 * x ** 5 + 2.5 * x ** 4      # integral of smootherstep from 0 to x  # noqa: E731
        for kind, t0, t1, p0, p1 in self.phases:
            if t > t1:
                continue
            if kind == "hold":
                return p0
            T, tau = t1 - t0, min(max(t - t0, 0.0), t1 - t0)
            dz = (p1 - p0) * self.Zm
            if tau < RAMP:
                z = self.v * RAMP * integ(tau / RAMP)
            elif tau > T - RAMP:
                z = dz - self.v * RAMP * integ((T - tau) / RAMP)
            else:
                z = self.v * RAMP / 2 + self.v * (tau - RAMP)
            return p0 + z / self.Zm
        return self.phases[-1][4]

    def window(self, t):
        phi = self.phi(t)
        s = self.size(phi)
        sg = next((g for g in self.segs if g["phi0"] <= phi <= g["phi1"]), self.segs[-1])
        A = sg["P"] - sg["r"] * s
        return (float(A[0]), float(A[0] + s[0]), float(A[1]), float(A[1] + s[1]))

    def zoom(self, t):
        return float(np.exp(np.mean(self.phi(t) * self.Z)))

    def label_times(self):
        def at(spec):
            if spec is None:
                return None
            if isinstance(spec, tuple):
                return self.arrive[spec[0]] + spec[1]
            return float(spec)
        out = [{"text": CAPTION, "in": 0.3, "out": START_HOLD, "where": "top-left"}]
        out += [{"text": txt, "in": at(a), "out": at(b), "where": "bottom-left"} for txt, a, b in LABELS]
        return out


# --------------------------------------------------------------------------------------------------------------------
# text


def find_font(weight="Medium"):
    for p in (f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf", "/usr/share/fonts/opentype/inter/Inter-Regular.otf"):
        if Path(p).exists():
            return p
    import matplotlib.font_manager as fm
    return fm.findfont("DejaVu Sans")


def text_patch(text, size, weight="Medium", tabular=False):
    """(alpha of the glyphs, alpha of a soft shadow) as float32 arrays, with a margin for the shadow. tabular: figures
    of equal width (OpenType tnum, needs libraqm), so a right-aligned counter does not twitch as its digits change."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont, features
    font = ImageFont.truetype(find_font(weight), size)
    kw = {"features": ["tnum"]} if tabular and features.check("raqm") else {}
    x0, y0, x1, y1 = font.getbbox(text, **kw)
    if tabular:                                 # fixed box height, so the baseline never moves either
        _, y0, _, y1 = font.getbbox("\u00d70123456789.", **kw)
    pad = 14
    im = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - x0, pad - y0), text, font=font, fill=255, **kw)
    glyph = np.asarray(im, np.float32) / 255.0
    halo = im.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(4.0))
    shadow = np.clip(np.asarray(halo, np.float32) / 255.0 * 1.5, 0, 1)
    return glyph, shadow, pad


def stamp(img, patch, x, y, opacity, anchor="tl"):
    """Composite white text with a dark shadow onto img (float32 h x w x 3) at (x, y) of its glyph box."""
    if opacity <= 0:
        return
    glyph, shadow, pad = patch
    ph, pw = glyph.shape
    if anchor == "tr":
        x = x - (pw - 2 * pad)
    elif anchor == "bl":
        y = y - (ph - 2 * pad)
    x, y = x - pad, y - pad
    sl = (slice(max(y, 0), min(y + ph, img.shape[0])), slice(max(x, 0), min(x + pw, img.shape[1])))
    g = glyph[sl[0].start - y:sl[0].stop - y, sl[1].start - x:sl[1].stop - x][..., None]
    s = shadow[sl[0].start - y:sl[0].stop - y, sl[1].start - x:sl[1].stop - x][..., None]
    reg = img[sl]
    reg *= 1.0 - 0.72 * opacity * s
    reg += (1.0 - reg) * (0.94 * opacity * g)


def fade(t, t_in, t_out):
    a = float(smootherstep((t - t_in) / FADE)) if t_in is not None else 1.0
    if t_out is not None:
        a *= 1.0 - float(smootherstep((t - t_out) / FADE))
    return a


def zoom_text(z):
    return f"×{z:.1f}" if z < 9.95 else f"×{z:.0f}"


# --------------------------------------------------------------------------------------------------------------------
# frames

_G = {}


def _init_worker(seconds_per_efold, single_point):
    _G["levels"] = load_levels()
    _G["cam"] = Camera(_G["levels"], seconds_per_efold, single_point)
    _G["labels"] = _G["cam"].label_times()
    _G["patches"] = [text_patch(lb["text"], 24 if lb["where"] == "top-left" else 34) for lb in _G["labels"]]
    _G["zoom_cache"] = {}


def frame_image(t):
    cam, levels = _G["cam"], _G["levels"]
    img = render_map(levels, cam.window(t))
    for lb, patch in zip(_G["labels"], _G["patches"]):
        op = fade(t, lb["in"], lb["out"])
        if lb["where"] == "top-left":
            stamp(img, patch, 48, 40, op)
        else:
            stamp(img, patch, 48, H - 46, op, anchor="bl")
    zt = zoom_text(cam.zoom(t))
    if zt not in _G["zoom_cache"]:
        _G["zoom_cache"][zt] = text_patch(zt, 26, tabular=True)
    stamp(img, _G["zoom_cache"][zt], W - 48, 40, 0.85, anchor="tr")
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _render_one(job):
    from PIL import Image
    i, t, path = job
    if Path(path).exists():
        return i
    tmp = Path(str(path) + ".tmp.png")
    Image.fromarray(frame_image(t)).save(tmp, compress_level=1)
    tmp.replace(path)
    return i


# --------------------------------------------------------------------------------------------------------------------
# entry points


def check_sampling(levels):
    """Maximum difference of the separable bilinear sampling from scipy.ndimage.map_coordinates(order=1)."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(0)
    worst = 0.0
    for lv in levels:
        M, N = lv["nt"], lv["na"]
        r = np.sort(rng.uniform(0, M - 1, 97))
        c = np.sort(rng.uniform(0, N - 1, 131))
        sep = bilinear_separable(lv["rgb"], r, c)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        ref = np.stack([map_coordinates(lv["rgb"][..., ch].astype(np.float64), [rr, cc], order=1, mode="nearest")
                        for ch in range(3)], axis=-1)
        worst = max(worst, float(np.abs(sep - ref).max()))
    return worst


def timeline(cam):
    return {"seconds": round(cam.duration, 3), "frames": int(round(cam.duration * FPS)),
            "phases": [{"kind": k, "from": round(a, 3), "to": round(b, 3), "phi": [round(p, 5), round(q, 5)]}
                       for k, a, b, p, q in cam.phases],
            "arrive": {k: round(v, 3) for k, v in cam.arrive.items()},
            "keys": [{"name": n, "phi": round(p, 5), "window": list(map(float, (A[0], A[0] + cam.size(p)[0], A[1],
                                                                                 A[1] + cam.size(p)[1])))}
                     for n, p, A in cam.keys],
            "travels": [{"from": g["from"], "to": g["to"], "r_alpha": round(float(g["r"][0]), 5),
                         "r_log10T": round(float(g["r"][1]), 5), "P": [float(g["P"][0]), float(g["P"][1])]}
                        for g in cam.segs],
            "labels": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in lb.items()}
                       for lb in cam.label_times()],
            "zoom_end": round(cam.zoom(cam.duration), 2),
            "zoom_end_per_axis": [round(float(x), 2) for x in np.exp(cam.Z)]}


def contact_sheet(frames_dir, n_frames, times, out):
    from PIL import Image, ImageDraw, ImageFont
    tw, th, cols = 480, 270, 4
    rows = math.ceil(len(times) / cols)
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 8, rows * (th + 30) + 8), (24, 24, 24))
    font = ImageFont.truetype(find_font("Regular"), 18)
    d = ImageDraw.Draw(sheet)
    for k, t in enumerate(times):
        i = min(int(round(t * FPS)), n_frames - 1)
        im = Image.open(frames_dir / f"{i:05d}.png").convert("RGB").resize((tw, th), Image.LANCZOS)
        x, y = 8 + (k % cols) * (tw + 8), 8 + (k // cols) * (th + 30)
        sheet.paste(im, (x, y))
        d.text((x, y + th + 4), f"t = {i / FPS:5.2f} s   frame {i}", font=font, fill=(220, 220, 220))
    sheet.save(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="silent video (MP4)")
    ap.add_argument("--frames", type=Path, help="scratch folder for the PNG frames (outside tmpfs)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seconds-per-efold", type=float, default=9.0)
    ap.add_argument("--single-point", action="store_true", help="one fixed point from the wide map to the end")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--timeline-only", action="store_true")
    ap.add_argument("--still-time", type=float, nargs="+")
    ap.add_argument("--still-out", type=Path)
    ap.add_argument("--check-sampling", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    if args.check_sampling:
        print(json.dumps({"max_abs_difference_vs_map_coordinates": check_sampling(load_levels())}))
        return
    cam = Camera(load_levels(with_rgb=False), args.seconds_per_efold, args.single_point)
    tl = timeline(cam)
    if args.timeline_only:
        print(json.dumps(tl, indent=1))
        return
    if args.still_time:
        from PIL import Image
        _init_worker(args.seconds_per_efold, args.single_point)
        for t in args.still_time:
            st = {}
            win = _G["cam"].window(t)
            render_map(_G["levels"], win, w=8, h=8, stats=st)
            out = args.still_out if len(args.still_time) == 1 else \
                args.still_out.with_name(f"{args.still_out.stem}-{t:06.2f}{args.still_out.suffix}")
            Image.fromarray(frame_image(t)).save(out)
            print(json.dumps({"t": t, "window": win, "zoom": round(_G["cam"].zoom(t), 3)} | st))
        return
    if args.out is None or args.frames is None:
        raise SystemExit("--out and --frames are required")
    if str(args.frames.resolve()).startswith("/tmp"):
        raise SystemExit("put the frames on disk, not on tmpfs")
    args.frames.mkdir(parents=True, exist_ok=True)
    n = tl["frames"]
    jobs = [(i, i / FPS, str(args.frames / f"{i:05d}.png")) for i in range(n)]
    t0 = time.time()
    from multiprocessing import get_context
    with get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                   initargs=(args.seconds_per_efold, args.single_point)) as pool:
        for k, _ in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=4)):
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{n} frames, {time.time() - t0:.0f} s", flush=True)
    render_s = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = args.out.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(args.frames / "%05d.png"),
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,"
                  "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", "slow",
           "-crf", "18", "-threads", "8", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-color_range", "tv", "-movflags", "+faststart", str(tmp_out)]
    subprocess.run(cmd, check=True)
    tmp_out.replace(args.out)
    times = [1.5, 9.0, 16.0, 23.0, cam.arrive["shrimp"] + 1.0, cam.arrive["shrimp"] + REST_HOLD + 6.0,
             cam.arrive["kite"] - 6.0, cam.arrive["kite"] + 1.0, cam.arrive["kite"] + REST_HOLD + 3.0,
             cam.arrive["end"] - 2.5, cam.arrive["end"] - 0.6, cam.duration - 0.5]
    sheet = args.out.with_name(args.out.stem + "-contact.png")
    contact_sheet(args.frames, n, times, sheet)
    rec = {"video": str(args.out), "sha256": sha256(args.out), "bytes": args.out.stat().st_size,
           "contact_sheet": str(sheet), "render_seconds": round(render_s, 1), "workers": args.workers,
           "source": {"tool": "tools/render_zoom_to_hook.py", "tool_sha256": sha256(Path(__file__)),
                      "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                               text=True).stdout.strip(),
                      "tool_committed": subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch",
                                                        "tools/render_zoom_to_hook.py"], capture_output=True).returncode == 0},
           "maps": [{k: v for k, v in lv.items() if k != "rgb"} for lv in load_levels(with_rgb=False)],
           "coloring": {"gray": "lambda <= 1e-3: clip(lambda, -0.6, 0) black to white",
                        "chaos": "lambda > 1e-3: plasma over 0 to 0.45"},
           "seconds_per_efold": args.seconds_per_efold, "ramp_seconds": RAMP, "fade": [FADE_FROM, FADE_TO],
           "feather": FEATHER, "timeline": tl, "ffmpeg": " ".join(cmd)}
    args.out.with_name(args.out.name + ".json").write_text(json.dumps(rec, indent=1))
    if not args.keep_frames:
        for i in range(n):
            (args.frames / f"{i:05d}.png").unlink(missing_ok=True)
        try:
            args.frames.rmdir()
        except OSError:
            pass
    print(json.dumps({"out": str(args.out), "seconds": tl["seconds"], "frames": n, "render_seconds": round(render_s, 1),
                      "contact_sheet": str(sheet)}))


if __name__ == "__main__":
    main()
