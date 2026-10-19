"""Time maps and slow push-ins for the explainer's data clips (render_text_split.py, render_batch_divergence.py and
plot_explainer_answers.py), so that a clip can be fitted to measured narration without dropping or repeating frames.

A time map is a JSON list of knots [[t_clip, t_anim], ...]: frame time t_clip of the rendered clip is drawn at time
t_anim of the animation's own schedule, interpolated linearly between knots (an event at animation time e therefore
appears where the map reaches e). The first knot must be [0, 0]-like in clip time (t_clip = 0), both coordinates must
not decrease, and the clip time must strictly increase. A push-in zooms every frame by a factor rising geometrically
from 1 to Z over the clip, about a center given in fractions of the frame, with subpixel resampling, so that no
stretch of the clip is a frozen frame.
"""
import json

from PIL import Image


def parse_time_map(text, seconds_clip, seconds_anim):
    """Validated knots; without a map, the identity (then the clip must keep the animation's own length)."""
    if not text:
        if abs(seconds_clip - seconds_anim) > 1e-9:
            raise SystemExit(f"time map: a clip of {seconds_clip} s needs a time map for an animation of "
                             f"{seconds_anim} s")
        return [[0.0, 0.0], [float(seconds_clip), float(seconds_anim)]]
    knots = json.loads(text) if isinstance(text, str) else text
    ok = isinstance(knots, list) and len(knots) >= 2 and all(isinstance(k, list) and len(k) == 2 for k in knots)
    if not ok:
        raise SystemExit(f"time map: malformed knots {text!r}")
    knots = [[float(a), float(b)] for a, b in knots]
    if knots[0][0] != 0.0:
        raise SystemExit("time map: the first knot must be at clip time 0")
    for (a0, b0), (a1, b1) in zip(knots, knots[1:]):
        if not (a1 > a0 and b1 >= b0):
            raise SystemExit(f"time map: knots must increase ({a0}, {b0}) -> ({a1}, {b1})")
    if knots[-1][0] < seconds_clip - 1e-6:
        raise SystemExit(f"time map ends at {knots[-1][0]} s, before the clip's end at {seconds_clip} s")
    if knots[0][1] < 0 or knots[-1][1] > seconds_anim + 1e-6:
        raise SystemExit(f"time map leaves the animation's schedule (0 to {seconds_anim} s)")
    return knots


def warp(t, knots):
    """Animation time drawn at clip time t."""
    for (a0, b0), (a1, b1) in zip(knots, knots[1:]):
        if t <= a1:
            return b0 + (b1 - b0) * (t - a0) / (a1 - a0)
    return knots[-1][1]


def push(path, t, seconds, zoom, center=(0.5, 0.5)):
    """Zoom the saved frame at `path` in place by zoom**(t/seconds) about `center` (fractions of the frame)."""
    if zoom == 1.0:
        return
    im = Image.open(path).convert("RGB")
    w, h = im.size
    z = zoom ** (min(max(t, 0.0), seconds) / seconds)
    bw, bh = w / z, h / z
    cx = min(max(center[0] * w, bw / 2), w - bw / 2)
    cy = min(max(center[1] * h, bh / 2), h - bh / 2)
    im.resize((w, h), Image.LANCZOS, box=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)).save(path)
