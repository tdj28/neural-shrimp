"""Explainer cards (docs/explainer/plan.md, section 3.1): title overlay, equation, interval, coexistence, steering,
kernels, lessons, limits, delay and end cards, rendered as 1920x1080 PNGs from a JSON list.

A card is {"name": ..., "transparent": false, "stages": K, "blocks": [...]}. The blocks are laid out once, top to
bottom (a block without "y" follows the previous flowing block after "gap" pixels), and stage k draws every block whose
"stage" is at most k, so the stages of a card differ only by what they add; render_figure_shot.py crossfades them at
the reveal times of the shot. Block kinds:

  text   "text", or "runs": [{"text", "font", "color", "size", "dy"}, ...]; "font" (title, body, bold, mono, monob,
         foot, caption), "size", "color", "x", "y" (top), "w" (wrap width), "align" (left or center), "lh" (line
         height over size), "gap", "bullet" (a gold bullet hanging left of "x"), "shadow" (blur radius of a dark
         halo, for transparent cards), "box" (dark rounded backing: {"fill", "alpha", "pad", "radius"})
  math   "tex" (matplotlib mathtext), "size" (pixels of the math font), "x", "y", "w", "align", "color", "gap"
  image  "path" (relative to the repository or absolute), "crop" (fractions [x0, y0, x1, y1]), "x", "y", "w" or "h",
         "plate" (backing color, or null), "pad", "radius", "label" (a footnote under the image)
  rect   "x", "y", "w", "h", "fill", "alpha", "radius", "feather" (Gaussian blur of the edges), "outline"
  line   "xy": [x0, y0, x1, y1], "width", "color"
  ellipse "xy": [x0, y0, x1, y1], "width", "color" (outline), "fill"

Style (plan section 5.1): DejaVu fonts of matplotlib; titles 52 px bold white, body 32 px #C8C8D0, gold bullets
#FFD93F, token strings in DejaVu Sans Mono, footnotes 24 px #9A9AA6, background #0B0B0F. A text containing {arxiv}
gets the --arxiv value (placeholder XXXX.XXXXX until the listing exists). A card that overflows the frame or names a
missing image is refused.

Usage: uv run --no-sync python tools/render_cards.py SPEC.json --out DIR [--only NAME ...] [--arxiv ID]
Writes DIR/<name>-stage<k>.png for each stage (DIR/<name>.png for a one-stage card) and DIR/cards-receipt.json.
"""
import argparse
import functools
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
BG = "#0b0b0f"
GOLD = "#ffd93f"
MARGIN = 120
FONT_FILES = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf",
              "monob": "DejaVuSansMono-Bold.ttf"}
STYLES = {"title": ("bold", 52, "#ffffff"), "body": ("sans", 32, "#c8c8d0"), "bold": ("bold", 32, "#ffffff"),
          "mono": ("mono", 32, GOLD), "monob": ("monob", 32, GOLD), "foot": ("sans", 24, "#9a9aa6"),
          "caption": ("sans", 26, "#9a9aa6")}


def font_path(family):
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / FONT_FILES[family]


@functools.lru_cache(maxsize=None)
def get_font(family, size):
    return ImageFont.truetype(str(font_path(family)), int(round(size)))


def rgba(color, alpha=1.0):
    r, g, b = ImageColor.getrgb(color)[:3]
    return (r, g, b, int(round(255 * alpha)))


def runs_of(block, arxiv):
    """Normalized runs of a text block: dicts with text, family, size, color, dy."""
    fam, size, color = STYLES[block.get("font", "body")]
    size = block.get("size", size)
    color = block.get("color", color)
    raw = block.get("runs") or [{"text": block["text"]}]
    out = []
    for r in raw:
        rf, _, rc = STYLES[r["font"]] if "font" in r else (fam, size, color)
        out.append({"text": r["text"].replace("{arxiv}", arxiv), "family": rf, "size": r.get("size", size),
                    "color": r.get("color", rc), "dy": r.get("dy", 0)})
    return out, size


def wrap(runs, width):
    """Greedy wrap on ASCII spaces (a no-break space keeps words together); returns lines of (token, run, width)."""
    lines = [[]]
    cur = 0.0
    for run in runs:
        f = get_font(run["family"], run["size"])
        for pi, part in enumerate(run["text"].split("\n")):
            if pi:
                lines.append([])
                cur = 0.0
            for tok in re.findall(r"[ ]+|[^ ]+", part):
                tw = f.getlength(tok)
                if tok.startswith(" "):
                    if lines[-1]:
                        lines[-1].append((tok, run, tw))
                        cur += tw
                    continue
                if lines[-1] and cur + tw > width + 0.5:
                    while lines[-1] and lines[-1][-1][0].startswith(" "):
                        cur -= lines[-1].pop()[2]
                    lines.append([])
                    cur = 0.0
                lines[-1].append((tok, run, tw))
                cur += tw
    for ln in lines:
        while ln and ln[-1][0].startswith(" "):
            ln.pop()
    return lines


def render_math(tex, size, color):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(4, 1), dpi=100)
    FigureCanvasAgg(fig)
    fig.text(0.0, 0.2, tex, fontsize=size * 72.0 / 100.0, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.04)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    im.load()
    return im


def load_image(block):
    p = Path(block["path"])
    p = p if p.is_absolute() else ROOT / p
    if not p.exists():
        sys.exit(f"refused: image {p} not found")
    im = Image.open(p).convert("RGBA")
    flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
    flat.alpha_composite(im)
    im = flat.convert("RGB")
    if "crop" in block:
        x0, y0, x1, y1 = block["crop"]
        im = im.crop((round(x0 * im.width), round(y0 * im.height), round(x1 * im.width), round(y1 * im.height)))
    if "w" in block:
        size = (int(block["w"]), int(round(block["w"] * im.height / im.width)))
    else:
        size = (int(round(block["h"] * im.width / im.height)), int(block["h"]))
    return im.resize(size, Image.LANCZOS), p


def layout(card, arxiv):
    """Positions of every block (first pass, independent of stages)."""
    placed = []
    cursor = 100
    for b in card["blocks"]:
        kind = b.get("kind", "text")
        item = {"block": b, "kind": kind, "stage": b.get("stage", 1)}
        if kind == "text":
            runs, size = runs_of(b, arxiv)
            x = b.get("x", MARGIN)
            width = b.get("w", W - MARGIN - x)
            lines = wrap(runs, width)
            lh = b.get("lh", 1.32) * size
            y = b["y"] if "y" in b else cursor + b.get("gap", 24)
            item.update(x=x, y=y, w=width, lines=lines, lh=lh, size=size,
                        ascent=get_font(runs[0]["family"], size).getmetrics()[0])
            height = lh * len(lines)
            item["h"] = height
            if b.get("flow", True):
                cursor = y + height
        elif kind == "math":
            im = render_math(b["tex"], b.get("size", 60), b.get("color", "#ffffff"))
            x = b.get("x", MARGIN)
            width = b.get("w", W - MARGIN - x)
            y = b["y"] if "y" in b else cursor + b.get("gap", 24)
            px = x + (width - im.width) / 2 if b.get("align", "center") == "center" else x
            item.update(im=im, x=px, y=y, h=im.height)
            cursor = y + im.height
        elif kind == "image":
            im, p = load_image(b)
            item.update(im=im, path=str(p), x=b["x"], y=b["y"], h=im.height)
            if b.get("flow"):
                cursor = b["y"] + im.height
        placed.append(item)
    return placed


def check_bounds(card, placed):
    for it in placed:
        if it["kind"] == "text":
            for ln in it["lines"]:
                wl = sum(t[2] for t in ln)
                if wl > it["w"] + 1:
                    sys.exit(f"refused: card {card['name']}: a word wider than its box: {''.join(t[0] for t in ln)!r}")
            bottom = it["y"] + it["h"]
        elif it["kind"] in ("math", "image"):
            if it["x"] < 0 or it["x"] + it["im"].width > W:
                sys.exit(f"refused: card {card['name']}: {it['kind']} block leaves the frame horizontally")
            bottom = it["y"] + it["im"].height
        else:
            continue
        if bottom > H - 10:
            sys.exit(f"refused: card {card['name']}: a {it['kind']} block ends at y={bottom:.0f}, below the frame")


def draw_text_item(canvas, it):
    b = it["block"]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    align = b.get("align", "left")
    for i, ln in enumerate(it["lines"]):
        wl = sum(t[2] for t in ln)
        x = it["x"] + (it["w"] - wl) / 2 if align == "center" else it["x"]
        base = it["y"] + i * it["lh"] + it["ascent"]
        if i == 0 and b.get("bullet"):
            bf = get_font("bold", it["size"])
            d.text((it["x"] - 0.95 * it["size"], base), "•", font=bf, fill=rgba(b.get("bullet_color", GOLD)),
                   anchor="ls")
        for tok, run, tw in ln:
            d.text((x, base + run["dy"]), tok, font=get_font(run["family"], run["size"]), fill=rgba(run["color"]),
                   anchor="ls")
            x += tw
    if "box" in b:
        bx = b["box"]
        pad = bx.get("pad", 20)
        widths = [sum(t[2] for t in ln) for ln in it["lines"]]
        if align == "center":
            x0 = it["x"] + (it["w"] - max(widths)) / 2 - pad
            x1 = x0 + max(widths) + 2 * pad
        else:
            x0, x1 = it["x"] - pad - (1.0 * it["size"] if b.get("bullet") else 0), it["x"] + max(widths) + pad
        back = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(back).rounded_rectangle((x0, it["y"] - pad * 0.6, x1, it["y"] + it["h"] + pad * 0.4),
                                               radius=bx.get("radius", 14),
                                               fill=rgba(bx.get("fill", BG), bx.get("alpha", 0.85)))
        canvas.alpha_composite(back)
    if b.get("shadow"):
        a = layer.getchannel("A").filter(ImageFilter.GaussianBlur(b["shadow"]))
        a = a.point(lambda v: min(255, int(v * 2.2)))
        sh = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        sh.putalpha(a)
        canvas.alpha_composite(sh)
    canvas.alpha_composite(layer)


def draw_rect(canvas, b):
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    box = (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])
    d.rounded_rectangle(box, radius=b.get("radius", 0), fill=rgba(b.get("fill", "#000000"), b.get("alpha", 1.0)),
                        outline=rgba(b["outline"]) if b.get("outline") else None, width=b.get("outline_width", 2))
    if b.get("feather"):
        a = layer.getchannel("A").filter(ImageFilter.GaussianBlur(b["feather"]))
        layer.putalpha(a)
    canvas.alpha_composite(layer)


def draw_image_item(canvas, it):
    b = it["block"]
    im = it["im"]
    pad = b.get("pad", 0)
    x, y = int(round(it["x"])), int(round(it["y"]))
    if b.get("plate"):
        back = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(back).rounded_rectangle((x - pad, y - pad, x + im.width + pad, y + im.height + pad),
                                               radius=b.get("radius", 10), fill=rgba(b["plate"]))
        canvas.alpha_composite(back)
    canvas.paste(im, (x, y))
    if b.get("label"):
        f = get_font("sans", b.get("label_size", 22))
        ImageDraw.Draw(canvas).text((x - pad, y + im.height + pad + 8), b["label"], font=f,
                                    fill=rgba(b.get("label_color", "#9a9aa6")), anchor="la")


def render_card(card, out_dir, arxiv):
    placed = layout(card, arxiv)
    check_bounds(card, placed)
    stages = card.get("stages", 1)
    files = []
    for k in range(1, stages + 1):
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0) if card.get("transparent") else rgba(card.get("bg", BG)))
        for it in placed:
            if it["stage"] > k:
                continue
            b = it["block"]
            if it["kind"] == "text":
                draw_text_item(canvas, it)
            elif it["kind"] == "math":
                canvas.alpha_composite(it["im"], (int(round(it["x"])), int(round(it["y"]))))
            elif it["kind"] == "image":
                draw_image_item(canvas, it)
            elif it["kind"] == "rect":
                draw_rect(canvas, b)
            elif it["kind"] == "line":
                ImageDraw.Draw(canvas).line(b["xy"], fill=rgba(b.get("color", "#9a9aa6")), width=b.get("width", 2))
            elif it["kind"] == "ellipse":
                ImageDraw.Draw(canvas).ellipse(b["xy"], fill=rgba(b["fill"]) if b.get("fill") else None,
                                               outline=rgba(b.get("color", "#9a9aa6")), width=b.get("width", 2))
        name = f"{card['name']}.png" if stages == 1 else f"{card['name']}-stage{k}.png"
        path = out_dir / name
        (canvas if card.get("transparent") else canvas.convert("RGB")).save(path)
        files.append({"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    texts = []
    for it in placed:
        if it["kind"] == "text":
            texts.append({"stage": it["stage"], "text": "".join(r["text"] for r in runs_of(it["block"], arxiv)[0])})
        elif it["kind"] == "math":
            texts.append({"stage": it["stage"], "tex": it["block"]["tex"]})
        elif it["kind"] == "image":
            texts.append({"stage": it["stage"], "image": it["path"], "crop": it["block"].get("crop")})
    return {"name": card["name"], "stages": stages, "files": files, "content": texts}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="JSON list of cards")
    ap.add_argument("--out", required=True, help="output directory for the card PNGs")
    ap.add_argument("--only", nargs="*", help="render only these card names")
    ap.add_argument("--arxiv", default="XXXX.XXXXX", help="arXiv identifier for {arxiv} (placeholder by default)")
    args = ap.parse_args()
    spec_path = Path(args.spec)
    cards = json.loads(spec_path.read_text())
    names = [c["name"] for c in cards]
    if len(set(names)) != len(names):
        sys.exit("refused: duplicate card names in the spec")
    if args.only:
        missing = sorted(set(args.only) - set(names))
        if missing:
            sys.exit(f"refused: no card named {missing}")
        cards = [c for c in cards if c["name"] in args.only]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "cards-receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"cards": {}}
    for card in cards:
        rec = render_card(card, out, args.arxiv)
        receipt["cards"][card["name"]] = rec
        print(f"{card['name']}: {rec['stages']} stage(s) -> {', '.join(Path(f['file']).name for f in rec['files'])}")
    receipt.update(spec=str(spec_path.resolve()), spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                   arxiv=args.arxiv, tool="tools/render_cards.py")
    receipt_path.write_text(json.dumps(receipt, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
