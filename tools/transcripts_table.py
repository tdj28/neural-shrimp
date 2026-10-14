"""LaTeX table from tools/transcripts.py output: for each cell the regime (period or chaotic exponent from the matching
scan), the leading tokens over a fixed number of consecutive steps (periodic cells: whole cycles; a dot-free
separator marks each cycle start) and the mean entropy of the soft token in nats.

Usage: python tools/transcripts_table.py TRANSCRIPTS_DIR STEPS LABELS_JSON
       LABELS_JSON maps "LOG10T,ALPHA" to a short regime label, e.g. {"-0.115,0.071674": "period-8 window"}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_", "#": r"\#", "%": r"\%", "&": r"\&",
           "$": r"\$", "^": r"\^{}", "~": r"\~{}"}


def tex_token(t):
    out = []
    for ch in t:
        if ch == "\n":
            out.append(r"$\hookleftarrow$")
        elif ch == " ":
            out.append(r"\ ")
        elif ch in SPECIAL:
            out.append(SPECIAL[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def main():
    import explore_llm_attractors as ela

    folder, steps, labels = sys.argv[1], int(sys.argv[2]), json.loads(Path(sys.argv[3]).read_text())
    data = json.loads((ela.DATA / "explore" / folder / "transcripts.json").read_text())
    rows = []
    for c in data["cells"]:
        key = f"{c['log10_temperature']},{c['alpha']}"
        toks = [s["token"] for s in c["steps"][:steps]]
        q = c["period"]
        pieces = []
        for k, t in enumerate(toks):
            if q and k and k % q == 0:
                pieces.append(r"\,$\mid$\,")
            pieces.append(tex_token(t))
        ent = sum(s["entropy"] for s in c["steps"]) / len(c["steps"])
        regime = labels.get(key, f"period {q}" if q else "no period")
        rows.append(f"${c['alpha']:.6f}$ & {regime} & {ent:.1f} & \\texttt{{{''.join(pieces)}}} \\\\")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
