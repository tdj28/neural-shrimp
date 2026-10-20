"""Resolve every cited bibliography entry of an article against public metadata (third review, 20 Oct 2026).

For each entry cited in MAIN: a DOI is looked up at Crossref (falling back to DataCite, which registers arXiv DOIs);
an arXiv eprint is looked up through the arXiv API. Title (normalized word overlap), first author's family name, year,
volume and first page are compared with the bib entry, and every difference is reported. Entries with neither
identifier are listed for a manual check. Network reads only; nothing is sent but the identifiers.

Usage: python tools/bib_resolve.py --bib BIB --main MAIN --out docs/bib/resolver-YYYY-MM-DD
"""
import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "neural-shrimp-bib-audit/1.0 (research bibliography check)"}


def fetch(url, timeout=60, tries=5):
    """GET with backoff on rate limits (HTTP 429/503) and timeouts."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            code = getattr(e, "code", None)
            if k == tries - 1 or (code is not None and code not in (429, 503)):
                raise
            time.sleep(10 * (k + 1))


def norm(s):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = re.sub(r"[{}$\\^_]", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def overlap(a, b):
    A, B = set(norm(a)), set(norm(b))
    return len(A & B) / max(1, len(A | B))


def field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}|\"[^\"]*\"|\d+)", body, re.I)
    if not m:
        return None
    v = m.group(1).strip()
    return v[1:-1].strip() if v[0] in "{\"" else v


def first_family(authors):
    if not authors:
        return None
    a = re.split(r"\s+and\s+", authors)[0].strip()
    fam = a.split(",")[0] if "," in a else a.split()[-1]
    return " ".join(norm(fam))


def crossref(doi):
    try:
        m = json.loads(fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
        auth = m.get("author") or []
        return {"source": "crossref", "title": (m.get("title") or [""])[0],
                "family": " ".join(norm(auth[0].get("family", ""))) if auth else None,
                "year": ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0],
                "volume": m.get("volume"), "page": (m.get("page") or "").split("-")[0] or m.get("article-number"),
                "container": (m.get("container-title") or [""])[0]}
    except Exception as e:
        try:
            d = json.loads(fetch("https://api.datacite.org/dois/" + urllib.parse.quote(doi)))["data"]["attributes"]
            cr = d.get("creators") or []
            return {"source": "datacite", "title": (d.get("titles") or [{}])[0].get("title", ""),
                    "family": " ".join(norm(cr[0].get("familyName") or cr[0].get("name", "").split(",")[0])) if cr else None,
                    "year": d.get("publicationYear"), "volume": None, "page": None, "container": d.get("publisher")}
        except Exception as e2:
            return {"error": f"crossref: {e}; datacite: {e2}"}


def arxiv(eid):
    try:
        x = fetch("http://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(eid))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        e = ET.fromstring(x).find("a:entry", ns)
        if e is None or e.find("a:title", ns) is None:
            return {"error": "no arXiv entry"}
        names = [n.text for n in e.findall("a:author/a:name", ns)]
        return {"source": "arxiv", "title": re.sub(r"\s+", " ", e.find("a:title", ns).text).strip(),
                "family": " ".join(norm(names[0].split()[-1])) if names else None,
                "year": int(e.find("a:published", ns).text[:4]), "authors": names}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bib", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bib = Path(args.bib).read_text()
    tex = Path(args.main).read_text()
    cited = []
    for m in re.finditer(r"\\(?:cite|onlinecite|citet|citep)\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            if k.strip() and k.strip() not in cited:
                cited.append(k.strip())
    entries = {k.strip(): (ty, b) for ty, k, b in re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S)}
    rows = []
    for k in cited:
        ty, b = entries.get(k, (None, ""))
        rec = {"key": k, "type": ty, "title": field(b, "title"), "author_family": first_family(field(b, "author")),
               "year": field(b, "year"), "volume": field(b, "volume"), "pages": field(b, "pages"),
               "doi": field(b, "doi"), "eprint": field(b, "eprint"), "venue": field(b, "journal") or field(b, "booktitle"),
               "checks": []}
        if ty is None:
            rec["checks"].append({"error": "cited key missing from the bib"})
        for kind, ident in (("doi", rec["doi"]), ("eprint", rec["eprint"])):
            if not ident:
                continue
            got = crossref(ident) if kind == "doi" else arxiv(ident)
            time.sleep(0.4 if kind == "doi" else 6.0)
            if "error" in got:
                rec["checks"].append({"id": ident, "kind": kind, "error": got["error"]})
                continue
            diffs = []
            ov = overlap(rec["title"], got["title"])
            if ov < 0.6:
                diffs.append(f"title overlap {ov:.2f}: record '{got['title'][:120]}'")
            if got.get("family") and rec["author_family"] and got["family"] != rec["author_family"]:
                diffs.append(f"first author '{got['family']}' vs bib '{rec['author_family']}'")
            if kind == "doi" and got.get("year") and rec["year"] and str(got["year"]) != str(rec["year"]):
                diffs.append(f"year {got['year']} vs bib {rec['year']}")
            if kind == "doi" and got.get("volume") and rec["volume"] and str(got["volume"]) != str(rec["volume"]):
                diffs.append(f"volume {got['volume']} vs bib {rec['volume']}")
            if kind == "doi" and got.get("page") and rec["pages"]:
                bp = re.split(r"[-\u2013]+", rec["pages"])[0].strip()
                if str(got["page"]).strip() != bp:
                    diffs.append(f"first page {got['page']} vs bib {bp}")
            rec["checks"].append({"id": ident, "kind": kind, "source": got["source"], "record_title": got["title"],
                                  "record_year": got.get("year"), "record_container": got.get("container"),
                                  "differences": diffs})
        if not rec["doi"] and not rec["eprint"]:
            rec["checks"].append({"manual": "no DOI or eprint; check by hand"})
        rows.append(rec)
        print(k, "->", "; ".join(str(d) for c in rec["checks"] for d in (c.get("differences") or [c.get("error") or c.get("manual") or "ok"])), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
    flagged = [r for r in rows if any(c.get("differences") or c.get("error") or c.get("manual") for c in r["checks"])]
    md = [f"# Bibliography resolver report ({len(rows)} cited entries; {len(flagged)} flagged for a manual look)", ""]
    for r in flagged:
        md.append(f"- `{r['key']}`: " + "; ".join(str(d) for c in r["checks"] for d in (c.get("differences") or [c.get("error") or c.get("manual") or "ok"])))
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"{len(rows)} entries, {len(flagged)} flagged")


if __name__ == "__main__":
    main()
