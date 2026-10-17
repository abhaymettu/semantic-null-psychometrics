"""Curate item text + item metadata for DASS-42 and IPIP-NEO-300 from the pinned sources.

Run:  uv run --with xlrd==2.0.2 python -I scripts/build_items.py
Needs the cache filled by scripts/fetch_data.py. Writes data/items/*.csv and
prints the coverage diagnostic. Item text is copied verbatim from the primary
source (DASS: the UNSW response form; IPIP: Johnson's OSF item key). The only
normalisation is whitespace collapsing of wrapped lines; the secondary source is
compared after stripping a trailing period and HTML entities.
"""
import csv, html, os, re, sys, zipfile
from pathlib import Path

import xlrd

REPO = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEMNULL_CACHE", REPO / "data" / "cache"))
RAW = CACHE / "raw"
OUT = Path(os.environ.get("SEMNULL_OUT", REPO / "data")) / "items"  # override only for rebuild checks

# DASS scoring template (Dass_template.pdf) covers one 21-item page; the form note says
# "For the full (42-item) DASS, apply template to both sides", so items 22-42 repeat it.
DASS_TEMPLATE_PAGE = "SADADSASADSSDSADDSAAD"


def norm(s):
    return " ".join(html.unescape(s).replace("’", "'").split()).rstrip(".").strip()


def dass_form_items():
    """Parse the numbered items from the UNSW Dass42.pdf (pdftotext -layout)."""
    import subprocess
    txt = subprocess.run(["pdftotext", "-layout", str(RAW / "dass42_pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    items, cur = {}, None
    for line in txt.splitlines():
        m = re.match(r"^\s*(\d{1,2})\s{2,}(.*?)\s{2,}0\s+1\s+2\s+3\s*$", line)
        if m:
            cur = int(m.group(1))
            items[cur] = m.group(2).strip()
        elif cur and line.strip() and re.match(r"^\s{4,}\S", line):
            items[cur] += " " + line.strip()  # wrapped continuation directly under an item
        else:
            cur = None
    return {k: " ".join(v.split()) for k, v in items.items()}


def dass_codebook_items():
    cb = (CACHE / "x" / "dass" / "DASS_data_21.02.19" / "codebook.txt")
    if not cb.exists():
        with zipfile.ZipFile(RAW / "dass_raw_zip") as z:
            z.extract("DASS_data_21.02.19/codebook.txt", CACHE / "x" / "dass")
    out = {}
    for line in cb.read_text(errors="replace").splitlines():
        m = re.match(r"^Q(\d+)\t(.+)$", line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def build_dass():
    form, cb = dass_form_items(), dass_codebook_items()
    rows = []
    for k in range(1, 43):
        text = form.get(k)
        rows.append({
            "item_id": f"DASS_Q{k:02d}", "position": k, "data_column": f"Q{k}A",
            "scale": "DAS"["DAS".index(DASS_TEMPLATE_PAGE[(k - 1) % 21])],
            "key": 1, "text": text or "",
            "text_source": "UNSW Dass42.pdf" if text else "UNRESOLVED",
            "secondary_match": bool(text) and norm(text) == norm(cb.get(k, "")),
            "secondary_text": norm(cb.get(k, "")),
        })
    return rows


def ipip_questionnaire_items():
    """Numbered item lines from 'IPIP-NEO-300 Questionnaire Suitable for Printing.docx'."""
    with zipfile.ZipFile(RAW / "ipipneo300_questionnaire_docx") as z:
        xml = z.read("word/document.xml").decode("utf8")
    paras = [html.unescape(re.sub(r"<[^>]+>", "", p)) for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
    cells = [" ".join(p.split()) for p in paras if p.strip()]
    return cells


def build_ipip():
    sh = xlrd.open_workbook(str(RAW / "ipipneo_itemkey_xls")).sheet_by_index(0)
    head = sh.row_values(0)
    assert head == ["Full#", "Short#", "Sign", "Key", "Facet", "Item"], head
    cells = ipip_questionnaire_items()
    # questionnaire numbers each item: the cell after "<n>" is item n's text (order-aware check)
    q = {int(a): norm(b) for a, b in zip(cells, cells[1:]) if a.isdigit()}
    rows = []
    for r in range(1, sh.nrows):
        full, short, sign, key, facet, item = sh.row_values(r)
        k = int(full)
        assert sign[0] in "+-" and sign[1:] == key, (k, sign, key)
        rows.append({
            "item_id": f"IPIP_I{k:03d}", "position": k, "data_column": f"I{k}",
            "domain": key[0], "facet_code": key, "facet_name": facet,
            "key": 1 if sign[0] == "+" else -1,
            "ipip_neo_120_position": int(short) if str(short).strip() not in ("", "0") else "",
            "text": item.strip(), "text_source": "OSF ycvdk IPIP-NEO-ItemKey.xls",
            "secondary_match": q.get(k) == norm(item),
        })
    rows.sort(key=lambda r: r["position"])
    assert [r["position"] for r in rows] == list(range(1, 301))
    return rows


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    report = []
    for name, rows in (("dass42", build_dass()), ("ipipneo300", build_ipip())):
        write(rows, OUT / f"{name}_items.csv")
        n = len(rows)
        have = sum(1 for r in rows if r["text"])
        match = sum(1 for r in rows if r["secondary_match"])
        report.append(f"{name}: items={n} text_resolved={have} ({100*have/n:.1f}%) "
                      f"secondary_source_match={match} ({100*match/n:.1f}%)")
        for r in rows:
            if not r["text"] or not r["secondary_match"]:
                report.append(f"  UNMATCHED {r['item_id']}: {r['text']!r}" +
                              (f" vs {r['secondary_text']!r}" if 'secondary_text' in r else ""))
    print("\n".join(report))
    return 1 if any(r.startswith("  UNMATCHED") for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
