#!/usr/bin/env python
"""Everything in one page: $WORK/report/index.html.

Collects what exists (nothing is searched): the status table, the TRiC report(s) at both cutoffs,
the ribosome control against the authors, the picked TRiC templates for closed and open state on
the same tiles (tools/search_figures.py: poses on the micrograph and template/patch pairs), and the
FAQ. Re-run it any time; figures are made only once per search.

  cd $PYSMAP
  python examples/decolace_tric/tric_summary.py --dest $WORK
  xdg-open $WORK/report/index.html
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CSS = """
:root { --ink:#1f2328; --ink2:#59636e; --rule:#d8dee4; --bg:#ffffff; --panel:#f6f8fa; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) { :root { --ink:#e6edf3; --ink2:#9198a1; --rule:#30363d; --bg:#0d1117;
  --panel:#161b22; --accent:#58a6ff; } }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1500px; margin:0 auto; padding:16px; }
nav { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--rule); padding:8px 16px;
  display:flex; flex-wrap:wrap; gap:14px; z-index:2; }
nav a { color:var(--accent); text-decoration:none; font-size:14px; }
h1 { font-size:22px; margin:8px 0 2px; } h2 { font-size:19px; margin:28px 0 8px; border-top:1px solid var(--rule);
  padding-top:16px; } h3 { font-size:16px; margin:18px 0 6px; } h4 { font-size:15px; margin:14px 0 4px; }
.meta { color:var(--ink2); font-size:13px; }
pre, code { font:12.5px/1.45 ui-monospace, monospace; } pre { background:var(--panel); padding:10px; overflow-x:auto;
  border-radius:6px; } p code, li code { background:var(--panel); padding:1px 4px; border-radius:4px; }
img { max-width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; background:#fff; }
.pair { display:grid; grid-template-columns:1fr 1fr; gap:12px; } @media (max-width:800px) { .pair { grid-template-columns:1fr; } }
table { border-collapse:collapse; font-size:13px; margin:6px 0 12px; } td, th { border:1px solid var(--rule);
  padding:3px 8px; text-align:right; } th { background:var(--panel); } td:first-child, th:first-child { text-align:left; }
.note { color:var(--ink2); font-size:13px; }
"""


def esc(s) -> str:
    return html.escape(str(s))


def md_to_html(text: str) -> str:
    """Small Markdown subset: headings, fenced code, lists, pipe tables, inline code and bold."""
    def inline(s: str) -> str:
        s = esc(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + esc("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            n = min(len(m.group(1)) + 1, 4)
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif ln.startswith("|"):
            rows_ = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows_.append(cells)
                i += 1
            t = "<table>" + "".join(("<tr>" + "".join(f"<{'th' if k == 0 else 'td'}>{inline(c)}</{'th' if k == 0 else 'td'}>"
                                                     for c in r) + "</tr>") for k, r in enumerate(rows_)) + "</table>"
            out.append(t)
            continue
        elif re.match(r"^\s*- ", ln):
            items = []
            while i < len(lines) and (re.match(r"^\s*- ", lines[i]) or (lines[i].startswith("  ") and items)):
                if re.match(r"^\s*- ", lines[i]):
                    items.append(re.sub(r"^\s*- ", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            continue
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.strip():
            para = [ln.strip()]
            while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|```|\||\s*- |---)", lines[i + 1]):
                i += 1
                para.append(lines[i].strip())
            out.append("<p>" + inline(" ".join(para)) + "</p>")
        i += 1
    return "\n".join(out)


def csv_table(path: Path, cols=None) -> str:
    if not path.exists():
        return ""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return ""
    cols = [c for c in (cols or rows[0].keys()) if c in rows[0]]
    return ("<table><tr>" + "".join(f"<th>{esc(c)}</th>" for c in cols) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{esc(r[c])}</td>" for c in cols) + "</tr>" for r in rows) + "</table>")


def img(page_dir: Path, p: Path, caption: str = "") -> str:
    rel = os.path.relpath(p, page_dir)
    cap = f"<div class='note'>{esc(caption)}</div>" if caption else ""
    return f"<figure style='margin:6px 0'><a href='{esc(rel)}'><img src='{esc(rel)}' loading='lazy'></a>{cap}</figure>"


def finished(f: Path) -> bool:
    return (f / ".done").exists() or (f / "particles.csv").exists()


def tile_key(name: str) -> str:
    m = re.search(r"_(\d{5})_", name)
    return m.group(1) if m else name


def make_figures(repo: Path, result: Path, out: Path, cutoff: float, label: str, log) -> bool:
    """tools/search_figures.py once per search (redone if the search is newer than the figures)."""
    if (out / "summary.png").exists() and (out / "summary.png").stat().st_mtime > (result / "particles.csv").stat().st_mtime:
        return True
    if out.exists():
        shutil.rmtree(out)
    cmd = [sys.executable, str(repo / "tools" / "search_figures.py"), str(result), "--cutoff", f"{cutoff:g}",
           "--out", str(out), "--label", label]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  figures failed for {result.name}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--strict", type=float, default=7.8, help="cutoff for the picked-template figures")
    ap.add_argument("--tiles", nargs="*", default=None, help="tiles for closed/open figures (default: all with open done)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    d = Path(a.dest).expanduser().resolve()
    kit = Path(__file__).resolve().parent
    repo = Path(__file__).resolve().parents[2]
    rep = d / "report"
    rep.mkdir(exist_ok=True)
    log = lambda m: print(m, flush=True)  # noqa: E731
    secs = []

    # status
    st = subprocess.run([sys.executable, str(kit / "status.py")], capture_output=True, text=True)
    secs.append(("status", "Status of every search", f"<pre>{esc(st.stdout.replace(str(Path.home()), '~'))}</pre>"))

    # TRiC reports
    body = ""
    for tag, title in (("", "Refinement threshold 7 (as searched)"), ("_opt6.5", "Refinement threshold 6.5 (re-refined)")):
        png, tab = rep / f"tric_report{tag}.png", rep / f"tric_report{tag}.csv"
        if png.exists():
            body += f"<h3>{esc(title)}</h3>" + img(rep, png)
            body += csv_table(tab, ["tile", "clean", "clean_strict", "noise_measured", "noise_strict", "open_clean",
                                    "open_strict", "decoy_clean", "decoy_strict", "on_lsu", "chance_on_lsu"])
    body += ("<p class='note'>clean = hits outside image-artifact zones at the per-tile cutoff (7.42); *_strict = at the "
             f"noise-calibrated cutoff ({a.strict:g}). noise = the same TRiC search on a phase-randomised copy of the tile.</p>")
    secs.append(("tric", "TRiC: closed, open, decoy and noise", body))

    # picked templates, closed vs open
    body = ""
    open_dir, closed_dir = d / "results_TRiC_open", d / "results_TRiC"
    tiles = a.tiles
    if tiles is None:
        tiles = sorted({tile_key(p.name) for p in open_dir.glob("*") if p.is_dir() and finished(p)})
    for k in tiles:
        c = next((p for p in closed_dir.glob(f"*_{k}_*") if finished(p)), None)
        o = next((p for p in open_dir.glob(f"*_{k}_*") if finished(p)), None)
        if not c or not o:
            continue
        cols = []
        for res, lab, sub in ((c, "TRiC closed", "closed"), (o, "TRiC open", "open")):
            out = rep / "figs" / f"{k}_{sub}"
            ok = True if a.no_figures else make_figures(repo, res, out, a.strict, lab, log)
            cell = f"<h4>{esc(lab)}</h4>"
            if ok and (out / "poses_pairs.png").exists():
                cell += img(rep, out / "poses_pairs.png", "micrograph patch | template at the found pose (top candidates)")
                cell += img(rep, out / "summary.png", f"poses at >= {a.strict:g} | z map | landscape")
            else:
                cell += "<p class='note'>no figure (see the terminal)</p>"
            cols.append(f"<div>{cell}</div>")
        body += f"<h3>Tile {esc(k)}</h3><div class='pair'>{''.join(cols)}</div>"
        log(f"tile {k}: closed/open figures")
    if not body:
        body = "<p class='note'>no tile has both a closed and an open search yet</p>"
    secs.append(("poses", "TRiC templates at the found poses: closed vs open, same tiles", body))

    # ribosome control
    body = ""
    for p in sorted(rep.glob("lsu_gap*.png")):
        body += f"<h3>{esc(p.stem)}</h3>" + img(rep, p)
    secs.append(("lsu", "Ribosome control vs the authors' list", body or "<p class='note'>run lsu_gap.py</p>"))

    # FAQ
    faq = repo / "docs" / "FAQ.md"
    secs.append(("faq", "FAQ and common issues", md_to_html(faq.read_text()) if faq.exists() else ""))

    nav = "".join(f"<a href='#{k}'>{esc(t.split(':')[0])}</a>" for k, t, _ in secs)
    page = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>TRiC results</title><style>{CSS}</style></head><body><nav>{nav}</nav><main>"
            f"<h1>TRiC on DeCo-LACE and ribosome control</h1><p class='meta'>Updated {time.strftime('%Y-%m-%d %H:%M')} &middot; "
            f"regenerate: <code>python examples/decolace_tric/tric_summary.py --dest {esc(str(d).replace(str(Path.home()), '~'))}</code></p>"
            + "".join(f"<h2 id='{k}'>{esc(t)}</h2>{b}" for k, t, b in secs) + "</main></body></html>")
    (rep / "index.html").write_text(page)
    log(f"-> {rep / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
