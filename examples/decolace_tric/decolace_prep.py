#!/usr/bin/env python
"""DeCo-LACE (EMPIAR-11063) tiles for pysmap searches: selection, CTF, templates, downloads.

  select : download the cisTEM project database of each lamella, read CTF (DEFOCUS1/2/ANGLE,
           pixel, kV, Cs) and the deposited 2DTM peaks of the LSU (per-tile count with SNR >= 7.75),
           rank tiles by LSU count (cytoplasm-rich first), compute the template potentials once,
           write one .par per tile and target, and the job lists.
  fetch  : download the aligned micrographs (+ masks) in priority order and write the search
           images (dark area outside the beam filled with Gaussian noise of the beam's mean/SD, as
           in the DeCo-LACE paper). Resumable; run it next to the searches.

  cd $PYSMAP
  python examples/decolace_tric/decolace_prep.py select --dest $WORK --n 50
  python examples/decolace_tric/decolace_prep.py fetch  --dest $WORK
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/11063/data"
LAMELLAE = ["EUC_Lamella1", "EUC_Lamella2", "EUC_Lamella3", "EUC_Lamella4",
            "FFF_Lamella1", "FFF_Lamella2", "FFF_Lamella3", "FFF_Lamella4"]
KIT = Path(__file__).resolve().parent


# ------------------------------------------------------------------ web

def _retry(fn, *args, tries: int = 6, wait: float = 30.0):
    """Retry on network errors (refused/reset/timeout), not on HTTP errors such as 404."""
    import time
    import urllib.error
    for k in range(tries):
        try:
            return fn(*args)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if k == tries - 1:
                raise
            print(f"  network error ({e}); retry in {wait * (k + 1):.0f} s", flush=True)
            time.sleep(wait * (k + 1))


def listing(url: str) -> list:
    return _retry(_listing, url)


def _listing(url: str) -> list:
    with urllib.request.urlopen(url, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    names = re.findall(r'href="([^"?]+)"', html)
    return [n for n in names if not n.startswith("/") and not n.startswith("http") and n != "../"]


def size_of(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or -1)


def download(url: str, path: Path) -> Path:
    """Resumable download to path (via path.part); skipped if path exists; retried on network errors."""
    return _retry(_download, url, path)


def _download(url: str, path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=300) as r:
        mode = "ab" if start and r.status == 206 else "wb"
        with open(part, mode) as fh:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b)
    part.rename(path)
    return path


# ------------------------------------------------------------------ cisTEM database

def _tables(con):
    return [r[0] for r in con.execute("select name from sqlite_master where type='table'")]


def _cols(con, t):
    return [r[1] for r in con.execute(f"pragma table_info('{t}')")]


def read_db(db: Path, snr_min: float = 7.75) -> tuple:
    """Rows {asset, filename, pixel, kv, cs_mm, df1, df2, angle, fit_A, score, lsu} and a report."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    tabs = {t: _cols(con, t) for t in _tables(con)}
    img = next((t for t, c in tabs.items() if {"IMAGE_ASSET_ID", "FILENAME"} <= set(c)
                and "CTF_ESTIMATION_ID" in c), None)
    ctf = next((t for t, c in tabs.items() if {"DEFOCUS1", "DEFOCUS2", "DEFOCUS_ANGLE"} <= set(c)), None)
    rep = dict(db=str(db), tables=sorted(tabs), image_table=img, ctf_table=ctf)
    if not img or not ctf:
        con.close()
        raise SystemExit(f"{db}: image or CTF table not found; tables: {sorted(tabs)}")
    ic, cc = tabs[img], tabs[ctf]

    def pick(cols, *names):
        return next((n for n in names if n in cols), None)

    fit = pick(cc, "DETECTED_RING_RESOLUTION", "FIT_RESOLUTION")
    score = pick(cc, "SCORE")
    q = (f"select i.IMAGE_ASSET_ID, i.FILENAME, {'i.PIXEL_SIZE' if 'PIXEL_SIZE' in ic else 'c.PIXEL_SIZE'}, "
         f"{'i.VOLTAGE' if 'VOLTAGE' in ic else 'c.VOLTAGE'}, "
         f"{'i.SPHERICAL_ABERRATION' if 'SPHERICAL_ABERRATION' in ic else 'c.SPHERICAL_ABERRATION'}, "
         f"c.DEFOCUS1, c.DEFOCUS2, c.DEFOCUS_ANGLE, {('c.' + fit) if fit else 'NULL'}, "
         f"{('c.' + score) if score else 'NULL'} "
         f"from {img} i join {ctf} c on i.CTF_ESTIMATION_ID = c.CTF_ESTIMATION_ID")
    rows = [dict(asset=r[0], filename=r[1], pixel=r[2], kv=r[3], cs_mm=r[4], df1=r[5], df2=r[6],
                 angle=r[7], fit_A=r[8], score=r[9], lsu=None) for r in con.execute(q)]
    # deposited 2DTM peaks: TEMPLATE_MATCH_PEAK_LIST_<id> + a table mapping TEMPLATE_MATCH_ID -> IMAGE_ASSET_ID
    peak_tabs = {t: c for t, c in tabs.items() if "PEAK" in t.upper() and "PEAK_HEIGHT" in c}
    mapt = next((t for t, c in tabs.items() if {"TEMPLATE_MATCH_ID", "IMAGE_ASSET_ID"} <= set(c)), None)
    counts = {}
    if peak_tabs and mapt:
        tm2img = dict(con.execute(f"select TEMPLATE_MATCH_ID, IMAGE_ASSET_ID from {mapt}"))
        for t in peak_tabs:
            m = re.search(r"(\d+)$", t)
            if not m or int(m.group(1)) not in tm2img:
                continue
            n = con.execute(f"select count(*) from {t} where PEAK_HEIGHT >= ?", (snr_min,)).fetchone()[0]
            a = tm2img[int(m.group(1))]
            counts[a] = max(counts.get(a, 0), int(n))
    for r in rows:
        r["lsu"] = counts.get(r["asset"]) if counts else None
    rep.update(n_images=len(rows), peak_tables=len(peak_tabs), match_table=mapt,
               images_with_lsu_counts=len(counts))
    con.close()
    return rows, rep


# ------------------------------------------------------------------ .par helpers

def set_keys(lines, values: dict) -> list:
    out, done = [], set()
    for ln in lines:
        t = ln.split()
        if t and t[0] in values:
            if t[0] not in done:
                out.append(f"{t[0]} {values[t[0]]}")
                done.add(t[0])
            continue
        out.append(ln)
    out += [f"{k} {v}" for k, v in values.items() if k not in done]
    return out


def check_par(par: Path) -> None:
    """Read back with the port's reader (v2.1 rule: a key matches anywhere in a line)."""
    from smap import io as sio
    p, _ = sio.read_params_file(par)
    given = {t[0]: " ".join(t[1:]) for t in (ln.split() for ln in par.read_text().splitlines())
             if t and "#" not in t[0]}
    for f in sio._SEARCH_GLOBAL_STRING_FIELDS:
        v, d = p.get(f), sio.SEARCH_GLOBAL_DEFAULTS.get(f)
        if v != d and str(v) != given.get(f):
            raise SystemExit(f"{par}: key '{f}' reads as '{v}'; a path contains the key name")


def ungz(path: Path, out_dir: Path) -> Path:
    """A model stored as .cif.gz in the repository, unpacked once into out_dir."""
    import gzip
    path = Path(path).expanduser()
    if path.suffix != ".gz":
        return path
    out = Path(out_dir) / path.stem
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(gzip.decompress(path.read_bytes()))
    return out


def make_sp(template_par: Path, model: Path, name: str, pixel: float, out_dir: Path) -> Path:
    """Template potential once (as search_global would write it), at the tiles' pixel size."""
    from smap import io as sio
    from smap.structure import calculate_sp
    sp = out_dir / f"{name}_SP.mrc"
    if sp.exists():
        return sp
    p, _ = sio.read_params_file(template_par)
    model = ungz(model, out_dir)
    p.update(structureFile=str(model), modelName=name, aPerPix=pixel, outputDir=str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    _, fn = calculate_sp(p, out_dir)
    ep = out_dir / f"{name}_EP.mrc"
    if ep.exists():
        ep.unlink()                                             # not needed; 0.5 GB
    return Path(fn)


# ------------------------------------------------------------------ commands

def cmd_select(a) -> int:
    dest = Path(a.dest).expanduser().resolve()
    rows = []
    for lam in a.lamella:
        names = listing(f"{a.base}/{lam}/")
        dbs = [n for n in names if n.endswith(".db")]
        if not dbs:
            print(f"{lam}: no .db in {a.base}/{lam}/ ({names[:10]})")
            continue
        url = f"{a.base}/{lam}/{dbs[0]}"
        sz = size_of(url)
        if sz > a.max_db_gb * 1e9:
            print(f"{lam}: {dbs[0]} is {sz / 1e9:.1f} GB > --max-db-gb; skipped")
            continue
        print(f"{lam}: {dbs[0]} ({sz / 1e6:.0f} MB)", flush=True)
        db = download(url, dest / lam / dbs[0])
        r, rep = read_db(db)
        print(f"  {rep['n_images']} images with CTF; LSU peak tables {rep['peak_tables']}, "
              f"images with counts {rep['images_with_lsu_counts']}")
        (dest / lam / "db_report.json").write_text(json.dumps(rep, indent=1))
        for x in r:
            x["lamella"] = lam
        rows += r
    if not rows:
        raise SystemExit("no tiles")
    # per-tile pixel size (aPerPix = aPerPix_search = the tile's own value, no resampling); the
    # template potential is computed once at the median, so tiles must agree within --pixel-tol
    pixel = float(np.median([float(r["pixel"]) for r in rows]))
    far = [r for r in rows if abs(float(r["pixel"]) / pixel - 1) > a.pixel_tol]
    if far:
        print(f"excluded {len(far)} tiles whose pixel size differs from {pixel:.4f} A by more than "
              f"{100 * a.pixel_tol:g}%: {sorted({round(float(r['pixel']), 4) for r in far})}")
        rows = [r for r in rows if r not in far]
    spread = sorted({round(float(r["pixel"]), 4) for r in rows})
    ok = [r for r in rows if r["df1"] and r["df1"] > 0 and (r["fit_A"] is None or r["fit_A"] <= a.max_fit_A)]
    have_counts = any(r["lsu"] is not None for r in ok)
    if have_counts:
        ok.sort(key=lambda r: (-(r["lsu"] or 0), -(r["score"] or 0)))
    else:
        print("WARNING: no deposited LSU peaks found; tiles ranked by CTF score (may include nucleus)")
        ok.sort(key=lambda r: -(r["score"] or 0))
    sel = ok[:a.n]

    # masks: names from the Images listing of each lamella
    masks = {}
    for lam in sorted({r["lamella"] for r in sel}):
        try:
            masks[lam] = [n for n in listing(f"{a.base}/{lam}/Assets/Images/") if n.endswith("_mask.mrc")]
        except Exception as e:                                  # pragma: no cover
            print(f"{lam}: Images listing failed ({e}); masks guessed")
            masks[lam] = []
    for r in sel:
        fn = Path(str(r["filename"]).replace("\\", "/")).name
        stem = Path(fn).stem
        base = stem[:-2] if stem.endswith("_0") else stem
        ms = masks.get(r["lamella"], [])
        cand = [m for m in ms if m in (f"{stem}_mask.mrc", f"{base}_mask.mrc")] or \
               [m for m in ms if m.startswith(base + "_")]
        r.update(stem=f"{r['lamella']}_{stem}", image_url=f"{a.base}/{r['lamella']}/Assets/Images/{fn}",
                 mask_url=(f"{a.base}/{r['lamella']}/Assets/Images/{cand[0]}" if cand else ""),
                 search_image=str(dest / "images" / f"{r['lamella']}_{stem}.mrc"))

    with open(dest / "tiles.csv", "w", newline="") as fh:
        keys = ["stem", "lamella", "asset", "lsu", "df1", "df2", "angle", "fit_A", "score", "pixel", "kv",
                "cs_mm", "image_url", "mask_url", "search_image"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sel)

    # templates (once) and .par files
    models = dest / "models"
    targets = {"TRiC": (KIT / "tric_decolace.par", Path(a.tric).expanduser(), "TRiC_closed_mouse")}
    if a.lsu_control:
        targets["LSU_control"] = (KIT / "tric_decolace.par", Path(a.lsu).expanduser(), "LSU_6EK0")
    sps = {}
    for t, (tp, model, name) in targets.items():
        print(f"template potential {name} at {pixel} A ...", flush=True)
        sps[t] = make_sp(tp, model, name, pixel, models)
    for t, (tp, _, name) in targets.items():
        pdir = dest / f"pars_{t}"
        pdir.mkdir(parents=True, exist_ok=True)
        jobs = []
        todo = sel[:a.lsu_control] if t == "LSU_control" else sel
        for r in todo:
            lines = set_keys(Path(tp).read_text().splitlines(), {
                "imageFile": r["search_image"], "structureFile": sps[t], "modelName": name,
                "defocus": f"{float(r['df1']):.1f} {float(r['df2']):.1f} {float(r['angle']):.2f}",
                "aPerPix": f"{float(r['pixel']):.4f}", "aPerPix_search": f"{float(r['pixel']):.4f}",
                "V_acc": f"{float(r['kv']) * 1000:.1f}",
                "Cs": f"{float(r['cs_mm']) * 1e-3:.6f}", "outputDir": dest / f"results_{t}" / r["stem"]})
            par = pdir / f"{r['stem']}.par"
            par.write_text("\n".join(lines) + "\n")
            check_par(par)
            jobs.append(str(par))
        (dest / f"jobs_{t}.txt").write_text("\n".join(jobs) + "\n")
    lsu = [r["lsu"] for r in sel if r["lsu"] is not None]
    kv = sorted({float(r["kv"]) for r in sel})
    cs = sorted({float(r["cs_mm"]) for r in sel})
    print(f"per-tile settings: pixel {spread} A, {kv} kV, Cs {cs} mm, defocus {min(float(r['df1']) for r in sel):.0f}-"
          f"{max(float(r['df1']) for r in sel):.0f} A")
    print(f"{len(sel)} tiles selected from {len(rows)} ({len(ok)} with CTF fit <= {a.max_fit_A} A); "
          f"pixel {pixel} A; " + (f"deposited LSU per tile {min(lsu)}-{max(lsu)}" if lsu else "no LSU counts"))
    print(f"tiles.csv, pars_*/, jobs_*.txt in {dest}")
    return 0


def fill_outside(im: np.ndarray, mask: np.ndarray | None, seed: int = 0) -> tuple:
    """Fill the unilluminated area with Gaussian noise of the illuminated mean/SD."""
    if mask is None or mask.shape != im.shape:
        return im, "no mask"
    m = np.clip(mask.astype(np.float64), 0, 1)
    inside = m > 0.5
    if inside.sum() < 1000:
        return im, "mask nearly empty"
    mu, sd = float(im[inside].mean()), float(im[inside].std())
    out_sd = float(im[~inside].std()) if (~inside).any() else 0.0
    if out_sd > 0.5 * sd:
        return im, "already filled"
    noise = np.random.default_rng(seed).normal(mu, sd, im.shape)
    return (m * im + (1 - m) * noise).astype(np.float32), f"filled {100 * (1 - inside.mean()):.0f}%"


def candidates(url: str, available) -> list:
    """The database names the latest alignment (``*_2.mrc``); EMPIAR may only hold ``*_0.mrc``."""
    name = Path(url).name
    alts = [name] + [re.sub(r"_\d+\.mrc$", f"_{k}.mrc", name) for k in (0, 1, 2)]
    alts = list(dict.fromkeys(alts))
    if available:
        alts = [n for n in alts if n in available] or alts
    return [url.rsplit("/", 1)[0] + "/" + n for n in alts]


def cmd_fetch(a) -> int:
    from smap import io as sio
    dest = Path(a.dest).expanduser().resolve()
    rows = list(csv.DictReader(open(dest / "tiles.csv")))
    listings, failed = {}, []
    for i, r in enumerate(rows, 1):
        out = Path(r["search_image"])
        if out.exists():
            continue
        lam = r["lamella"]
        if lam not in listings:
            try:
                listings[lam] = set(listing(r["image_url"].rsplit("/", 1)[0] + "/"))
            except Exception as e:                              # pragma: no cover
                print(f"{lam}: Images listing failed ({e})", flush=True)
                listings[lam] = set()
        raw = dest / lam / "Images"
        img, used = None, None
        for url in candidates(r["image_url"], listings[lam]):
            try:
                img = download(url, raw / Path(url).name)
                used = Path(url).name
                break
            except Exception as e:
                print(f"[{i}/{len(rows)}] {Path(url).name}: {e}", flush=True)
        if img is None:
            failed.append(r["stem"])
            print(f"[{i}/{len(rows)}] {r['stem']}: SKIPPED (no deposited image)", flush=True)
            continue
        base = re.sub(r"_\d+$", "", Path(used).stem)
        masks = [m for m in listings[lam] if m.endswith("_mask.mrc") and (m.startswith(Path(used).stem) or m.startswith(base + "_"))]
        msk = None
        if masks:
            try:
                msk = download(r["image_url"].rsplit("/", 1)[0] + "/" + sorted(masks)[0], raw / sorted(masks)[0])
            except Exception as e:                              # pragma: no cover
                print(f"  mask: {e}", flush=True)
        try:
            im = sio.mr(img).astype(np.float32)
            m = sio.mr(msk) if msk else None
            f, how = fill_outside(im, m, seed=i)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            sio.mw(np.asarray(f, dtype=np.float32), tmp, float(r["pixel"]))
            os.replace(tmp, out)
        except Exception as e:
            failed.append(r["stem"])
            print(f"[{i}/{len(rows)}] {r['stem']}: SKIPPED ({e})", flush=True)
            continue
        note = "" if used == Path(r["image_url"]).name else f" (deposited as {used})"
        print(f"[{i}/{len(rows)}] {r['stem']}: {im.shape[1]}x{im.shape[0]} px, {how}{note}", flush=True)
    (dest / "fetch_failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    print(f"fetch finished: {len(failed)} tiles without an image (fetch_failed.txt)", flush=True)
    return 0


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repository root (smap)                           # run from the Py_smap_v2.1 clone
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--dest", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--lamella", nargs="+", default=LAMELLAE)
    s.add_argument("--base", default=BASE)
    s.add_argument("--max-db-gb", type=float, default=3.0)
    s.add_argument("--max-fit-A", type=float, default=8.0)
    s.add_argument("--pixel-tol", type=float, default=0.001, help="relative pixel-size tolerance")
    s.add_argument("--tric", default=str(KIT / "models" / "7NVL_mouse.cif.gz"))
    s.add_argument("--lsu", default="examples/ribosome_in_situ/model/6ek0_LSU.cif")
    s.add_argument("--lsu-control", type=int, default=2, help="LSU control on the first N tiles (0: none)")
    f = sub.add_parser("fetch")
    f.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    return cmd_select(a) if a.cmd == "select" else cmd_fetch(a)


if __name__ == "__main__":
    raise SystemExit(main())
