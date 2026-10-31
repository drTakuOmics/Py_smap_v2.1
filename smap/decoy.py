"""Target-decoy calibration of the final SMAP score (new; not part of v2.1).

Run the same search twice on the same image with identical settings: once normally
(target) and once with a decoy: a *different protein* of similar mass and size as
structureFile/modelFile (``decoy_label NAME`` records it), or ``decoy_image phase_random``.
A mirror-image template is not a decoy (its projections equal the target's at M R M).
If the decoy cannot produce true hits, the number of decoy candidates at or above a cutoff
estimates the number of false target candidates at that cutoff:

    FDR(c) ~= N_decoy(score >= c) / N_target(score >= c)
    q(candidate) = min over c <= score of FDR(c)        (monotone, as in proteomics)

Several tiles/images can be pooled (repeat --pair). The estimate assumes the decoy's noise
behaves like the target's noise: a different protein should match mass and size (SNR scales
with template signal) and share no domains with the target; the phase-randomised image keeps
the power spectrum but removes all structure (so it tends to under-estimate false positives
from real, non-target density). Report which decoy was used. For identity, omitted-domain
(anchor) searches with reconstruction of the omitted density are the stronger control.

usage:
    python -m smap.decoy --pair TARGET_OUT DECOY_OUT [--pair ...] [--score final|refined]
                         [--out decoy_calibration] [--cutoff 8]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .compare import _matrix

SCORE_FILES = {"final": ("particles.txt", 1), "refined": ("list_final.txt", 0)}


def read_scores(out_dir: Path, score: str = "final") -> np.ndarray:
    name, col = SCORE_FILES[score]
    m = _matrix(Path(out_dir) / name)
    return m[:, col].astype(np.float64) if m.size else np.zeros(0)


def count_ge(scores: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """count(scores >= c) for each cutoff (the RNF213 review uses >=)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    return s.size - np.searchsorted(s, cutoffs, side="left")


def fdr_curve(target: np.ndarray, decoy: np.ndarray, cutoffs: np.ndarray):
    nt = count_ge(target, cutoffs)
    nd = count_ge(decoy, cutoffs)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr = np.where(nt > 0, np.minimum(1.0, nd / np.maximum(nt, 1)), 0.0)
    # q(c) = min FDR over all cutoffs <= c  (cutoffs ascending)
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    return nt, nd, fdr, q


def q_values(target: np.ndarray, decoy: np.ndarray) -> np.ndarray:
    """q-value of every target candidate (evaluated at its own score)."""
    t = np.asarray(target, dtype=np.float64)
    if t.size == 0:
        return t
    cut = np.unique(np.concatenate([t, np.asarray(decoy, dtype=np.float64)]))
    _, _, _, q = fdr_curve(t, decoy, cut)
    return q[np.searchsorted(cut, t)]


def calibrate(pairs, score="final", cutoffs=None, report_cutoff=8.0) -> dict:
    tgt = np.concatenate([read_scores(t, score) for t, _ in pairs]) if pairs else np.zeros(0)
    dec = np.concatenate([read_scores(d, score) for _, d in pairs]) if pairs else np.zeros(0)
    if cutoffs is None:
        lo = min([5.0] + list(tgt) + list(dec))
        hi = max([12.0] + list(tgt) + list(dec))
        cutoffs = np.round(np.arange(np.floor(lo), np.ceil(hi) + 1e-9, 0.05), 4)
    nt, nd, fdr, q = fdr_curve(tgt, dec, cutoffs)

    def first_cut(level):
        ok = np.nonzero((q <= level) & (nt > 0))[0]
        return float(cutoffs[ok[0]]) if ok.size else None

    i8 = int(np.argmin(np.abs(cutoffs - report_cutoff)))
    decoys_meta = []
    for _, d in pairs:
        st = Path(d) / "search_status.json"
        decoys_meta.append(json.loads(st.read_text()).get("decoy") if st.exists() else None)
    return dict(score=score, n_pairs=len(pairs), n_target=int(tgt.size), n_decoy=int(dec.size),
                at_cutoff=dict(cutoff=float(cutoffs[i8]), n_target=int(nt[i8]), n_decoy=int(nd[i8]),
                               fdr=float(fdr[i8]), q=float(q[i8])),
                cutoff_for_q_0p05=first_cut(0.05), cutoff_for_q_0p01=first_cut(0.01),
                decoy_settings=decoys_meta,
                curve=dict(cutoff=cutoffs.tolist(), n_target=nt.tolist(), n_decoy=nd.tolist(),
                           fdr=fdr.tolist(), q=q.tolist()),
                target_q_values=q_values(tgt, dec).tolist(),
                note="decoy-based FDR estimate; assumes decoy noise matches target noise")


def plot(rep: dict, path_base: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    c = rep["curve"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6), sharex=True, layout="constrained")
    axes[0].step(c["cutoff"], c["n_target"], where="post", color="#2878B5", lw=1.9, label="target")
    axes[0].step(c["cutoff"], c["n_decoy"], where="post", color="#888888", lw=1.6, label="decoy")
    axes[0].set(ylabel="Candidates with final score ≥ cutoff", yscale="symlog")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(c["cutoff"], c["q"], color="#2878B5", lw=1.9)
    axes[1].set(xlabel="Final flat-fielded SMAP SNR cutoff", ylabel="Estimated FDR (q)", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.axvline(rep["at_cutoff"]["cutoff"], color="#555555", ls=":", lw=1.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.16)
    a = rep["at_cutoff"]
    axes[0].set_title(f"Target–decoy calibration · {rep['n_pairs']} image(s) · at {a['cutoff']:g}: "
                      f"{a['n_target']} target, {a['n_decoy']} decoy, FDR {a['fdr']:.2f}", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_base}.{ext}", dpi=300)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("TARGET_OUT", "DECOY_OUT"))
    ap.add_argument("--score", choices=sorted(SCORE_FILES), default="final")
    ap.add_argument("--cutoff", type=float, default=8.0, help="cutoff to report (default 8)")
    ap.add_argument("--out", default="decoy_calibration", help="output file base name")
    a = ap.parse_args(argv)
    rep = calibrate([(Path(t), Path(d)) for t, d in a.pair], a.score, report_cutoff=a.cutoff)
    base = Path(a.out)
    base.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{base}.json").write_text(json.dumps(rep, indent=1))
    with open(f"{base}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cutoff", "n_target", "n_decoy", "fdr", "q"])
        c = rep["curve"]
        w.writerows(zip(c["cutoff"], c["n_target"], c["n_decoy"], c["fdr"], c["q"]))
    plot(rep, base)
    at = rep["at_cutoff"]
    print(f"score={rep['score']}  target={rep['n_target']}  decoy={rep['n_decoy']}")
    print(f"at cutoff {at['cutoff']:g}: {at['n_target']} target, {at['n_decoy']} decoy, "
          f"estimated FDR {at['fdr']:.3f}")
    print(f"cutoff for q<=0.05: {rep['cutoff_for_q_0p05']}   q<=0.01: {rep['cutoff_for_q_0p01']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
