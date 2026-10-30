Small numerical experiments behind the optional improvements. Run from the repository root
(`python experiments/<script>.py`); CPU only, seconds to ~1 min each.

| Script | Question | Result when written (2026-10-30) |
|---|---|---|
| `e1_template_accuracy.py` | How close are SMAP's Fourier-slice templates to exact projections of a Gaussian-atom volume? | CC 0.987 (v2.1, no padding) → 0.9997 with 2× oversampling (`oversample 2`); edge (gridding) correction alone 0.989 |
| `e2_defocus_step.py` | CC kept when the particle lies between defocus planes (sample .par physics, 1.5 Å/px, Cs 1e-6 m) | 50 nm step: mean 0.85/0.93/0.97 for B = 0/50/100 Å²; 25 nm: 0.96/0.98/0.99 (set `df_inc 25`, no code change) |
| `e3_template_envelope.py` | Matched-filter efficiency of a template without the image's B-factor envelope | 0.93/0.80/0.69 for B = 50/100/150 Å² in a flat-spectrum model (upper bound on the loss; `template_bfactor` must be fitted on data) |
| `e4_refinement_accuracy.py` | Stage-C accuracy on synthetic ground truth (off-grid rotations 4–10°, sub-pixel shifts, +12 nm defocus; 48-px potential at 2 Å/px, 20° search, 2° refinement; 5 images, 18 matched particles; same stage-B input for all modes) | median angle error 2.18° (v2.1) / 1.80° (`fix_optimize_angle`) / 1.72° (`refine_continuous`), stage B 2.20°; position 0.19 / 0.16 / 0.13 px; \|Δdf\| 15.0 / 15.0 / 12.7 nm; final score +0.075 / +0.104 vs v2.1 (higher in 18/18 particles); 0.1 / 1.1 / 1.4 s per particle on CPU. Angular precision here is limited by the small synthetic particle; repeat on a real tile |

These are model calculations, not measurements on experimental images.
