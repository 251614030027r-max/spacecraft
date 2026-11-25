# LaTeX source of the ECC–RWG article (IEEE TAES template)

- `main.tex` — entry point (compile with pdfLaTeX + BibTeX; on Overleaf set Compiler = pdfLaTeX, Main document = main.tex).
- `macros.tex` — single source of notation; `\todo{}` marks author information still to be entered (names, funding).
- `SYMBOLS.md` — symbol self-check table (one symbol, one meaning); not part of the paper.
- `sections/` — one file per section: abstract, Section I (introduction), Sections II–IV (methods), Section V
  (simulations with the round-2 formal results and the supplementary assets, `docs/paper/facts/08`), Section VI
  (conclusion), Appendix A (proofs) and Appendix B (ECC without critical-window correction).
- `figures/fig_*.pdf` — compiled figures included by the paper (Overleaf needs only these PDFs).
  Fig. 1 `fig_scene`; Fig. 2 `fig_timeline`; Fig. 3 `fig_framework`; Fig. 4 `fig_rwg_{a,b,c}`;
  Fig. 5 `fig_screen_{a,b}`; Fig. 6 `fig_res_learn`; Fig. 7 `fig_res_caseA_{traj,err}` (task A trajectory and errors);
  Fig. 8 `fig_res_caseA_{con,wr}` (task A constraints and wrench); Fig. 9 `fig_res_caseB` (task B);
  Fig. 10 `fig_res_outcome` (outcomes, time, Δv); Fig. 11 `fig_res_cond` (task conditions).
- `figures/src/make_r2_data.py` — converts `evidence/R2_FORMAL/` (training curves, outcomes, strata, 2-s case rows)
  and `evidence/R2_ASSETS/` (0.1-s case traces, out-of-distribution conditions) into `figures/src/data/`; run it
  before `build.sh` when the evidence changes.
- `figures/src/` — TikZ/pgfplots sources of the figures and `build.sh` (run `sh figures/src/build.sh` to rebuild all
  PDFs; `gen_scene.py` generates the 3-D projection of Fig. 1). Edit sources, then rebuild; never edit the PDFs.
- `refs.bib` — references (checked against the PDFs in References/ and the publisher records).
- `IEEEtaes.cls`, `IEEEtaes.bst` — official TAES template files (unchanged).

Facts come from `docs/paper/facts/`; writing sources and the structural template are in `References/REFERENCE_ANALYSIS.md`.
Terminology: "execution-opportunity coordination", "critical candidate set", "coordination map" (keep / correct / launch), "coordinated kernel".
