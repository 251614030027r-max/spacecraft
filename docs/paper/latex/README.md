# LaTeX source of the ECC–RWG article (IEEE TAES template)

- `main.tex` — entry point (compile with pdfLaTeX + BibTeX; on Overleaf set Compiler = pdfLaTeX, Main document = main.tex).
- `macros.tex` — single source of notation; `\todo{}` marks content that waits for the formal results or author input.
- `SYMBOLS.md` — symbol self-check table (one symbol, one meaning); not part of the paper.
- `sections/` — one file per section. Sections II–IV are the methods, Section V the simulations with the round-2
  formal results (`docs/paper/facts/08`), Appendix A the proofs and Appendix B the ECC variant without critical-window
  correction; the abstract, Section I (except the Notations paragraph) and Section VI are not written yet.
- `figures/fig_*.pdf` — compiled figures included by the paper (Overleaf needs only these PDFs).
  Fig. 1 `fig_scene`; Fig. 2 `fig_timeline`; Fig. 3 `fig_framework`; Fig. 4 `fig_rwg_{a,b,c}`;
  Fig. 5 `fig_screen_{a,b}`; Fig. 6 `fig_res_learn`; Fig. 7 `fig_res_case_{a,...,f}` (task A, complete process);
  Fig. 8 `fig_res_caseB` (task B, ECC vs SAC–MPC); Fig. 9 `fig_res_strata`. Table cells marked in red (`\tbd`) and
  `\todo{}` notes wait for the additional assets (0.1-s records of task A, terminal errors, out-of-distribution runs).
- `figures/src/make_r2_data.py` — converts `evidence/R2_FORMAL/` (training curves, strata, case trajectories) into `figures/src/data/`; run it before `build.sh` when the evidence changes.
- `figures/src/` — TikZ/pgfplots sources of the figures and `build.sh` (run `sh figures/src/build.sh` to rebuild all
  PDFs; `gen_scene.py` generates the 3-D projection of Fig. 1). Edit sources, then rebuild; never edit the PDFs.
- `refs.bib` — references; entries preceded by `% TODO` must be verified.
- `IEEEtaes.cls`, `IEEEtaes.bst` — official TAES template files (unchanged).

Facts come from `docs/paper/facts/`; writing sources and the structural template are in `References/REFERENCE_ANALYSIS.md`.
Terminology: "execution-opportunity coordination", "critical candidate set", "coordination map" (keep / correct / launch), "coordinated kernel".
