# LaTeX source of the ECC–RWG article (IEEE TAES template)

- `main.tex` — entry point (compile with pdfLaTeX + BibTeX; on Overleaf set Compiler = pdfLaTeX, Main document = main.tex).
- `macros.tex` — single source of notation; `\todo{}` marks content that waits for the formal results or author input.
- `SYMBOLS.md` — symbol self-check table (one symbol, one meaning); not part of the paper.
- `sections/` — one file per section. Sections II–IV and the appendix are the methods; Section V is the first version of
  the simulation results (time-limited capture); the abstract, Section I (except the Notations paragraph) and
  Section VI are not written yet.
- `figures/fig_*.pdf` — compiled figures included by the paper (Overleaf needs only these PDFs).
  Fig. 1 `fig_scene`; Fig. 2 `fig_timeline`; Fig. 3 `fig_framework`; Fig. 4 `fig_rwg_{a,b,c}`;
  Fig. 5 `fig_screen_{a,b}`; Fig. 6 `fig_res_main_{a,c}`; Fig. 7 `fig_res_attr_{a,b}`; Fig. 8 `fig_res_scase_{a,b}`;
  Fig. 9 `fig_res_case_{a,b,c}`; Fig. 10 `fig_res_ref`; Fig. 11 `fig_res_learn`.
- `figures/src/make_results_data.py` — converts the round-1 readout tables (`evidence/FORMAL/ROUND1_READOUT/`) into
  `figures/src/data/*.dat`; `figures/src/make_case_data.py` converts the case trajectories (`evidence/PAPER_FIGS/`)
  into `figures/src/data/case_*` and `screen_*`. Run both before `build.sh` when the evidence changes.
- `figures/src/` — TikZ/pgfplots sources of the figures and `build.sh` (run `sh figures/src/build.sh` to rebuild all
  PDFs; `gen_scene.py` generates the 3-D projection of Fig. 1). Edit sources, then rebuild; never edit the PDFs.
- `refs.bib` — references; entries preceded by `% TODO` must be verified.
- `IEEEtaes.cls`, `IEEEtaes.bst` — official TAES template files (unchanged).

Facts come from `docs/paper/facts/`; writing sources and the structural template are in `References/REFERENCE_ANALYSIS.md`.
Terminology: "execution-opportunity screening", "admissible candidate set", "correction map", "screened kernel".
