# LaTeX source of the ECC–RWG article (IEEE TAES template)

- `main.tex` — entry point (compile with pdfLaTeX + BibTeX; on Overleaf set Compiler = pdfLaTeX, Main document = main.tex).
- `macros.tex` — single source of notation; `\todo{}` marks content that waits for the formal results or author input.
- `SYMBOLS.md` — symbol self-check table (one symbol, one meaning); not part of the paper.
- `sections/` — one file per section. Sections II–IV and the appendix are the methods draft; Section V-A holds the
  simulation setup (scenario and parameter tables); the abstract, Section I (except the Notations paragraph),
  the rest of Section V, and Section VI wait for the formal results.
- `figures/fig_*.pdf` — compiled figures included by the paper (Overleaf needs only these PDFs).
  Fig. 1 `fig_scene`; Fig. 2 `fig_timeline`; Fig. 3 `fig_framework`; Fig. 4 `fig_rwg_{a,b,c}`;
  Fig. 5 `fig_screen_{a,b,c}`; Fig. 6 `fig_timescales`.
- `figures/src/` — TikZ/pgfplots sources of the figures and `build.sh` (run `sh figures/src/build.sh` to rebuild all
  PDFs; `gen_scene.py` generates the 3-D projection of Fig. 1). Edit sources, then rebuild; never edit the PDFs.
- `refs.bib` — references; entries preceded by `% TODO` must be verified.
- `IEEEtaes.cls`, `IEEEtaes.bst` — official TAES template files (unchanged).

Facts come from `docs/paper/facts/`; writing sources and the structural template are in `References/REFERENCE_ANALYSIS.md`.
Terminology: "execution-opportunity screening", "admissible candidate set", "correction map", "screened kernel".
