# LaTeX source of the ECC–RWG article (IEEE TAES template)

- `main.tex` — entry point (compile with pdfLaTeX + BibTeX; on Overleaf set Compiler = pdfLaTeX, Main document = main.tex).
- `macros.tex` — notation; `\todo{}` marks content that waits for the formal results or author input.
- `sections/` — one file per section; Sections I, V, VI and the abstract are placeholders until the formal results are delivered.
- `figures/` — TikZ figures (frames, framework, sweep cone).
- `refs.bib` — references; entries preceded by `% TODO` must be verified.
- `IEEEtaes.cls`, `IEEEtaes.bst` — official TAES template files (unchanged).

Facts come from `docs/paper/facts/`; writing sources and the structural template are in `References/REFERENCE_ANALYSIS.md`.
