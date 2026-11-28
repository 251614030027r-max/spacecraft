#!/bin/sh
# Compile every standalone figure source in this directory to ../<name>.pdf.
# Usage: sh build.sh [name ...]   (run from figures/src)
set -e
cd "$(dirname "$0")"
python3 gen_scene.py scene_body.tex
names="$*"
[ -z "$names" ] && names=$(ls fig_*.tex | sed 's/\.tex$//')
for n in $names; do
  pdflatex -interaction=nonstopmode -halt-on-error "$n.tex" > "$n.stdout" 2>&1 || { tail -30 "$n.stdout"; exit 1; }
  cp "$n.pdf" "../$n.pdf"
done
rm -f fig_*.aux fig_*.log fig_*.stdout fig_*.pdf
