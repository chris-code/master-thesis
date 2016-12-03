#!/bin/bash

filename="Report"

pdflatex "$filename".tex
pdflatex "$filename".tex
bibtex "$filename".aux
pdflatex "$filename".tex
pdflatex "$filename".tex

# clean up
rm "$filename".aux "$filename".bbl "$filename".blg "$filename".log "$filename".toc "$filename".lof "$filename".lot "$filename".loa
