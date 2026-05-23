# Mini-DLSS Technical Summary

This folder contains a research-paper-style technical breakdown of the Mini-DLSS repository.

## Files

- `summary.tex`: main LaTeX document.
- `references.bib`: bibliography for papers, datasets, frameworks, and methods.
- `figures/`: TikZ diagrams included by `summary.tex`.
- `spellcheck_report.md`: secondary review report covering grammar, spelling, clarity, and consistency.

## Compile

From this directory:

```bash
latexmk -pdf summary.tex
```

If `latexmk` is unavailable, run:

```bash
pdflatex summary.tex
bibtex summary
pdflatex summary.tex
pdflatex summary.tex
```

The document uses TikZ figures and a BibTeX bibliography. A typical TeX Live or MacTeX installation should include the required packages.

