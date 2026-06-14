# Mini-DLSS Technical Summary

This folder contains the concise technical summary and evaluation report for
Mini-DLSS.

## Files

- `summary.tex`: main LaTeX document.
- `summary.pdf`: compiled report.
- `references.bib`: retained source bibliography from the earlier long-form report.
- `figures/`: retained source diagrams from the earlier long-form report.
- `spellcheck_report.md`: review notes for the current revision.

## Compile

The current document uses an inline bibliography and compiles with Tectonic:

```bash
tectonic -X compile summary.tex
```

The report includes the final qualitative preview from `results/final/`, so
compile it from inside the repository rather than copying `summary.tex` alone.
