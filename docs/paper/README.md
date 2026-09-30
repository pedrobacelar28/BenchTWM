# Technical Specification

`main.tex` is a self-contained, conference-style technical specification. It
documents the benchmark suite and deliberately contains no empirical results.

Compile it from this directory when a LaTex distribution is available:

```bash
pdflatex main.tex
pdflatex main.tex
```

The factual benchmark descriptions are based on the local paper copies in
`../../repos_usados/papers_txt/` and the vendored runtime code. The document
distinguishes those source benchmarks from algorithms implemented locally.
