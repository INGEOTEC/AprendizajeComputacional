# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Spanish-language Quarto book, *Aprendizaje Computacional* (machine learning course notes, INFOTEC / MCDI), published at https://ingeotec.github.io/AprendizajeComputacional. There is no application code: the "source" is the `.qmd` chapters under `capitulos/`, which embed executable Python (numpy, scikit-learn, pandas/seaborn, JAX/optax, CVXPY, IngeoML, EvoMSA, CompStats). All prose, commit messages and PR titles are written in Spanish.

## Commands

Quarto is not installed by any script; it comes from the devcontainer (`.devcontainer/`, rocker quarto-cli feature with TinyTeX and Chromium). Python deps: `pip install -r requirements.txt`.

```bash
python scripts/render.py                               # install missing deps, render whole book, verify freeze cache
python scripts/render.py capitulos/06Agrupamiento.qmd  # render one chapter (the equivalent of "run one test")
python scripts/render.py --force capitulos/06Agrupamiento.qmd  # drop its frozen result and re-execute the code
python scripts/render.py --verify-only                 # no render; just check _freeze/ is in sync with sources
quarto preview                                         # live preview while editing
```

`python3 scripts/render.py` is also the `test_command` in `.issue-pilot.json`; a run is "green" when it exits 0. Rendering the full book executes every chapter (model fits, UMAP, etc.) and takes several minutes.

## Build / publish model: the freeze cache is load-bearing

- `_quarto.yml` sets `execute: freeze: auto`. Quarto stores each chapter's executed output in `_freeze/capitulos/<Chapter>/execute-results/{html,tex}.json` plus figure files, keyed by the **md5 of the `.qmd` source**.
- `.github/workflows/publish.yml` (push to `master`) installs Quarto + TinyTeX + Chrome Headless Shell but **no Python**. It renders purely from `_freeze/`. If any chapter's hash does not match its frozen result, the publish job fails.
- Therefore: after editing any chapter that has code chunks, re-render it with `scripts/render.py` and commit the chapter **together with** its `_freeze/` changes. Never touch a `.qmd` after rendering without rendering again (even whitespace changes the hash). `scripts/render.py` prints exactly which paths to commit.
- `_freeze/` and `_book/` are both tracked in git (despite the docstring in `render.py` saying `_book/` is ignored); `.quarto/`, `capitulos/*_files` and `*.quarto_ipynb` are ignored.
- The mermaid diagram in `01Introduccion.qmd` is the reason for the headless-Chrome setup in both the workflow and `postCreate.sh`; keep those apt lists in sync if either changes.

## Branching

`develop` is the working branch (also `base_branch` in `.issue-pilot.json`); `master` is what CI publishes. Open PRs from `develop` (or feature branches) toward `master`.

## Chapter conventions

Every chapter in `capitulos/` follows the same skeleton; new sections or chapters should too:

1. `# Título {#sec-slug}` heading, then an opening paragraph starting `El **objetivo** de la unidad es ...`.
2. `## Paquetes usados {.unnumbered}` with one `#| echo: true` chunk holding *all* imports for the chapter, followed by a hidden (`#| echo: false`) setup chunk (`from IPython.display import Markdown`, `sns.set_style('whitegrid')`).
3. A YouTube embed wrapped in `::: {.content-visible when-format="html"}` (PDF output has `echo: false`, HTML has `echo: true` and `code-tools`).
4. Figures: labeled chunks `#| label: fig-...`, `#| fig-cap: "..."`, `#| code-fold: true`, referenced in prose as `@fig-...`. Tables use `tbl-` labels / `#| tbl-cap`.
5. Numbers quoted in prose are never hard-coded: compute them in a hidden chunk as `x_f = Markdown(f'${x:0.4f}$')` and reference with inline `` `{python} x_f` `` so the text stays consistent with the executed code.
6. Pedagogical blocks: `::: {.callout-tip collapse="true"}` with `### Actividad` for exercises, `::: {.callout-note}` for asides.
7. Cross-references use `@sec-...` across chapters (58 distinct anchors exist, so renaming a `{#sec-...}` id requires grepping all of `capitulos/`). Citations are `@bibkey` from `references.bib`; the bibliography renders in `17Referencias.qmd`.

Chapter order and the appendix split live only in `_quarto.yml` (`book.chapters` / `book.appendices`). `capitulos/15Codigo.qmd` exists but is **not** listed there, so it is not rendered.

## Spell checking

The repo uses cSpell in Spanish (`.vscode/settings.json`, `cSpell.language: es`), with math, code spans, Python chunks, citation keys and `#sec-` anchors excluded by regex. Legitimate technical / English terms and proper names go into the `cSpell.words` list in that file rather than being rewritten in the prose.
