# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Spanish-language Quarto book, *Aprendizaje Computacional* (machine learning course notes, INFOTEC / MCDI), published at https://ingeotec.github.io/AprendizajeComputacional. There is no application code: the "source" is the `.qmd` chapters under `capitulos/`, which embed executable Python (numpy, scikit-learn, pandas/seaborn, JAX/optax, CVXPY, IngeoML, EvoMSA, CompStats). All prose, commit messages and PR titles are written in Spanish.

## Commands

Quarto is not installed by any script; it comes from the devcontainer (`.devcontainer/`, rocker quarto-cli feature with TinyTeX and Chromium). Python deps: `pip install -r requirements.txt`.

```bash
python scripts/render.py                               # install missing deps, render whole book, verify freeze cache
python scripts/render.py capitulos/13Agrupamiento.qmd  # render one chapter (the equivalent of "run one test")
python scripts/render.py --force capitulos/13Agrupamiento.qmd  # drop its frozen result and re-execute the code
python scripts/render.py --verify-only                 # no render; just check _freeze/ is in sync with sources
python scripts/check_visible.py capitulos/13Agrupamiento.qmd  # ejecuta solo el código visible de un capítulo (sin argumentos: todo el libro)
quarto preview                                         # live preview while editing
python3 scripts/test_sitio.py                          # pruebas de scripts/sitio.py (unittest, sin red ni Quarto)
python3 scripts/test_nombres.py                        # ningún capítulo se llama como uno del MCDI (unittest, sin red ni Quarto)
python3 scripts/sitio.py <render> <checkout_gh_pages> <raiz|dev|mcdi>  # despliega una edición en su carpeta de gh-pages (lo usa el flujo)
quarto render --profile borrador --to html --output-dir "$TMPDIR/borrador"  # probar el perfil del borrador fuera de _book/
```

`python3 scripts/render.py` is also the `test_command` in `.issue-pilot.json`; a run is "green" when it exits 0. Rendering the full book executes every chapter (model fits, UMAP, etc.) and takes several minutes.

## Build / publish model: the freeze cache is load-bearing

- `_quarto.yml` sets `execute: freeze: auto`. Quarto stores each chapter's executed output in `_freeze/capitulos/<Chapter>/execute-results/{html,tex}.json` plus figure files, keyed by the **md5 of the `.qmd` source**.
- `.github/workflows/publish.yml` (push to `master` or `develop`, or `workflow_dispatch`) installs Quarto + TinyTeX + Chrome Headless Shell but **no Python**. It renders purely from `_freeze/`. If any chapter's hash does not match its frozen result, the publish job fails.
- El sitio (rama `gh-pages`, GitHub Pages «legacy» en `/`) tiene tres ediciones y **cada rama publica solo su carpeta**, sin borrar las otras: `master` → raíz (edición vigente, `quarto render`), `develop` → `dev/` (borrador, `quarto render --profile borrador`, perfil `_quarto-borrador.yml` con aviso y `noindex`), `mcdi` → `mcdi/` (edición histórica). La sincronización la hace `scripts/sitio.py` (protege `dev/`, `mcdi/` y `.git`, y escribe siempre `.nojekyll` y `404.html`, copia de `ediciones/404.html`); el flujo hace commit y push a `gh-pages` sin `--force`, reintentando hasta 3 veces si otra publicación llegó antes. Ya no se usa `quarto-actions/publish` ni `quarto publish`. El bloque `quarto-actions/setup` debe conservar la forma `with:` / `version:` que lee `scripts/render.py`.
- La edición histórica solo se construye si falta `mcdi/index.html` en `gh-pages` o si se lanza el flujo a mano con `reconstruir_historico: true`: el flujo hace checkout de `mcdi`, copia `ediciones/mcdi/_quarto-historico.yml` a su raíz y renderiza con `--profile historico` (agrega el aviso de edición histórica y el menú «Ediciones»). El perfil vive aquí, no en `mcdi`.
- El menú «Ediciones» (`book.navbar` en `_quarto.yml`), la sección «Ediciones» de `index.qmd`, la tabla de `README.md` y `ediciones/404.html` listan las mismas ediciones y ramas; si cambia una, cambiar las cuatro. Los avisos usan `format.html.include-before-body` porque `book.announcement` no se aplica a libros en Quarto 1.10.
- Los perfiles se prueban con `--output-dir` fuera del repositorio (p. ej. `$TMPDIR`), nunca sobre `_book/`, que está versionado y no debe llevar el aviso de borrador. Para la histórica: `git worktree add $TMPDIR/mcdi mcdi`, copiar el perfil, `quarto render --profile historico --to html --output-dir ...` y `git worktree remove --force`.
- Therefore: after editing any chapter that has code chunks, re-render it with `scripts/render.py` and commit the chapter **together with** its `_freeze/` changes. Never touch a `.qmd` after rendering without rendering again (even whitespace changes the hash). `scripts/render.py` prints exactly which paths to commit.
- Renderizar un capítulo suelto (`quarto render capitulos/X.qmd` o `scripts/render.py capitulos/X.qmd`) siempre ejecuta su código aunque exista resultado congelado; `freeze: auto` solo se respeta en el render del libro completo. Además `quarto render` acepta un solo archivo: para varios capítulos, una llamada por capítulo.
- `_freeze/` and `_book/` are both tracked in git (despite the docstring in `render.py` saying `_book/` is ignored); `.quarto/`, `capitulos/*_files` and `*.quarto_ipynb` are ignored.
- The mermaid diagram in `01Intro.qmd` is the reason for the headless-Chrome setup in both the workflow and `postCreate.sh`; keep those apt lists in sync if either changes.

## Data (release `datos-v1`)

The processed public-policy datasets of the appendix (`capitulos/20ConjuntosDatos.qmd`) are not in the repo: they are assets of the GitHub release `datos-v1`, and the appendix and the regression chapters (`capitulos/03Regresion.qmd` and `capitulos/03SeleccionVariables.qmd`) read them from the release URL (`pd.read_csv('https://github.com/INGEOTEC/AprendizajeComputacional/releases/download/datos-v1/<file>')`). Rendering those chapters locally therefore needs network; CI does not, since it publishes from `_freeze/`. `scripts/datos/` has the scripts that generate the files (see its README). For a new version of the data create a new release (`datos-v2`) instead of overwriting `datos-v1`.

## Branching

`develop` is the working branch (also `base_branch` in `.issue-pilot.json`) and is published as the draft in `/dev/`; `master` is the current edition, published at the site root. Open PRs from `develop` (or feature branches) toward `master`. `mcdi` (apunta a `8e4544b`) es la edición histórica del curso de la MCDI, publicada en `/mcdi/`: está **congelada y nunca recibe commits**. Las ramas `2018`, `v1.0`, `v2.0` y `v2.1` son versiones anteriores sin sitio; solo se enlazan en GitHub.

## Chapter conventions

Every chapter in `capitulos/` follows the same skeleton; new sections or chapters should too:

1. `# Título {#sec-slug}` heading, then an opening paragraph starting `El **objetivo** de la unidad es ...`.
2. `## Paquetes usados {.unnumbered}` with one `#| echo: true` chunk holding *all* imports for the chapter, followed by a hidden (`#| echo: false`) setup chunk (`from IPython.display import Markdown`, `sns.set_style('whitegrid')`).
3. A YouTube embed wrapped in `::: {.content-visible when-format="html"}` (PDF output has `echo: false`, HTML has `echo: true` and `code-tools`).
4. Figures: labeled chunks `#| label: fig-...`, `#| fig-cap: "..."`, `#| code-fold: true`, referenced in prose as `@fig-...`. Tables use `tbl-` labels / `#| tbl-cap`.
5. Numbers quoted in prose are never hard-coded: compute them in a hidden chunk as `x_f = Markdown(f'${x:0.4f}$')` and reference with inline `` `{python} x_f` `` so the text stays consistent with the executed code.
6. Pedagogical blocks: `::: {.callout-tip collapse="true"}` with `### Actividad` for exercises, `::: {.callout-note}` for asides.
7. Todo lo que el lector ve debe poder ejecutarse por sí solo: toda variable usada en un chunk mostrado se define en un chunk mostrado anterior (los imports y constantes como `PALETTE` van en el chunk visible de `## Paquetes usados`). Los chunks `echo: false` son solo para `Markdown(...)`, estilos y cálculos que alimentan texto en línea. `scripts/check_visible.py` lo comprueba: extrae los chunks visibles, los concatena y los ejecuta (no forma parte de `test_command` ni de CI; correrlo a mano al editar un capítulo).
8. Cross-references use `@sec-...` across chapters (58 distinct anchors exist, so renaming a `{#sec-...}` id requires grepping all of `capitulos/`). Citations are `@bibkey` from `references.bib`; the bibliography renders in `18Referencias.qmd`.

Chapter order and the appendix split live only in `_quarto.yml` (`book.chapters` / `book.appendices`). File numbers are deliberately sparse: 15, 16, 17 and 21 are reserved for planned chapters (interpretability, causal inference, text, etc.; `03Regresion.qmd` and `03SeleccionVariables.qmd` already exist and share the 03 prefix), so a gap in the numbering is intentional and does not affect Quarto, which numbers chapters by their position in `_quarto.yml`. The current edition is oriented to the METPOL course (CIDE), so the Introduction frames the book around public-policy problems. `capitulos/22Codigo.qmd` exists but is **not** listed there, so it is not rendered.

Ningún capítulo de la edición vigente puede llamarse como uno de la edición MCDI (rama `mcdi`): las ligas viejas a `/capitulos/<capítulo MCDI>.html` deben caer en la página 404 (`ediciones/404.html`) y no mostrar sin aviso otro contenido. Por eso la Introducción y Redes Neuronales se llaman `01Intro.qmd` y `11Redes.qmd` (ex `01Introduccion.qmd` y `11RedesNeuronales.qmd`). `python3 scripts/test_nombres.py` lo comprueba con la lista fija de los 16 nombres del MCDI (no forma parte de `test_command` ni de CI; correrlo a mano al agregar o renombrar un capítulo).

## Spell checking

The repo uses cSpell in Spanish (`.vscode/settings.json`, `cSpell.language: es`), with math, code spans, Python chunks, citation keys and `#sec-` anchors excluded by regex. Legitimate technical / English terms and proper names go into the `cSpell.words` list in that file rather than being rewritten in the prose.
