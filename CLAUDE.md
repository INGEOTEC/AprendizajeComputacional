# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Spanish-language Quarto book, *Aprendizaje Computacional* (machine learning course notes, INFOTEC / MCDI), published at https://ingeotec.github.io/AprendizajeComputacional. There is no application code: the "source" is the `.qmd` chapters under `capitulos/`, which embed executable Python (numpy, scikit-learn, pandas/seaborn, JAX/optax, CVXPY, IngeoML, EvoMSA, CompStats, shap). All prose, commit messages and PR titles are written in Spanish.

## Commands

Quarto is not installed by any script; it comes from the devcontainer (`.devcontainer/`, rocker quarto-cli feature with TinyTeX and Chromium). Python deps: `pip install -r requirements.txt`.

```bash
python scripts/render.py                               # install missing deps, render whole book, verify freeze cache
python scripts/render.py capitulos/13Agrupamiento.qmd  # render one chapter (the equivalent of "run one test")
python scripts/render.py --force capitulos/13Agrupamiento.qmd  # drop its frozen result and re-execute the code
python scripts/render.py --verify-only                 # no render; just check _freeze/ is in sync with sources
python scripts/check_visible.py capitulos/13Agrupamiento.qmd  # ejecuta solo el código visible de un capítulo (sin argumentos: todo el libro)
python3 scripts/check_redaccion.py -v capitulos/13Agrupamiento.qmd  # mide dos puntos, punto y coma e inicios de párrafo tipo título (sin argumentos: todo el libro)
python3 scripts/test_redaccion.py                      # pruebas de scripts/check_redaccion.py (unittest, sin red ni Quarto)
quarto preview                                         # live preview while editing
python3 scripts/test_sitio.py                          # pruebas de scripts/sitio.py (unittest, sin red ni Quarto)
python3 scripts/test_nombres.py                        # ningún capítulo se llama como uno del MCDI (unittest, sin red ni Quarto)
python3 scripts/sitio.py <render> <checkout_gh_pages> <raiz|dev|mcdi>  # despliega una edición en su carpeta de gh-pages (lo usa el flujo)
quarto render --profile borrador --to html --output-dir "$TMPDIR/borrador"  # probar el perfil del borrador fuera de _book/
```

`python3 scripts/render.py` is also the `test_command` in `.issue-pilot.json`; a run is "green" when it exits 0. Rendering the full book executes every chapter (model fits, UMAP, etc.) and takes several minutes. Before rendering it runs `scripts/check_redaccion.py` and exits 1 without rendering if any file breaks the writing rules (see «Redacción»).

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

The processed public-policy datasets of the appendix (`capitulos/20ConjuntosDatos.qmd`) are not in the repo: they are assets of the GitHub release `datos-v1`, and the appendix and the regression chapters (`capitulos/03Regresion.qmd` and `capitulos/03SeleccionVariables.qmd`) read them from the release URL (`pd.read_csv('https://github.com/INGEOTEC/AprendizajeComputacional/releases/download/datos-v1/<file>')`). Rendering those chapters locally therefore needs network; CI does not, since it publishes from `_freeze/`. `scripts/datos/` has the scripts that generate the files (see its README). For a new version of the data create a new release (`datos-v2`) instead of overwriting `datos-v1`. `datos-v1` also holds `legalia_instrumentos.csv` and `legalia_similitud.csv.gz`, the directed instrument × instrument similarity matrix of LegalIA's Atlas (Qwen3-Embedding-4B, Apache 2.0) documented in `@sec-datos-instrumentos`; they are produced by `scripts/datos/legalia_instrumentos.py`, which needs a LegalIA clone and `uv` and takes about an hour on its first run. Adding a new file to `datos-v1` (`gh release upload` without `--clobber`) is not overwriting it; replacing an existing asset is.

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
9. La prosa sigue las reglas de la sección «Redacción» de este archivo; `scripts/check_redaccion.py` las mide.

Chapter order and the appendix split live only in `_quarto.yml` (`book.chapters` / `book.appendices`). File numbers are deliberately sparse: 16, 17 and 21 are reserved for planned chapters (causal inference, text, etc.; `03Regresion.qmd` and `03SeleccionVariables.qmd` already exist and share the 03 prefix), so a gap in the numbering is intentional and does not affect Quarto, which numbers chapters by their position in `_quarto.yml`. The current edition is oriented to the METPOL course (CIDE), so the Introduction frames the book around public-policy problems. `capitulos/22Codigo.qmd` exists but is **not** listed there, so it is not rendered.

Ningún capítulo de la edición vigente puede llamarse como uno de la edición MCDI (rama `mcdi`): las ligas viejas a `/capitulos/<capítulo MCDI>.html` deben caer en la página 404 (`ediciones/404.html`) y no mostrar sin aviso otro contenido. Por eso la Introducción y Redes Neuronales se llaman `01Intro.qmd` y `11Redes.qmd` (ex `01Introduccion.qmd` y `11RedesNeuronales.qmd`). `python3 scripts/test_nombres.py` lo comprueba con la lista fija de los 16 nombres del MCDI (no forma parte de `test_command` ni de CI; correrlo a mano al agregar o renombrar un capítulo).

## Redacción

Reglas del autor para toda la prosa del libro (capítulos, `index.qmd`, pies de figura y de tabla). Se cumplen al escribir, no solo al revisar: después de escribir o editar prosa en un capítulo, correr `python3 scripts/check_redaccion.py -v capitulos/<Capítulo>.qmd` y corregir hasta que diga `ok`.

1. **Dos puntos solo para introducir.** Los dos puntos se usan al final de un párrafo para introducir una ecuación, un bloque de código o una lista, y nada más. No se usan a media oración para anunciar una explicación («una observación incómoda: un bosque…»), ni después de un término en negritas («el **efecto Rashomon**: con frecuencia…»), ni para dar una consecuencia («aunque el promedio sea bueno: los hogares…»). En su lugar la oración se une con palabras: «que», «porque», «es decir», «de manera que», «ya que», «esto es», «el cual», o se termina con punto y se empieza otra oración completa.
2. **Punto y coma casi nunca.** Dos ideas se unen con una conjunción («y», «pero», «mientras que», «en cambio», «por lo que») o se separan con punto. El punto y coma solo se admite para separar los elementos de una enumeración cuando esos elementos ya llevan comas.
3. **Nunca dos de esos signos en el mismo párrafo.** Un párrafo de prosa tiene a lo más uno (dos puntos a media oración o punto y coma), y en el capítulo completo no hay más de uno por cada diez párrafos.
4. **El párrafo no empieza con un título disfrazado.** La primera oración de un párrafo no es un enunciado corto que el resto desarrolla («Conviene separar dos preguntas.», «Hay dos razones.», «Esta medida tiene dos limitaciones.», «En política pública la pregunta no es académica.»). La primera oración ya dice algo completo: se funde el enunciado corto con la oración que lo desarrolla («Esta medida tiene dos limitaciones. La primera es que…» pasa a «La primera limitación de esta medida es que…»). Como guía, la primera oración de un párrafo de varias oraciones tiene al menos diez palabras.
5. **Tampoco frases de efecto.** No se cierra un párrafo con una oración corta y contundente que repite lo dicho, ni se usan contrastes del tipo «no es X, es Y» o «X, no Y» como recurso; se dice directamente qué es.
6. **No se esquiva la regla cambiando el formato.** No se convierte prosa en listas, ni se sustituyen los dos puntos por rayas («—») o paréntesis, ni se parte una oración larga en muchas oraciones cortas. El estilo del libro es de párrafos con oraciones completas unidas por conectores, como en `capitulos/02Rendimiento.qmd` y `capitulos/07Lineal.qmd`.

`scripts/check_redaccion.py` mide las reglas 1 a 4 y es obligatorio: `scripts/render.py` (el `test_command`) lo corre antes de renderizar y termina con código 1, sin renderizar, si algún archivo no pasa. Todos los archivos del libro pasan y deben seguir pasando. Las reglas 5 y 6 no se miden y se cuidan al escribir. `scripts/test_redaccion.py` prueba el script.

## Spell checking

The repo uses cSpell in Spanish (`.vscode/settings.json`, `cSpell.language: es`), with math, code spans, Python chunks, citation keys and `#sec-` anchors excluded by regex. Legitimate technical / English terms and proper names go into the `cSpell.words` list in that file rather than being rewritten in the prose.
