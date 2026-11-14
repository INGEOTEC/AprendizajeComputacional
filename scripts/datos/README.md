# Scripts de datos

Cada script descarga una fuente oficial y escribe el archivo procesado que usa el
apéndice «Conjunto de Datos» (`capitulos/20ConjuntosDatos.qmd`). Los archivos
procesados **no están en el repositorio**: se publican como assets del release
[`datos-v1`](https://github.com/INGEOTEC/AprendizajeComputacional/releases/tag/datos-v1)
y el apéndice los lee de ahí. Las fichas completas (variables, objetivo, tarea de
aprendizaje) están en el apéndice.

| Archivo | Script | Fuente | Licencia |
|---|---|---|---|
| `enigh_pobreza.csv.gz` | `enigh_pobreza.py` | ENIGH 2022 Nueva Serie, INEGI; línea de pobreza por ingresos de agosto de 2022, CONEVAL | Términos de libre uso de la información del INEGI |
| `envipe_seguridad.csv.gz` | `envipe_seguridad.py` | ENVIPE 2026 (año de referencia 2025), INEGI | Términos de libre uso de la información del INEGI |
| `ensanut_salud.csv.gz` | `ensanut_salud.py` | ENSANUT Continua 2022, Instituto Nacional de Salud Pública | Acceso libre para investigación y docencia, citando al INSP |
| `sep_f911_educacion.csv.gz` | `sep_f911_educacion.py` | Estadística del Formato 911 (educación básica, secundaria, ciclos 2023-2024 y 2024-2025), SEP | CC-BY-4.0 |
| `censo_denue_territorio.csv` | `censo_denue_territorio.py` | Censo de Población y Vivienda 2020 (AGEB urbana de Aguascalientes) y DENUE 05_2026, INEGI | Términos de libre uso de la información del INEGI |
| `legalia_instrumentos.csv` | `legalia_instrumentos.py` | LegalIA, colecciones de la SCJN (leyes, reglamentos, lineamientos y tratados), modelo Qwen3-Embedding-4B | Apache 2.0 |
| `legalia_similitud.csv.gz` | `legalia_instrumentos.py` | LegalIA, matriz de similitud entre instrumentos del Atlas (modelo Qwen3-Embedding-4B) | Apache 2.0 |

## Reglas

- Los archivos no contienen identificadores directos de personas
  (`comun.guardar` lo verifica) y no se muestrean: se publican todos los registros
  que cumplen los filtros de cada script.
- Los scripts son idempotentes y deterministas: correr
  uno dos veces produce exactamente el mismo archivo.
- Las descargas crudas van a un caché fuera del repositorio,
  `~/.cache/aprendizaje-computacional/` (se cambia con `APRENDIZAJE_CACHE`), y los
  archivos procesados a `salida/` dentro del caché (se cambia con
  `APRENDIZAJE_SALIDA`). Nada de esto se commitea.
- Una versión nueva de los datos es un release nuevo (`datos-v2`, ...); `datos-v1`
  no se sobrescribe, porque el apéndice cita cifras calculadas con esos archivos.

## Regenerar

Desde la raíz del repositorio, con las dependencias de `requirements.txt`:

```bash
python scripts/datos/enigh_pobreza.py
python scripts/datos/envipe_seguridad.py
python scripts/datos/ensanut_salud.py
python scripts/datos/sep_f911_educacion.py   # requiere además: pip install curl_cffi
python scripts/datos/censo_denue_territorio.py
```

La matriz de instrumentos jurídicos se genera aparte, porque no descarga una fuente
oficial sino que recalcula la matriz del Atlas de
[LegalIA](https://github.com/INGEOTEC/LegalIA) con sus propios scripts:

```bash
python scripts/datos/legalia_instrumentos.py   # requiere un clon de LegalIA y uv
```

Depende de un clon de LegalIA (`--legalia RUTA`, la variable `LEGALIA_REPO` o, por
omisión, `../LegalIA` junto al clon principal de este repositorio; si no existe, lo
clona en el caché) y de [`uv`](https://docs.astral.sh/uv/), con el que corre
`prepare_umap_input.py --unique-names` e `instrument_matrix.py` dentro de LegalIA. Las
dependencias de LegalIA se resuelven en un entorno propio dentro del caché
(`legalia/venv`, vía `UV_PROJECT_ENVIRONMENT`), no en `requirements.txt` ni en el
`.venv` del clon, que `uv sync` dejaría sólo con lo que pide LegalIA;
este script sólo necesita además `pyarrow` para leer `instruments.parquet`. LegalIA no
se modifica: el directorio de trabajo
(`~/.cache/aprendizaje-computacional/legalia/emb-run-atlas-4b`) y los cachés de
vectores y corpus viven en el caché del libro. Antes de escribir los archivos, el
script verifica la matriz contra el atlas público (`atlas-qwen3-4b.json`): mismos
instrumentos en el mismo orden, mismos fragmentos contados y casillas con valor, y
los mismos cinco destinos más fuertes por instrumento; si algo difiere, termina sin
escribir nada. La única diferencia aceptada es un empate en la frontera de la
tolerancia de LegalIA (1e-6 sobre la similitud coseno en float32), que otra biblioteca
de álgebra lineal puede resolver distinto: a lo más cinco instrumentos, con sus
fragmentos contados iguales y ningún peso movido más de medio voto, y el script los
imprime. En la corrida que publicó los archivos (LegalIA `f403872b1`) fue uno, la Ley
Orgánica de Nacional Financiera, con 13 y 8 votos hacia sus dos primeros destinos
contra 12.5 y 8.5 en el atlas.

**Advertencia:** la primera corrida descarga del orden de 1 a 2 GB (vectores del
modelo 4B y los corpus de la SCJN) y tarda alrededor de una hora:
`prepare_umap_input.py` unos 13 minutos e `instrument_matrix.py` de 15 a 40 minutos
con unos 4.3 GB de memoria. Conviene correrlo en segundo plano con la salida a una
bitácora. Los dos pasos son idempotentes (`prepare.done` e
`instrument-matrix/.done`), así que una segunda corrida no recalcula la matriz y
produce archivos idénticos.

Las descargas son de decenas a cientos de MB en total y necesitan red. El script de
la SEP necesita `curl_cffi` porque su servidor rechaza a los clientes que no se
presentan como un navegador; junto con `pyarrow` (ver abajo) es de las pocas
dependencias que no están en `requirements.txt`, y sólo se usa para descargar.

## Publicar

Los archivos procesados quedan en `~/.cache/aprendizaje-computacional/salida/`. Para
una versión nueva:

```bash
gh release create datos-v2 --target develop --title "Datos de política pública v2" \
  --notes "..." ~/.cache/aprendizaje-computacional/salida/*
```

y se actualiza la constante con la URL base del release en
`capitulos/20ConjuntosDatos.qmd`. Para corregir un asset de un release ya publicado
se usa `gh release upload <tag> <archivo> --clobber`, sólo si las cifras del apéndice
no cambian.

Agregar un archivo nuevo a un release existente (`gh release upload <tag> <archivo>`,
sin `--clobber`) no lo sobrescribe: así se agregaron a `datos-v1` los dos archivos de
LegalIA, sin tocar los cinco anteriores.
