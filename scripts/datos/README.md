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

## Reglas

- Cada archivo pesa menos de 5 MB (`comun.guardar` lo verifica) y no contiene
  identificadores directos de personas.
- Los scripts son idempotentes y deterministas (muestreos con semilla fija): correr
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

Las descargas son de decenas a cientos de MB en total y necesitan red. El script de
la SEP necesita `curl_cffi` porque su servidor rechaza a los clientes que no se
presentan como un navegador; es la única dependencia que no está en
`requirements.txt` y sólo se usa para descargar.

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
