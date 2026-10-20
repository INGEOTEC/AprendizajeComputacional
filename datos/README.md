# Datos

Archivos procesados que usa el libro, versionados en el repositorio. Cada archivo
lo genera un script de [`scripts/datos/`](../scripts/datos/) a partir de la
fuente oficial; las fichas completas (variables, objetivo, tarea de aprendizaje)
están en el apéndice «Conjunto de Datos» (`capitulos/20ConjuntosDatos.qmd`).

| Archivo | Script | Fuente | Licencia |
|---|---|---|---|
| `enigh_pobreza.csv.gz` | `scripts/datos/enigh_pobreza.py` | ENIGH 2022 Nueva Serie, INEGI; línea de pobreza por ingresos de agosto de 2022, CONEVAL | Términos de libre uso de la información del INEGI |
| `envipe_seguridad.csv.gz` | `scripts/datos/envipe_seguridad.py` | ENVIPE 2026 (año de referencia 2025), INEGI | Términos de libre uso de la información del INEGI |
| `ensanut_salud.csv.gz` | `scripts/datos/ensanut_salud.py` | ENSANUT Continua 2022, Instituto Nacional de Salud Pública | Acceso libre para investigación y docencia, citando al INSP |
| `sep_f911_educacion.csv.gz` | `scripts/datos/sep_f911_educacion.py` | Estadística del Formato 911 (educación básica, secundaria, ciclos 2023-2024 y 2024-2025), SEP | CC-BY-4.0 |
| `censo_denue_territorio.csv` | `scripts/datos/censo_denue_territorio.py` | Censo de Población y Vivienda 2020 (AGEB urbana de Aguascalientes) y DENUE 05_2026, INEGI | Términos de libre uso de la información del INEGI |

## Reglas

- Cada archivo pesa menos de 5 MB (`comun.guardar` lo verifica) y no contiene
  identificadores directos de personas.
- El render del libro **nunca descarga**: los capítulos sólo leen estos archivos
  con `pd.read_csv('../datos/...')`.
- Los archivos crudos que descargan los scripts se guardan en un caché fuera del
  repositorio, `~/.cache/aprendizaje-computacional/` (se cambia con la variable de
  entorno `APRENDIZAJE_CACHE`); nunca se commitean.
- Los scripts son idempotentes y deterministas (muestreos con semilla fija): correr
  uno dos veces produce exactamente el mismo archivo.

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
`requirements.txt` y sólo se usa para descargar. Si cambia una fuente (otra edición,
otra URL), se actualiza el script, se vuelve a correr y se revisan las cifras que
cita el apéndice, que se calculan al renderizar.
