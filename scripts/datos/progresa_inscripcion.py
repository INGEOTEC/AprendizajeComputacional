"""Prepara ``progresa_inscripcion.csv``: inscripción escolar en el experimento de Progresa.

Fuente: archivo de réplica de Lee y Shaikh (2014), «Multiple Testing and
Heterogeneous Treatment Effects: Re-evaluating the Effect of PROGRESA on School
Enrollment», Journal of Applied Econometrics 29(4), 612-626, publicado en el
archivo de datos de la revista (ZBW Journal Data Archive):
https://journaldata.zbw.eu/dataset/4dbd19b5-808f-464c-b9e5-795900a77d3d
Los autores lo extrajeron de las bases de la evaluación de Progresa que publicaba
el gobierno de México (evaluacion.oportunidades.gob.mx, hoy fuera de línea).
Licencia: el archivo de réplica no declara una licencia; se redistribuye
procesado, con atribución, para docencia. Fecha de consulta: 2026-11-18.

Unidad: niña o niño de un hogar elegible (pobre) de las localidades de la
evaluación, observado en la ronda 1 (1997, antes del programa). El archivo
original tiene un renglón por persona y ronda (5 rondas); aquí queda un renglón
por persona con sus características de la ronda 1 y su inscripción en las
rondas 3 (otoño de 1998) y 5 (otoño de 1999), posteriores al inicio del
programa. El tratamiento (``progresa``) se asignó al azar por localidad.

Columnas:
  - ``localidad``: identificador anónimo de la localidad (conglomerado).
  - ``ninos_localidad``: personas de la localidad en el archivo original, ronda 1.
  - ``progresa``: 1 si la localidad recibió el programa desde 1998, 0 si fue
    de control.
  - ``mujer``, ``edad``, ``grado`` (último grado aprobado, de 0 a 9) e
    ``inscrito_1997``: características en la ronda 1.
  - ``inscrito_1998`` (objetivo) e ``inscrito_1999``: 1 si asistía a la escuela.

El identificador de persona del archivo original no se publica.

El servidor responde a los agentes de navegador con una página de verificación,
por eso la descarga se hace con ``agente=None``.

Dependencias extra: ninguna (pandas de requirements.txt).
Uso: ``python scripts/datos/progresa_inscripcion.py``
"""

import pandas as pd

import comun

URL = ("https://journaldata.zbw.eu/dataset/4dbd19b5-808f-464c-b9e5-795900a77d3d/"
       "resource/916e124d-037a-471b-a289-b45d4c64b9af/download/ls-data.csv")
COLUMNAS = ["wave", "sooloca", "indivill_seq", "villsize_t", "progresa1",
            "sooind_id", "sex1", "age1", "hgc1", "poor1", "school"]


def main() -> None:
    d = pd.read_csv(comun.descargar(URL, nombre="progresa_ls-data.csv",
                                    agente=None))
    if list(d.columns) != COLUMNAS:
        raise SystemExit(f"columnas inesperadas: {list(d.columns)}")
    if sorted(d["wave"].unique()) != [1, 2, 3, 4, 5]:
        raise SystemExit("se esperaban las rondas 1 a 5")
    if d.duplicated(["sooind_id", "wave"]).any():
        raise SystemExit("hay personas repetidas dentro de una ronda")
    if (d["poor1"] != 1).any():
        raise SystemExit("se esperaban solo hogares elegibles")

    ronda = {k: g.set_index("sooind_id") for k, g in d.groupby("wave")}
    base = ronda[1].sort_index()
    sal = pd.DataFrame({
        "localidad": base["sooloca"],
        "ninos_localidad": base["villsize_t"],
        "progresa": base["progresa1"],
        "mujer": base["sex1"],
        "edad": base["age1"],
        "grado": base["hgc1"],
        "inscrito_1997": base["school"],
        "inscrito_1998": ronda[3]["school"].reindex(base.index),
        "inscrito_1999": ronda[5]["school"].reindex(base.index),
    })
    if sal.isna().any().any():
        raise SystemExit("hay personas de la ronda 1 sin ronda 3 o 5")
    if sal.groupby("localidad")["progresa"].nunique().max() != 1:
        raise SystemExit("una localidad tiene más de un valor de tratamiento")
    sal = sal.astype(int).reset_index(drop=True)
    comun.guardar(sal, "progresa_inscripcion.csv")


if __name__ == "__main__":
    main()
