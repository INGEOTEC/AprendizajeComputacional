"""Prepara ``datos/censo_denue_territorio.csv``: escolaridad y economía local por AGEB.

Fuentes (INEGI, datos abiertos):
  - Censo de Población y Vivienda 2020, resultados por AGEB urbana y manzana,
    Aguascalientes (se usan los renglones de total por AGEB):
    https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/ageb_manzana/ageb_mza_urbana_01_cpv2020_csv.zip
  - Directorio Estadístico Nacional de Unidades Económicas (DENUE),
    Aguascalientes, edición 05_2026 (actualización del 2026-05-20):
    https://www.inegi.org.mx/contenidos/masiva/denue/denue_01_csv.zip
Licencia: Términos de libre uso de la información del INEGI. Fecha de
consulta: 2026-10-20.

Unidad: AGEB urbana del estado de Aguascalientes. Objetivo
(``grado_promedio_escolaridad``): grado promedio de escolaridad (``GRAPROES``)
de la población de 15 años o más en 2020. Predictoras: conteo de
establecimientos del DENUE por sector SCIAN (2 dígitos) y por estrato de
personal ocupado, más el número de establecimientos. Las columnas
``pct_viv_internet``, ``pct_sin_servicio_salud`` y ``pct_jefatura_femenina`` son
otros indicadores censales que sirven como objetivos alternativos; no deben
usarse como predictoras cuando el objetivo es la escolaridad. ``pob_total`` y
``viviendas_habitadas`` son el tamaño del AGEB.

Advertencias que se documentan en el libro:
  - El censo es de 2020 y el DENUE de 2026: los establecimientos son más
    recientes que las características de la población.
  - El censo suprime algunos valores por confidencialidad (``*``) o los
    reporta como no disponibles (``N/D``). Aquí se tratan explícitamente: se
    descartan los AGEB cuyo objetivo está suprimido, y los porcentajes de los
    otros indicadores quedan vacíos (NaN) cuando el numerador o el
    denominador está suprimido.
  - Se descartan los AGEB con menos de ``POB_MINIMA`` habitantes: el grado
    promedio de escolaridad es inestable en AGEB casi deshabitados
    (zonas industriales, por ejemplo).
  - Los establecimientos del DENUE sin clave de AGEB o fuera de las AGEB
    urbanas no se cuentan.

Dependencias extra: ninguna (pandas y numpy de requirements.txt).
Uso: ``python scripts/datos/censo_denue_territorio.py``
"""

import zipfile

import numpy as np
import pandas as pd

import comun

URL_CENSO = ("https://www.inegi.org.mx/contenidos/programas/ccpv/2020/"
             "datosabiertos/ageb_manzana/ageb_mza_urbana_01_cpv2020_csv.zip")
URL_DENUE = "https://www.inegi.org.mx/contenidos/masiva/denue/denue_01_csv.zip"
POB_MINIMA = 100

# sector SCIAN (2 dígitos) -> nombre de la columna
SECTORES = {
    "11": "agropecuario", "21": "mineria", "22": "energia_y_agua",
    "23": "construccion", "31": "manufactura", "32": "manufactura",
    "33": "manufactura", "43": "comercio_mayoreo", "46": "comercio_menudeo",
    "48": "transporte", "49": "transporte", "51": "informacion_medios",
    "52": "finanzas_seguros", "53": "inmobiliarios", "54": "servicios_profesionales",
    "55": "corporativos", "56": "servicios_apoyo", "61": "educacion",
    "62": "salud_asistencia", "71": "esparcimiento", "72": "alojamiento_alimentos",
    "81": "otros_servicios", "93": "gobierno_organismos"}
ESTRATOS = {
    "0 a 5 personas": "est_0_5_personas", "6 a 10 personas": "est_6_10_personas",
    "11 a 30 personas": "est_11_30_personas",
    "31 a 50 personas": "est_31_50_personas",
    "51 a 100 personas": "est_51_100_personas",
    "101 a 250 personas": "est_101_250_personas",
    "251 y más personas": "est_251_o_mas_personas"}


def leer_zip(ruta, sufijo: str, **kw) -> pd.DataFrame:
    with zipfile.ZipFile(ruta) as zf:
        nombre = [n for n in zf.namelist()
                  if "conjunto_de_datos/" in n and n.endswith(sufijo)][0]
        with zf.open(nombre) as f:
            return pd.read_csv(f, dtype=str, low_memory=False, **kw)


def numero(serie: pd.Series) -> pd.Series:
    """Valores suprimidos del censo (``*``) o no disponibles (``N/D``) -> NaN."""
    return pd.to_numeric(serie.replace({"*": np.nan, "N/D": np.nan}),
                         errors="coerce")


def main() -> None:
    censo = leer_zip(comun.descargar(URL_CENSO), ".csv")
    ageb = censo[(censo["MZA"] == "000") & (censo["AGEB"] != "0000")].copy()

    denue = leer_zip(comun.descargar(URL_DENUE), ".csv", encoding="latin1",
                     usecols=["codigo_act", "per_ocu", "cve_mun", "cve_loc",
                              "ageb"])
    denue = denue.dropna(subset=["ageb", "codigo_act", "per_ocu"])
    denue["sector"] = denue["codigo_act"].str[:2].map(SECTORES)
    denue["estrato"] = denue["per_ocu"].map(ESTRATOS)
    desconocidos = denue[denue["sector"].isna() | denue["estrato"].isna()]
    if len(desconocidos):
        raise SystemExit(f"{len(desconocidos)} establecimientos sin sector o "
                         "estrato reconocido")
    llaves = ["cve_mun", "cve_loc", "ageb"]
    por_sector = denue.groupby(llaves + ["sector"]).size().unstack(fill_value=0)
    por_sector.columns = [f"est_{c}" for c in por_sector.columns]
    por_estrato = denue.groupby(llaves + ["estrato"]).size().unstack(fill_value=0)

    d = ageb.merge(por_sector, left_on=["MUN", "LOC", "AGEB"],
                   right_index=True, how="left")
    d = d.merge(por_estrato, left_on=["MUN", "LOC", "AGEB"],
                right_index=True, how="left")
    conteos = list(por_sector.columns) + list(por_estrato.columns)
    d[conteos] = d[conteos].fillna(0).astype(int)  # sin establecimientos = 0

    pob = numero(d["POBTOT"])
    viv = numero(d["TVIVPARHAB"])
    sal = pd.DataFrame({
        "clave_ageb": d["ENTIDAD"] + d["MUN"] + d["LOC"] + d["AGEB"],
        "municipio": d["NOM_MUN"],
        "pob_total": pob,
        "viviendas_habitadas": viv,
        "grado_promedio_escolaridad": numero(d["GRAPROES"]),
        "pct_viv_internet": (100 * numero(d["VPH_INTER"]) / viv).round(2),
        "pct_sin_servicio_salud": (100 * numero(d["PSINDER"]) / pob).round(2),
        "pct_jefatura_femenina": (100 * numero(d["HOGJEF_F"])
                                  / numero(d["TOTHOG"])).round(2),
    })
    sal["est_total"] = d[list(por_estrato.columns)].sum(axis=1)
    for c in sorted(por_sector.columns):
        sal[c] = d[c]
    for c in ESTRATOS.values():
        sal[c] = d[c]

    sal = sal[sal["grado_promedio_escolaridad"].notna()
              & (sal["pob_total"] >= POB_MINIMA)]
    sal = sal.sort_values("clave_ageb").reset_index(drop=True)
    for c in ["pob_total", "viviendas_habitadas"]:
        sal[c] = sal[c].astype("Int64")
    comun.guardar(sal, "censo_denue_territorio.csv")


if __name__ == "__main__":
    main()
