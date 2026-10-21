"""Prepara ``ensanut_salud.csv.gz``: diagnóstico previo de diabetes en adultos.

Fuente: Encuesta Nacional de Salud y Nutrición Continua (ENSANUT) 2022,
Instituto Nacional de Salud Pública (INSP). Se usan tres archivos CSV de la
página de descargas de la encuesta (el sitio los sirve con un formulario
POST, sin registro):
    https://ensanut.insp.mx/encuestas/ensanutcontinua2022/descargas.php
  - cuestionario de salud de adultos (20 años o más), ``ensadul2022_entrega_w``
  - cuestionario de antropometría y tensión arterial, ``ensaantro2022_entrega_w``
  - índice de bienestar del hogar, ``ENSANUT2022_ICB``
Licencia: la ENSANUT es de acceso libre para fines de investigación y
docencia; las bases son del INSP y deben citarse como fuente. Fecha de
consulta: 2026-10-20.

Unidad: adulto (20 a 69 años en la muestra final, pues la medición
antropométrica y de tensión arterial no cubre a más edad) con ambas mediciones. Objetivo (``diabetes``): 1 si un médico le dijo alguna vez que
tiene diabetes (o azúcar alta en la sangre), 0 si no. Se excluyen las
personas con diabetes sólo durante el embarazo (el objetivo sería
ambiguo), a las mujeres embarazadas o en lactancia y a quienes no tienen
peso, talla o tensión arterial válidos. Diabetes es viable en esta encuesta
(alrededor del 9% de la muestra final), por lo que no se usó hipertensión.

Predictoras: sexo, edad, entidad, tamaño de localidad, quintil del índice de
bienestar del hogar, peso, talla, índice de masa corporal y tensión arterial
medidos, diagnóstico previo de hipertensión, antecedentes familiares de
diabetes (padre, madre, hermanos), tabaquismo y consumo de alcohol. No se
incluye ninguna pregunta sobre tratamiento ni complicaciones de la diabetes.
Se conserva el ponderador (``factor``).

Dependencias extra: ninguna (pandas y numpy de requirements.txt).
Uso: ``python scripts/datos/ensanut_salud.py``
"""

import base64
import zipfile

import numpy as np
import pandas as pd

import comun

URL = "https://ensanut.insp.mx/encuestas/ensanutcontinua2022/descargas.php"
ARCHIVOS = {
    "ensadul2022_entrega_w.csv.csv.zip":
        "01-Componente de SALUD/04-Cuestionario de salud de adultos "
        "(20 años o más)/ensadul2022_entrega_w.csv.csv.zip",
    "ensaantro2022_entrega_w.csv.csv.zip":
        "02-Componente de NUTRICIÓN/01-Cuestionario de antropometría y "
        "tensión arterial/ensaantro2022_entrega_w.csv.csv.zip",
    "ENSANUT2022_ICB.csv.csv.zip":
        "01-Componente de SALUD/01-Cuestionario de hogar/03-Indice de "
        "bienestar/ENSANUT2022_ICB.csv.csv.zip",
}
TAM_LOCALIDAD = {1: "rural (menos de 2,500)", 2: "urbano (2,500 a 99,999)",
                 3: "metropolitano (100 mil o más)"}
SI_NO_NS = {1: "sí", 2: "no", 9: "no sabe"}
FUMA = {1: "todos los días", 2: "algunos días", 3: "no fuma"}
ALCOHOL = {1: "diario", 2: "semanal", 3: "mensual", 4: "anual",
           5: "no en los últimos 12 meses", 6: "nunca"}


def descargar(nombre: str):
    """Descarga un archivo de la ENSANUT (el sitio usa un botón de formulario)."""
    llave = "ArchId" + base64.b64encode(ARCHIVOS[nombre].encode()).decode()
    ruta = comun.descargar(URL, nombre, datos={llave: ""})
    if not zipfile.is_zipfile(ruta):
        ruta.unlink()
        raise RuntimeError(f"El INSP no regresó un ZIP para {nombre}")
    return ruta


def leer(nombre: str, columnas: list[str]) -> pd.DataFrame:
    ruta = descargar(nombre)
    with zipfile.ZipFile(ruta) as zf:
        csv = [n for n in zf.namelist() if n.endswith(".csv")
               and not n.endswith(("_valores.csv", "_variables.csv"))][0]
        with zf.open(csv) as f:
            return pd.read_csv(f, usecols=columnas, encoding="latin1",
                               dtype=str)


def num(serie: pd.Series) -> pd.Series:
    """Texto -> número; algunos archivos usan coma decimal y espacios vacíos."""
    return pd.to_numeric(serie.str.replace(",", ".").str.strip(),
                         errors="coerce")


def promedio(d: pd.DataFrame, columnas: list[str], no_valido: float) -> pd.Series:
    x = d[columnas].apply(num)
    x = x.where(x < no_valido)  # 222.2 / 999 = no se midió
    return x.mean(axis=1)


def main() -> None:
    ad = leer("ensadul2022_entrega_w.csv.csv.zip", [
        "FOLIO_I", "FOLIO_INT", "entidad", "edad", "sexo", "ponde_f",
        "estrato", "a0301", "a0401", "a0701p", "a0701m", "a0701h", "a1301",
        "a1308"])
    an = leer("ensaantro2022_entrega_w.csv.csv.zip", [
        "FOLIO_INT", "an01_1", "an01_2", "an04_1", "an04_2", "an06",
        "an27_02s", "an27_02d", "an27_03s", "an27_03d"])
    ic = leer("ENSANUT2022_ICB.csv.csv.zip", ["FOLIO_I", "nse5F"])

    d = ad.merge(an, on="FOLIO_INT", how="inner", validate="1:1")
    d = d.merge(ic, on="FOLIO_I", how="inner", validate="m:1")

    d["peso"] = promedio(d, ["an01_1", "an01_2"], 222)
    d["talla"] = promedio(d, ["an04_1", "an04_2"], 222)
    d["sistolica"] = promedio(d, ["an27_02s", "an27_03s"], 999)
    d["diastolica"] = promedio(d, ["an27_02d", "an27_03d"], 999)
    d["embarazo"] = num(d["an06"]).isin([1, 2, 3])
    d["dx"] = num(d["a0301"])

    d = d[(d["dx"].isin([1, 3])) & ~d["embarazo"]]
    d = d.dropna(subset=["peso", "talla", "sistolica", "diastolica"])

    sal = pd.DataFrame({
        "diabetes": (d["dx"] == 1).astype(int),
        "factor": num(d["ponde_f"]).round(1),
        "entidad": num(d["entidad"]).astype(int).map(comun.ENTIDADES),
        "tam_localidad": num(d["estrato"]).astype(int).map(TAM_LOCALIDAD),
        "quintil_bienestar": num(d["nse5F"]).astype(int),
        "sexo": num(d["sexo"]).map({1: "hombre", 2: "mujer"}),
        "edad": num(d["edad"]).astype(int),
        "peso_kg": d["peso"].round(1),
        "talla_cm": d["talla"].round(1),
        "imc": (d["peso"] / (d["talla"] / 100) ** 2).round(2),
        "presion_sistolica": d["sistolica"].round(1),
        "presion_diastolica": d["diastolica"].round(1),
        "diagnostico_hipertension": np.where(num(d["a0401"]) == 1, "sí", "no"),
        "padre_diabetes": num(d["a0701p"]).map(SI_NO_NS),
        "madre_diabetes": num(d["a0701m"]).map(SI_NO_NS),
        # " " significa que no tiene hermanos.
        "hermanos_diabetes": num(d["a0701h"]).map(SI_NO_NS).fillna("sin hermanos"),
        "fuma": num(d["a1301"]).map(FUMA),
        "alcohol": num(d["a1308"]).map(ALCOHOL),
    })
    # "No responde" (9) en tabaco y alcohol queda sin categoría: se descartan.
    sal = sal.dropna().reset_index(drop=True)
    comun.guardar(sal, "ensanut_salud.csv.gz")


if __name__ == "__main__":
    main()
