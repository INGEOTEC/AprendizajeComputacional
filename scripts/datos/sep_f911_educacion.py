"""Prepara ``sep_f911_educacion.csv.gz``: abandono escolar por secundaria.

Fuente: Estadística del Formato 911 de la Secretaría de Educación Pública
(SEP), "Registro del alumnado y personal docente de educación básica, media
superior y superior (formato 911)", publicada en datos.gob.mx por la SEP con
licencia Creative Commons Atribución 4.0 (CC-BY-4.0); los archivos están en
el repositorio de datos abiertos de la SEP. Se usan dos ciclos de educación
básica (inicio de cursos), nivel secundaria:
  - 2023-2024: https://repodatos.atdt.gob.mx/s_educacion_publica/f911/ESTANDAR_BASICA_I2324.csv
  - 2024-2025: https://repodatos.atdt.gob.mx/api_update/secretaria_educacion/registro_alumnado_personal_docente_educacion_basica_media_superior_formato_911/educacion_basica_2024_2025.csv
    (es el ciclo más reciente disponible al 2026-10-20)
Catálogo: https://www.datos.gob.mx/dataset/registro_alumnado_personal_docente_educacion_basica_media_superior_formato_911
Fecha de consulta: 2026-10-20.

Unidad: secundaria-turno (clave de centro de trabajo, ``clave_cct``, más
turno). Objetivo (``tasa_abandono``): tasa de abandono escolar del ciclo
2023-2024 calculada con la definición de la SEP, que compara matrículas de
ciclos consecutivos:

    abandono = 100 * (1 - (M[t+1] - NI[t+1]) / (M[t] - E[t]))

con M[t] la matrícula total al inicio del ciclo 2023-2024, M[t+1] la del ciclo
2024-2025, NI[t+1] los alumnos de nuevo ingreso a primer grado en 2024-2025 y
E[t] los egresados de 2023-2024 (el archivo de inicio de cursos de 2024-2025
reporta los egresados del ciclo anterior). Entre planteles hay traslados, de
modo que a nivel de plantel la tasa puede ser negativa (el plantel recibe más
alumnos de los que pierde); se acota a 0 y el script reporta cuántos
planteles se acotaron.

Predictoras: características del plantel en el ciclo 2023-2024 (entidad,
sostenimiento, tipo de servicio, turno, organización de docentes, alumnos,
grupos, docentes, alumnos por docente y por grupo, proporción de mujeres, de
alumnos de primer grado y de repetidores, aulas en uso, laboratorios y
talleres). No se usa ninguna cifra del ciclo 2024-2025 como predictora.

Se descartan los planteles que no aparecen en los dos ciclos y los que tienen
menos de ``MIN_BASE`` alumnos en ``M[t] - E[t]`` (las tasas son muy
inestables) o que no reportan docentes o grupos. No se incluyen nombres de escuelas ni domicilios.

Dependencias extra: ``curl_cffi`` (``pip install curl_cffi``): el servidor de
la SEP rechaza a los clientes que no se presentan como un navegador
(respuesta "Access Denied"); el paquete sólo se usa para descargar.
Uso: ``python scripts/datos/sep_f911_educacion.py``
"""

import numpy as np
import pandas as pd

import comun

URL_2324 = "https://repodatos.atdt.gob.mx/s_educacion_publica/f911/ESTANDAR_BASICA_I2324.csv"
URL_2425 = ("https://repodatos.atdt.gob.mx/api_update/secretaria_educacion/"
            "registro_alumnado_personal_docente_educacion_basica_media_superior_"
            "formato_911/educacion_basica_2024_2025.csv")
MIN_BASE = 30

TEXTO = {  # el ciclo 2023-2024 viene en mayúsculas
    "PÚBLICO": "público", "PRIVADO": "privado",
    "FEDERAL": "federal", "FEDERAL TRANSFERIDO": "federal transferido",
    "FEDERAL TRANSFERIDO-ESTATAL": "federal transferido-estatal",
    "ESTATAL": "estatal", "PARTICULAR": "particular", "SUBSIDIO": "subsidio",
    "AUTÓNOMO": "autónomo",
    "TELESECUNDARIA": "telesecundaria", "SECUNDARIA GENERAL": "general",
    "SECUNDARIA TECNICA": "técnica", "SECUNDARIA CONAFE": "Conafe",
    "MATUTINO": "matutino", "VESPERTINO": "vespertino",
    "NOCTURNO": "nocturno", "DISCONTINUO": "discontinuo",
    "CONTINUO": "continuo"}


def leer(url: str, columnas: list[str]) -> pd.DataFrame:
    ruta = comun.descargar(url, navegador=True)
    cols = pd.read_csv(ruta, nrows=0).columns
    # el ciclo 2024-2025 nombra ins_01, ins_02...; el 2023-2024, ins_1, ins_2...
    pedidas = [c for c in columnas if c in cols]
    d = pd.read_csv(ruta, usecols=pedidas, low_memory=False,
                    dtype={"turno": str, "entidad": str})
    return d[d["nivel"].str.upper() == "SECUNDARIA"].copy()


def main() -> None:
    a = leer(URL_2324, [
        "entidad", "clavecct", "turno", "nivel", "subnivel", "control",
        "subcontrol", "tipo_org_docente", "muj_t",
        "ins_1", "rei_t", "tot_doc", "gpos_t", "aula_u_t", "laborat", "taller",
        "ins_t"]).rename(columns={"clavecct": "clave_cct"})
    b = leer(URL_2425, ["clave_cct", "turno", "nivel", "ins_t", "ing_01",
                        "egre_tot"])
    b = b.rename(columns={"ins_t": "matricula_sig", "ing_01": "ingreso_sig",
                          "egre_tot": "egresados"}).drop(columns="nivel")
    d = a.merge(b, on=["clave_cct", "turno"], how="inner", validate="1:1")

    base = d["ins_t"] - d["egresados"]
    bruta = 100 * (1 - (d["matricula_sig"] - d["ingreso_sig"]) / base)
    ok = (base >= MIN_BASE) & (d["tot_doc"] > 0) & (d["gpos_t"] > 0)
    d, bruta = d[ok], bruta[ok]
    print(f"planteles en los dos ciclos con base >= {MIN_BASE} y con docentes y grupos: {len(d)}; "
          f"tasa negativa acotada a 0 en {int((bruta < 0).sum())}")

    sal = pd.DataFrame({
        "tasa_abandono": bruta.clip(lower=0).round(3),
        "clave_cct": d["clave_cct"],
        "entidad": d["entidad"].astype(int).map(comun.ENTIDADES),
        "sostenimiento": d["control"].map(TEXTO),
        "tipo_sostenimiento": d["subcontrol"].map(TEXTO),
        "tipo_servicio": d["subnivel"].map(TEXTO),
        "turno": d["turno"].astype(int).map({1: "matutino", 2: "vespertino",
                                              3: "nocturno", 4: "discontinuo",
                                              5: "continuo"}),
        "organizacion_docentes": d["tipo_org_docente"].str.lower(),
        "alumnos": d["ins_t"],
        "grupos": d["gpos_t"],
        "docentes": d["tot_doc"],
        "alumnos_por_docente": (d["ins_t"] / d["tot_doc"]).round(3),
        "alumnos_por_grupo": (d["ins_t"] / d["gpos_t"]).round(3),
        "pct_mujeres": (100 * d["muj_t"] / d["ins_t"]).round(3),
        "pct_primer_grado": (100 * d["ins_1"] / d["ins_t"]).round(3),
        "pct_repetidores": (100 * d["rei_t"] / d["ins_t"]).round(3),
        "aulas_en_uso": d["aula_u_t"],
        "laboratorios": d["laborat"],
        "talleres": d["taller"],
    })
    sal = sal.replace([np.inf, -np.inf], np.nan)
    faltantes = sal.isna().sum()
    if faltantes.any():
        print("Valores sin recodificar o inválidos:\n", faltantes[faltantes > 0])
        raise SystemExit(1)
    sal = sal.sort_values(["clave_cct", "turno"], kind="stable") \
             .reset_index(drop=True)
    comun.guardar(sal, "sep_f911_educacion.csv.gz")


if __name__ == "__main__":
    main()
