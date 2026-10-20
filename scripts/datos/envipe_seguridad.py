"""Prepara ``datos/envipe_seguridad.csv.gz``: victimización de las personas.

Fuente: Encuesta Nacional de Victimización y Percepción sobre Seguridad
Pública (ENVIPE) 2026, INEGI (datos abiertos; periodo de referencia: el año
2025). Es la edición más reciente publicada al 2026-10-20 (publicada en
septiembre de 2026). Tablas ``tper_vic1``, ``tper_vic2``, ``tsdem``,
``tvivienda`` y ``thogar``.
    https://www.inegi.org.mx/contenidos/programas/envipe/2026/datosabiertos/conjunto_de_datos_envipe_2026_csv.zip
Licencia: datos abiertos de INEGI, Términos de libre uso de la información
del INEGI. Fecha de consulta: 2026-10-20.

Unidad: persona de 18 años o más seleccionada como informante. Objetivo
(``victima_delito``): 1 si la persona declara haber sido víctima, durante
2025, de algún delito personal del Grupo B de la ENVIPE (robo o asalto,
otro robo, fraude, extorsión, amenazas, lesiones, secuestro, hostigamiento o
violación sexual, otros); 0 en otro caso. Se excluyen las 79 personas que
respondieron "no sabe". Los delitos que afectan al hogar (vehículo, casa
habitación) no forman parte del objetivo.

Predictoras: características sociodemográficas de la persona, tamaño del
hogar y dominio (urbano, complemento urbano, rural), y las preguntas de
percepción de la sección IV (temas que preocupan, percepción de seguridad
por lugar, incivilidades y problemas en la colonia, sensación de ser víctima
y expectativas). **No** se incluyen variables que sean consecuencia de la
victimización: medidas de protección y gasto en protección, cambios de
conducta por temor, contacto con autoridades o denuncia, ni nada del módulo
de victimización. Se conserva el factor de expansión (``factor``).

Se toma una muestra aleatoria de ``N_MUESTRA`` personas con semilla fija.

Dependencias extra: ninguna (pandas y numpy de requirements.txt).
Uso: ``python scripts/datos/envipe_seguridad.py``
"""

import zipfile

import pandas as pd

import comun

URL = ("https://www.inegi.org.mx/contenidos/programas/envipe/2026/"
       "datosabiertos/conjunto_de_datos_envipe_2026_csv.zip")
N_MUESTRA = 30000

NIVEL = {0: "ninguno", 1: "preescolar", 2: "primaria", 3: "secundaria",
         4: "carrera técnica con secundaria", 5: "normal básica",
         6: "preparatoria", 7: "carrera técnica con preparatoria",
         8: "licenciatura", 9: "maestría", 10: "doctorado"}
ACTIVIDAD = {1: "trabajó", 2: "tenía trabajo pero no trabajó",
             3: "buscó trabajo", 4: "estudiante", 5: "quehaceres del hogar",
             6: "jubilado o pensionado", 7: "incapacitado para trabajar",
             8: "no trabajó", 9: "no especificado"}
DOMINIO = {"U": "urbano", "C": "complemento urbano", "R": "rural"}
TIEMPO_VIVIENDA = {1: "menos de seis meses", 2: "seis meses a un año",
                   3: "más de un año", 9: "no sabe"}
SEGURIDAD = {1: "seguro", 2: "inseguro", 3: "no aplica", 9: "no sabe"}
SEGURIDAD_NOCHE = {1: "muy seguro", 2: "seguro", 3: "inseguro",
                   4: "muy inseguro", 5: "no aplica", 9: "no sabe"}
SI_NO = {1: "sí", 2: "no", 3: "no aplica", 9: "no sabe"}
TENDENCIA = {1: "mejorará", 2: "seguirá igual de bien",
             3: "seguirá igual de mal", 4: "empeorará", 9: "no sabe"}

# nombre legible -> columna de la ENVIPE
TEMAS = {"preocupa_pobreza": "AP4_2_01", "preocupa_desempleo": "AP4_2_02",
         "preocupa_narcotrafico": "AP4_2_03", "preocupa_precios": "AP4_2_04",
         "preocupa_inseguridad": "AP4_2_05", "preocupa_corrupcion": "AP4_2_08",
         "preocupa_salud": "AP4_2_10"}
SEGURIDAD_LUGAR = {
    "seguridad_colonia": "AP4_3_1", "seguridad_municipio": "AP4_3_2",
    "seguridad_estado": "AP4_3_3", "seguridad_casa": "AP4_4_01",
    "seguridad_trabajo": "AP4_4_02", "seguridad_calle": "AP4_4_03",
    "seguridad_escuela": "AP4_4_04", "seguridad_mercado": "AP4_4_05",
    "seguridad_banco": "AP4_4_07", "seguridad_cajero": "AP4_4_08",
    "seguridad_transporte_publico": "AP4_4_09",
    "seguridad_automovil": "AP4_4_10", "seguridad_parque": "AP4_4_12"}
INCIVILIDADES = {"colonia_alcohol_calle": "AP4_5_01",
                 "colonia_pandillas": "AP4_5_02",
                 "colonia_consumo_drogas": "AP4_5_08",
                 "colonia_robos_asaltos": "AP4_5_09",
                 "colonia_venta_drogas": "AP4_5_10",
                 "colonia_disparos": "AP4_5_11",
                 "colonia_sin_alumbrado": "AP4_5_19"}
SENSACION = {"temor_robo_asalto": "AP4_6_1", "temor_lesiones": "AP4_6_2",
             "temor_extorsion_secuestro": "AP4_6_3"}
EXPECTATIVA = {"expectativa_colonia": "AP4_7_1",
               "expectativa_municipio": "AP4_7_2",
               "expectativa_estado": "AP4_7_3"}
PROBLEMAS = {"problema_alumbrado": "AP4_8_1", "problema_agua": "AP4_8_2",
             "problema_baches": "AP4_8_3", "problema_pandillerismo": "AP4_8_4",
             "problema_robos": "AP4_8_5", "problema_delincuencia": "AP4_8_6"}


def leer(zf: zipfile.ZipFile, tabla: str, columnas: list[str]) -> pd.DataFrame:
    ruta = (f"{tabla}_envipe2026/conjunto_de_datos/"
            f"conjunto_de_datos_{tabla}_envipe2026.csv")
    with zf.open(ruta) as f:
        return pd.read_csv(f, usecols=columnas, low_memory=False)


def main() -> None:
    zip_path = comun.descargar(URL)
    cols1 = (["ID_PER", "ID_HOG", "CVE_ENT", "SEXO", "EDAD", "AP4_1", "AP4_4A", "FAC_ELE",
              "DOMINIO"]
             + list(TEMAS.values()) + list(SEGURIDAD_LUGAR.values())
             + [f"AP4_5_{i:02d}" for i in range(1, 20)]
             + list(SENSACION.values()) + list(EXPECTATIVA.values())
             + list(PROBLEMAS.values()))
    with zipfile.ZipFile(zip_path) as zf:
        p1 = leer(zf, "tper_vic1", cols1)
        p2 = leer(zf, "tper_vic2", ["ID_PER", "AP7_2"])
        sd = leer(zf, "tsdem", ["ID_PER", "NIV", "AP3_8"])
        hg = leer(zf, "thogar", ["ID_HOG", "TOT_PER"])

    d = p1.merge(p2, on="ID_PER", validate="1:1")
    d = d.merge(sd, on="ID_PER", how="left", validate="1:1")
    d = d.merge(hg, on="ID_HOG", how="left", validate="m:1")
    d = d[d["AP7_2"].isin([1, 2])].copy()

    sal = pd.DataFrame({
        "victima_delito": (d["AP7_2"] == 1).astype(int),
        "factor": d["FAC_ELE"],
        "entidad": d["CVE_ENT"].map(comun.ENTIDADES),
        "dominio": d["DOMINIO"].map(DOMINIO),
        "sexo": d["SEXO"].map({1: "hombre", 2: "mujer"}),
        "edad": d["EDAD"],
        "nivel_instruccion": d["NIV"].map(NIVEL).fillna("no sabe"),
        "condicion_actividad": d["AP3_8"].map(ACTIVIDAD),
        "personas_hogar": d["TOT_PER"],
        "tiempo_en_vivienda": d["AP4_1"].map(TIEMPO_VIVIENDA),
    })
    for nombre, col in TEMAS.items():
        sal[nombre] = d[col].astype(int)
    for nombre, col in SEGURIDAD_LUGAR.items():
        sal[nombre] = d[col].map(SEGURIDAD)
    sal["seguridad_caminar_de_noche"] = d["AP4_4A"].map(SEGURIDAD_NOCHE)
    for nombre, col in INCIVILIDADES.items():
        sal[nombre] = d[col].astype(int)
    sal["incivilidades_colonia"] = d[[f"AP4_5_{i:02d}" for i in range(1, 20)]] \
        .sum(axis=1).astype(int)
    for nombre, col in SENSACION.items():
        sal[nombre] = d[col].map(SI_NO)
    for nombre, col in EXPECTATIVA.items():
        sal[nombre] = d[col].map(TENDENCIA)
    for nombre, col in PROBLEMAS.items():
        sal[nombre] = d[col].map(SI_NO)

    sal = sal.reset_index(drop=True)
    faltantes = sal.isna().sum()
    if faltantes.any():
        print("Valores sin recodificar:\n", faltantes[faltantes > 0])
        raise SystemExit(1)
    muestra = sal.sample(n=N_MUESTRA, random_state=comun.SEMILLA).sort_index()
    comun.guardar(muestra, "envipe_seguridad.csv.gz")


if __name__ == "__main__":
    main()
