"""Prepara ``enigh_pobreza.csv.gz``: focalización de la pobreza por ingresos.

Fuente: Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2022,
Nueva Serie, INEGI (datos abiertos, tablas ``concentradohogar``, ``viviendas``
y ``hogares``).
    https://www.inegi.org.mx/contenidos/programas/enigh/nc/2022/datosabiertos/conjunto_de_datos_enigh_ns_2022_csv.zip
Línea de pobreza por ingresos de agosto de 2022 (CONEVAL, valor mensual por
persona, canasta alimentaria más no alimentaria): rural $2,970.19 y urbano
$4,157.63. Se tomaron de la serie oficial
    https://www.coneval.org.mx/Medicion/Documents/Lineas_de_Pobreza_por_Ingresos/Lineas_de_Pobreza_por_Ingresos.zip
Licencia: datos abiertos de INEGI, Términos de libre uso de la información
del INEGI. Fecha de consulta: 2026-10-20.

Unidad: hogar. Objetivo (``pobre_ingresos``): 1 si el ingreso corriente per
cápita mensual del hogar (``ing_cor`` trimestral entre 3 y entre el número de
integrantes) es menor que la línea de pobreza por ingresos de su ámbito
(rural: localidades de menos de 2,500 habitantes; urbano: el resto). Es una
aproximación a la medición oficial, no la replica. Las predictoras son sólo
características observables de la vivienda y el hogar (lo que usaría un
*proxy means test*): no se incluye ninguna variable de ingreso ni de gasto.
Se conserva el factor de expansión (``factor``) como columna aparte.

Se conservan todos los hogares con información completa, sin muestreo.

Dependencias extra: ninguna (pandas y numpy de requirements.txt).
Uso: ``python scripts/datos/enigh_pobreza.py``
"""

import zipfile

import numpy as np
import pandas as pd

import comun

URL = ("https://www.inegi.org.mx/contenidos/programas/enigh/nc/2022/"
       "datosabiertos/conjunto_de_datos_enigh_ns_2022_csv.zip")
LINEA_RURAL = 2970.19
LINEA_URBANA = 4157.63

TAM_LOCALIDAD = {1: "100 mil o más", 2: "15 mil a 99,999", 3: "2,500 a 14,999",
                 4: "menos de 2,500"}
SEXO = {1: "hombre", 2: "mujer"}
TIPO_HOGAR = {1: "unipersonal", 2: "nuclear", 3: "ampliado", 4: "compuesto",
              5: "corresidente"}
ESCOLARIDAD = {1: "sin instrucción", 2: "preescolar", 3: "primaria incompleta",
               4: "primaria completa", 5: "secundaria incompleta",
               6: "secundaria completa", 7: "preparatoria incompleta",
               8: "preparatoria completa", 9: "profesional incompleta",
               10: "profesional completa", 11: "posgrado"}
TIPO_VIVIENDA = {1: "casa independiente", 2: "departamento", 3: "vecindad",
                 4: "cuarto de azotea", 5: "local no habitacional"}
MAT_PARED = {1: "desecho", 2: "lámina de cartón", 3: "lámina de asbesto o metálica",
             4: "carrizo, bambú o palma", 5: "embarro o bajareque", 6: "madera",
             7: "adobe", 8: "tabique, ladrillo, block o concreto"}
MAT_TECHO = {1: "desecho", 2: "lámina de cartón", 3: "lámina metálica",
             4: "lámina de asbesto", 5: "fibrocemento", 6: "palma o paja",
             7: "madera o tejamanil", 8: "terrado con viguería", 9: "teja",
             10: "losa de concreto"}
MAT_PISO = {1: "tierra", 2: "cemento o firme", 3: "madera, mosaico u otro"}
AGUA = {1: "entubada dentro de la vivienda", 2: "entubada dentro del terreno",
        3: "llave pública o hidrante", 4: "captación de lluvia",
        5: "acarreo de otra vivienda", 6: "pipa", 7: "pozo, río u otra"}
DRENAJE = {1: "red pública", 2: "fosa séptica", 3: "barranca o grieta",
           4: "río, lago o mar", 5: "no tiene"}
ELECTRICIDAD = {1: "servicio público", 2: "planta particular",
                3: "panel solar", 4: "otra fuente", 5: "no tiene"}
COMBUSTIBLE = {1: "leña", 2: "carbón", 3: "gas de tanque",
               4: "gas natural o de tubería", 5: "electricidad", 6: "otro"}
BASURA = {1: "camión o carrito", 2: "basurero público", 3: "contenedor",
          4: "la queman", 5: "la entierran", 6: "terreno baldío o calle",
          7: "barranca o grieta", 8: "río, lago o mar"}
TENENCIA = {1: "rentada", 2: "prestada", 3: "propia, la pagan", 4: "propia",
            5: "intestada o en litigio", 6: "otra"}


def leer(zf: zipfile.ZipFile, tabla: str, columnas: list[str]) -> pd.DataFrame:
    ruta = (f"conjunto_de_datos_{tabla}_enigh2022_ns/conjunto_de_datos/"
            f"conjunto_de_datos_{tabla}_enigh2022_ns.csv")
    with zf.open(ruta) as f:
        return pd.read_csv(f, usecols=columnas, low_memory=False)


def main() -> None:
    zip_path = comun.descargar(URL)
    with zipfile.ZipFile(zip_path) as zf:
        hog = leer(zf, "concentradohogar", [
            "folioviv", "foliohog", "ubica_geo", "tam_loc", "factor",
            "clase_hog", "sexo_jefe", "edad_jefe", "educa_jefe", "tot_integ",
            "menores", "p12_64", "p65mas", "ing_cor"])
        viv = leer(zf, "viviendas", [
            "folioviv", "tipo_viv", "mat_pared", "mat_techos", "mat_pisos",
            "num_cuarto", "cuart_dorm", "disp_agua", "excusado", "drenaje",
            "disp_elect", "combustible", "eli_basura", "tenencia", "tinaco_azo",
            "cisterna", "calent_sol", "calent_gas", "medidor_luz"])
        bienes = ["num_auto", "num_van", "num_pickup", "num_moto", "num_compu",
                  "num_refri", "num_lavad", "num_micro", "num_estuf",
                  "num_tvd"]
        aux = leer(zf, "hogares", ["folioviv", "foliohog", "telefono", "celular",
                                   "tv_paga", "conex_inte"] + bienes)

    d = hog.merge(viv, on="folioviv", how="left", validate="m:1")
    d = d.merge(aux, on=["folioviv", "foliohog"], how="left", validate="1:1")

    # Algunas columnas traen "&" (no especificado) y se leen como texto.
    for col in ["tipo_viv", "mat_pisos", "combustible"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    entidad = d["ubica_geo"] // 1000 if d["ubica_geo"].max() > 99 else d["ubica_geo"]
    rural = d["tam_loc"] == 4
    linea = np.where(rural, LINEA_RURAL, LINEA_URBANA)
    ingreso_pc = d["ing_cor"] / 3 / d["tot_integ"]

    def si_no(col):  # 1 = sí, 2 = no en los cuestionarios de INEGI
        return (d[col] == 1).astype(int)

    sal = pd.DataFrame({
        "pobre_ingresos": (ingreso_pc < linea).astype(int),
        "factor": d["factor"],
        "entidad": entidad.map(comun.ENTIDADES),
        "tam_localidad": d["tam_loc"].map(TAM_LOCALIDAD),
        "ambito": np.where(rural, "rural", "urbano"),
        "tipo_hogar": d["clase_hog"].map(TIPO_HOGAR),
        "integrantes": d["tot_integ"],
        "menores_12": d["menores"],
        "personas_12_64": d["p12_64"],
        "mayores_65": d["p65mas"],
        "sexo_jefe": d["sexo_jefe"].map(SEXO),
        "edad_jefe": d["edad_jefe"],
        "escolaridad_jefe": d["educa_jefe"].map(ESCOLARIDAD),
        "tipo_vivienda": d["tipo_viv"].map(TIPO_VIVIENDA),
        "material_paredes": d["mat_pared"].map(MAT_PARED),
        "material_techo": d["mat_techos"].map(MAT_TECHO),
        "material_piso": d["mat_pisos"].map(MAT_PISO),
        "cuartos": d["num_cuarto"],
        "dormitorios": d["cuart_dorm"],
        "personas_por_cuarto": (d["tot_integ"] / d["num_cuarto"]).round(3),
        "agua": d["disp_agua"].map(AGUA),
        "drenaje": d["drenaje"].map(DRENAJE),
        "electricidad": d["disp_elect"].map(ELECTRICIDAD),
        "combustible_cocina": d["combustible"].map(COMBUSTIBLE),
        "basura": d["eli_basura"].map(BASURA),
        "tenencia": d["tenencia"].map(TENENCIA),
        "tiene_excusado": si_no("excusado"),
        "tiene_tinaco": si_no("tinaco_azo"),
        "tiene_cisterna": si_no("cisterna"),
        "tiene_calentador_solar": si_no("calent_sol"),
        "tiene_calentador_gas": si_no("calent_gas"),
        "tiene_medidor_luz": si_no("medidor_luz"),
        "tiene_telefono_fijo": si_no("telefono"),
        "tiene_celular": si_no("celular"),
        "tiene_tv_paga": si_no("tv_paga"),
        "tiene_internet": si_no("conex_inte"),
    })
    autos = d[["num_auto", "num_van", "num_pickup"]].fillna(0).sum(axis=1)
    sal["vehiculos"] = autos.astype(int)
    for nuevo, col in [("motocicletas", "num_moto"), ("computadoras", "num_compu"),
                       ("refrigeradores", "num_refri"), ("lavadoras", "num_lavad"),
                       ("microondas", "num_micro"), ("estufas", "num_estuf"),
                       ("televisores_digitales", "num_tvd")]:
        sal[nuevo] = d[col].fillna(0).astype(int)

    sal = sal.dropna().reset_index(drop=True)  # unos ~110 hogares con "&"
    comun.guardar(sal, "enigh_pobreza.csv.gz")


if __name__ == "__main__":
    main()
