"""Lectura del corpus Delitos de INGEOTEC (https://ingeotec.github.io/Delitos).

El corpus son 1,800 tuits en español etiquetados por tres personas según
reporten o no un delito. Se distribuye en ``delitos.zip`` (173 KB), cifrado con
la clave ``ingeotec``, que ``zipfile`` de la biblioteca estándar abre sin
paquetes adicionales. Cada archivo es un JSON por línea con las llaves
``annotations``, ``id``, ``klass`` y ``text``.

``registros()`` regresa los 1,800 registros en un orden fijo, primero los 1,440
de entrenamiento y luego los 360 de prueba. Los archivos se leen del zip en
memoria y nunca se escriben descomprimidos a disco. El zip se guarda en el
caché del proyecto (``comun.cache_dir()``); la descarga necesita red, así que
en el cluster se hace desde el nodo de acceso (los nodos de cómputo pueden no
tenerla) con ``python3 scripts/datos/delitos.py``.

Sólo usa la biblioteca estándar.
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

URL = "https://github.com/INGEOTEC/Delitos/releases/download/Datos/delitos.zip"
CLAVE = b"ingeotec"
ARCHIVOS = (("entrenamiento", "delitos/delitos_ingeotec_Es_train.json"),
            ("prueba", "delitos/delitos_ingeotec_Es_test.json"))
TAMANOS = {"entrenamiento": 1440, "prueba": 360}
PARTICIONES = tuple(p for p, _ in ARCHIVOS)


def ruta_zip() -> Path:
    """Ruta de ``delitos.zip`` en el caché (puede no existir)."""
    return comun.cache_dir() / "delitos.zip"


def obtener_zip() -> Path:
    """Regresa la ruta del zip; lo descarga si no está en el caché.

    Si no está y no hay red, termina con un mensaje que explica cómo
    bajarlo desde una máquina con red."""
    ruta = ruta_zip()
    if ruta.exists() and ruta.stat().st_size > 0:
        return ruta
    try:
        return comun.descargar(URL, nombre=ruta.name, intentos=2, espera=5)
    except RuntimeError as e:
        raise SystemExit(
            f"No está {ruta} y no se pudo descargar ({e}). Descárgalo desde "
            "una máquina con red (en el cluster, el nodo de acceso) con "
            "python3 scripts/datos/delitos.py") from e


def leer_zip(ruta) -> list[dict]:
    """Lee los registros del zip en el orden fijo (entrenamiento y prueba).

    Cada registro tiene ``particion``, ``indice`` (posición dentro de su
    archivo, desde 0), ``klass``, ``annotations`` y ``text``."""
    salida = []
    with zipfile.ZipFile(ruta) as zf:
        for particion, nombre in ARCHIVOS:
            contenido = zf.read(nombre, pwd=CLAVE).decode("utf-8")
            lineas = [linea for linea in contenido.splitlines() if linea.strip()]
            for indice, linea in enumerate(lineas):
                d = json.loads(linea)
                salida.append(dict(particion=particion, indice=indice,
                                   klass=int(d["klass"]),
                                   annotations=list(d["annotations"]),
                                   text=d["text"]))
    return salida


def registros() -> list[dict]:
    """Los 1,800 registros del corpus, verificando los tamaños conocidos."""
    salida = leer_zip(obtener_zip())
    for particion, n in TAMANOS.items():
        m = sum(r["particion"] == particion for r in salida)
        if m != n:
            raise RuntimeError(f"Delitos: {particion} tiene {m} registros, "
                               f"se esperaban {n}")
    return salida


if __name__ == "__main__":
    regs = registros()
    for particion in PARTICIONES:
        sub = [r for r in regs if r["particion"] == particion]
        positivos = sum(r["klass"] for r in sub)
        acuerdo = sum(len(set(r["annotations"])) == 1 for r in sub)
        print(f"{particion}: {len(sub)} registros, {len(sub) - positivos} de "
              f"clase 0 y {positivos} de clase 1, las tres personas coinciden "
              f"en {acuerdo}")
    print(f"zip en {ruta_zip()}")
