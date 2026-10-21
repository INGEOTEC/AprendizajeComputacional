"""Utilidades compartidas por los scripts de preparación de datos.

Cada script de ``scripts/datos/`` descarga la fuente oficial a un caché que
vive **fuera del repositorio** y escribe el archivo procesado en un directorio
de salida, también fuera del repositorio. Los archivos procesados se publican
como assets del release ``datos-v1`` de GitHub (ver ``README.md``); el libro
los lee de ahí.

El caché es ``~/.cache/aprendizaje-computacional/`` y se puede cambiar con la
variable de entorno ``APRENDIZAJE_CACHE``. La salida va por omisión a la
subcarpeta ``salida/`` del caché y se puede cambiar con
``APRENDIZAJE_SALIDA``.

Sólo usa la biblioteca estándar; los scripts que la importan necesitan además
``pandas`` y ``numpy`` (ya incluidos en ``requirements.txt``). Cuando una
descarga pide ``navegador=True`` (el servidor de la SEP rechaza a los clientes
que no se presentan como un navegador, aun con un agente de usuario de
navegador) se usa el paquete opcional ``curl_cffi``.
"""

import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode

SEMILLA = 0
LIMITE_BYTES = 5 * 1024 * 1024

# Claves de entidad federativa (INEGI) y su nombre.
ENTIDADES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas", 8: "Chihuahua",
    9: "Ciudad de México", 10: "Durango", 11: "Guanajuato", 12: "Guerrero",
    13: "Hidalgo", 14: "Jalisco", 15: "México", 16: "Michoacán", 17: "Morelos",
    18: "Nayarit", 19: "Nuevo León", 20: "Oaxaca", 21: "Puebla",
    22: "Querétaro", 23: "Quintana Roo", 24: "San Luis Potosí", 25: "Sinaloa",
    26: "Sonora", 27: "Tabasco", 28: "Tamaulipas", 29: "Tlaxcala",
    30: "Veracruz", 31: "Yucatán", 32: "Zacatecas"}

# Algunos sitios oficiales rechazan el agente por omisión de urllib.
AGENTE = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")


def cache_dir() -> Path:
    """Carpeta del caché de descargas (se crea si no existe)."""
    ruta = Path(os.environ.get("APRENDIZAJE_CACHE",
                               Path.home() / ".cache" / "aprendizaje-computacional"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def salida_dir() -> Path:
    """Carpeta donde se escriben los archivos procesados (se crea si no existe)."""
    ruta = Path(os.environ.get("APRENDIZAJE_SALIDA", cache_dir() / "salida"))
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def _bajar_como_navegador(url: str, parcial: Path) -> None:
    from curl_cffi import requests
    resp = requests.get(url, impersonate="chrome", timeout=600)
    resp.raise_for_status()
    parcial.write_bytes(resp.content)


def descargar(url: str, nombre: str | None = None, datos: dict | None = None,
              navegador: bool = False, intentos: int = 5,
              espera: float = 15.0) -> Path:
    """Descarga ``url`` al caché y regresa la ruta del archivo.

    Es idempotente: si el archivo ya existe en el caché no vuelve a bajarlo.
    ``datos`` hace la petición con POST (formulario). ``navegador`` usa
    ``curl_cffi`` (ver el encabezado del módulo). Se reintenta con espera
    creciente ante errores de red o del servidor.
    """
    destino = cache_dir() / (nombre or url.rstrip("/").rsplit("/", 1)[-1])
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    if navegador:
        try:
            import curl_cffi  # noqa: F401
        except ImportError as e:
            raise RuntimeError("Esta fuente requiere el paquete curl_cffi: "
                               "pip install curl_cffi") from e
    cuerpo = urlencode(datos).encode() if datos is not None else None
    ultimo = None
    for i in range(intentos):
        parcial = destino.with_name(destino.name + ".part")
        try:
            if navegador:
                _bajar_como_navegador(url, parcial)
            else:
                peticion = urllib.request.Request(url, data=cuerpo,
                                                  headers={"User-Agent": AGENTE})
                with urllib.request.urlopen(peticion, timeout=120) as resp, \
                        open(parcial, "wb") as fout:
                    shutil.copyfileobj(resp, fout, length=1 << 20)
            if parcial.stat().st_size == 0:
                raise OSError("respuesta vacía")
            parcial.replace(destino)
            return destino
        except Exception as e:  # red, HTTP o del propio curl_cffi
            ultimo = e
            print(f"  intento {i + 1}/{intentos} falló para {url}: {e}", flush=True)
            time.sleep(espera * (i + 1))
    raise RuntimeError(f"No se pudo descargar {url}: {ultimo}")


def guardar(df, nombre: str) -> Path:
    """Escribe ``df`` en ``salida_dir()/<nombre>`` (CSV, comprimido si el nombre
    termina en ``.gz``) de forma determinista y verifica el límite de tamaño."""
    ruta = salida_dir() / nombre
    if nombre.endswith(".gz"):
        # mtime fijo: el mismo contenido produce exactamente el mismo archivo.
        import gzip
        with open(ruta, "wb") as raw, \
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    else:
        df.to_csv(ruta, index=False, lineterminator="\n")
    tam = ruta.stat().st_size
    if tam >= LIMITE_BYTES:
        raise RuntimeError(f"{ruta} pesa {tam} bytes, supera el límite de 5 MB")
    print(f"{ruta}: {len(df)} filas, {df.shape[1]} columnas, "
          f"{tam / 1024:.0f} KiB", flush=True)
    return ruta
