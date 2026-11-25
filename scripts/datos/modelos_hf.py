"""Descarga y borrado de modelos de Hugging Face con un disco escaso.

El disco del cluster (``/home``) es compartido y tiene poco espacio, así que
los scripts que necesitan GPU (``delitos_qwen.py`` y ``delitos_juez.py``)
trabajan con un modelo a la vez: se descarga desde el nodo de acceso (los
nodos de cómputo corren con ``HF_HUB_OFFLINE=True``), se usa y se borra por
completo. Es la versión mínima de ``download_model``, ``wait_for_disk``,
``cleanup_model`` y ``cleanup_blob_store`` de Chimalli (``submit_jobs.py``).

Uso:

    python3 scripts/datos/modelos_hf.py descargar Qwen/Qwen3.5-4B
    python3 scripts/datos/modelos_hf.py borrar Qwen/Qwen3.5-4B
    python3 scripts/datos/modelos_hf.py           # estado: marca, caché y disco

``descargar`` no hace nada si el modelo ya estaba en la caché de Hugging Face
antes de la corrida (puede estar usándolo otro proyecto). Si no estaba, lo
anota en la marca (``modelos_descargados.txt`` en el caché del proyecto),
espera a que el espacio libre sea al menos el tamaño del modelo más 10 GB,
revisando cada 10 minutos, y corre ``hf download``. Si la descarga falla, borra
lo que haya quedado. Se niega a descargar si la marca ya tiene otro modelo,
para que nunca haya más de uno a la vez.

``borrar`` sólo actúa sobre modelos anotados en la marca. Borra
``models--<org>--<nombre>`` y sus copias en el almacén ``blobs/`` que
``hf download`` 1.x también llena: quita las líneas del modelo de cada
``.refs`` y borra el blob, su ``.refs`` y su ``.lock`` cuando ningún otro
modelo lo usa. Al final lo quita de la marca.

Sólo usa la biblioteca estándar; ``huggingface_hub`` se importa dentro de la
función que consulta el tamaño del modelo.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

MARGEN = 10 * 1024 ** 3
ESPERA = 600
GIB = 1024 ** 3


def hub_dir() -> Path:
    """Caché de Hugging Face (``HF_HUB_CACHE``, ``HF_HOME/hub`` o la de omisión)."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def ruta_marca() -> Path:
    return comun.cache_dir() / "modelos_descargados.txt"


def carpeta_modelo(modelo: str, hub: Path) -> Path:
    """``Qwen/Qwen3.5-4B`` -> ``<hub>/models--Qwen--Qwen3.5-4B``."""
    return Path(hub) / ("models--" + modelo.replace("/", "--"))


def leer_marca(marca: Path) -> list[str]:
    marca = Path(marca)
    if not marca.exists():
        return []
    return [m.strip() for m in marca.read_text(encoding="utf-8").splitlines()
            if m.strip()]


def escribir_marca(marca: Path, modelos: list[str]) -> None:
    Path(marca).write_text("".join(m + "\n" for m in modelos), encoding="utf-8")


def limpiar_blobs(modelo: str, hub: Path) -> list[Path]:
    """Quita ``modelo`` del almacén ``<hub>/blobs`` y regresa los blobs
    borrados (los que ya no usa ningún otro modelo)."""
    almacen = Path(hub) / "blobs"
    if not almacen.is_dir():
        return []
    prefijo = carpeta_modelo(modelo, hub).name + "/"
    borrados = []
    for refs in sorted(almacen.glob("*/*.refs")):
        lineas = [x.strip() for x in refs.read_text(encoding="utf-8").splitlines()
                  if x.strip()]
        quedan = [x for x in lineas if not x.startswith(prefijo)]
        if len(quedan) == len(lineas):
            continue
        blob = refs.with_name(refs.name[:-len(".refs")])
        if quedan:
            refs.write_text("".join(x + "\n" for x in quedan), encoding="utf-8")
            continue
        for ruta in (blob, refs, blob.with_name(blob.name + ".lock")):
            if ruta.is_file():
                ruta.unlink()
        borrados.append(blob)
    return borrados


def borrar(modelo: str, hub: Path | None = None,
           marca: Path | None = None) -> bool:
    """Borra ``modelo`` de la caché si esta corrida lo descargó (está en la
    marca). Regresa ``True`` si lo borró."""
    hub = Path(hub) if hub else hub_dir()
    marca = Path(marca) if marca else ruta_marca()
    anotados = leer_marca(marca)
    if modelo not in anotados:
        print(f"{modelo} no está en {marca}: no lo descargó esta corrida y "
              "no se toca.", flush=True)
        return False
    carpeta = carpeta_modelo(modelo, hub)
    if carpeta.exists():
        shutil.rmtree(carpeta)
    blobs = limpiar_blobs(modelo, hub)
    escribir_marca(marca, [m for m in anotados if m != modelo])
    print(f"borrado {carpeta} y {len(blobs)} blobs del almacén", flush=True)
    return True


def tamano_modelo(modelo: str) -> int:
    from huggingface_hub import HfApi
    info = HfApi().model_info(modelo, files_metadata=True)
    return sum(s.size or 0 for s in info.siblings)


def libre(hub: Path) -> int:
    ruta = Path(hub)
    while not ruta.exists():
        ruta = ruta.parent
    return shutil.disk_usage(ruta).free


def descargar(modelo: str) -> None:
    hub, marca = hub_dir(), ruta_marca()
    anotados = leer_marca(marca)
    carpeta = carpeta_modelo(modelo, hub)
    if carpeta.exists() and modelo not in anotados:
        print(f"{modelo} ya estaba en {hub}; no se descarga ni se borrará.",
              flush=True)
        return
    otros = [m for m in anotados if m != modelo]
    if otros:
        raise SystemExit(f"Ya hay un modelo descargado por esta corrida "
                         f"({', '.join(otros)}); bórralo antes con "
                         "modelos_hf.py borrar.")
    if modelo not in anotados:
        escribir_marca(marca, anotados + [modelo])
    necesario = tamano_modelo(modelo) + MARGEN
    while (hay := libre(hub)) < necesario:
        print(f"esperando espacio para {modelo}: hay {hay / GIB:.1f} GiB libres, "
              f"se necesitan {necesario / GIB:.1f} GiB (modelo + 10 GiB), "
              f"faltan {(necesario - hay) / GIB:.1f} GiB", flush=True)
        time.sleep(ESPERA)
    print(f"descargando {modelo} ({(necesario - MARGEN) / GIB:.2f} GiB), "
          f"libres {libre(hub) / GIB:.1f} GiB", flush=True)
    try:
        subprocess.run(["hf", "download", modelo], check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt):
        borrar(modelo, hub, marca)
        raise
    print(f"listo; libres {libre(hub) / GIB:.1f} GiB", flush=True)


def estado() -> None:
    hub, marca = hub_dir(), ruta_marca()
    print(f"marca {marca}: {leer_marca(marca) or 'vacía'}")
    modelos = sorted(p.name for p in hub.glob("models--*")) if hub.is_dir() else []
    print(f"modelos en {hub}: {modelos or 'ninguno'}")
    blobs = list((hub / "blobs").glob("*/*")) if hub.is_dir() else []
    blobs = [b for b in blobs if not b.name.endswith((".refs", ".lock"))]
    print(f"blobs en el almacén: {len(blobs)}")
    print(f"libres: {libre(hub) / GIB:.1f} GiB")


def main(argv):
    if not argv:
        estado()
    elif len(argv) == 2 and argv[0] == "descargar":
        descargar(argv[1])
    elif len(argv) == 2 and argv[0] == "borrar":
        borrar(argv[1])
    else:
        raise SystemExit("uso: modelos_hf.py [descargar|borrar] <modelo>")


if __name__ == "__main__":
    main(sys.argv[1:])
