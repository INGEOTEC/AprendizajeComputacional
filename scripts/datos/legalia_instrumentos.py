"""Matriz de similitud entre instrumentos jurídicos federales (LegalIA).

Recalcula, desde los vectores publicados por el proyecto LegalIA
(https://github.com/INGEOTEC/LegalIA, Apache 2.0), la matriz dirigida de
instrumentos × instrumentos del Atlas con el modelo
``Qwen/Qwen3-Embedding-4B``, la verifica contra el atlas público
(``atlas-qwen3-4b.json``) y escribe dos archivos procesados:

* ``legalia_instrumentos.csv``: un renglón por instrumento, en el orden de la
  matriz (``i``, ``coleccion``, ``clave``, ``nombre``, ``fragmentos``,
  ``fragmentos_contados``, ``peso_entrante``).
* ``legalia_similitud.csv.gz``: la matriz en forma dispersa, un renglón por
  casilla con valor (``origen``, ``destino``, ``peso``).

``A[i, j]`` es el número de fragmentos del instrumento ``i`` cuyo texto más
parecido (similitud coseno, fuera de ``i``) pertenece a ``j``; si el texto
ganador es de ``m`` instrumentos, cada uno recibe ``1/m``.

El cálculo lo hacen los scripts de LegalIA (``scripts/embeddings/``), que se
ejecutan con ``uv`` dentro de un clon de LegalIA; este script sólo los
orquesta. El clon se busca con ``--legalia``, la variable ``LEGALIA_REPO`` o,
por omisión, junto al clon principal de este repositorio
(``../LegalIA``); si no existe se clona en el caché. LegalIA no se modifica:
el directorio de trabajo, los vectores y los corpus viven en el caché
(``~/.cache/aprendizaje-computacional/legalia/``).

La primera corrida descarga del orden de 1 a 2 GB y tarda de 30 minutos a una
hora (``prepare_umap_input.py`` unos 13 minutos; ``instrument_matrix.py`` de
15 a 40 minutos con unos 4.3 GB de memoria). Los dos pasos son idempotentes
(``prepare.done`` e ``instrument-matrix/.done``): una segunda corrida no los
repite y produce archivos idénticos byte a byte.

    python scripts/datos/legalia_instrumentos.py
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import cache_dir, descargar, guardar  # noqa: E402

MODELO = "Qwen/Qwen3-Embedding-4B"
COLECCIONES = ("leyes", "reglamentos", "lineamientos", "tratados")
REPO_LEGALIA = "https://github.com/INGEOTEC/LegalIA"
ATLAS = "https://ingeotec.github.io/LegalIA/pages/atlas/atlas-qwen3-4b.json"

# Cifras del atlas publicado (28 de octubre de 2026) que la matriz debe reproducir.
INSTRUMENTOS = 2670
CONTADOS = 156432
CASILLAS = 47157

# Los pesos del atlas están redondeados a un decimal.
TOLERANCIA_ATLAS = 0.05 + 1e-6

# Instrumentos que pueden diferir del atlas sólo por un empate (ver verificar).
MAX_EMPATES = 5


def raiz_principal() -> Path:
    """Raíz del clon principal de este repositorio (el primer renglón de
    ``git worktree list``), aunque se trabaje en un worktree."""
    aqui = Path(__file__).resolve().parent
    try:
        salida = subprocess.run(["git", "worktree", "list", "--porcelain"],
                                cwd=aqui, check=True, capture_output=True,
                                text=True).stdout
        return Path(salida.splitlines()[0].split(" ", 1)[1])
    except (subprocess.CalledProcessError, IndexError, FileNotFoundError):
        return aqui.parents[1]


def localizar_legalia(ruta: str | None) -> Path:
    """Clon de LegalIA: ``--legalia``, ``LEGALIA_REPO``, el clon hermano o uno
    nuevo en el caché."""
    ruta = ruta or os.environ.get("LEGALIA_REPO")
    if ruta:
        legalia = Path(ruta).expanduser().resolve()
        if not (legalia / "scripts" / "embeddings").is_dir():
            raise SystemExit(f"{legalia} no parece un clon de LegalIA")
        return legalia
    hermano = raiz_principal().parent / "LegalIA"
    if (hermano / "scripts" / "embeddings").is_dir():
        return hermano.resolve()
    clon = cache_dir() / "legalia" / "LegalIA"
    if not (clon / "scripts" / "embeddings").is_dir():
        clon.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO_LEGALIA, str(clon)], check=True)
    return clon


def entorno() -> dict:
    """Variables para los subprocesos: un entorno de Python propio para
    LegalIA y cachés propios de ``legalvec`` y ``scjn``, todo dentro del caché
    del libro y, si ``gh`` está autenticado, un token para no agotar el límite
    de la API de GitHub.

    El entorno propio (``UV_PROJECT_ENVIRONMENT``) importa: ``uv sync`` deja el
    entorno exactamente con las dependencias de LegalIA y desinstala todo lo
    demás, así que sobre el ``.venv`` del clon, que puede ser un enlace a un
    entorno compartido con otros proyectos (este libro incluido), borraría sus
    paquetes."""
    env = dict(os.environ)
    base = cache_dir() / "legalia"
    env["UV_PROJECT_ENVIRONMENT"] = str(base / "venv")
    env.pop("VIRTUAL_ENV", None)
    env["LEGALVEC_CACHE_DIR"] = str(base / "legalvec")
    env["SCJN_CACHE_DIR"] = str(base / "scjn")
    if not (env.get("GH_TOKEN") or env.get("GITHUB_TOKEN")):
        try:
            token = subprocess.run(["gh", "auth", "token"], check=True,
                                   capture_output=True, text=True).stdout.strip()
            if token:
                env["GH_TOKEN"] = token
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return env


def correr(legalia: Path, env: dict, *args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(list(args), cwd=legalia, env=env, check=True)


def calcular(legalia: Path, trabajo: Path) -> None:
    """Corre la cadena de LegalIA si la matriz no está ya calculada."""
    if (trabajo / "instrument-matrix" / ".done").exists():
        print(f"{trabajo / 'instrument-matrix'} ya está calculada", flush=True)
        return
    env = entorno()
    uv = ["uv", "run", "--frozen", "--group", "viz"]
    correr(legalia, env, "uv", "sync", "--frozen", "--group", "viz")
    if not (trabajo / "prepare.done").exists():
        correr(legalia, env, *uv, "legalvec", "download", "--collection", "all",
               "--model", MODELO)
        for coleccion in COLECCIONES:
            correr(legalia, env, *uv, "scjn", "download", "--coleccion", coleccion)
        correr(legalia, env, *uv, "python", "scripts/embeddings/prepare_umap_input.py",
               "--work-dir", str(trabajo), "--model", MODELO, "--unique-names")
    correr(legalia, env, *uv, "python", "scripts/embeddings/instrument_matrix.py",
           "--work-dir", str(trabajo))
    if not (trabajo / "instrument-matrix" / ".done").exists():
        raise SystemExit("instrument_matrix.py terminó sin escribir .done")


def r_ok(fila: np.ndarray, out: list, tolerancia: float) -> bool:
    """Los pesos de ``out`` (pares ``[destino, peso]`` del atlas) coinciden con
    ``fila`` dentro de ``tolerancia``."""
    return all(abs(fila[j] - w) <= tolerancia for j, w in out)


def verificar(instrumentos: pd.DataFrame, A: np.ndarray, resumen: dict) -> None:
    """Compara la matriz recalculada con el atlas público; ``SystemExit`` con
    lo que difiere."""
    atlas = json.loads(descargar(ATLAS, "legalia/atlas-qwen3-4b.json")
                       .read_text(encoding="utf-8"))
    errores = []
    if atlas["meta"].get("model") != MODELO:
        errores.append(f"el atlas es de {atlas['meta'].get('model')}, no de {MODELO}")
    ref = atlas["instruments"]
    if len(instrumentos) != INSTRUMENTOS or len(ref) != INSTRUMENTOS:
        errores.append(f"instrumentos: {len(instrumentos)} recalculados, "
                       f"{len(ref)} en el atlas, se esperaban {INSTRUMENTOS}")
    else:
        orden = [(r["c"], r["k"]) for r in ref]
        propio = list(zip(instrumentos["coleccion"], instrumentos["clave"].astype(str)))
        distintos = sum(a != b for a, b in zip(orden, propio))
        if distintos:
            errores.append(f"{distintos} instrumentos en distinta posición que en el atlas")
    if resumen.get("counted_rows") != CONTADOS:
        errores.append(f"counted_rows {resumen.get('counted_rows')}, se esperaba {CONTADOS}")
    casillas = int(np.count_nonzero(A))
    if casillas != CASILLAS:
        errores.append(f"casillas con valor {casillas}, se esperaban {CASILLAS}")
    if np.any(np.diag(A) != 0):
        errores.append("la diagonal no es cero")
    if abs(A.sum() - CONTADOS) > 0.5:
        errores.append(f"la suma de la matriz es {A.sum():.3f}, se esperaba {CONTADOS}")
    if not errores:
        vecinos = []
        for i, r in enumerate(ref):
            fila = A[i]
            k = len(r["out"])
            nnz = int(np.count_nonzero(fila))
            mayores = np.sort(fila[fila > 0])[::-1][:k]
            ok = (k == min(5, nnz)
                  and np.allclose(mayores, [w for _, w in r["out"]], atol=TOLERANCIA_ATLAS, rtol=0)
                  and all(abs(fila[j] - w) <= TOLERANCIA_ATLAS for j, w in r["out"])
                  and r["p"] == int(instrumentos["units"].iloc[i]))
            if not ok:
                vecinos.append(i)
        # Un empate en la frontera de la tolerancia de LegalIA (1e-6 sobre la
        # similitud coseno en float32) puede resolverse distinto con otra
        # biblioteca de álgebra lineal: un fragmento reparte su voto 1/2 y 1/2
        # en un equipo y 1 y 0 en otro. Se aceptan pocos casos así, siempre que
        # los fragmentos contados del instrumento coincidan y ningún peso se
        # mueva más de medio voto; cualquier otra diferencia detiene el script.
        empates = [i for i in vecinos
                   if abs(A[i].sum() - ref[i]["pc"]) <= 0.5
                   and r_ok(A[i], ref[i]["out"], 0.5 + TOLERANCIA_ATLAS)]
        for i in empates:
            r = ref[i]
            propios = [[j, round(float(A[i, j]), 2)] for j, _ in r["out"]]
            print(f"empate: {r['c']}/{r['k']} ({r['n']}) tiene {propios}, "
                  f"el atlas {r['out']}", flush=True)
        otros = [i for i in vecinos if i not in empates]
        if len(empates) > MAX_EMPATES:
            errores.append(f"{len(empates)} instrumentos difieren por empates, "
                           f"se aceptan a lo más {MAX_EMPATES}")
        if otros:
            errores.append(f"{len(otros)} instrumentos no coinciden en sus cinco destinos "
                           f"más fuertes o en sus fragmentos (primeros: {otros[:10]})")
    if errores:
        raise SystemExit("La matriz recalculada no coincide con el atlas público:\n  "
                         + "\n  ".join(errores))
    print("verificación contra el atlas público: correcta", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legalia", default=None,
                        help="clon de LegalIA (por omisión LEGALIA_REPO o ../LegalIA)")
    args = parser.parse_args(argv)

    legalia = localizar_legalia(args.legalia)
    commit = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=legalia,
                            check=True, capture_output=True, text=True).stdout.strip()
    print(f"LegalIA: {legalia} (commit {commit})", flush=True)

    trabajo = (cache_dir() / "legalia" / "emb-run-atlas-4b").resolve()
    trabajo.mkdir(parents=True, exist_ok=True)
    calcular(legalia, trabajo)

    salida = trabajo / "instrument-matrix"
    A = np.load(salida / "matrix.npy").astype(np.float64)
    resumen = json.loads((salida / "matrix.json").read_text(encoding="utf-8"))
    # pyarrow directo y no pd.read_parquet: no depende de que las versiones de
    # pandas y pyarrow instaladas sean compatibles entre sí.
    import pyarrow.parquet as pq
    instrumentos = pd.DataFrame(pq.read_table(trabajo / "instruments.parquet").to_pydict())
    instrumentos = instrumentos.sort_values("i").reset_index(drop=True)
    if not np.array_equal(instrumentos["i"].to_numpy(), np.arange(len(instrumentos))):
        raise SystemExit("instruments.parquet no numera sus renglones de 0 a n-1")
    if A.shape != (len(instrumentos), len(instrumentos)):
        raise SystemExit(f"la matriz es {A.shape} y hay {len(instrumentos)} instrumentos")
    verificar(instrumentos, A, resumen)

    # Cada fragmento contado reparte un peso total de 1: la suma de un renglón
    # es un entero salvo por el redondeo de float32.
    contados = np.rint(A.sum(axis=1))
    if np.abs(A.sum(axis=1) - contados).max() > 1e-2:
        raise SystemExit("la suma de algún renglón no es un número entero de fragmentos")
    tabla = pd.DataFrame({
        "i": instrumentos["i"].astype(int),
        "coleccion": instrumentos["coleccion"].astype(str),
        "clave": instrumentos["clave"].astype(str),
        "nombre": instrumentos["nombre"].astype(str),
        "fragmentos": instrumentos["units"].astype(int),
        "fragmentos_contados": contados.astype(int),
        "peso_entrante": np.round(A.sum(axis=0), 4),
    })
    origen, destino = np.nonzero(A)  # orden por renglón: origen y luego destino
    similitud = pd.DataFrame({"origen": origen, "destino": destino,
                              "peso": np.round(A[origen, destino], 6)})
    # `nombre` es el nombre de un instrumento jurídico, no de una persona.
    guardar(tabla, "legalia_instrumentos.csv", no_personas=("nombre",))
    guardar(similitud, "legalia_similitud.csv.gz")
    print(f"suma de peso {similitud['peso'].sum():.3f}; LegalIA commit {commit}", flush=True)


if __name__ == "__main__":
    main()
