"""Vectores de Qwen3-Embedding-0.6B para los 1,800 tuits del corpus Delitos.

Escribe ``delitos_qwen3_embedding.csv.gz`` en la carpeta de salida
(``comun.salida_dir()``) con las columnas ``particion``, ``indice`` y ``v0`` a
``v1023``, un renglón por tuit en el orden de ``delitos.registros()``. No lleva
el texto ni el identificador de los tuits: el libro une por ``particion`` e
``indice`` con el corpus que descarga de su fuente.

La codificación sigue a LegalIA (``scripts/embeddings/encode_shard.py``), que
es una sola llamada a ``transformers.pipeline("feature-extraction")``. El
vector de un texto es el estado oculto del último segmento (*last-token
pooling*), por lo que el relleno tiene que ir a la izquierda. No se antepone
instrucción al texto y el vector no se normaliza. Antes de escribir, el script
comprueba que los primeros 8 textos codificados en lote y codificados uno por
uno tengan similitud coseno mayor que 0.999 (la prueba de que el relleno quedó
bien); si no, termina con error sin escribir.

Necesita GPU: en el cluster se envía con ``scripts/datos/enviar_gpu.sh`` (ver
``README.md``), después de descargar el modelo con ``modelos_hf.py`` y el
corpus con ``delitos.py`` desde el nodo de acceso. Si la salida ya existe, no
recalcula nada.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402
import delitos  # noqa: E402

NOMBRE = "delitos_qwen3_embedding.csv.gz"
MODELO = "Qwen/Qwen3-Embedding-0.6B"
DIMENSION = 1024
N_RELLENO = 8
UMBRAL_RELLENO = 0.999


def argumentos():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modelo", default=MODELO)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--dispositivo", default=None,
                   help="p. ej. cuda:0 o cpu; por omisión device_map='auto'")
    return p.parse_args()


def codificar(pipe, textos, lote):
    """Matriz float32 (n, d) con el último segmento de cada texto."""
    import numpy as np
    salidas = pipe(textos, batch_size=lote, return_tensors=True,
                   truncation=True, max_length=512)
    return np.stack([s[0][-1].float().cpu().numpy() for s in salidas]
                    ).astype(np.float32)


def coseno(a, b):
    import numpy as np
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1)
                                  * np.linalg.norm(b, axis=1))


def main():
    args = argumentos()
    destino = comun.salida_dir() / NOMBRE
    if destino.exists():
        print(f"{destino} ya existe; no se recalcula.", flush=True)
        return
    import numpy as np
    import pandas as pd
    import torch
    from transformers import pipeline

    regs = delitos.registros()
    textos = [r["text"] for r in regs]
    print(f"{len(textos)} textos; cuda disponible: "
          f"{torch.cuda.is_available()}", flush=True)
    colocacion = (dict(device=args.dispositivo) if args.dispositivo
                  else dict(device_map="auto"))
    pipe = pipeline("feature-extraction", model=args.modelo, dtype="auto",
                    **colocacion)
    pipe.tokenizer.padding_side = "left"

    inicio = time.time()
    X = codificar(pipe, textos, args.lote)
    print(f"codificados en lote ({args.lote}) en {time.time() - inicio:.0f} s; "
          f"forma {X.shape}", flush=True)
    uno = codificar(pipe, textos[:N_RELLENO], 1)
    cos = coseno(X[:N_RELLENO], uno)
    print(f"prueba de relleno: coseno lote contra uno por uno, mínimo "
          f"{cos.min():.6f}, máximo {cos.max():.6f} (umbral {UMBRAL_RELLENO})",
          flush=True)

    errores = []
    if X.shape != (len(regs), DIMENSION):
        errores.append(f"forma {X.shape}, se esperaba ({len(regs)}, {DIMENSION})")
    if np.isnan(X).any():
        errores.append("hay NaN en los vectores")
    if not (cos > UMBRAL_RELLENO).all():
        errores.append("la prueba de relleno a la izquierda falló")
    if errores:
        raise SystemExit("No se escribe nada: " + "; ".join(errores))

    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
    df.insert(0, "indice", [r["indice"] for r in regs])
    df.insert(0, "particion", [r["particion"] for r in regs])
    comun.guardar(df, NOMBRE, float_format="%.6g")


if __name__ == "__main__":
    main()
