"""Un modelo de lenguaje como juez sobre el corpus Delitos.

Un modelo generativo de la familia Qwen3.5 decide, sin entrenamiento, si cada
uno de los 1,800 tuits del corpus reporta un delito, y lo hace con tres
redacciones del prompt (``PROMPTS``) que piden lo mismo. La forma sigue a
``juez.py`` del proyecto Chimalli: ``pipeline("text-generation")``, cada
entrada es ``[dict(role='user', content=prompt)]``, generación determinista y
sin razonamiento (``enable_thinking=False``), y la etiqueta se extrae de la
línea ``Veredicto: delito`` o ``Veredicto: no_delito`` con una expresión
regular tolerante (``extraer_veredicto``); lo que no se puede leer queda como
``unk``.

Una corrida por modelo escribe el parcial ``juez-<slug>.csv`` en el caché, con
``particion``, ``indice``, ``modelo``, ``prompt`` y ``etiqueta`` (5,400
renglones). Se escribe por lotes y se reanuda: si el parcial ya tiene
renglones, sólo se calculan los que faltan. No se guarda el texto generado,
porque puede citar el tuit. ``--limite N`` corre sólo los primeros N textos,
escribe en ``juez-<slug>-prueba.csv`` e imprime las respuestas crudas en la
bitácora. ``--unir`` junta los tres parciales en ``delitos_juez.csv.gz``
(16,200 renglones) y se niega a escribir si falta alguno. ``--verificar`` lee
los dos archivos publicados e imprime la tabla de verificación (no carga
ningún modelo).

La corrida del juez necesita GPU: en el cluster se envía con
``scripts/datos/enviar_gpu.sh`` (ver ``README.md``). ``--unir`` y
``--verificar`` corren en cualquier máquina. Las funciones puras sólo usan la
biblioteca estándar; ``torch``, ``transformers`` y ``pandas`` se importan
dentro de las funciones que los usan.
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402
import delitos  # noqa: E402

NOMBRE = "delitos_juez.csv.gz"
MODELOS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")
COLUMNAS = ("particion", "indice", "modelo", "prompt", "etiqueta")
ETIQUETAS = ("delito", "no_delito", "unk")
MAX_NUEVOS = 64

# Las tres redacciones aplican los mismos cuatro criterios de la regla con que
# se etiquetó el corpus: se puede contestar "¿cuál es el delito?", el mensaje
# es un hecho, el objetivo del mensaje es el delito (no es una opinión ni usa
# el delito para comunicar otro evento) y no se trata de un accidente.
PROMPTS = (
    # 1. Regla
    "Vas a clasificar un mensaje de una red social según la siguiente regla. "
    "Un mensaje reporta un delito cuando se cumplen las cuatro condiciones: "
    "se puede contestar la pregunta \"¿cuál es el delito?\", el mensaje "
    "describe un hecho, el objetivo del mensaje es el delito (no es una "
    "opinión ni usa el delito para comunicar otro evento) y no se trata de "
    "un accidente. Si falta cualquiera de ellas, el mensaje no reporta un "
    "delito.\n\n"
    "Aplica la regla al siguiente mensaje.\n\n"
    "Mensaje: {mensaje}\n\n"
    "Responde con una sola oración breve y termina con una línea que diga "
    "exactamente \"Veredicto: delito\" o \"Veredicto: no_delito\".",
    # 2. Pregunta
    "Lee el siguiente mensaje de una red social y contesta primero, en una "
    "oración breve, la pregunta \"¿cuál es el delito que el mensaje reporta "
    "como un hecho?\". El veredicto es no_delito cuando la pregunta no tiene "
    "respuesta, cuando el mensaje es una opinión, cuando el delito sólo "
    "acompaña a otra noticia o cuando lo ocurrido fue un accidente. En "
    "cualquier otro caso el veredicto es delito.\n\n"
    "Mensaje: {mensaje}\n\n"
    "Después de tu respuesta, termina con una línea que diga exactamente "
    "\"Veredicto: delito\" o \"Veredicto: no_delito\".",
    # 3. Rol
    "Eres analista de una secretaría de seguridad pública y tu trabajo es "
    "separar los mensajes de redes sociales que reportan un hecho delictivo "
    "de los demás. Un mensaje es un reporte cuando se puede decir cuál es el "
    "delito, cuenta un hecho, su propósito es informar sobre ese delito (no "
    "es una opinión ni lo usa para hablar de otro evento) y no describe un "
    "accidente.\n\n"
    "Mensaje: {mensaje}\n\n"
    "Explica tu decisión en una sola oración breve y termina con una línea "
    "que diga exactamente \"Veredicto: delito\" o \"Veredicto: no_delito\".",
)

# Tolerante a mayúsculas, asteriscos, corchetes y "no delito" con espacio o
# guion, como PATRON_JUDGMENT de Chimalli; sin anclar al inicio de línea.
PATRON_VEREDICTO = re.compile(
    r"(?i)\bveredicto\s*\**\s*:\s*[\*\[\"'`\s]*(no[\s_-]*delito|delito)\b")


def extraer_veredicto(respuesta: str) -> str:
    """``delito``, ``no_delito`` o ``unk`` a partir de la respuesta del modelo.

    Se usa sólo lo que sigue al último ``</think>`` y, si hay varios
    veredictos, el último."""
    respuesta = respuesta.split("</think>")[-1]
    encontrados = PATRON_VEREDICTO.findall(respuesta)
    if not encontrados:
        return "unk"
    ultimo = encontrados[-1].lower()
    return "delito" if ultimo == "delito" else "no_delito"


def armar_prompt(numero: int, mensaje: str) -> str:
    """Prompt ``numero`` (1, 2 o 3) con el mensaje dentro."""
    return PROMPTS[numero - 1].format(mensaje=mensaje)


def slug(modelo: str) -> str:
    """``Qwen/Qwen3.5-0.8B`` -> ``qwen3.5-0.8b``."""
    return modelo.rstrip("/").rsplit("/", 1)[-1].lower()


def ruta_parcial(modelo: str, prueba: bool = False) -> Path:
    sufijo = "-prueba" if prueba else ""
    return comun.cache_dir() / f"juez-{slug(modelo)}{sufijo}.csv"


def leer_parcial(ruta) -> list[dict]:
    """Renglones válidos de un parcial, sin duplicados (gana el primero).

    Un renglón truncado (un trabajo que murió a media escritura) o con una
    etiqueta desconocida se descarta y se vuelve a calcular."""
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    vistos, filas = set(), []
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if (None in fila or any(fila.get(c) in (None, "") for c in COLUMNAS)
                    or fila["etiqueta"] not in ETIQUETAS
                    or fila["particion"] not in delitos.PARTICIONES
                    or fila["prompt"] not in ("1", "2", "3")
                    or not fila["indice"].isdigit()):
                continue
            clave = (fila["particion"], int(fila["indice"]), int(fila["prompt"]))
            if clave in vistos:
                continue
            vistos.add(clave)
            filas.append(dict(particion=fila["particion"],
                              indice=int(fila["indice"]),
                              modelo=fila["modelo"], prompt=int(fila["prompt"]),
                              etiqueta=fila["etiqueta"]))
    return filas


def escribir_parcial(ruta, filas, agregar: bool = False) -> None:
    ruta = Path(ruta)
    nuevo = not agregar or not ruta.exists()
    with open(ruta, "w" if not agregar else "a", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS, lineterminator="\n")
        if nuevo:
            w.writeheader()
        w.writerows(filas)


def pendientes(regs, hechos) -> list[tuple[int, dict]]:
    """Pares (prompt, registro) que faltan, dado el conjunto de claves
    ``(particion, indice, prompt)`` ya calculadas. Orden fijo: por prompt y
    luego en el orden del corpus."""
    return [(p, r) for p in range(1, len(PROMPTS) + 1) for r in regs
            if (r["particion"], r["indice"], p) not in hechos]


def unir_parciales(parciales: dict, regs) -> list[dict]:
    """Junta los parciales ``{modelo: filas}`` en el orden de ``MODELOS``,
    prompt, partición e índice. Lanza ``ValueError`` si a algún modelo le falta
    algún par (registro, prompt)."""
    orden = {(r["particion"], r["indice"]): i for i, r in enumerate(regs)}
    salida = []
    for modelo in MODELOS:
        filas = parciales.get(modelo, [])
        porclave = {(f["particion"], f["indice"], f["prompt"]): f for f in filas}
        faltan = pendientes(regs, set(porclave))
        if faltan:
            raise ValueError(f"al parcial de {modelo} le faltan {len(faltan)} "
                             f"de {len(regs) * len(PROMPTS)} renglones")
        sobran = [k for k in porclave if k[:2] not in orden]
        if sobran:
            raise ValueError(f"el parcial de {modelo} tiene {len(sobran)} "
                             "renglones que no están en el corpus")
        claves = sorted(porclave, key=lambda k: (k[2], orden[k[:2]]))
        for k in claves:
            salida.append(dict(porclave[k], modelo=slug(modelo)))
    return salida


def argumentos():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modelo", choices=MODELOS,
                   help="identificador de Hugging Face del juez")
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--salida", default=None,
                   help="ruta del parcial (por omisión juez-<slug>.csv en el caché)")
    p.add_argument("--limite", type=int, default=None,
                   help="sólo los primeros N textos; escribe en un archivo de "
                        "prueba aparte e imprime las respuestas crudas")
    p.add_argument("--unir", action="store_true",
                   help=f"junta los tres parciales en {NOMBRE}")
    p.add_argument("--verificar", action="store_true",
                   help="imprime la tabla de verificación de los dos archivos")
    return p.parse_args()


def juzgar(args):
    import torch
    from transformers import GenerationConfig, pipeline

    regs = delitos.registros()
    prueba = args.limite is not None
    if prueba:
        regs = regs[:args.limite]
        ruta = Path(args.salida) if args.salida else ruta_parcial(args.modelo, True)
        if ruta.exists():
            ruta.unlink()
    else:
        ruta = Path(args.salida) if args.salida else ruta_parcial(args.modelo)
    filas = leer_parcial(ruta)
    # Reescribe sin renglones truncados ni duplicados antes de continuar.
    escribir_parcial(ruta, filas)
    hechos = {(f["particion"], f["indice"], f["prompt"]) for f in filas}
    tareas = pendientes(regs, hechos)
    total = len(regs) * len(PROMPTS)
    print(f"{args.modelo}: {len(hechos)} de {total} renglones hechos, "
          f"faltan {len(tareas)}; parcial {ruta}; cuda disponible: "
          f"{torch.cuda.is_available()}", flush=True)
    if not tareas:
        print("nada que calcular.", flush=True)
        return

    pipe = pipeline("text-generation", model=args.modelo, device_map="auto",
                    dtype="auto")
    pipe.tokenizer.padding_side = "left"
    try:
        cfg = GenerationConfig.from_pretrained(args.modelo)
    except OSError:
        cfg = GenerationConfig()
    cfg.do_sample = False
    cfg.temperature = None
    cfg.top_p = None
    cfg.top_k = None
    cfg.max_new_tokens = MAX_NUEVOS

    nombre = slug(args.modelo)
    lotes = [tareas[i:i + args.lote] for i in range(0, len(tareas), args.lote)]
    inicio = time.time()
    for j, lote in enumerate(lotes, 1):
        entradas = [[dict(role="user", content=armar_prompt(p, r["text"]))]
                    for p, r in lote]
        salidas = pipe(entradas, generation_config=cfg,
                       tokenizer_encode_kwargs=dict(enable_thinking=False),
                       batch_size=args.lote)
        nuevas = []
        for (p, r), s in zip(lote, salidas):
            respuesta = s[0]["generated_text"][-1]["content"]
            etiqueta = extraer_veredicto(respuesta)
            if prueba:
                print(f"--- {r['particion']} {r['indice']} prompt {p} -> "
                      f"{etiqueta}\n{respuesta}", flush=True)
            nuevas.append(dict(particion=r["particion"], indice=r["indice"],
                               modelo=nombre, prompt=p, etiqueta=etiqueta))
        escribir_parcial(ruta, nuevas, agregar=True)
        unk = sum(f["etiqueta"] == "unk" for f in nuevas)
        print(f"lote {j}/{len(lotes)}: {len(nuevas)} renglones, {unk} unk, "
              f"{time.time() - inicio:.0f} s", flush=True)

    filas = leer_parcial(ruta)
    unk = sum(f["etiqueta"] == "unk" for f in filas)
    print(f"{ruta}: {len(filas)} de {total} renglones, {unk} unk "
          f"({unk / max(len(filas), 1):.1%})", flush=True)


def unir():
    import pandas as pd
    destino = comun.salida_dir() / NOMBRE
    if destino.exists():
        print(f"{destino} ya existe; no se recalcula.", flush=True)
        return
    regs = delitos.registros()
    parciales = {m: leer_parcial(ruta_parcial(m)) for m in MODELOS}
    try:
        filas = unir_parciales(parciales, regs)
    except ValueError as e:
        raise SystemExit(f"No se escribe {NOMBRE}: {e}")
    comun.guardar(pd.DataFrame(filas, columns=list(COLUMNAS)), NOMBRE)


def verificar():
    """Tabla de verificación del Paso 7 del issue 135."""
    import pandas as pd
    salida = comun.salida_dir()
    regs = pd.DataFrame(delitos.registros())[["particion", "indice", "klass"]]
    ok = True

    ruta = salida / "delitos_qwen3_embedding.csv.gz"
    if ruta.exists():
        emb = pd.read_csv(ruta)
        nan = int(emb.isna().sum().sum())
        print(f"vectores: {emb.shape[0]} renglones, {emb.shape[1]} columnas, "
              f"{nan} NaN, columnas text/id: "
              f"{sorted({'text', 'id'} & set(emb.columns)) or 'ninguna'}")
        ok &= (emb.shape == (1800, 1026) and nan == 0
               and not {"text", "id"} & set(emb.columns))
        ok &= (emb[["particion", "indice"]].values.tolist()
               == regs[["particion", "indice"]].values.tolist())
    else:
        print(f"falta {ruta}")
        ok = False

    ruta = salida / NOMBRE
    if not ruta.exists():
        print(f"falta {ruta}")
        raise SystemExit(1)
    juez = pd.read_csv(ruta)
    print(f"juez: {juez.shape[0]} renglones, columnas {list(juez.columns)}")
    ok &= juez.shape[0] == 16200 and not {"text", "id"} & set(juez.columns)
    conteo = juez.groupby(["modelo", "prompt"]).size()
    unk = juez.assign(u=juez.etiqueta.eq("unk")).groupby(
        ["modelo", "prompt"]).u.mean()
    print("\nrenglones y proporción de unk por modelo y prompt")
    print(pd.DataFrame(dict(renglones=conteo, unk=unk.round(4))).to_string())
    ok &= bool((conteo == 1800).all()) and len(conteo) == 9
    ok &= bool((unk <= 0.10).all())

    ancho = juez.pivot_table(index=["modelo", "particion", "indice"],
                             columns="prompt", values="etiqueta",
                             aggfunc="first").reset_index()
    ancho = ancho.merge(regs, on=["particion", "indice"])
    tabla = []
    for modelo, g in ancho.groupby("modelo", sort=False):
        e = g[[1, 2, 3]]
        consistencia = (e.nunique(axis=1) == 1).mean()
        votos = (e == "delito").sum(axis=1)
        mayoria = (votos >= 2).astype(int)
        exactitud = (mayoria == g.klass).mean()
        tabla.append(dict(modelo=modelo, consistencia=round(consistencia, 4),
                          exactitud_mayoria=round(exactitud, 4)))
    tabla = pd.DataFrame(tabla).set_index("modelo")
    base = (regs.klass == 0).mean()
    print(f"\nconsistencia y exactitud del voto de mayoría (siempre no_delito: "
          f"{base:.4f})")
    print(tabla.to_string())
    if "qwen3.5-4b" in tabla.index:
        ok &= tabla.loc["qwen3.5-4b", "exactitud_mayoria"] >= base
    print(f"\nverificación: {'ok' if ok else 'FALLA'}")
    if not ok:
        raise SystemExit(1)


def main():
    args = argumentos()
    if args.verificar:
        verificar()
    elif args.unir:
        unir()
    elif args.modelo:
        juzgar(args)
    else:
        raise SystemExit("Indica --modelo, --unir o --verificar")


if __name__ == "__main__":
    main()
