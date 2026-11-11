#!/usr/bin/env python3
"""Despliega una edición del libro en su carpeta de un checkout de gh-pages.

El sitio publicado en GitHub Pages (rama ``gh-pages``) contiene tres ediciones
que provienen de ramas distintas:

    raíz    edición vigente          rama master
    dev/    borrador                 rama develop
    mcdi/   edición histórica MCDI   rama mcdi (congelada)

Cada publicación reemplaza solo la carpeta de su edición y deja intactas las
otras. Además, siempre escribe en la raíz ``.nojekyll`` y ``404.html`` (copia
de ``ediciones/404.html``), que son comunes a todo el sitio.

Uso (lo llama .github/workflows/publish.yml; el commit y el push los hace el
flujo):

    python3 scripts/sitio.py <directorio_renderizado> <checkout_gh_pages> <raiz|dev|mcdi>

Solo usa la biblioteca estándar. Pruebas: ``python3 scripts/test_sitio.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Destino -> subcarpeta de gh-pages ("" es la raíz). Único lugar donde se
# definen las carpetas de las ediciones: las subcarpetas son las que un
# despliegue a la raíz nunca debe borrar.
DESTINOS = {"raiz": "", "dev": "dev", "mcdi": "mcdi"}
CARPETAS_PROTEGIDAS = frozenset(c for c in DESTINOS.values() if c) | {".git"}

PAGINA_404 = REPO / "ediciones" / "404.html"


class ErrorDespliegue(Exception):
    """Entrada inválida para el despliegue."""


def _vaciar(directorio: Path, conservar: frozenset[str] = frozenset()) -> None:
    """Borra el contenido de ``directorio`` salvo los nombres en ``conservar``."""
    for entrada in directorio.iterdir():
        if entrada.name in conservar:
            continue
        if entrada.is_dir() and not entrada.is_symlink():
            shutil.rmtree(entrada)
        else:
            entrada.unlink()


def desplegar(
    render: Path, gh_pages: Path, destino: str, pagina_404: Path = PAGINA_404
) -> Path:
    """Copia ``render`` en la carpeta de ``destino`` dentro de ``gh_pages``.

    Devuelve la carpeta escrita.
    """
    render, gh_pages = Path(render), Path(gh_pages)
    if destino not in DESTINOS:
        raise ErrorDespliegue(
            f"destino desconocido {destino!r}; debe ser uno de: {', '.join(DESTINOS)}"
        )
    if not (render / "index.html").is_file():
        raise ErrorDespliegue(
            f"{render} no contiene index.html; ¿se renderizó el libro?"
        )
    if not gh_pages.is_dir():
        raise ErrorDespliegue(f"{gh_pages} no es un directorio")
    if not Path(pagina_404).is_file():
        raise ErrorDespliegue(f"no existe la página 404 {pagina_404}")
    if destino == "raiz":
        reservadas = sorted(e.name for e in render.iterdir() if e.name in CARPETAS_PROTEGIDAS)
        if reservadas:
            # Copiarlas pisaría otra edición (o el .git del checkout).
            raise ErrorDespliegue(
                f"el render contiene {', '.join(reservadas)}, reservado para otra edición"
            )

    carpeta = gh_pages / DESTINOS[destino]
    if destino == "raiz":
        _vaciar(gh_pages, conservar=CARPETAS_PROTEGIDAS)
    elif carpeta.exists():
        shutil.rmtree(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    for entrada in render.iterdir():
        if entrada.is_dir():
            shutil.copytree(entrada, carpeta / entrada.name)
        else:
            shutil.copy2(entrada, carpeta / entrada.name)

    (gh_pages / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(pagina_404, gh_pages / "404.html")
    return carpeta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("render", type=Path, help="directorio renderizado (_book)")
    parser.add_argument("gh_pages", type=Path, help="checkout de la rama gh-pages")
    parser.add_argument("destino", help=f"uno de: {', '.join(DESTINOS)}")
    args = parser.parse_args(argv)
    try:
        carpeta = desplegar(args.render, args.gh_pages, args.destino)
    except ErrorDespliegue as error:
        print(f"sitio.py: error: {error}", file=sys.stderr)
        return 1
    print(f"sitio.py: {args.render} -> {carpeta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
