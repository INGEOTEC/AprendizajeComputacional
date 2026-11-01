#!/usr/bin/env python3
"""Comprueba que el código que el lector ve en un capítulo se puede ejecutar.

Los capítulos tienen chunks ocultos (`#| echo: false`, `#| include: false` o
`#| eval: false`) donde se definen variables que luego usa el código mostrado.
Quarto los ejecuta todos en orden, así que el libro se renderiza bien, pero un
lector que copia solo el código visible obtiene un `NameError`. Este script
reproduce ese caso: extrae únicamente los chunks ```{python} visibles, descarta
las líneas `#|`, los concatena en un solo programa (cada uno precedido por un
comentario `# --- chunk @ línea N ---`) y lo ejecuta en un directorio temporal
con `MPLBACKEND=Agg`.

Convención: toda variable usada en un chunk mostrado debe definirse en un
chunk mostrado anterior; los chunks ocultos son solo para `Markdown(...)`,
estilos y cálculos que alimentan texto en línea.

Uso, desde cualquier lugar del repositorio:

    python scripts/check_visible.py                                # todo el libro
    python scripts/check_visible.py capitulos/13Agrupamiento.qmd   # un capítulo

Sin argumentos revisa los .qmd de `book.chapters` y `book.appendices` en
_quarto.yml. Sale con código distinto de cero si algún capítulo falla; si pasa,
imprime `ok capitulos/<nombre>.qmd`. Algunos capítulos (p. ej.
20ConjuntosDatos.qmd) descargan datos, por lo que necesitan red.

Solo usa la biblioteca estándar para extraer el código; ejecutarlo requiere
las dependencias de requirements.txt.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUARTO_YML = REPO / "_quarto.yml"

FENCE_OPEN = re.compile(r"^```\{python[^}]*\}\s*$")
FENCE_CLOSE = re.compile(r"^```\s*$")
HIDDEN = re.compile(r"^#\|\s*(echo|include|eval)\s*:\s*(false|no)\s*$")
MARKER = "# --- chunk @ línea "


def pages() -> list[Path]:
    """Los .qmd que lista _quarto.yml, en orden."""
    text = QUARTO_YML.read_text(encoding="utf-8")
    entries = re.findall(r"^\s*-\s+(\S+\.qmd)\s*$", text, re.MULTILINE)
    return [Path(entry) for entry in dict.fromkeys(entries)]


def visible_chunks(path: Path) -> list[tuple[int, list[str]]]:
    """Los chunks ```{python} visibles de `path` como (línea, código)."""
    chunks = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        if not FENCE_OPEN.match(lines[i]):
            i += 1
            continue
        start = i + 1
        i += 1
        body = []
        while i < len(lines) and not FENCE_CLOSE.match(lines[i]):
            body.append(lines[i])
            i += 1
        i += 1
        options = [line for line in body if line.startswith("#|")]
        if any(HIDDEN.match(line) for line in options):
            continue
        code = [line for line in body if not line.startswith("#|")]
        chunks.append((start + 1, code))
    return chunks


def build_program(chunks: list[tuple[int, list[str]]]) -> str:
    parts = []
    for line, code in chunks:
        parts.append(f"{MARKER}{line} ---")
        parts.extend(code)
        parts.append("")
    return "\n".join(parts)


def culprit(program: str, stderr: str) -> int | None:
    """La línea del .qmd del chunk donde falló, según el traceback."""
    hits = re.findall(r'File ".*?", line (\d+)', stderr)
    if not hits:
        return None
    # el último frame del script principal es el más cercano al error
    program_lines = program.splitlines()
    for hit in reversed(hits):
        n = int(hit)
        if n > len(program_lines):
            continue
        for k in range(n - 1, -1, -1):
            if program_lines[k].startswith(MARKER):
                return int(program_lines[k][len(MARKER):].split()[0])
    return None


def check(page: Path) -> bool:
    path = page if page.is_absolute() else REPO / page
    program = build_program(visible_chunks(path))
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "visible.py"
        script.write_text(program, encoding="utf-8")
        env = dict(os.environ, MPLBACKEND="Agg")
        result = subprocess.run(
            [sys.executable, "-I", str(script)],
            cwd=tmp, env=env, capture_output=True, text=True,
        )
    try:
        shown = path.relative_to(REPO)
    except ValueError:
        shown = path
    if result.returncode == 0:
        print(f"ok {shown}", flush=True)
        return True
    print(f"FALLA {shown}", flush=True)
    print(result.stderr, file=sys.stderr, end="")
    line = culprit(program, result.stderr)
    if line is not None:
        print(f"chunk culpable: {shown}:{line}", file=sys.stderr)
    return False


def main() -> int:
    targets = [Path(arg) for arg in sys.argv[1:]] or pages()
    failed = [page for page in targets if not check(page)]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
