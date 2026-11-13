#!/usr/bin/env python3
"""Ningún capítulo de la edición vigente se llama como uno de la edición MCDI.

Antes de separar las ediciones, el curso de la MCDI vivía en la raíz del
sitio, así que hay ligas externas a
``/capitulos/<capítulo MCDI>.html``. Esas ligas deben caer en la página 404
(``ediciones/404.html``), que avisa que el libro se reorganizó; si un
capítulo de la edición vigente tuviera el mismo nombre de archivo, la liga
mostraría sin aviso otro contenido.

La lista del MCDI es fija porque la rama ``mcdi`` está congelada. Se usa
solo la biblioteca estándar (``_quarto.yml`` se lee con una expresión
regular), sin red ni Quarto::

    python3 scripts/test_nombres.py
"""
import re
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# Capítulos de la edición MCDI (rama ``mcdi``, ``_quarto.yml``).
CAPITULOS_MCDI = frozenset([
    "01Introduccion",
    "02Teoria_Decision",
    "03Parametricos",
    "04Rendimiento",
    "05ReduccionDim",
    "06Agrupamiento",
    "07NoParametricos",
    "08Arboles",
    "09Lineal",
    "10Optimizacion",
    "11RedesNeuronales",
    "12Ensambles",
    "13Comparacion",
    "14Estadistica",
    "16ConjuntosDatos",
    "17Referencias",
])

PATRON = re.compile(r"^\s*-\s*capitulos/([^/\s]+)\.qmd\s*$")


def capitulos_quarto(texto):
    """Nombres base de los capítulos listados en ``_quarto.yml``."""
    return [m.group(1) for linea in texto.splitlines()
            if (m := PATRON.match(linea))]


def colisiones(nombres):
    """Nombres que coinciden con un capítulo del MCDI."""
    return sorted(set(nombres) & CAPITULOS_MCDI)


class PruebasNombres(unittest.TestCase):
    def test_lectura_de_quarto(self):
        nombres = capitulos_quarto((RAIZ / "_quarto.yml").read_text(
            encoding="utf-8"))
        self.assertIn("01Intro", nombres)
        self.assertIn("11Redes", nombres)
        self.assertIn("20ConjuntosDatos", nombres)  # apéndice

    def test_sin_colisiones(self):
        nombres = capitulos_quarto((RAIZ / "_quarto.yml").read_text(
            encoding="utf-8"))
        self.assertEqual(colisiones(nombres), [])

    def test_detecta_colision(self):
        texto = ("  chapters:\n    - index.qmd\n"
                 "    - capitulos/01Intro.qmd\n"
                 "    - capitulos/11RedesNeuronales.qmd\n"
                 "  appendices:\n    - capitulos/14Estadistica.qmd\n")
        self.assertEqual(colisiones(capitulos_quarto(texto)),
                         ["11RedesNeuronales", "14Estadistica"])


if __name__ == "__main__":
    unittest.main()
