#!/usr/bin/env python3
"""Pruebas de scripts/check_redaccion.py (unittest, sin red ni Quarto).

Uso:
    python3 scripts/test_redaccion.py
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_redaccion  # noqa: E402


def revisa(texto):
    """Escribe el texto en un archivo temporal y regresa check_redaccion.revisa."""
    with tempfile.NamedTemporaryFile('w', suffix='.qmd', delete=False,
                                     encoding='utf-8') as archivo:
        archivo.write(texto)
    try:
        return check_redaccion.revisa(archivo.name)
    finally:
        os.unlink(archivo.name)


class TestRedaccion(unittest.TestCase):
    def test_dos_puntos_a_media_oracion(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'El capítulo terminó con una observación incómoda: un bosque '
            'predice mejor que un árbol pero ya no se puede leer.\n')
        self.assertEqual((total, signos), (1, 1))
        self.assertEqual(len(sueltos), 1)
        self.assertEqual(cargados, [])

    def test_parrafo_cargado(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'Hay dos enfoques conocidos: el primero modela los datos; el '
            'segundo modela el algoritmo que los predice.\n')
        self.assertEqual(signos, 2)
        self.assertEqual(len(cargados), 1)

    def test_dos_puntos_al_final_no_cuentan(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'El error estándar de la media se calcula con la siguiente '
            'ecuación:\n\n$$\nx = 1\n$$\n')
        self.assertEqual((total, signos), (1, 0))

    def test_inicio_corto(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'Conviene separar dos preguntas. La primera es qué variables usa '
            'el modelo en general y la segunda es por qué hizo una '
            'predicción.\n')
        self.assertEqual(len(inicios), 1)

    def test_pie_de_tabla_no_es_prosa(self):
        total, signos, cargados, inicios, sueltos = revisa(
            '| a | b |\n|---|---|\n| 1 | 2 |\n\n: Notación {#tbl-notacion}\n')
        self.assertEqual((total, signos), (0, 0))

    def test_ecuacion_de_una_linea_con_etiqueta(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'El riesgo de cada acción se define de la siguiente manera\n\n'
            '$$R = \\sum_k a_k,$$ {#eq-riesgo}\n\n'
            'Este párrafo sí es prosa; antes no se revisaba por estar '
            'después de la ecuación.\n\n'
            '$$\nx = 1;\n$$ {#eq-otra}\n\n'
            'Otro párrafo de prosa que no tiene ninguno de los signos '
            'medidos.\n')
        self.assertEqual((total, signos), (3, 1))

    def test_continuacion_en_minuscula_no_es_inicio_corto(self):
        total, signos, cargados, inicios, sueltos = revisa(
            'donde $a$ es el número de elementos. Es importante notar que '
            'este valor no depende del orden de los datos.\n')
        self.assertEqual(inicios, [])


if __name__ == '__main__':
    unittest.main()
