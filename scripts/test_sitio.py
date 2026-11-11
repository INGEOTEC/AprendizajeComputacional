#!/usr/bin/env python3
"""Pruebas de scripts/sitio.py: python3 scripts/test_sitio.py

Solo biblioteca estándar (unittest y directorios temporales); sin red ni Quarto.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sitio  # noqa: E402


def escribir(ruta: Path, texto: str) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def contenido(directorio: Path) -> dict[str, str]:
    """Mapa ruta relativa -> texto de todos los archivos bajo ``directorio``."""
    return {
        str(p.relative_to(directorio)): p.read_text(encoding="utf-8")
        for p in sorted(directorio.rglob("*"))
        if p.is_file()
    }


class PruebasSitio(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.render = base / "render"
        self.gh = base / "gh-pages"
        self.p404 = base / "404.html"
        escribir(self.p404, "<html>404</html>")
        # Render nuevo
        escribir(self.render / "index.html", "nuevo index")
        escribir(self.render / "capitulos" / "01Intro.html", "nuevo cap 1")
        escribir(self.render / "search.json", "[]")
        # gh-pages existente: raíz vieja más las otras dos ediciones
        escribir(self.gh / ".git" / "HEAD", "ref: refs/heads/gh-pages")
        escribir(self.gh / "index.html", "viejo index")
        escribir(self.gh / "capitulos" / "02Teoria_Decision.html", "obsoleto")
        escribir(self.gh / "viejo.txt", "obsoleto")
        escribir(self.gh / "dev" / "index.html", "dev index")
        escribir(self.gh / "dev" / "obsoleto.html", "dev obsoleto")
        escribir(self.gh / "mcdi" / "index.html", "mcdi index")
        escribir(self.gh / "mcdi" / "capitulos" / "x.html", "mcdi x")

    def tearDown(self):
        self._tmp.cleanup()

    def desplegar(self, destino):
        return sitio.desplegar(self.render, self.gh, destino, pagina_404=self.p404)

    def fuera_de(self, antes, carpeta):
        """Archivos de ``antes`` fuera de ``carpeta`` y de los comunes de la raíz."""
        comunes = {".nojekyll", "404.html"}
        return {
            k: v
            for k, v in antes.items()
            if not k.startswith(carpeta + "/") and k not in comunes
        }

    def test_raiz_conserva_ediciones_y_borra_obsoletos(self):
        self.desplegar("raiz")
        archivos = contenido(self.gh)
        self.assertEqual(archivos["index.html"], "nuevo index")
        self.assertEqual(archivos["capitulos/01Intro.html"], "nuevo cap 1")
        self.assertNotIn("viejo.txt", archivos)
        self.assertNotIn("capitulos/02Teoria_Decision.html", archivos)
        self.assertEqual(archivos["dev/index.html"], "dev index")
        self.assertEqual(archivos["dev/obsoleto.html"], "dev obsoleto")
        self.assertEqual(archivos["mcdi/index.html"], "mcdi index")
        self.assertEqual(archivos["mcdi/capitulos/x.html"], "mcdi x")
        self.assertEqual(archivos[".git/HEAD"], "ref: refs/heads/gh-pages")

    def test_subcarpetas_solo_tocan_su_carpeta(self):
        for destino in ("dev", "mcdi"):
            with self.subTest(destino=destino):
                antes = contenido(self.gh)
                self.desplegar(destino)
                despues = contenido(self.gh)
                self.assertEqual(
                    self.fuera_de(antes, destino), self.fuera_de(despues, destino)
                )
                propios = {
                    k[len(destino) + 1 :]: v
                    for k, v in despues.items()
                    if k.startswith(destino + "/")
                }
                self.assertEqual(propios, contenido(self.render))

    def test_siempre_escribe_nojekyll_y_404(self):
        for destino in ("raiz", "dev", "mcdi"):
            with self.subTest(destino=destino):
                (self.gh / ".nojekyll").unlink(missing_ok=True)
                (self.gh / "404.html").unlink(missing_ok=True)
                self.desplegar(destino)
                self.assertTrue((self.gh / ".nojekyll").is_file())
                self.assertEqual(
                    (self.gh / "404.html").read_text(encoding="utf-8"),
                    "<html>404</html>",
                )

    def test_crea_subcarpeta_si_no_existe(self):
        import shutil

        shutil.rmtree(self.gh / "mcdi")
        self.desplegar("mcdi")
        self.assertEqual(
            (self.gh / "mcdi" / "index.html").read_text(encoding="utf-8"),
            "nuevo index",
        )

    def test_render_sin_index_falla(self):
        (self.render / "index.html").unlink()
        antes = contenido(self.gh)
        with self.assertRaisesRegex(sitio.ErrorDespliegue, "index.html"):
            self.desplegar("raiz")
        self.assertEqual(antes, contenido(self.gh))

    def test_destino_desconocido_falla(self):
        antes = contenido(self.gh)
        with self.assertRaisesRegex(sitio.ErrorDespliegue, "destino desconocido"):
            self.desplegar("produccion")
        self.assertEqual(antes, contenido(self.gh))

    def test_render_con_carpeta_reservada_falla_sin_tocar_nada(self):
        escribir(self.render / "dev" / "index.html", "intruso")
        antes = contenido(self.gh)
        with self.assertRaisesRegex(sitio.ErrorDespliegue, "reservado"):
            self.desplegar("raiz")
        self.assertEqual(antes, contenido(self.gh))

    def test_main_devuelve_error_legible(self):
        codigo = sitio.main([str(self.render), str(self.gh), "otro"])
        self.assertEqual(codigo, 1)

    def test_pagina_404_del_repositorio_existe(self):
        self.assertTrue(sitio.PAGINA_404.is_file())


if __name__ == "__main__":
    unittest.main()
