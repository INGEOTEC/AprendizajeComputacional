#!/usr/bin/env python3
"""Pruebas de los scripts de datos de texto (delitos_juez.py y modelos_hf.py).

unittest, sin torch, sin red y sin GPU: sólo las funciones puras.

Uso:
    python3 scripts/test_texto.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'datos'))
import delitos_juez as dj  # noqa: E402
import modelos_hf as mh  # noqa: E402


def corpus(n_entrenamiento=3, n_prueba=2):
    """Registros falsos con la forma de delitos.registros()."""
    regs = [dict(particion='entrenamiento', indice=i, klass=0, text=f't{i}')
            for i in range(n_entrenamiento)]
    regs += [dict(particion='prueba', indice=i, klass=1, text=f'p{i}')
             for i in range(n_prueba)]
    return regs


def filas_de(modelo, regs, prompts=(1, 2, 3)):
    return [dict(particion=r['particion'], indice=r['indice'],
                 modelo=dj.slug(modelo), prompt=p, etiqueta='no_delito')
            for p in prompts for r in regs]


class TestVeredicto(unittest.TestCase):
    def test_basico(self):
        self.assertEqual(dj.extraer_veredicto('Veredicto: delito'), 'delito')
        self.assertEqual(dj.extraer_veredicto('Veredicto: no_delito'),
                         'no_delito')

    def test_mayusculas_y_asteriscos(self):
        self.assertEqual(dj.extraer_veredicto('**VEREDICTO:** **DELITO**'),
                         'delito')
        self.assertEqual(dj.extraer_veredicto('**Veredicto**: [No_Delito]'),
                         'no_delito')
        self.assertEqual(dj.extraer_veredicto('Veredicto: no delito'),
                         'no_delito')

    def test_texto_antes(self):
        respuesta = ('El mensaje reporta un robo a mano armada como un hecho.\n'
                     'Veredicto: delito')
        self.assertEqual(dj.extraer_veredicto(respuesta), 'delito')

    def test_think(self):
        respuesta = ('<think>Veredicto: delito</think>\n'
                     'Es una opinión.\nVeredicto: no_delito')
        self.assertEqual(dj.extraer_veredicto(respuesta), 'no_delito')

    def test_sin_veredicto(self):
        self.assertEqual(dj.extraer_veredicto('Es un delito de robo.'), 'unk')
        self.assertEqual(dj.extraer_veredicto(''), 'unk')
        self.assertEqual(dj.extraer_veredicto('Veredicto: tal vez'), 'unk')


class TestPrompts(unittest.TestCase):
    def test_tres_prompts(self):
        self.assertEqual(len(dj.PROMPTS), 3)

    def test_mensaje_y_linea_de_veredicto(self):
        mensaje = 'Asaltan una farmacia en el centro {sin formato}'
        for p in (1, 2, 3):
            texto = dj.armar_prompt(p, mensaje)
            self.assertIn(mensaje, texto)
            self.assertIn('"Veredicto: delito"', texto)
            self.assertIn('"Veredicto: no_delito"', texto)
            # La línea exigida en el prompt se lee con el mismo patrón.
            self.assertEqual(dj.extraer_veredicto('Veredicto: no_delito'),
                             'no_delito')

    def test_slug(self):
        self.assertEqual(dj.slug('Qwen/Qwen3.5-0.8B'), 'qwen3.5-0.8b')
        self.assertEqual([dj.slug(m) for m in dj.MODELOS],
                         ['qwen3.5-0.8b', 'qwen3.5-2b', 'qwen3.5-4b'])


class TestReanudacion(unittest.TestCase):
    def test_pendientes(self):
        regs = corpus()
        hechos = {('entrenamiento', 0, 1), ('prueba', 1, 3)}
        faltan = dj.pendientes(regs, hechos)
        self.assertEqual(len(faltan), 3 * len(regs) - 2)
        self.assertEqual((faltan[0][0], faltan[0][1]['indice']), (1, 1))
        self.assertNotIn((3, 'prueba', 1),
                         [(p, r['particion'], r['indice']) for p, r in faltan])

    def test_parcial_con_renglon_truncado(self):
        regs = corpus()
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / 'juez-x.csv'
            dj.escribir_parcial(ruta, filas_de('Qwen/Qwen3.5-2B', regs[:2], (1,)))
            with open(ruta, 'a', encoding='utf-8') as f:
                f.write('entrenamiento,2,qwen3.5-2b,1,no_del')
            filas = dj.leer_parcial(ruta)
            self.assertEqual(len(filas), 2)
            hechos = {(f['particion'], f['indice'], f['prompt']) for f in filas}
            self.assertEqual(len(dj.pendientes(regs, hechos)), 3 * len(regs) - 2)

    def test_duplicados(self):
        regs = corpus()
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / 'juez-x.csv'
            filas = filas_de('Qwen/Qwen3.5-2B', regs[:1], (1,))
            dj.escribir_parcial(ruta, filas)
            dj.escribir_parcial(ruta, filas, agregar=True)
            self.assertEqual(len(dj.leer_parcial(ruta)), 1)

    def test_parcial_inexistente(self):
        self.assertEqual(dj.leer_parcial('/no/existe/juez.csv'), [])


class TestUnir(unittest.TestCase):
    def test_completo(self):
        regs = corpus()
        parciales = {m: filas_de(m, regs)[::-1] for m in dj.MODELOS}
        filas = dj.unir_parciales(parciales, regs)
        self.assertEqual(len(filas), 3 * 3 * len(regs))
        self.assertEqual(filas[0], dict(particion='entrenamiento', indice=0,
                                        modelo='qwen3.5-0.8b', prompt=1,
                                        etiqueta='no_delito'))
        self.assertEqual(filas[-1]['modelo'], 'qwen3.5-4b')
        self.assertEqual((filas[-1]['prompt'], filas[-1]['particion'],
                          filas[-1]['indice']), (3, 'prueba', 1))

    def test_rechaza_incompleto(self):
        regs = corpus()
        parciales = {m: filas_de(m, regs) for m in dj.MODELOS}
        parciales['Qwen/Qwen3.5-2B'] = parciales['Qwen/Qwen3.5-2B'][:-1]
        with self.assertRaises(ValueError):
            dj.unir_parciales(parciales, regs)

    def test_rechaza_modelo_faltante(self):
        regs = corpus()
        parciales = {m: filas_de(m, regs) for m in dj.MODELOS[:2]}
        with self.assertRaises(ValueError):
            dj.unir_parciales(parciales, regs)


class TestModelosHF(unittest.TestCase):
    def blob(self, hub, sha, usuarios):
        carpeta = hub / 'blobs' / sha[:2]
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / sha).write_text('pesos')
        (carpeta / (sha + '.lock')).write_text('')
        (carpeta / (sha + '.refs')).write_text(
            ''.join(f'{u}/blobs/{sha}\n' for u in usuarios))
        return carpeta / sha

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        self.hub = raiz / 'hub'
        self.marca = raiz / 'modelos_descargados.txt'
        self.propio = 'Qwen/Qwen3.5-0.8B'
        self.ajeno = 'Qwen/Qwen3-Embedding-4B'
        for m in (self.propio, self.ajeno):
            (mh.carpeta_modelo(m, self.hub) / 'snapshots').mkdir(parents=True)
        p = mh.carpeta_modelo(self.propio, self.hub).name
        a = mh.carpeta_modelo(self.ajeno, self.hub).name
        self.solo = self.blob(self.hub, 'aa11', [p])
        self.compartido = self.blob(self.hub, 'bb22', [p, a])
        self.del_otro = self.blob(self.hub, 'cc33', [a])
        mh.escribir_marca(self.marca, [self.propio])

    def tearDown(self):
        self.tmp.cleanup()

    def test_carpeta(self):
        self.assertEqual(mh.carpeta_modelo('Qwen/Qwen3.5-4B', Path('/h')),
                         Path('/h/models--Qwen--Qwen3.5-4B'))

    def test_borrar(self):
        self.assertTrue(mh.borrar(self.propio, self.hub, self.marca))
        self.assertFalse(mh.carpeta_modelo(self.propio, self.hub).exists())
        self.assertFalse(self.solo.exists())
        self.assertFalse(self.solo.with_name('aa11.refs').exists())
        self.assertFalse(self.solo.with_name('aa11.lock').exists())
        self.assertTrue(self.compartido.exists())
        refs = self.compartido.with_name('bb22.refs').read_text().split()
        self.assertEqual(refs, ['models--Qwen--Qwen3-Embedding-4B/blobs/bb22'])
        self.assertTrue(self.del_otro.exists())
        self.assertTrue(mh.carpeta_modelo(self.ajeno, self.hub).exists())
        self.assertEqual(mh.leer_marca(self.marca), [])

    def test_no_toca_modelo_fuera_de_la_marca(self):
        self.assertFalse(mh.borrar(self.ajeno, self.hub, self.marca))
        self.assertTrue(mh.carpeta_modelo(self.ajeno, self.hub).exists())
        self.assertTrue(self.del_otro.exists())
        self.assertTrue(self.compartido.exists())
        self.assertEqual(mh.leer_marca(self.marca), [self.propio])


if __name__ == '__main__':
    unittest.main()
