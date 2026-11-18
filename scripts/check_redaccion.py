#!/usr/bin/env python3
"""Mide tres vicios de redacción en la prosa de los capítulos.

Uso:
    python3 scripts/check_redaccion.py                      # todos los capítulos
    python3 scripts/check_redaccion.py capitulos/15Interpretabilidad.qmd
    python3 scripts/check_redaccion.py -v capitulos/15Interpretabilidad.qmd

Solo lee los archivos; no ejecuta código ni necesita Quarto. Revisa los
párrafos de prosa (no el código, las ecuaciones, los títulos, las listas ni
las tablas) y cuenta:

1. Signos: los dos puntos a media oración y los punto y coma. Los dos puntos
   con que termina un párrafo, que introducen una ecuación, un bloque de
   código o una lista, no se cuentan.
2. Párrafos cargados: los que tienen dos o más de esos signos.
3. Inicios cortos: párrafos de varias oraciones cuya primera oración tiene
   menos de MIN_PALABRAS palabras (un enunciado tipo título que el resto del
   párrafo desarrolla).

Un capítulo pasa si no tiene párrafos cargados, si los signos no superan
MAX_SIGNOS por párrafo y si los inicios cortos no superan MAX_INICIOS por
párrafo. Sale con código 1 si algún archivo revisado no pasa. Las reglas
están en la sección «Redacción» de CLAUDE.md.
"""
import glob
import re
import sys

MIN_PALABRAS = 10
MAX_SIGNOS = 0.10
MAX_INICIOS = 0.05
NO_PROSA = ('#', ':::', '|', '<', '![', '-', '*', '+', '>', '```')


def parrafos(ruta):
    """Regresa [(línea, texto)] con los párrafos de prosa del archivo."""
    salida, actual, inicio = [], [], 0
    codigo = ecuacion = yaml = False
    with open(ruta, encoding='utf-8') as archivo:
        lineas = archivo.read().split('\n')
    for num, linea in enumerate(lineas + [''], 1):
        texto = linea.strip()
        if num == 1 and texto == '---':
            yaml = True
            continue
        if yaml:
            yaml = texto != '---'
            continue
        if texto.startswith('```'):
            codigo = not codigo
            texto = ''
        elif codigo:
            continue
        elif texto.startswith('$$'):
            if not (len(texto) > 4 and texto.endswith('$$')):
                ecuacion = not ecuacion
            texto = ''
        elif ecuacion:
            continue
        if (not texto or texto.startswith(NO_PROSA)
                or re.match(r'\d+\.\s', texto)):
            if actual:
                salida.append((inicio, ' '.join(actual)))
                actual = []
            continue
        if not actual:
            inicio = num
        actual.append(texto)
    return salida


def limpia(texto):
    """Quita código en línea, ecuaciones, ligas y atributos."""
    texto = re.sub(r'`[^`]*`', 'C', texto)
    texto = re.sub(r'\$[^$]*\$', 'M', texto)
    texto = re.sub(r'\]\([^)]*\)', ']', texto)
    texto = re.sub(r'\{[^}]*\}', '', texto)
    texto = re.sub(r'https?://\S+', 'U', texto)
    return texto.replace('**', '').strip()


def revisa(ruta):
    """Regresa (número de párrafos, signos, cargados, inicios cortos)."""
    signos, cargados, inicios = 0, [], []
    lista = parrafos(ruta)
    for num, crudo in lista:
        texto = limpia(crudo)
        cuerpo = texto[:-1] if texto.endswith(':') else texto
        cuenta = len(re.findall(r':\s', cuerpo)) + cuerpo.count(';')
        signos += cuenta
        if cuenta >= 2:
            cargados.append((num, cuenta, crudo))
        primera = re.match(r'(.+?\.)\s+[A-ZÁÉÍÓÚÑ¿¡@(]', texto)
        if primera and len(primera.group(1).split()) < MIN_PALABRAS:
            inicios.append((num, primera.group(1)))
    return len(lista), signos, cargados, inicios


def main(argv):
    detalle = '-v' in argv
    rutas = [a for a in argv if a != '-v']
    if not rutas:
        rutas = ['index.qmd'] + sorted(glob.glob('capitulos/*.qmd'))
    fallan = 0
    for ruta in rutas:
        total, signos, cargados, inicios = revisa(ruta)
        if total == 0:
            continue
        pasa = (not cargados and signos <= MAX_SIGNOS * total
                and len(inicios) <= MAX_INICIOS * total)
        fallan += not pasa
        print(f"{'ok   ' if pasa else 'FALLA'} {ruta}: {total} párrafos, "
              f"{signos} signos (máx. {int(MAX_SIGNOS * total)}), "
              f"{len(cargados)} párrafos cargados (máx. 0), "
              f"{len(inicios)} inicios cortos (máx. {int(MAX_INICIOS * total)})")
        if detalle:
            for num, cuenta, crudo in cargados:
                print(f'    línea {num}: {cuenta} signos: {crudo[:90]}...')
            for num, oracion in inicios:
                print(f'    línea {num}: inicio corto: {oracion}')
    return 1 if fallan else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
