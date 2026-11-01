"""Comprobaciones estáticas de los cuadernos: formato, sintaxis, importaciones, rutas y nombres.

No ejecutan los cuadernos (eso lo hace ``test_cuadernos_reducidos.py``): detectan errores que se
manifestarían al ejecutarlos, como una importación que ya no existe, un nombre sin definir o una ruta
absoluta del equipo de quien escribió el cuaderno.
"""

import ast
import builtins
import importlib
import json
import re
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

CUADERNOS = sorted(RAIZ.glob("[0-9][0-9]_*.ipynb"))


def cargar(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))


def fuente(celda):
    s = celda["source"]
    return "".join(s) if isinstance(s, list) else s


def codigo(ruta):
    """Celdas de código con las instrucciones mágicas de IPython neutralizadas."""
    celdas = []
    for celda in cargar(ruta)["cells"]:
        if celda["cell_type"] == "code":
            celdas.append("\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in fuente(celda).splitlines()))
    return celdas


def nombres_ligados(arbol):
    ligados = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Name) and isinstance(nodo.ctx, (ast.Store, ast.Del)):
            ligados.add(nodo.id)
        elif isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            ligados.add(nodo.name)
        elif isinstance(nodo, ast.arg):
            ligados.add(nodo.arg)
        elif isinstance(nodo, ast.ExceptHandler) and nodo.name:
            ligados.add(nodo.name)
        elif isinstance(nodo, (ast.Import, ast.ImportFrom)):
            for alias in nodo.names:
                ligados.add((alias.asname or alias.name).split(".")[0])
    return ligados


class TestCuadernos(unittest.TestCase):
    def test_hay_cuadernos_y_la_numeracion_es_consecutiva(self):
        self.assertGreaterEqual(len(CUADERNOS), 8)
        numeros = [int(c.name[:2]) for c in CUADERNOS]
        self.assertEqual(numeros, list(range(1, len(numeros) + 1)))

    def test_el_cuaderno_de_sintesis_es_el_ultimo(self):
        self.assertIn("synthesis", CUADERNOS[-1].name)
        self.assertEqual(sum("synthesis" in c.name for c in CUADERNOS), 1)

    def test_formato_nbformat(self):
        for ruta in CUADERNOS:
            with self.subTest(ruta.name):
                nb = cargar(ruta)
                self.assertEqual(nb["nbformat"], 4)
                ids = [c["id"] for c in nb["cells"]]
                self.assertEqual(len(ids), len(set(ids)), "identificadores de celda repetidos")
                for celda in nb["cells"]:
                    self.assertIn(celda["cell_type"], ("code", "markdown"))
                    self.assertIsInstance(celda["source"], list)
                    if celda["cell_type"] == "code":
                        self.assertIn("outputs", celda)
                        self.assertIn("execution_count", celda)

    def test_las_celdas_de_codigo_compilan(self):
        for ruta in CUADERNOS:
            for i, celda in enumerate(codigo(ruta)):
                with self.subTest(f"{ruta.name} celda de código {i}"):
                    ast.parse(celda)

    def test_las_importaciones_del_paquete_existen(self):
        for ruta in CUADERNOS:
            for celda in codigo(ruta):
                for nodo in ast.walk(ast.parse(celda)):
                    if isinstance(nodo, ast.ImportFrom) and (nodo.module or "").startswith("thinfilm_mc"):
                        modulo = importlib.import_module(nodo.module)
                        for alias in nodo.names:
                            with self.subTest(f"{ruta.name}: {nodo.module}.{alias.name}"):
                                # El nombre puede ser un objeto del módulo o un submódulo del paquete
                                existe = hasattr(modulo, alias.name)
                                if not existe:
                                    try:
                                        importlib.import_module(f"{nodo.module}.{alias.name}")
                                        existe = True
                                    except ModuleNotFoundError:
                                        pass
                                self.assertTrue(existe)

    def test_no_hay_nombres_sin_definir(self):
        permitidos = set(dir(builtins)) | {"display", "get_ipython"}
        for ruta in CUADERNOS:
            with self.subTest(ruta.name):
                arboles = [ast.parse(c) for c in codigo(ruta)]
                ligados = set().union(*(nombres_ligados(a) for a in arboles))
                usados = {n.id for a in arboles for n in ast.walk(a) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                self.assertEqual(sorted(usados - ligados - permitidos), [])

    def test_no_hay_rutas_absolutas_de_ningun_equipo(self):
        patron = re.compile(r"[A-Za-z]:\\|/home/|/Users/|/content/|/mnt/|/tmp/")
        for ruta in CUADERNOS:
            for celda in cargar(ruta)["cells"]:
                if celda["cell_type"] == "code":
                    with self.subTest(ruta.name):
                        self.assertIsNone(patron.search(fuente(celda)), fuente(celda)[:120])

    def test_las_salidas_van_a_la_carpeta_de_resultados(self):
        for ruta in CUADERNOS:
            for celda in codigo(ruta):
                for m in re.finditer(r"savefig\(\s*(?!ruta_figura\()", celda):
                    self.fail(f"{ruta.name}: savefig sin ruta_figura: {celda[m.start():m.start() + 60]!r}")
                for m in re.finditer(r"to_csv\(\s*(?!ruta_tabla\()", celda):
                    self.fail(f"{ruta.name}: to_csv sin ruta_tabla: {celda[m.start():m.start() + 60]!r}")

    def test_los_cuadernos_no_redefinen_funciones_del_paquete(self):
        exportadas = set()
        for nombre in ("analisis", "estadistica", "motor", "barridos", "graficos", "entorno"):
            modulo = importlib.import_module(f"thinfilm_mc.{nombre}")
            exportadas |= {n for n, o in vars(modulo).items() if callable(o) and getattr(o, "__module__", "") == modulo.__name__
                           or hasattr(o, "py_func")}
        exportadas = {n for n in exportadas if not n.startswith("_")}
        for ruta in CUADERNOS:
            definidas = {n.name for c in codigo(ruta) for n in ast.parse(c).body if isinstance(n, ast.FunctionDef)}
            with self.subTest(ruta.name):
                self.assertEqual(sorted(definidas & exportadas), [])

    def test_las_referencias_a_otros_cuadernos_existen(self):
        total = len(CUADERNOS)
        for ruta in CUADERNOS:
            for celda in cargar(ruta)["cells"]:
                if celda["cell_type"] != "markdown":
                    continue
                for m in re.finditer(r"[Cc]uadernos?\s+(\d+(?:\s*(?:,|y|a)\s*\d+)*)", fuente(celda)):
                    for n in re.findall(r"\d+", m.group(1)):
                        with self.subTest(f"{ruta.name}: {m.group(0)}"):
                            self.assertTrue(1 <= int(n) <= total)

    def test_ninguna_salida_guardada_es_un_error(self):
        for ruta in CUADERNOS:
            for celda in cargar(ruta)["cells"]:
                if celda["cell_type"] == "code":
                    for salida in celda["outputs"]:
                        with self.subTest(ruta.name):
                            self.assertNotEqual(salida.get("output_type"), "error")


if __name__ == "__main__":
    unittest.main()
