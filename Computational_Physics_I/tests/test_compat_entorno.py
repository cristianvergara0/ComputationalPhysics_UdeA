"""Pruebas de la compatibilidad con Numba y del módulo de entorno (rutas, guardado y versiones)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from thinfilm_mc import compat, entorno


class TestCompat(unittest.TestCase):
    def test_njit_se_puede_usar_con_y_sin_argumentos(self):
        @compat.njit
        def suma(a, b):
            return a + b

        @compat.njit(cache=False)
        def producto(a, b):
            return a * b

        self.assertEqual(suma(2, 3), 5)
        self.assertEqual(producto(2, 3), 6)

    def test_prange_recorre_el_rango_fuera_de_una_funcion_compilada(self):
        self.assertEqual(sum(compat.prange(5)), 10)

    def test_la_bandera_es_booleana(self):
        self.assertIsInstance(compat.NUMBA_DISPONIBLE, bool)


class TestEntorno(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        parche = mock.patch.object(entorno, "DIR_RESULTADOS", Path(self.tmp.name) / "resultados")
        parche.start()
        self.addCleanup(parche.stop)

    def test_la_raiz_del_proyecto_contiene_el_paquete(self):
        self.assertTrue((entorno.RAIZ_PROYECTO / "thinfilm_mc").is_dir())

    def test_las_rutas_no_dependen_del_directorio_de_trabajo(self):
        actual = os.getcwd()
        self.addCleanup(os.chdir, actual)
        os.chdir(self.tmp.name)
        ruta = Path(entorno.ruta_figura("fig.png"))
        self.assertEqual(ruta.parent, Path(self.tmp.name) / "resultados" / "figuras")
        self.assertTrue(ruta.parent.is_dir())

    def test_guardar_tabla_escribe_un_csv_legible(self):
        tabla = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]})
        self.assertIsNone(entorno.guardar_tabla(tabla, "prueba"))
        ruta = Path(entorno.ruta_tabla("prueba.csv"))
        pd.testing.assert_frame_equal(pd.read_csv(ruta), tabla)
        entorno.guardar_tabla(tabla, "otra.csv")
        self.assertTrue(Path(entorno.ruta_tabla("otra.csv")).is_file())
        self.assertFalse(Path(entorno.ruta_tabla("otra.csv.csv")).exists())

    def test_registrar_entorno_imprime_y_guarda_las_versiones(self):
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            info = entorno.registrar_entorno("cuaderno_de_prueba")
        for clave in ("python", "numpy", "scipy", "pandas", "matplotlib", "numba", "fecha"):
            self.assertIn(clave, info)
        self.assertIn("Entorno de ejecución", salida.getvalue())
        guardado = json.loads((entorno.DIR_RESULTADOS / "entorno_cuaderno_de_prueba.json").read_text(encoding="utf-8"))
        self.assertEqual(guardado["numpy"], info["numpy"])

    def test_registrar_entorno_sin_cuaderno_no_escribe_archivos(self):
        with contextlib.redirect_stdout(io.StringIO()):
            entorno.registrar_entorno()
        self.assertFalse(entorno.DIR_RESULTADOS.exists())

    def test_crear_fila_alinea_las_etiquetas(self):
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            entorno.crear_fila(8)("W", 1.25)
        self.assertEqual(salida.getvalue(), "W       : 1.25\n")


if __name__ == "__main__":
    unittest.main()
