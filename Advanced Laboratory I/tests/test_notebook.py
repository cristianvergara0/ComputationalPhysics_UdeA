# -*- coding: utf-8 -*-
"""Comprueba que el notebook del método usa el mismo código que el programa y da los mismos resultados.

El notebook es autocontenido: las funciones están copiadas en sus celdas. Esta prueba evita que las dos copias
se separen. Se ejecuta con:  python -m unittest discover -s tests -v
"""
import ast
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import spring_magnet_analyzer as sma  # noqa: E402

NOTEBOOK = RAIZ / "notebooks" / "spring_magnet_induced_emf_analysis.ipynb"
NOMBRES_COMPARTIDOS = (
    "ErrorAnalisis", "Montaje", "OpcionesAnalisis", "simular_registro", "preparar_senal", "frecuencia_dominante",
    "envolvente_y_picos", "ajustar_decaimiento", "lobulos_de_signo_constante", "sigma_ruido", "formatear",
    "resolver_parametros", "comparar_con_simulador", "analizar_registro")


@unittest.skipUnless(NOTEBOOK.exists(), "no está el notebook")
class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
        cls.codigo = ["".join(c["source"]) for c in cls.celdas]
        script = ast.parse(Path(sma.__file__).read_text(encoding="utf-8"))
        cls.del_programa = {n.name: ast.dump(n) for n in script.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

    def test_las_funciones_y_clases_son_las_del_programa(self):
        vistos = set()
        for fuente in self.codigo:
            for nodo in ast.parse(fuente).body:
                if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
                    self.assertEqual(ast.dump(nodo), self.del_programa[nodo.name], nodo.name)
                    vistos.add(nodo.name)
        self.assertEqual(vistos, set(NOMBRES_COMPARTIDOS))

    def test_las_constantes_son_las_del_programa(self):
        ns = {}
        for fuente in self.codigo:
            for nodo in ast.parse(fuente).body:
                if isinstance(nodo, ast.Assign) and isinstance(nodo.targets[0], ast.Name) \
                        and nodo.targets[0].id.isupper() and isinstance(nodo.value, ast.Constant):
                    ns[nodo.targets[0].id] = nodo.value.value
        self.assertGreaterEqual(len(ns), 15)
        for nombre, valor in ns.items():
            self.assertEqual(getattr(sma, nombre), valor, nombre)

    def test_las_salidas_guardadas_corresponden_a_cada_celda(self):
        self.assertTrue(all(c["execution_count"] == i + 1 for i, c in enumerate(self.celdas)))
        figuras = sum(o["output_type"] == "display_data" for c in self.celdas for o in c["outputs"])
        self.assertGreaterEqual(figuras, 6)

    def test_ejecutar_el_notebook_da_los_mismos_resultados_que_el_programa(self):
        ns = {"__name__": "__main__"}
        with contextlib.redirect_stdout(io.StringIO()):
            for i, fuente in enumerate(self.codigo):
                exec(compile(fuente, f"<celda {i + 1}>", "exec"), ns)
        plt.close("all")

        t, cuentas, fs, _ = sma.simular_registro()
        medido, _ = sma.analizar_registro(t, cuentas, fs, 3.3 / 4095)
        self.assertAlmostEqual(ns["f0"], medido["f0_Hz"], places=12)
        self.assertAlmostEqual(ns["gamma_a"], medido["gamma_area_1_s"], places=12)
        self.assertAlmostEqual(ns["gamma_h"], medido["gamma_altura_1_s"], places=12)
        self.assertAlmostEqual(ns["integral"], medido["integral_emf_Vs"], places=12)
        deducido = sma.resolver_parametros(medido, sma.Montaje(vueltas=3400, radio=0.0415, masa=0.2))["valores"]
        for clave in ("k_N_m", "b_kg_s", "B_T", "Q"):
            self.assertAlmostEqual(ns["res"]["valores"][clave], deducido[clave], places=12, msg=clave)


if __name__ == "__main__":
    unittest.main()
