# -*- coding: utf-8 -*-
"""Comprueba que el notebook del método usa el mismo código que el programa y da los mismos resultados.

El notebook es autocontenido: las funciones están copiadas en sus celdas. Esta prueba evita que las dos copias
se separen y que las salidas guardadas queden desactualizadas. Se ejecuta con:  python -m unittest discover -s tests -v
"""
import ast
import contextlib
import io
import json
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
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


def _numeros(texto):
    """Texto con cada número reemplazado por # y la lista de esos números."""
    patron = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
    return patron.sub("#", texto), [float(x) for x in patron.findall(texto)]


class TestNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not NOTEBOOK.exists():
            raise AssertionError(f"Falta {NOTEBOOK.relative_to(RAIZ)}: las pruebas esperan el notebook en esa ruta.")
        nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
        cls.codigo = ["".join(c["source"]) for c in cls.celdas]
        script = ast.parse(Path(sma.__file__).read_text(encoding="utf-8"))
        cls.del_programa = {n.name: ast.dump(n) for n in script.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

        # Una ejecución desde cero, que guarda lo que imprime cada celda
        cls.ns = {"__name__": "__main__"}
        cls.salidas = []
        with mock.patch.object(plt, "show"):
            for i, fuente in enumerate(cls.codigo):
                texto = io.StringIO()
                with contextlib.redirect_stdout(texto):
                    exec(compile(fuente, f"<celda {i + 1}>", "exec"), cls.ns)
                cls.salidas.append(texto.getvalue())
        plt.close("all")

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

    def test_el_texto_guardado_es_el_de_una_ejecucion_nueva(self):
        # Mismo texto y mismos números (con tolerancia de 0.2 %, por si cambia el último dígito impreso entre versiones)
        for i, (celda, nueva) in enumerate(zip(self.celdas, self.salidas), 1):
            guardada = "".join("".join(o["text"]) for o in celda["outputs"] if o["output_type"] == "stream")
            esqueleto_guardado, numeros_guardados = _numeros(guardada.strip())
            esqueleto_nuevo, numeros_nuevos = _numeros(nueva.strip())
            self.assertEqual(esqueleto_guardado, esqueleto_nuevo, f"celda de código {i}")
            np.testing.assert_allclose(numeros_guardados, numeros_nuevos, rtol=2e-3, atol=1e-9,
                                       err_msg=f"celda de código {i}")

    def test_ejecutar_el_notebook_da_los_mismos_resultados_que_el_programa(self):
        ns = self.ns
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
