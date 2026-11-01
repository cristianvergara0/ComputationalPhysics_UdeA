"""Coherencia de la cadena de modelos entre los cuadernos que la forman.

Cada cuaderno de la cadena define los parámetros comunes de la serie y el diccionario ``MODELOS`` con los
modelos hasta el suyo. La prueba comprueba, sin ejecutar los cuadernos, que los parámetros comunes valen lo
mismo en todos y que los modelos de cada cuaderno son los del anterior, con el mismo nombre y las mismas
banderas, más uno nuevo. Así el modelo de cada cuaderno contiene exactamente al del anterior y las semillas
de un mismo modelo coinciden en toda la serie. El cuaderno de síntesis, el último, compara los modelos de la
cadena completa sin añadir ninguno: sus modelos son exactamente los del penúltimo.
"""

import ast
import json
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
COMUNES = ["NX", "NY", "T_ESCALA", "MAX_PASOS_DIFUSION", "TEMP_BASE", "SIGMA_INCIDENCIA_INIT", "SIGMA_SCATTERING_GAS",
           "MFP_PRESION_INTERMEDIA", "SOURCE_DISTANCE", "MAX_PASOS_RAYO", "SEMILLA"]


def asignaciones(ruta):
    """Asignaciones de nivel superior de las celdas de código: {nombre: nodo del valor}."""
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    salida = {}
    for celda in nb["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        try:
            arbol = ast.parse(fuente)
        except SyntaxError:
            continue
        for nodo in arbol.body:
            if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name):
                salida[nodo.targets[0].id] = nodo.value
    return salida


def cuadernos_de_la_cadena():
    """Cuadernos, en orden, que definen el diccionario MODELOS."""
    return [(ruta, a) for ruta in sorted(RAIZ.glob("[0-9][0-9]_*.ipynb")) if "MODELOS" in (a := asignaciones(ruta))]


class TestCadenaDeModelos(unittest.TestCase):
    def setUp(self):
        self.cadena = cuadernos_de_la_cadena()
        if not self.cadena:
            self.skipTest("ningún cuaderno define la cadena de modelos")

    def test_los_parametros_comunes_coinciden(self):
        referencia_ruta, referencia = self.cadena[0]
        for ruta, valores in self.cadena[1:]:
            for nombre in COMUNES:
                with self.subTest(cuaderno=ruta.name, parametro=nombre):
                    self.assertIn(nombre, valores)
                    self.assertEqual(ast.literal_eval(valores[nombre]), ast.literal_eval(referencia[nombre]),
                                     f"distinto de {referencia_ruta.name}")

    def test_cada_cuaderno_extiende_la_cadena_del_anterior(self):
        anteriores = None
        for ruta, valores in self.cadena:
            modelos = valores["MODELOS"]
            self.assertIsInstance(modelos, ast.Dict)
            actuales = [(ast.literal_eval(k), ast.dump(v)) for k, v in zip(modelos.keys, modelos.values)]
            with self.subTest(cuaderno=ruta.name):
                if anteriores is not None:
                    self.assertEqual(actuales[:len(anteriores)], anteriores)
                    if "synthesis" in ruta.name:
                        self.assertEqual(actuales, anteriores, "la síntesis compara la cadena completa")
                    else:
                        self.assertEqual(len(actuales), len(anteriores) + 1, "cada cuaderno añade un modelo")
            anteriores = actuales

    def test_los_nombres_empiezan_por_su_clave(self):
        for ruta, valores in self.cadena:
            claves = [ast.literal_eval(k) for k in valores["MODELOS"].keys]
            with self.subTest(cuaderno=ruta.name):
                self.assertEqual([c.split(":")[0] for c in claves], [f"M{i}" for i in range(1, len(claves) + 1)])


if __name__ == "__main__":
    unittest.main()
