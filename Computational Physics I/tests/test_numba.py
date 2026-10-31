"""Equivalencia entre las rutinas compiladas con Numba y las mismas rutinas sin compilar.

Se omite si Numba no está instalada. Los casos de ``casos_numba.py`` se calculan dos veces: en este
proceso, con las funciones compiladas, y en otro proceso con ``NUMBA_DISABLE_JIT=1``, donde las mismas
funciones se ejecutan como Python puro. Las semillas se fijan con ``motor.sembrar_generador`` o dentro
de las rutinas de réplicas, de modo que cada versión siembra su propio generador. Las superficies, los
contadores y las celdas de impacto deben coincidir exactamente, y los ángulos y las rugosidades con una
tolerancia relativa de 1e-12.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thinfilm_mc.compat import NUMBA_DISPONIBLE


@unittest.skipUnless(NUMBA_DISPONIBLE, "Numba no está instalada")
class TestNumbaEquivaleAPythonPuro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from casos_numba import calcular

        cls.compilado = calcular()
        with tempfile.TemporaryDirectory() as tmp:
            salida = Path(tmp) / "sin_compilar.npz"
            entorno = dict(os.environ, NUMBA_DISABLE_JIT="1")
            proceso = subprocess.run([sys.executable, str(Path(__file__).with_name("casos_numba.py")), str(salida)],
                                     capture_output=True, text=True, env=entorno, timeout=1800)
            if proceso.returncode != 0:
                raise RuntimeError("El cálculo sin compilar falló:\n" + proceso.stdout + proceso.stderr)
            with np.load(salida) as datos:
                cls.sin_compilar = {clave: datos[clave] for clave in datos.files}

    def comparar(self, prefijo):
        claves = sorted(k for k in self.compilado if k.startswith(prefijo))
        self.assertTrue(claves)
        self.assertEqual(claves, sorted(k for k in self.sin_compilar if k.startswith(prefijo)))
        for clave in claves:
            a, b = np.asarray(self.compilado[clave]), self.sin_compilar[clave]
            with self.subTest(clave):
                if np.issubdtype(a.dtype, np.integer):
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_trazador_de_rayos(self):
        a, b = self.compilado["trazador"], self.sin_compilar["trazador"]
        np.testing.assert_array_equal(a[:, [0, 1, 2, 5, 6, 7, 8]], b[:, [0, 1, 2, 5, 6, 7, 8]])   # enteros
        np.testing.assert_allclose(a[:, 3:5], b[:, 3:5], rtol=1e-12, atol=1e-12)

    def test_lazo_de_replica(self):
        self.comparar("replica_")

    def test_replicas_en_paralelo(self):
        self.comparar("replicas")

    def test_motor_muestreado(self):
        self.comparar("muestreado_")

    def test_vuelos_libres(self):
        self.comparar("vuelos_")

    def test_contacto_lateral(self):
        a, b = self.compilado["contacto_lateral"], self.sin_compilar["contacto_lateral"]
        enteros = [0, 1, 2, 5, 6, 7, 8]                     # celda, colisiones, truncado, contacto lateral
        np.testing.assert_array_equal(a[:, enteros], b[:, enteros])
        np.testing.assert_allclose(a[:, 3:5], b[:, 3:5], rtol=1e-12, atol=1e-12)

    def test_motor_con_volumen(self):
        self.comparar("volumen_")


if __name__ == "__main__":
    unittest.main()
