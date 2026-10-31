"""Cifras de referencia a escala completa de los motores del paquete.

Las cifras se obtuvieron con 50 000 partículas en una malla de 200 x 200 celdas y la semilla 42: con
``ejecutar_modelo_muestreado`` para ocho configuraciones de banderas (15 réplicas cada una) y con
``correr_con_semillas`` para tres puntos de un barrido con gas (10 réplicas cada uno), sin compilar. Comprobarlas
verifica que los núcleos compartidos y la derivación de semillas reproducen el cálculo completo, no solo casos
pequeños, y, con Numba, que la versión compilada da las mismas cifras. Los nombres de las configuraciones solo
fijan sus semillas.

Estas pruebas son lentas en Python puro (unos cuatro minutos en total), así que solo se ejecutan si se define
``TFMC_PRUEBAS_LENTAS=1``. Con Numba tardan segundos.

    TFMC_PRUEBAS_LENTAS=1 python -m unittest discover -s tests -p "test_escala_completa.py" -v
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import io

import numpy as np

from thinfilm_mc.barridos import ParametrosMotor, ParametrosUnificado, correr_con_semillas, ejecutar_modelo_muestreado

LENTAS = os.environ.get("TFMC_PRUEBAS_LENTAS") == "1"

PARAMS = ParametrosUnificado(
    nx=200, ny=200, t_escala=500, max_pasos_difusion=5, sigma_incidencia=15.0, sigma_dispersion=20.0,
    mfp=10.0, distancia_fuente=40.0,
)
N_EVENTOS, N_REPLICAS, SEMILLA = 50_000, 15, 42


def banderas(angulo, gas, sombreado, relajacion, temp, lateral=False):
    return {"enable_angle": angulo, "enable_scattering": gas, "enable_shadowing": sombreado,
            "enable_diffusion": relajacion, "temp": temp, "enable_lateral": lateral}


# configuración (el nombre completo entra en la derivación de las semillas): (banderas, W medio, desviación de W
# entre réplicas, columna más alta media, colisiones por átomo, huecos medios)
MODELOS = {
    "M1: Deposición aleatoria": (banderas(False, False, False, False, 0), 1.119065, 0.003787, 8.2, 0.0, 0.0),
    "M2: Relajación térmica (300 K)": (banderas(False, False, False, True, 300), 0.744575, 0.003576, 6.066667, 0.0, 0.0),
    "M3: Relajación térmica (400 K)": (banderas(False, False, False, True, 400), 0.627669, 0.002317, 4.733333, 0.0, 0.0),
    "M4: Incidencia angular": (banderas(True, False, False, True, 400), 0.627667, 0.00215, 4.866667, 0.0, 0.0),
    "M5: Sombreado geométrico": (banderas(True, False, True, True, 400), 0.656279, 0.003471, 5.866667, 0.0, 0.0),
    "M6: Dispersión en fase gaseosa": (banderas(True, True, True, True, 400), 0.75327, 0.006337, 9.4, 6.320808, 0.0),
    "M7: Adhesión lateral y huecos": (banderas(True, True, True, True, 400, True), 0.643834, 0.00276, 5.0,
                                      6.331463, 397.466667),
    "M7 frío: Adhesión lateral sin relajación": (banderas(True, True, True, False, 0, True), 1.364306, 0.0081, 8.8,
                                                 6.321361, 9858.066667),
}


@unittest.skipUnless(LENTAS, "prueba lenta: defina TFMC_PRUEBAS_LENTAS=1")
class TestEscalaCompleta(unittest.TestCase):
    def comprobar(self, clave):
        flags, w, sd, pico, colisiones, huecos = MODELOS[clave]
        with contextlib.redirect_stdout(io.StringIO()):
            datos = ejecutar_modelo_muestreado(PARAMS, clave, flags, N_REPLICAS, SEMILLA,
                                               np.array([N_EVENTOS], dtype=np.int64), np.array([False]))
        rug = datos["rugosidad"][:, -1]
        self.assertEqual(len(rug), N_REPLICAS)
        self.assertAlmostEqual(float(np.mean(rug)), w, places=5)
        self.assertAlmostEqual(float(np.std(rug, ddof=1)), sd, places=5)
        self.assertAlmostEqual(float(np.mean(datos["relieve"][:, -1] + datos["espesor"][:, -1])), pico, places=5)
        self.assertAlmostEqual(float(np.mean(datos["colisiones_por_atomo"])), colisiones, places=5)
        self.assertAlmostEqual(float(np.mean(datos["huecos_finales"])), huecos, places=5)
        np.testing.assert_array_equal(datos["depositados"], N_EVENTOS)
        np.testing.assert_array_equal(datos["truncados"], 0)

    def test_m1_deposicion_aleatoria(self):
        self.comprobar("M1: Deposición aleatoria")

    def test_m2_relajacion_300_k(self):
        self.comprobar("M2: Relajación térmica (300 K)")

    def test_m3_relajacion_400_k(self):
        self.comprobar("M3: Relajación térmica (400 K)")

    def test_m4_incidencia_angular(self):
        self.comprobar("M4: Incidencia angular")

    def test_m5_sombreado_geometrico(self):
        self.comprobar("M5: Sombreado geométrico")

    def test_m6_dispersion_en_fase_gaseosa(self):
        self.comprobar("M6: Dispersión en fase gaseosa")

    def test_m7_adhesion_lateral(self):
        self.comprobar("M7: Adhesión lateral y huecos")

    def test_m7_adhesion_lateral_sin_relajacion(self):
        self.comprobar("M7 frío: Adhesión lateral sin relajación")

    def test_puntos_de_un_barrido_con_gas(self):
        params = ParametrosMotor(nx=200, ny=200, t_escala=500, max_pasos_difusion=5, sigma_dispersion=20.0,
                                 distancia_fuente=40.0, max_pasos_rayo=1_000_000, semilla=42)
        # (camino libre medio, columna del barrido, rugosidad media de referencia con 15 grados de emisión)
        for lam, j, w in [(5.0, 0, 0.778262), (30.0, 3, 0.697374), (np.inf, 6, 0.652687)]:
            with self.subTest(camino_libre=lam):
                semillas = np.random.SeedSequence(42, spawn_key=(2, 3, j)).generate_state(10)
                superficies = correr_con_semillas(params, semillas, 400.0, 15, lam, 50_000)[0]
                self.assertAlmostEqual(float(superficies.reshape(10, -1).std(axis=1).mean()), w, places=5)


if __name__ == "__main__":
    unittest.main()
