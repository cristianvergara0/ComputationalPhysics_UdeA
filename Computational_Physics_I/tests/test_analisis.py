"""Pruebas de las métricas de superficie, el análisis espectral y los exponentes de escalamiento."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thinfilm_mc.analisis import (
    beta_ventana,
    calcular_metricas,
    calcular_observables,
    correlacion_alturas_radial,
    espectro_potencia_radial,
    exponente_efectivo,
    islas_periodicas,
    longitud_correlacion,
    pendiente_local,
    resumir_superficie,
    tiempo_de_cruce,
)


def tablero(n, bajo=0, alto=1):
    """Superficie en tablero de ajedrez entre dos alturas."""
    return (np.indices((n, n)).sum(axis=0) % 2 * (alto - bajo) + bajo).astype(np.int32)


class TestMetricas(unittest.TestCase):
    def test_superficie_plana(self):
        h = np.full((8, 8), 5, dtype=np.int32)
        espesor, rugosidad = calcular_metricas(h, 8, 8)
        self.assertEqual((espesor, rugosidad), (5.0, 0.0))

    def test_tablero_de_ajedrez(self):
        espesor, rugosidad = calcular_metricas(tablero(8), 8, 8)
        self.assertAlmostEqual(espesor, 0.5, places=12)
        self.assertAlmostEqual(rugosidad, 0.5, places=12)

    def test_coincide_con_numpy(self):
        h = np.random.default_rng(0).integers(0, 20, size=(30, 17)).astype(np.int32)
        espesor, rugosidad = calcular_metricas(h, 30, 17)
        self.assertAlmostEqual(espesor, h.mean(), places=12)
        self.assertAlmostEqual(rugosidad, h.std(), places=12)

    def test_la_rugosidad_no_depende_de_una_altura_base_grande(self):
        # Con alturas del orden de 1e9 una fórmula de un solo paso pierde toda la precisión.
        h = (tablero(8) + 10 ** 9).astype(np.int64)
        _, rugosidad = calcular_metricas(h, 8, 8)
        self.assertAlmostEqual(rugosidad, 0.5, places=9)


class TestObservables(unittest.TestCase):
    def test_superficie_plana_sin_cubrir(self):
        obs = calcular_observables(np.zeros((16, 16), dtype=np.int32))
        self.assertEqual(obs["rugosidad"], 0.0)
        self.assertEqual(obs["vacias"], 100.0)
        self.assertEqual(obs["escalones"], 0.0)

    def test_escalones_en_un_tablero_de_dos_capas(self):
        obs = calcular_observables(tablero(16, 0, 2))
        self.assertEqual(obs["escalones"], 100.0)
        self.assertEqual(obs["maximo"], 2)
        self.assertEqual(obs["vacias"], 50.0)

    def test_resumir_superficie(self):
        resumen = resumir_superficie(tablero(8, 1, 4))
        self.assertAlmostEqual(resumen["espesor"], 2.5)
        self.assertEqual((resumen["maximo"], resumen["minimo"]), (4, 1))
        self.assertEqual(resumen["vacias"], 0.0)


class TestEspectroYCorrelacion(unittest.TestCase):
    def test_el_pico_del_espectro_esta_en_la_frecuencia_de_la_onda(self):
        n, ciclos = 64, 8
        x = np.arange(n)
        h = np.round(20 + 5 * np.cos(2 * np.pi * ciclos * x / n))[:, None] * np.ones((n, n))
        k, potencia = espectro_potencia_radial(h.astype(np.int32), True)
        ancho_bin = k[1] - k[0]
        self.assertLessEqual(abs(k[np.argmax(potencia)] - ciclos / n), ancho_bin)

    def test_el_ruido_blanco_casi_no_esta_correlacionado(self):
        h = np.random.default_rng(1).poisson(10, size=(120, 120)).astype(np.int32)
        _, correlacion, xi = correlacion_alturas_radial(h, True)
        self.assertLess(xi, 1.6)
        self.assertGreater(correlacion[0], 0.0)

    def test_una_superficie_suave_tiene_longitud_de_correlacion_grande(self):
        n = 96
        x = np.arange(n)
        h = np.round(30 + 6 * np.cos(2 * np.pi * x / n)[:, None] + 6 * np.cos(2 * np.pi * x / n)[None, :])
        _, _, xi = correlacion_alturas_radial(h.astype(np.int32), True)
        self.assertGreater(xi, 8.0)

    def test_la_version_vectorizada_coincide_con_la_definicion(self):
        h = np.random.default_rng(2).poisson(10, size=(64, 64)).astype(np.int32)
        _, _, xi = correlacion_alturas_radial(h, True)
        self.assertAlmostEqual(longitud_correlacion(h), xi, places=9)


class TestExponentes(unittest.TestCase):
    def test_recupera_un_exponente_conocido(self):
        eventos = np.arange(500, 100001, 500, dtype=float)
        n_celdas = 1000
        capas = eventos / n_celdas
        rugosidad = 0.4 * capas ** 0.25
        beta = exponente_efectivo(eventos, rugosidad, n_celdas, 1.0)
        self.assertAlmostEqual(float(beta), 0.25, places=9)

    def test_beta_ventana_recupera_un_exponente_conocido(self):
        capas = np.linspace(0.5, 40.0, 200)
        rugosidad = 0.3 * capas ** 0.2
        resultado = beta_ventana(capas, rugosidad, 5.0, 30.0)
        beta = resultado[0] if isinstance(resultado, (tuple, list)) else resultado
        self.assertAlmostEqual(float(beta), 0.2, places=9)

    def test_sin_puntos_suficientes_no_hay_exponente(self):
        beta = exponente_efectivo(np.array([100.0, 200.0]), np.array([1.0, 1.1]), 100, 50.0)
        self.assertTrue(np.isnan(beta))


class TestPendienteYCruce(unittest.TestCase):
    def test_la_pendiente_local_de_una_ley_de_potencias_es_su_exponente(self):
        t = np.geomspace(0.1, 1000.0, 60)
        np.testing.assert_allclose(pendiente_local(t, 2.0 * t ** 0.3), 0.3, rtol=0, atol=1e-12)

    def test_la_pendiente_local_sigue_un_cambio_de_regimen(self):
        t = np.geomspace(1.0, 1e4, 200)
        w = np.minimum(t ** 0.5, 10.0)                                   # saturación en t = 100
        pendiente = pendiente_local(t, w)
        self.assertAlmostEqual(float(pendiente[t < 50].mean()), 0.5, places=9)
        self.assertAlmostEqual(float(pendiente[t > 200].max()), 0.0, places=12)

    def test_el_cruce_se_interpola_en_escala_logaritmica(self):
        t = np.geomspace(1.0, 1e4, 9)
        w = np.log(t)                                                    # lineal en ln t: interpolación exacta
        self.assertAlmostEqual(tiempo_de_cruce(t, w, w_sat=10.0, fraccion=0.5), np.exp(5.0), places=6)

    def test_el_cruce_en_la_primera_muestra_y_sin_cruce(self):
        t = np.array([1.0, 2.0, 4.0])
        self.assertEqual(tiempo_de_cruce(t, np.array([0.9, 1.0, 1.1]), 1.0, 0.8), 1.0)
        self.assertTrue(np.isnan(tiempo_de_cruce(t, np.array([0.1, 0.2, 0.3]), 1.0, 0.8)))


class TestIslasPeriodicas(unittest.TestCase):
    def test_sin_regiones_sobre_el_umbral(self):
        self.assertEqual(islas_periodicas(np.zeros((6, 6)), 0.5).size, 0)

    def test_una_region_que_cruza_un_borde_se_cuenta_una_vez(self):
        h = np.zeros((8, 8), dtype=np.int32)
        h[3, 0] = h[3, 7] = h[4, 7] = 2                                  # cruza el borde izquierdo-derecho
        h[0, 4] = h[7, 4] = 2                                            # cruza el borde superior-inferior
        h[5, 3] = 2                                                      # celda aislada
        np.testing.assert_array_equal(islas_periodicas(h, 1.0), [3, 2, 1])

    def test_una_region_que_cruza_dos_bordes_a_la_vez(self):
        h = np.zeros((6, 6), dtype=np.int32)
        h[0, 0] = h[0, 5] = h[5, 0] = h[5, 5] = 1                       # las cuatro esquinas son vecinas
        np.testing.assert_array_equal(islas_periodicas(h, 0.5), [4])

    def test_las_diagonales_no_conectan(self):
        h = np.zeros((6, 6), dtype=np.int32)
        h[2, 2] = h[3, 3] = 1
        np.testing.assert_array_equal(islas_periodicas(h, 0.5), [1, 1])

    def test_el_area_total_es_el_numero_de_celdas_sobre_el_umbral(self):
        h = np.random.default_rng(3).poisson(5, size=(40, 40))
        umbral = h.mean() + h.std()
        areas = islas_periodicas(h, umbral)
        self.assertEqual(int(areas.sum()), int((h > umbral).sum()))
        self.assertTrue(np.all(np.diff(areas) <= 0))


if __name__ == "__main__":
    unittest.main()
