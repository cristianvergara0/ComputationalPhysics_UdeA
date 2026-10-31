"""Pruebas de la simulación con instantes de registro y del análisis del crecimiento (``thinfilm_mc.crecimiento``)."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thinfilm_mc import motor
from thinfilm_mc.barridos import ParametrosUnificado
from thinfilm_mc.crecimiento import (
    ajuste_por_replica,
    capas,
    codigo_fuente,
    espesor_de_inicio,
    estadistica_por_instante,
    fraccion_elevada,
    media_y_se,
    mediana_censurada,
    observables_de_instantaneas,
    resumir_simulacion,
    ritmo_del_relieve,
    saturacion_por_tamano,
    simular_modelo,
    tabla_de_historias,
)

PARAMS = ParametrosUnificado(nx=16, ny=16, t_escala=500, max_pasos_difusion=5, sigma_incidencia=15.0, sigma_dispersion=20.0,
                             mfp=10.0, distancia_fuente=40.0)
RELAJACION = {"enable_angle": False, "enable_scattering": False, "enable_shadowing": False, "enable_diffusion": True, "temp": 400}
RD = dict(RELAJACION, enable_diffusion=False, temp=0)


def en_silencio(funcion, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return funcion(*args, **kwargs)


def datos_sinteticos(h, valores, detenida=None, espesor_final=None, L=10):
    """Diccionario con la forma mínima de un resultado de ``simular_modelo`` para probar el análisis."""
    valores = np.atleast_2d(np.asarray(valores, dtype=float))
    n = valores.shape[0]
    return {"eventos": np.asarray(h, dtype=float) * L ** 2, "L": L, "rugosidad": valores, "relieve": valores,
            "detenida": np.zeros(n, bool) if detenida is None else np.asarray(detenida),
            "espesor_final": np.full(n, h[-1]) if espesor_final is None else np.asarray(espesor_final, dtype=float)}


class TestSimularModelo(unittest.TestCase):
    def test_instantaneas_y_metadatos(self):
        d = en_silencio(simular_modelo, PARAMS, "prueba", RELAJACION, 3, 7, 4.0, [4.0, 1.0], n_log=12)
        np.testing.assert_array_equal(d["objetivos"], [1.0, 4.0])
        self.assertEqual(d["L"], 16)
        self.assertEqual(d["instantaneas"].shape, (3, 2, 16, 16))
        np.testing.assert_array_equal(d["instantaneas"].reshape(3, 2, -1).sum(axis=2), [[256, 1024]] * 3)
        self.assertAlmostEqual(capas(d)[-1], 4.0)

    def test_la_realizacion_no_depende_de_los_instantes_registrados(self):
        a = en_silencio(simular_modelo, PARAMS, "prueba", RELAJACION, 2, 7, 4.0, [2.0], n_log=5)
        b = en_silencio(simular_modelo, PARAMS, "prueba", RELAJACION, 2, 7, 4.0, [2.0], n_log=40)
        np.testing.assert_array_equal(a["instantaneas"], b["instantaneas"])
        np.testing.assert_array_equal(a["superficies"], b["superficies"])

    def test_rechaza_instantaneas_que_coinciden(self):
        with self.assertRaises(ValueError):
            en_silencio(simular_modelo, PARAMS, "prueba", RD, 2, 7, 4.0, [1.0, 1.001])

    def test_observables_e_historias(self):
        d = en_silencio(simular_modelo, PARAMS, "prueba", RD, 3, 7, 2.0, [1.0, 2.0], n_log=10)
        obs = observables_de_instantaneas(d, "M1")
        self.assertEqual(len(obs), 6)
        fila = obs[(obs["replica"] == 1) & (obs["objetivo"] == 2.0)].iloc[0]
        superficie = d["instantaneas"][1, 1]
        self.assertAlmostEqual(fila["rugosidad"], superficie.std(), places=12)
        self.assertAlmostEqual(fila["relieve"], superficie.max() - 2.0, places=12)
        historia = tabla_de_historias({"M1": d})
        self.assertEqual(len(historia), 3 * len(d["eventos"]))


class TestEstadistica(unittest.TestCase):
    def test_estadistica_por_instante_ignora_las_replicas_detenidas(self):
        d = datos_sinteticos([1, 2, 3], [[1.0, 2.0, 3.0], [3.0, 4.0, np.nan]])
        h, media, se, n = estadistica_por_instante(d)
        np.testing.assert_array_equal(n, [2, 2, 1])
        np.testing.assert_allclose(media, [2.0, 3.0, 3.0])
        self.assertAlmostEqual(se[0], 1.0)
        self.assertTrue(np.isnan(se[2]))

    def test_media_y_se(self):
        self.assertEqual(media_y_se([1.0, 3.0, np.nan]), (2.0, 1.0))
        self.assertTrue(np.isnan(media_y_se([1.0])[0]))

    def test_ajuste_por_replica(self):
        x = np.linspace(0, 5, 11)
        y = np.vstack([2 * x + 1, -x, np.where(x > 4, np.nan, x)])
        ajuste = ajuste_por_replica(x, y, 0, 5)
        np.testing.assert_allclose(ajuste[:2], [[2, 1], [-1, 0]], atol=1e-12)
        self.assertTrue(np.isnan(ajuste[2]).all())
        np.testing.assert_allclose(ajuste_por_replica(x, y, 0, 3)[2], [1, 0], atol=1e-12)


class TestSaturacion(unittest.TestCase):
    def test_recupera_la_rugosidad_saturada_y_no_rechaza_una_meseta(self):
        rng = np.random.default_rng(0)
        resultados = {}
        for L, w2 in [(8, 0.4), (16, 0.5)]:
            h = np.linspace(0.01, 0.5, 200) * L ** 2
            valores = np.sqrt(w2 + rng.normal(0, 0.01, size=(30, h.size)))
            resultados[L] = datos_sinteticos(h, valores, L=L)
        tabla = saturacion_por_tamano(resultados, (0.2, 0.5))
        np.testing.assert_allclose(tabla["W_sat^2"], [0.4, 0.5], atol=0.002)
        self.assertTrue((tabla["p (t pareada)"] > 0.001).all())

    def test_detecta_una_rugosidad_que_aun_crece(self):
        h = np.linspace(0.01, 0.5, 200) * 64
        valores = np.sqrt(0.3 + 0.2 * h / h[-1] + np.random.default_rng(1).normal(0, 0.005, size=(20, h.size)))
        tabla = saturacion_por_tamano({8: datos_sinteticos(h, valores, L=8)}, (0.2, 0.5))
        self.assertLess(tabla["p (t pareada)"].iloc[0], 1e-6)


class TestInestabilidad(unittest.TestCase):
    def test_espesor_de_inicio_interpolado_y_con_corte(self):
        h = np.array([1.0, 2.0, 4.0, 8.0])
        relieve = np.array([[5.0, 10.0, 30.0, 50.0], [5.0, 6.0, np.nan, np.nan], [5.0, 6.0, 7.0, 8.0]])
        d = datos_sinteticos(h, relieve, detenida=[False, True, False], espesor_final=[8.0, 3.0, 8.0])
        inicio = espesor_de_inicio(d, 20.0)
        self.assertAlmostEqual(inicio[0], 2.0 * 2 ** 0.5, places=9)    # 20 está a mitad de camino entre 10 y 30
        self.assertEqual(inicio[1], 3.0)                              # detenida antes de registrar el cruce
        self.assertTrue(np.isnan(inicio[2]))
        self.assertTrue(np.isnan(espesor_de_inicio(d, 20.0, hasta=2.5)[1]))

    def test_ritmo_de_un_relieve_exponencial(self):
        h = np.linspace(1, 20, 40)
        d = datos_sinteticos(h, 2.0 * np.exp(0.3 * h))
        self.assertAlmostEqual(ritmo_del_relieve(d, 20.0, 100.0)[0], 0.3, places=9)


class TestCodigoFuente(unittest.TestCase):
    def test_devuelve_el_codigo_de_una_funcion_compilada(self):
        texto = codigo_fuente(motor.difundir_particula_pbc)
        self.assertIn("def difundir_particula_pbc", texto)


class TestFraccionElevada(unittest.TestCase):
    def test_cuenta_las_columnas_que_superan_la_media_en_cada_umbral(self):
        sup = np.zeros((10, 10), dtype=np.int32)
        sup[0, 0] = 30
        sup[1, 1] = 15
        datos = {"L": 10, "objetivos": np.array([1.0]), "instantaneas": sup[None, None],
                 "instantanea_valida": np.array([[True]])}
        f = fraccion_elevada(datos, (10.0, 20.0))
        media = sup.mean()
        self.assertAlmostEqual(f["fraccion_10"].iloc[0], np.mean(sup - media >= 10))
        self.assertAlmostEqual(f["fraccion_10"].iloc[0], 0.02)
        self.assertAlmostEqual(f["fraccion_20"].iloc[0], 0.01)

    def test_omite_las_instantaneas_no_alcanzadas(self):
        d = en_silencio(simular_modelo, PARAMS, "prueba", RD, 3, 7, 8.0, [1.0, 8.0], n_log=10, relieve_maximo=6.0)
        f = fraccion_elevada(d)
        self.assertEqual(len(f), int(d["instantanea_valida"].sum()))


class TestResumenes(unittest.TestCase):
    def test_mediana_censurada(self):
        self.assertEqual(mediana_censurada([1.0, 2.0, np.nan]), 2.0)
        self.assertEqual(mediana_censurada([1.0, np.nan, np.nan]), np.inf)
        self.assertEqual(mediana_censurada([3.0, 1.0, 2.0, 5.0]), 2.5)

    def test_resumir_simulacion(self):
        d = en_silencio(simular_modelo, PARAMS, "prueba", RD, 3, 7, 8.0, [1.0, 8.0], n_log=10, relieve_maximo=6.0)
        r = resumir_simulacion(d, 3.0)
        self.assertEqual(r["replicas"], 3)
        self.assertEqual(r["detenidas"], int(d["detenida"].sum()))
        if d["instantanea_valida"][:, 0].all():
            self.assertAlmostEqual(r["W_1"], d["instantaneas"][:, 0].reshape(3, -1).std(axis=1).mean(), places=12)
        if not d["instantanea_valida"][:, 1].all():
            self.assertTrue(np.isnan(r["W_8"]))
        self.assertLessEqual(r["h_umbral"], r["h_corte"])


if __name__ == "__main__":
    unittest.main()
