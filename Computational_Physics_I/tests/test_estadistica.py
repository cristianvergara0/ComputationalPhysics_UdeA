"""Pruebas de la estadística de las réplicas: bondad de ajuste, intervalos, interacciones y contrastes."""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from thinfilm_mc.estadistica import (
    a_cubo,
    ajuste_lineal_multiple,
    ajuste_ponderado,
    a_matriz,
    anova_dos_factores,
    cociente_con_error,
    contraste_welch,
    efectos_factorial,
    media_ic95,
    prueba_bondad_ajuste_poisson,
    residuos_interaccion,
    resumen_bondad_ajuste,
    resumir_por_punto,
)


def deposicion_aleatoria(rng, n, n_particulas):
    """Superficie de ``n x n`` celdas tras depositar ``n_particulas`` en celdas uniformes."""
    return rng.multinomial(n_particulas, np.full(n * n, 1.0 / (n * n))).reshape(n, n)


class TestBondadDeAjuste(unittest.TestCase):
    def test_una_deposicion_aleatoria_es_compatible_con_poisson(self):
        h = deposicion_aleatoria(np.random.default_rng(3), 200, 40000)
        _, p, gl, n_intervalos = prueba_bondad_ajuste_poisson(h)
        self.assertGreater(p, 1e-3)
        self.assertEqual(gl, n_intervalos - 2)

    def test_una_superficie_uniforme_no_es_poisson(self):
        h = np.full((100, 100), 10)
        _, p, _, _ = prueba_bondad_ajuste_poisson(h)
        self.assertLess(p, 1e-10)

    def test_la_prueba_rechaza_con_la_frecuencia_nominal_bajo_la_hipotesis_nula(self):
        # Con n_intervalos - 2 grados de libertad, los valores p de una deposición aleatoria
        # son uniformes: al nivel del 5 % debe rechazarse en torno al 5 % de las superficies.
        rng = np.random.default_rng(11)
        rechazos = sum(prueba_bondad_ajuste_poisson(deposicion_aleatoria(rng, 40, 1600))[1] < 0.05 for _ in range(400))
        self.assertTrue(0.02 <= rechazos / 400 <= 0.09, rechazos / 400)

    def test_resumen_de_varias_superficies(self):
        rng = np.random.default_rng(5)
        resumen = resumen_bondad_ajuste([deposicion_aleatoria(rng, 60, 3600) for _ in range(15)])
        self.assertEqual(resumen["n"], 15)
        self.assertEqual(len(resumen["valores_p"]), 15)
        self.assertGreater(resumen["p_conjunto"], 1e-3)
        self.assertGreater(resumen["gl_conjunto"], 15)


class TestIntervalosYCocientes(unittest.TestCase):
    def test_media_e_intervalo_de_confianza(self):
        media, semiancho = media_ic95([1.0, 2.0, 3.0, 4.0, 5.0])
        esperado = stats.t.ppf(0.975, 4) * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
        self.assertAlmostEqual(media, 3.0)
        self.assertAlmostEqual(semiancho, esperado, places=12)

    def test_cociente_y_su_error_por_el_metodo_delta(self):
        cociente, error = cociente_con_error(2.0, 0.1, 4.0, 0.2)
        self.assertAlmostEqual(cociente, 0.5)
        self.assertAlmostEqual(error, 0.5 * np.sqrt((0.1 / 2) ** 2 + (0.2 / 4) ** 2), places=12)


class TestContrasteWelch(unittest.TestCase):
    def test_coincide_con_scipy(self):
        x = np.array([1.00, 1.10, 0.90, 1.05, 0.95, 1.02])
        y = np.array([1.30, 1.20, 1.40, 1.25, 1.35])
        r = contraste_welch(x, y)
        ref = stats.ttest_ind(x, y, equal_var=False)
        self.assertAlmostEqual(r["t"], ref.statistic, places=10)
        self.assertAlmostEqual(r["p"], ref.pvalue, places=10)
        self.assertAlmostEqual(r["gl"], ref.df, places=8)
        self.assertAlmostEqual(r["diferencia"], x.mean() - y.mean(), places=12)
        self.assertEqual((r["n_x"], r["n_y"]), (6, 5))

    def test_incluye_la_comprobacion_de_normalidad(self):
        rng = np.random.default_rng(0)
        r = contraste_welch(rng.normal(0, 1, 40), rng.normal(0, 1, 40))
        self.assertGreater(r["p_shapiro_x"], 0.01)
        sesgada = contraste_welch(rng.exponential(1, 60) ** 3, rng.normal(0, 1, 40))
        self.assertLess(sesgada["p_shapiro_x"], 1e-3)

    def test_muestras_identicas(self):
        r = contraste_welch([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(r["p"], 1.0)
        self.assertTrue(np.isnan(contraste_welch([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])["p_shapiro_x"]))

    def test_exige_dos_valores_por_muestra(self):
        with self.assertRaises(ValueError):
            contraste_welch([1.0], [1.0, 2.0])


class TestInteracciones(unittest.TestCase):
    def test_los_residuos_de_un_modelo_aditivo_son_nulos(self):
        media = np.add.outer([0.0, 1.0, 2.0], [0.0, 3.0, 5.0, 6.0])
        residuos = residuos_interaccion(media, np.full_like(media, 1e-4))[0]
        np.testing.assert_allclose(residuos, 0.0, atol=1e-12)

    def test_los_residuos_de_un_modelo_multiplicativo_no_son_nulos(self):
        media = np.multiply.outer([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
        residuos = residuos_interaccion(media, np.full_like(media, 1e-4))[0]
        self.assertGreater(np.abs(residuos).max(), 0.1)

    def test_anova_distingue_efectos_aditivos_de_interaccion(self):
        rng = np.random.default_rng(8)
        aditivo = np.add.outer([0.0, 1.0, 2.0], [0.0, 3.0, 5.0, 6.0])[:, :, None] + rng.normal(0, 0.3, (3, 4, 10))
        a = anova_dos_factores(aditivo)
        self.assertGreater(a["p_ab"], 0.01)
        self.assertAlmostEqual(a["eta2_a"] + a["eta2_b"] + a["eta2_ab"] + a["eta2_e"], 1.0, places=10)
        producto = np.multiply.outer([1.0, 2.0, 3.0], [1.0, 2.0, 4.0, 6.0])[:, :, None] + rng.normal(0, 0.3, (3, 4, 10))
        self.assertLess(anova_dos_factores(producto)["p_ab"], 1e-6)

    def test_efectos_de_un_factorial_2_a_la_2(self):
        # y = 1 + 2 xA + 3 xB + 0.5 xA xB, con xA, xB = -1, +1: los efectos son 4, 6 y 1.
        filas = []
        for a, b in itertools.product([0, 1], repeat=2):
            for r in range(4):
                xa, xb = 2 * a - 1, 2 * b - 1
                filas.append({"A": a, "B": b, "rugosidad": 1 + 2 * xa + 3 * xb + 0.5 * xa * xb + 1e-3 * (r - 1.5)})
        efectos = efectos_factorial(pd.DataFrame(filas), ["A", "B"]).set_index("efecto")["valor"]
        self.assertAlmostEqual(efectos["A"], 4.0, places=3)
        self.assertAlmostEqual(efectos["B"], 6.0, places=3)
        self.assertAlmostEqual(efectos["A×B"], 1.0, places=3)


class TestResumenPorPunto(unittest.TestCase):
    def setUp(self):
        observables = ["rugosidad", "maximo", "vacias", "escalones", "xi", "colisiones", "sigma_eff_med", "sigma_red_med"]
        filas = []
        for i, j, r in itertools.product(range(2), range(3), range(4)):
            fila = {"i": i, "j": j, "replica": r}
            fila.update({o: 1.0 + i + 0.1 * j + 0.01 * r + k for k, o in enumerate(observables)})
            filas.append(fila)
        self.tabla = pd.DataFrame(filas)

    def test_media_error_estandar_y_numero_de_replicas(self):
        resumen = resumir_por_punto(self.tabla)
        self.assertEqual(len(resumen), 6)
        self.assertTrue((resumen["n_replicas"] == 4).all())
        fila = resumen[(resumen.i == 1) & (resumen.j == 2)].iloc[0]
        self.assertAlmostEqual(fila["rugosidad"], 1.0 + 1 + 0.2 + 0.015, places=12)
        self.assertAlmostEqual(fila["rugosidad_se"], np.std([0, 0.01, 0.02, 0.03], ddof=1) / 2.0, places=12)

    def test_reorganizacion_en_matrices_y_cubos(self):
        resumen = resumir_por_punto(self.tabla)
        matriz = a_matriz(resumen, "rugosidad", 2, 3)
        self.assertEqual(matriz.shape, (2, 3))
        self.assertAlmostEqual(matriz[1, 2], 2.215, places=12)
        cubo = a_cubo(self.tabla, "rugosidad", 2, 3)
        self.assertEqual(cubo.shape, (2, 3, 4))
        np.testing.assert_allclose(cubo.mean(axis=2), matriz, atol=1e-12)


class TestAjusteLineal(unittest.TestCase):
    def test_coincide_con_linregress_con_una_variable(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        y = 1.5 + 2.0 * x + rng.normal(0, 0.3, 40)
        tabla, r2 = ajuste_lineal_multiple({"x": x}, y)
        ref = stats.linregress(x, y)
        self.assertAlmostEqual(tabla.loc["x", "coef"], ref.slope, places=12)
        self.assertAlmostEqual(tabla.loc["constante", "coef"], ref.intercept, places=12)
        self.assertAlmostEqual(tabla.loc["x", "se"], ref.stderr, places=12)
        self.assertAlmostEqual(tabla.loc["x", "p"], ref.pvalue, places=12)
        self.assertAlmostEqual(r2, ref.rvalue ** 2, places=12)

    def test_recupera_los_coeficientes_de_dos_variables(self):
        rng = np.random.default_rng(1)
        x1, x2 = rng.normal(size=200), rng.normal(size=200)
        y = 3.0 - 1.0 * x1 + 0.5 * x2 + rng.normal(0, 0.05, 200)
        tabla, r2 = ajuste_lineal_multiple({"x1": x1, "x2": x2}, y)
        np.testing.assert_allclose(tabla["coef"].to_numpy(), [3.0, -1.0, 0.5], atol=0.02)
        np.testing.assert_array_less(np.abs(tabla["coef"].to_numpy() - [3.0, -1.0, 0.5]), 4.0 * tabla["se"].to_numpy())
        self.assertTrue((tabla["ic95_inf"] < tabla["coef"]).all() and (tabla["coef"] < tabla["ic95_sup"]).all())
        self.assertGreater(r2, 0.98)

    def test_una_variable_irrelevante_no_es_significativa(self):
        rng = np.random.default_rng(2)
        x1, ruido = rng.normal(size=60), rng.normal(size=60)
        y = 1.0 + x1 + rng.normal(0, 0.5, 60)
        tabla, _ = ajuste_lineal_multiple({"x1": x1, "ruido": ruido}, y)
        self.assertLess(tabla.loc["x1", "p"], 1e-6)
        self.assertGreater(tabla.loc["ruido", "p"], 0.01)

    def test_exige_mas_observaciones_que_coeficientes(self):
        with self.assertRaises(ValueError):
            ajuste_lineal_multiple({"a": [1.0, 2.0], "b": [2.0, 1.0]}, [1.0, 2.0])

    def test_un_ajuste_exacto_no_falla(self):
        x = np.arange(10.0)
        tabla, r2 = ajuste_lineal_multiple({"x": x}, 2.0 * x + 1.0)
        self.assertAlmostEqual(tabla.loc["x", "coef"], 2.0, places=10)
        self.assertAlmostEqual(r2, 1.0, places=12)


class TestAjustePonderado(unittest.TestCase):
    def test_recupera_una_recta_exacta(self):
        x = np.log([16.0, 32.0, 64.0, 128.0])
        r = ajuste_ponderado(x, 0.2 + 0.13 * x, np.full(4, 0.01))
        self.assertAlmostEqual(r["pendiente"], 0.13, places=12)
        self.assertAlmostEqual(r["ordenada"], 0.2, places=12)
        self.assertAlmostEqual(r["chi2"], 0.0, places=12)
        self.assertEqual(r["gl"], 2)

    def test_coincide_con_curve_fit_con_errores_absolutos(self):
        from scipy.optimize import curve_fit
        rng = np.random.default_rng(4)
        x = np.linspace(0.0, 3.0, 8)
        se = np.linspace(0.02, 0.08, 8)
        y = 1.0 - 0.5 * x + rng.normal(0.0, se)
        r = ajuste_ponderado(x, y, se)
        coef, cov = curve_fit(lambda t, a, b: a + b * t, x, y, sigma=se, absolute_sigma=True)
        np.testing.assert_allclose([r["ordenada"], r["pendiente"]], coef, rtol=1e-9)
        np.testing.assert_allclose([r["se_ordenada"], r["se_pendiente"]], np.sqrt(np.diag(cov)), rtol=1e-6)

    def test_el_chi2_sigue_su_distribucion_si_los_errores_son_correctos(self):
        rng = np.random.default_rng(5)
        x = np.arange(6.0)
        p = [ajuste_ponderado(x, 2.0 * x + rng.normal(0.0, 0.1, 6), np.full(6, 0.1))["p"] for _ in range(400)]
        self.assertGreater(stats.kstest(p, "uniform").pvalue, 0.01)

    def test_rechaza_errores_no_positivos_y_pocos_puntos(self):
        with self.assertRaises(ValueError):
            ajuste_ponderado([0, 1, 2], [0, 1, 2], [0.1, 0.0, 0.1])
        with self.assertRaises(ValueError):
            ajuste_ponderado([0, 1], [0, 1], [0.1, 0.1])


if __name__ == "__main__":
    unittest.main()
