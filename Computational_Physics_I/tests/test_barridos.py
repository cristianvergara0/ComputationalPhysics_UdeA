"""Pruebas de los barridos de parámetros, la ejecución de modelos y el análisis angular."""

import contextlib
import dataclasses
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from thinfilm_mc.barridos import (
    ParametrosMotor,
    ParametrosUnificado,
    colapso_por_punto,
    contraste_identidad,
    contraste_identidad_por_punto,
    correr_con_semillas,
    descriptores_llegada,
    ejecutar_barrido,
    ejecutar_modelo_muestreado,
    malla_de_muestreo,
    referencia_balistica,
    resumir_vuelos,
    semillas_de_modelo,
    sigma_reducida_gaussiana,
    simular_punto,
    tabla_colapso,
)
from thinfilm_mc.motor import EMISION_COSENO, ejecutar_suite_muestreada, muestrear_vuelos_libres

PARAMS = ParametrosMotor(nx=16, ny=16, t_escala=500, max_pasos_difusion=5, sigma_dispersion=20.0,
                         distancia_fuente=40.0, max_pasos_rayo=1_000_000, semilla=42)
PARAMS_U = ParametrosUnificado(nx=16, ny=16, t_escala=500, max_pasos_difusion=5, sigma_incidencia=15.0, sigma_dispersion=20.0,
                               mfp=10.0, distancia_fuente=40.0)
BANDERAS_M6 = {"enable_angle": True, "enable_scattering": True, "enable_shadowing": True, "enable_diffusion": True, "temp": 400}


def en_silencio(funcion, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return funcion(*args, **kwargs)


class TestParametros(unittest.TestCase):
    def test_los_parametros_son_inmutables(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            PARAMS.nx = 10
        with self.assertRaises(dataclasses.FrozenInstanceError):
            PARAMS_U.mfp = 5.0
        self.assertEqual(dataclasses.replace(PARAMS, nx=20).nx, 20)


class TestCorrerConSemillas(unittest.TestCase):
    def test_formas_y_conservacion(self):
        semillas = [3, 4, 5]
        superficies, colisiones, theta2, theta2_red, truncados = correr_con_semillas(PARAMS, semillas, 400.0, 15.0, 10.0, 300)
        self.assertEqual(superficies.shape, (3, 16, 16))
        self.assertTrue(np.all(superficies.sum(axis=(1, 2)) == 300))
        for v in (colisiones, theta2, theta2_red, truncados):
            self.assertEqual(np.shape(v), (3,))
        self.assertTrue(np.all(colisiones > 0))
        self.assertTrue(np.all(theta2_red <= theta2 + 1e-9))

    def test_cada_replica_depende_solo_de_su_semilla(self):
        conjunta = correr_con_semillas(PARAMS, [3, 4], 400.0, 15.0, 10.0, 200)[0]
        suelta = correr_con_semillas(PARAMS, [4], 400.0, 15.0, 10.0, 200)[0]
        np.testing.assert_array_equal(conjunta[1], suelta[0])

    def test_la_ley_del_coseno_se_selecciona_con_dos_argumentos(self):
        sup, col, *_ = correr_con_semillas(PARAMS, [1], 400.0, 0.0, 10.0, 200, 1, EMISION_COSENO, 30.0)
        self.assertEqual(int(sup.sum()), 200)
        self.assertGreater(col[0], 0)


class TestSimularPunto(unittest.TestCase):
    def test_columnas_y_numero_de_filas(self):
        tabla, primera = simular_punto(PARAMS, 1, 0, 2, 400.0, 15.0, 10.0, 3, 200)
        self.assertEqual(len(tabla), 3)
        for columna in ("replica", "rugosidad", "maximo", "vacias", "escalones", "xi", "colisiones", "sigma_eff_med", "sigma_red_med"):
            self.assertIn(columna, tabla.columns)
        self.assertEqual(primera.shape, (16, 16))

    def test_reproducible_y_con_semillas_distintas_por_punto(self):
        a = simular_punto(PARAMS, 1, 0, 2, 400.0, 15.0, 10.0, 3, 200)[0]
        b = simular_punto(PARAMS, 1, 0, 2, 400.0, 15.0, 10.0, 3, 200)[0]
        c = simular_punto(PARAMS, 1, 0, 3, 400.0, 15.0, 10.0, 3, 200)[0]
        np.testing.assert_array_equal(a["rugosidad"], b["rugosidad"])
        self.assertFalse(np.array_equal(a["rugosidad"], c["rugosidad"]))

    def test_la_semilla_raiz_cambia_los_resultados(self):
        otra = dataclasses.replace(PARAMS, semilla=43)
        a = simular_punto(PARAMS, 1, 0, 2, 400.0, 15.0, 10.0, 2, 200)[0]["rugosidad"]
        b = simular_punto(otra, 1, 0, 2, 400.0, 15.0, 10.0, 2, 200)[0]["rugosidad"]
        self.assertFalse(np.array_equal(a, b))


class TestEjecutarBarrido(unittest.TestCase):
    def test_copia_las_etiquetas_y_usa_valores_por_defecto(self):
        puntos = [dict(i=0, j=0, temp=400.0, sigma=0.0, mfp=np.inf, etiqueta="balístico"),
                  dict(i=0, j=1, temp=400.0, sigma=15.0, mfp=10.0, etiqueta="con gas", sombreado=0)]
        tabla, superficies = en_silencio(ejecutar_barrido, PARAMS, 2, puntos, 2, 200, "prueba")
        self.assertEqual(len(tabla), 4)
        self.assertEqual(set(tabla["etiqueta"]), {"balístico", "con gas"})
        self.assertEqual(sorted(tabla["j"].unique()), [0, 1])
        self.assertEqual(len(superficies), 2)


class TestEjecutarModelo(unittest.TestCase):
    EVENTOS = np.array([100, 200, 400], dtype=np.int64)
    MARCAS = np.array([False, False, True])

    def ejecutar(self, nombre="M6: prueba", banderas=BANDERAS_M6, semilla=7, **kwargs):
        return en_silencio(ejecutar_modelo_muestreado, PARAMS_U, nombre, banderas, 3, semilla, self.EVENTOS, self.MARCAS,
                           **kwargs)

    def test_resultado_y_conservacion(self):
        datos = self.ejecutar()
        self.assertEqual(datos["rugosidad"].shape, (3, 3))
        np.testing.assert_allclose(datos["espesor"], np.tile(self.EVENTOS / 256, (3, 1)))
        np.testing.assert_array_equal(datos["superficies"].reshape(3, -1).sum(axis=1), 400)
        self.assertGreater(np.mean(datos["colisiones_por_atomo"]), 0.0)
        np.testing.assert_array_equal(datos["huecos_finales"], 0)

    def test_el_nombre_y_la_semilla_determinan_las_replicas(self):
        a = self.ejecutar()["rugosidad"]
        b = self.ejecutar()["rugosidad"]
        c = self.ejecutar(nombre="otro modelo")["rugosidad"]
        d = self.ejecutar(semilla=8)["rugosidad"]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a, d))

    def test_con_adhesion_lateral_la_suma_de_alturas_cuenta_los_huecos(self):
        datos = self.ejecutar(banderas=dict(BANDERAS_M6, temp=0, enable_lateral=True))
        sumas = datos["superficies"].reshape(3, -1).sum(axis=1)
        self.assertTrue(np.all(datos["huecos_finales"] > 0))
        np.testing.assert_array_equal(sumas, 400 + datos["huecos_finales"])


class TestAnalisisAngular(unittest.TestCase):
    def test_sigma_reducida_crece_y_se_satura(self):
        valores = [sigma_reducida_gaussiana(s) for s in (5.0, 15.0, 35.0, 60.0, 120.0, 400.0)]
        self.assertTrue(np.all(np.diff(valores) > 0))
        self.assertAlmostEqual(sigma_reducida_gaussiana(5.0), 5.0, delta=0.01)
        self.assertAlmostEqual(valores[-1], 90.0 / np.sqrt(3.0), delta=0.5)

    def test_la_identidad_angular_se_cumple_en_vuelos_simulados(self):
        n_col, tx, ty = muestrear_vuelos_libres(3000, 20, 20, 15.0, 20.0, 10.0, 10.0, 5, 10 ** 9, 0, 0.0)
        resumen = resumir_vuelos(n_col, tx, ty, 15.0, 20.0)
        claves = [k for k in resumen if "z" in k.lower().split("_")]
        self.assertTrue(claves, resumen.keys())
        self.assertLess(abs(resumen[claves[0]]), 4.0)


class TestReferenciaYColapso(unittest.TestCase):
    def setUp(self):
        resumen = pd.DataFrame({"sigma_theta": [40.0, 0.0, 20.0, 10.0], "rugosidad": [1.9, 1.0, 1.5, 1.2],
                                "rugosidad_se": [0.02, 0.01, 0.02, 0.01], "sigma_red_med": [38.0, 0.0, 20.0, 10.0]})
        self.ref = referencia_balistica(resumen)

    def test_la_referencia_queda_ordenada_por_el_ancho_de_emision(self):
        self.assertEqual(self.ref["sigma"].tolist(), [0.0, 10.0, 20.0, 40.0])
        self.assertEqual(list(self.ref.columns), ["sigma", "W", "W_se", "sigma_red"])
        self.assertEqual(self.ref["W"].tolist(), [1.0, 1.2, 1.5, 1.9])

    def punto(self, **kw):
        base = {"i": 0, "j": 0, "lambda": 10.0, "sigma_theta": 10.0, "sigma_disp": 20.0, "colisiones": 0.25,
                "sigma_eff_med": 14.0, "sigma_red_med": 15.0, "rugosidad": 1.3, "rugosidad_se": 0.01}
        base.update(kw)
        return base

    def test_las_dos_predicciones_son_las_esperadas(self):
        resultado = colapso_por_punto(pd.DataFrame([self.punto()]), self.ref).iloc[0]
        sigma_eff = np.sqrt(10.0 ** 2 + 20.0 ** 2 * 0.25)
        esperado = float(PchipInterpolator(self.ref["sigma"], self.ref["W"], extrapolate=False)(sigma_eff))
        self.assertAlmostEqual(resultado["sigma_eff"], sigma_eff, places=12)
        self.assertAlmostEqual(resultado["W_pred"], esperado, places=12)
        self.assertAlmostEqual(resultado["desv_rel"], (1.3 - esperado) / esperado, places=12)
        self.assertAlmostEqual(resultado["W_pred_red"], 1.2 + (1.5 - 1.2) * 0.5, places=12)     # sigma_red = 15 entre 10 y 20
        self.assertAlmostEqual(resultado["desv_rel_red"], (1.3 - 1.35) / 1.35, places=12)

    def test_no_extrapola_fuera_de_la_curva_y_omite_los_puntos_sin_gas(self):
        tabla = pd.DataFrame([self.punto(sigma_theta=40.0, colisiones=8.0, sigma_red_med=45.0), self.punto(j=1, **{"lambda": np.inf})])
        resultado = colapso_por_punto(tabla, self.ref)
        self.assertEqual(len(resultado), 1)
        self.assertTrue(np.isnan(resultado.iloc[0]["W_pred"]))
        self.assertTrue(np.isnan(resultado.iloc[0]["W_pred_red"]))

    def test_la_version_para_mapas_equivale_a_la_general(self):
        angulos, caminos = [0.0, 10.0], [5.0, np.inf]
        res = pd.DataFrame({"i": [0, 0, 1, 1], "j": [0, 1, 0, 1], "colisiones": [3.0, 0.0, 2.0, 0.0],
                            "sigma_eff_med": [30.0, 0.0, 28.0, 10.0], "sigma_red_med": [28.0, 0.0, 26.0, 10.0],
                            "rugosidad": [1.1, 1.0, 1.4, 1.2], "rugosidad_se": [0.01] * 4})
        mapa = tabla_colapso(res, self.ref, angulos, caminos, 20.0)
        general = res.copy()
        general["lambda"] = [caminos[j] for j in general["j"]]
        general["sigma_theta"] = [angulos[i] for i in general["i"]]
        general["sigma_disp"] = 20.0
        pd.testing.assert_frame_equal(mapa, colapso_por_punto(general, self.ref))
        self.assertEqual(len(mapa), 2)


class TestContrasteDeIdentidad(unittest.TestCase):
    def tabla(self, sesgo):
        rng = np.random.default_rng(3)
        filas = []
        for i in range(2):
            for r in range(8):
                n = rng.uniform(1, 5)
                sigma_eff = np.sqrt(15.0 ** 2 + 20.0 ** 2 * n + sesgo + rng.normal(0, 2.0))
                filas.append({"i": i, "j": 0, "sigma_theta": 15.0, "sigma_disp": 20.0, "colisiones": n, "sigma_eff_med": sigma_eff})
        return pd.DataFrame(filas)

    def test_sin_sesgo_no_se_rechaza_y_con_sesgo_si(self):
        self.assertTrue((contraste_identidad_por_punto(self.tabla(0.0))["p"] > 0.01).all())
        self.assertTrue((contraste_identidad_por_punto(self.tabla(60.0))["p"] < 0.01).all())

    def test_la_version_para_mapas_equivale_a_la_general(self):
        tabla = self.tabla(0.0)
        general = tabla.copy()
        reducida = tabla.drop(columns=["sigma_theta", "sigma_disp"])
        pd.testing.assert_frame_equal(contraste_identidad(reducida, [15.0, 15.0], 20.0), contraste_identidad_por_punto(general))

    def test_sin_dispersion_el_valor_p_es_uno(self):
        tabla = pd.DataFrame({"i": [0] * 4, "j": [0] * 4, "sigma_theta": 0.0, "sigma_disp": 20.0, "colisiones": 0.0, "sigma_eff_med": 0.0})
        self.assertEqual(contraste_identidad_por_punto(tabla).iloc[0]["p"], 1.0)


class TestDescriptoresDeLlegada(unittest.TestCase):
    def test_flujo_normal(self):
        d = descriptores_llegada(np.zeros(100), np.zeros(100))
        self.assertEqual((d["sigma_red"], d["polar_mediano"], d["f_rasante"]), (0.0, 0.0, 0.0))

    def test_los_angulos_se_reducen_modulo_180_grados(self):
        d = descriptores_llegada(np.array([100.0]), np.array([0.0]))       # 100 grados equivale a -80 grados
        self.assertAlmostEqual(d["sigma_red"], np.sqrt(0.5 * 80.0 ** 2), places=9)
        self.assertAlmostEqual(d["polar_mediano"], 80.0, places=9)
        self.assertEqual(d["f_rasante"], 1.0)

    def test_el_umbral_rasante_es_configurable(self):
        tx = np.array([10.0, 30.0, 50.0, 70.0])
        self.assertEqual(descriptores_llegada(tx, np.zeros(4))["f_rasante"], 0.25)
        self.assertEqual(descriptores_llegada(tx, np.zeros(4), umbral_rasante=20.0)["f_rasante"], 0.75)


class TestDispersionPorPunto(unittest.TestCase):
    def test_cada_punto_puede_fijar_su_dispersion_por_colision(self):
        puntos = [dict(i=0, j=0, temp=400.0, sigma=15.0, mfp=10.0, sigma_disp=5.0),
                  dict(i=0, j=1, temp=400.0, sigma=15.0, mfp=10.0, sigma_disp=45.0)]
        tabla, _ = en_silencio(ejecutar_barrido, PARAMS, 3, puntos, 3, 300, "prueba")
        self.assertEqual(set(tabla["sigma_disp"]), {5.0, 45.0})
        medias = tabla.groupby("j")["sigma_eff_med"].mean()
        self.assertGreater(medias[1], medias[0])         # mayor dispersión por colisión, mayor ancho angular

    def test_sin_sigma_disp_se_usa_la_de_los_parametros(self):
        puntos = [dict(i=0, j=0, temp=400.0, sigma=15.0, mfp=10.0)]
        a = en_silencio(ejecutar_barrido, PARAMS, 3, puntos, 3, 300, "prueba")[0]["rugosidad"]
        b = en_silencio(ejecutar_barrido, PARAMS, 3, [dict(puntos[0], sigma_disp=20.0)], 3, 300, "prueba")[0]["rugosidad"]
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())

    def test_la_ley_del_coseno_se_declara_en_el_punto(self):
        puntos = [dict(i=0, j=0, temp=400.0, sigma=0.0, mfp=np.inf, ley_emision=EMISION_COSENO, theta_max=30.0)]
        tabla, _ = en_silencio(ejecutar_barrido, PARAMS, 3, puntos, 3, 300, "prueba")
        self.assertTrue((tabla["sigma_eff_med"] > 5.0).all())
        self.assertTrue((tabla["sigma_eff_med"] < 25.0).all())


class TestMallaDeMuestreo(unittest.TestCase):
    def test_instantes_crecientes_con_el_ultimo_en_el_espesor_maximo(self):
        eventos, marcas = malla_de_muestreo(256, 64.0, 40, espesores_instantaneas=(1.25, 8, 64))
        self.assertEqual(int(eventos[-1]), 64 * 256)
        self.assertTrue(np.all(np.diff(eventos) > 0))
        self.assertEqual(eventos.dtype, np.int64)
        np.testing.assert_array_equal(eventos[marcas], [320, 8 * 256, 64 * 256])

    def test_el_tramo_lineal_cubre_el_final(self):
        eventos, _ = malla_de_muestreo(100, 50.0, 10, desde_lineal=25.0, n_lineales=26)
        self.assertTrue(np.all(np.isin(np.arange(2500, 5001, 100), eventos)))

    def test_rechaza_instantaneas_fuera_de_rango(self):
        with self.assertRaises(ValueError):
            malla_de_muestreo(100, 10.0, 10, espesores_instantaneas=(20.0,))


class TestEjecutarModeloMuestreado(unittest.TestCase):
    def test_reproduce_las_replicas_del_motor_con_las_semillas_del_modelo(self):
        eventos, marcas = malla_de_muestreo(256, 400 / 256, 8, espesores_instantaneas=(1.0,))
        datos = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "M6: prueba", BANDERAS_M6, 3, 7, eventos, marcas)
        directo = ejecutar_suite_muestreada(
            semillas_de_modelo(7, "M6: prueba", 3), 16, 16, int(eventos[-1]), 400.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0,
            True, True, True, True, eventos, marcas, PARAMS.max_pasos_rayo, np.inf)
        np.testing.assert_array_equal(datos["superficies"], directo[0])
        np.testing.assert_allclose(datos["rugosidad"], directo[2], rtol=0, atol=1e-12)
        np.testing.assert_allclose(datos["colisiones_por_atomo"], directo[6], rtol=0, atol=1e-15)
        self.assertEqual(datos["instantaneas"].shape, (3, 1, 16, 16))
        np.testing.assert_array_equal(datos["eventos_instantaneas"], [256])
        np.testing.assert_array_equal(datos["truncados"], 0)

    def test_la_historia_en_los_instantes_comunes_no_depende_de_la_malla_de_registro(self):
        m3 = dict(BANDERAS_M6, enable_angle=False, enable_scattering=False, enable_shadowing=False)
        fina = np.arange(50, 401, 50, dtype=np.int64)
        gruesa = np.array([100, 400], dtype=np.int64)
        a = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "M3: prueba", m3, 2, 1, fina, np.zeros(8, dtype=bool))
        b = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "M3: prueba", m3, 2, 1, gruesa, np.zeros(2, dtype=bool))
        np.testing.assert_allclose(a["rugosidad"][:, [1, 7]], b["rugosidad"], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(a["superficies"], b["superficies"])

    def test_el_corte_por_relieve_detiene_cada_replica_por_separado(self):
        rd = {"enable_angle": False, "enable_scattering": False, "enable_shadowing": False, "enable_diffusion": False, "temp": 0}
        eventos, marcas = malla_de_muestreo(256, 8.0, 10, espesores_instantaneas=(1.0, 8.0))
        sin = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "RD", rd, 4, 3, eventos, marcas)
        con = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "RD", rd, 4, 3, eventos, marcas, relieve_maximo=6.0)
        self.assertFalse(np.any(sin["detenida"]))
        self.assertTrue(np.any(con["detenida"]))
        for r in range(4):
            alcanzados = eventos <= con["depositados"][r]
            np.testing.assert_array_equal(con["rugosidad"][r, alcanzados], sin["rugosidad"][r, alcanzados])
            self.assertTrue(np.all(np.isnan(con["rugosidad"][r, ~alcanzados])))
        np.testing.assert_array_equal(con["instantanea_valida"][:, 1], ~con["detenida"])
        np.testing.assert_allclose(con["espesor_final"], con["depositados"] / 256)
        alcanzados = np.isfinite(con["relieve"])
        self.assertTrue(np.all(con["relieve"][alcanzados] <= 6.0))
        np.testing.assert_allclose(sin["relieve"][:, -1], sin["superficies"].reshape(4, -1).max(axis=1) - 8.0)

    def test_rechaza_instantes_mal_ordenados(self):
        with self.assertRaises(ValueError):
            en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "x", BANDERAS_M6, 2, 1, np.array([10, 5]), np.zeros(2, bool))

    def test_gas_sin_sombreado_y_adhesion_sin_sombreado(self):
        # Sin sombreado el relieve no intercepta el vuelo: el gas solo desplaza el impacto y no hay contacto lateral.
        eventos, marcas = malla_de_muestreo(256, 4.0, 10, espesores_instantaneas=(4.0,))
        banderas = dict(BANDERAS_M6, enable_shadowing=False, enable_lateral=True)
        d = en_silencio(ejecutar_modelo_muestreado, PARAMS_U, "gas sin sombreado", banderas, 3, 1, eventos, marcas)
        self.assertTrue(np.all(d["colisiones_por_atomo"] > 0))
        np.testing.assert_array_equal(d["huecos_finales"], 0)
        np.testing.assert_array_equal(d["superficies"].reshape(3, -1).sum(axis=1), 1024)


if __name__ == "__main__":
    unittest.main()
