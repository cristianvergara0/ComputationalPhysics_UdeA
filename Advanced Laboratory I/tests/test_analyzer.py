# -*- coding: utf-8 -*-
"""Pruebas de spring_magnet_analyzer.py.

Se ejecutan desde la carpeta del proyecto con:  python -m unittest discover -s tests -v
No necesitan hardware: usan el simulador del propio programa, que conoce los valores verdaderos.
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spring_magnet_analyzer as sma  # noqa: E402

V_SCALE = 3.3 / 4095


def silencio():
    return contextlib.redirect_stdout(io.StringIO())


def escribir_config(carpeta, nombre="config.json", **secciones):
    """Escribe un config.json a partir de la plantilla con los cambios dados por sección."""
    cfg = sma.plantilla_config()
    cfg["fuente"]["puerto"] = "COM10"
    cfg["salida"]["graficas"] = False
    for seccion, cambios in secciones.items():
        cfg[seccion].update(cambios)
    ruta = Path(carpeta) / nombre
    ruta.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def medido_de_prueba(**cambios):
    m = {"f0_Hz": 1.45, "gamma_1_s": 0.022, "gamma_err_1_s": 1e-4, "integral_emf_Vs": 0.0516,
         "sigma_integral_Vs": 1e-5}
    m.update(cambios)
    return m


class TestFormato(unittest.TestCase):
    def test_incertidumbre_en_notacion_fija(self):
        self.assertEqual(sma.formatear(1.2346, 0.0123), "1.235 +/- 0.012")

    def test_incertidumbre_en_notacion_cientifica(self):
        self.assertEqual(sma.formatear(1.51699e-5, 2.7e-7), "(1.517 +/- 0.027)e-05")

    def test_sin_incertidumbre_o_sin_valor(self):
        self.assertEqual(sma.formatear(16.6), "16.6")
        self.assertEqual(sma.formatear(None), "n/d")
        self.assertEqual(sma.formatear(float("nan"), 0.1), "n/d")


class TestUtilidades(unittest.TestCase):
    def test_ruta_relativa(self):
        self.assertEqual(sma.ruta_relativa("/a/b/c.csv", "/a/b"), "c.csv")
        self.assertEqual(sma.ruta_relativa("/a/x.csv", "/a/b/c"), "../../x.csv")

    def test_texto_de_version(self):
        texto = sma.texto_version()
        for esperado in ("spring_magnet_analyzer.py", "numpy", "scipy", "Python"):
            self.assertIn(esperado, texto)


class TestSimulador(unittest.TestCase):
    def test_misma_semilla_mismo_registro(self):
        a = sma.simular_registro(n_muestras=5000, semilla=3)[1]
        b = sma.simular_registro(n_muestras=5000, semilla=3)[1]
        c = sma.simular_registro(n_muestras=5000, semilla=4)[1]
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_cuentas_enteras_y_dentro_del_rango(self):
        _, cuentas, _, _ = sma.simular_registro()
        self.assertTrue(np.all(cuentas == np.rint(cuentas)))
        self.assertGreaterEqual(cuentas.min(), 0)
        self.assertLessEqual(cuentas.max(), sma.ADC_MAX)
        self.assertEqual(np.median(cuentas), 0)   # el ADC recorta las tensiones negativas

    def test_valores_verdaderos(self):
        _, _, _, v = sma.simular_registro(m=0.2, k=16.6, b=0.0088)
        self.assertAlmostEqual(v["gamma_1_s"], 0.022, places=12)
        self.assertAlmostEqual(v["omega0_rad_s"], math.sqrt(16.6 / 0.2), places=12)
        self.assertAlmostEqual(v["omega1_rad_s"], math.sqrt(v["omega0_rad_s"] ** 2 - 0.022 ** 2), places=12)
        self.assertAlmostEqual(v["Q"], v["omega0_rad_s"] / 0.044, places=9)

    def test_sistema_sobreamortiguado_es_un_error(self):
        with self.assertRaises(sma.ErrorAnalisis):
            sma.simular_registro(b=20.0)


class TestAnalisisDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t, cls.cuentas, cls.fs, cls.verdad = sma.simular_registro()
        cls.medido, cls.series = sma.analizar_registro(cls.t, cls.cuentas, cls.fs, V_SCALE)
        cls.montaje = sma.Montaje(vueltas=3400, radio=0.0415, masa=0.200)
        cls.res = sma.resolver_parametros(cls.medido, cls.montaje)

    def test_frecuencia(self):
        self.assertAlmostEqual(self.medido["f0_Hz"], self.verdad["f0_Hz"], delta=0.003 * self.verdad["f0_Hz"])

    def test_gamma_por_area(self):
        self.assertEqual(self.medido["gamma_metodo"], "área de los lóbulos")
        self.assertAlmostEqual(self.medido["gamma_1_s"], self.verdad["gamma_1_s"], delta=0.05 * self.verdad["gamma_1_s"])
        self.assertGreater(self.medido["r2_gamma"], 0.99)

    def test_gamma_por_altura_sobreestima_poco_en_este_regimen(self):
        cociente = self.medido["gamma_altura_1_s"] / self.verdad["gamma_1_s"]
        self.assertTrue(1.0 <= cociente <= 1.10, cociente)

    def test_integral_y_campo(self):
        self.assertAlmostEqual(self.medido["integral_emf_Vs"], self.verdad["integral_emf_Vs"],
                               delta=0.005 * self.verdad["integral_emf_Vs"])
        self.assertAlmostEqual(self.res["valores"]["B_T"], self.verdad["B_T"], delta=0.01 * self.verdad["B_T"])

    def test_parametros_mecanicos_con_la_masa(self):
        v = self.res["valores"]
        self.assertAlmostEqual(v["k_N_m"], self.verdad["k_N_m"], delta=0.005 * self.verdad["k_N_m"])
        self.assertAlmostEqual(v["b_kg_s"], self.verdad["b_kg_s"], delta=0.05 * self.verdad["b_kg_s"])

    def test_sin_advertencias_en_el_caso_limpio(self):
        self.assertEqual(self.medido["advertencias"], [])

    def test_la_comparacion_con_el_simulador_incluye_lo_calculado(self):
        nombres = [n for n, _, _ in sma.comparar_con_simulador(self.medido, self.res, self.verdad)]
        for esperado in ("f0 (Hz)", "gamma (1/s)", "k (N/m)", "B (T)"):
            self.assertIn(esperado, nombres)

    def test_opciones_de_analisis_llegan_al_calculo(self):
        # Con f_min = 2 Hz el pico buscado es un armónico del tren de pulsos
        medido, _ = sma.analizar_registro(self.t, self.cuentas, self.fs, V_SCALE, sma.OpcionesAnalisis(f_min=2.0))
        self.assertAlmostEqual(medido["f0_Hz"], 2 * self.verdad["f0_Hz"], delta=0.02)

    def test_el_subarmonico_se_detecta_en_el_espectro(self):
        t = np.arange(40000) / 1000.0
        puro = np.cos(2 * np.pi * 2.9 * t)
        modulado = puro * (1 + 0.4 * np.cos(2 * np.pi * 1.45 * t))   # Aparecen bandas laterales en 1.45 Hz
        self.assertLess(sma.frecuencia_dominante(puro, 1000.0)["amp_subarmonico"], 0.01)
        self.assertGreaterEqual(sma.frecuencia_dominante(modulado, 1000.0)["amp_subarmonico"], sma.UMBRAL_SUBARMONICO)


class TestLimitaciones(unittest.TestCase):
    """Casos donde el método pierde exactitud y el programa debe decirlo."""

    def analizar(self, **kw):
        t, c, fs, v = sma.simular_registro(**kw)
        medido, _ = sma.analizar_registro(t, c, fs, V_SCALE)
        return medido, v

    def test_amplitud_grande_sesga_gamma_y_avisa(self):
        medido, v = self.analizar(altura=0.060, amplitud=0.040)
        self.assertGreater(medido["gamma_altura_1_s"] / v["gamma_1_s"], 1.25)
        self.assertTrue(any("más rápido que su área" in a for a in medido["advertencias"]), medido["advertencias"])

    def test_zona_muerta_del_adc_subestima_la_integral_y_avisa(self):
        medido, v = self.analizar(zona_muerta_V=0.10)
        self.assertLess(medido["integral_emf_Vs"], 0.5 * v["integral_emf_Vs"])
        self.assertTrue(any("zona muerta" in a for a in medido["advertencias"]), medido["advertencias"])


class TestRobustez(unittest.TestCase):
    def test_registro_que_empieza_a_mitad_de_un_lobulo(self):
        t, c, fs, v = sma.simular_registro(t_inicio=0.2)
        medido, _ = sma.analizar_registro(t, c, fs, V_SCALE)
        self.assertAlmostEqual(medido["integral_emf_Vs"], v["integral_emf_Vs"], delta=0.01 * v["integral_emf_Vs"])

    def test_lecturas_ilegibles_se_interpolan_y_se_avisa(self):
        t, c, fs, v = sma.simular_registro()
        c = c.copy()
        c[np.random.default_rng(3).choice(len(c), 800, replace=False)] = np.nan
        medido, _ = sma.analizar_registro(t, c, fs, V_SCALE)
        self.assertTrue(any("interpolaron" in a for a in medido["advertencias"]))
        self.assertAlmostEqual(medido["gamma_1_s"], v["gamma_1_s"], delta=0.1 * v["gamma_1_s"])

    def test_saturacion_del_adc_se_detecta(self):
        t, c, fs, _ = sma.simular_registro()
        c = c.copy()
        c[1000:1010] = sma.ADC_MAX
        medido, _ = sma.analizar_registro(t, c, fs, V_SCALE)
        self.assertTrue(any("fondo de escala" in a for a in medido["advertencias"]))

    def test_senal_con_offset_se_analiza_igual_y_ve_los_dos_lobulos(self):
        t, c, fs, v = sma.simular_registro(offset_V=1.65, ruido_V=1e-3, n_espurios=0)
        medido, _ = sma.analizar_registro(t, c, fs, V_SCALE)
        self.assertAlmostEqual(medido["nivel_reposo_adc"], 1.65 / V_SCALE, delta=2.0)
        self.assertAlmostEqual(medido["integral_emf_Vs"], v["integral_emf_Vs"], delta=0.02 * v["integral_emf_Vs"])

    def test_registro_sin_oscilacion_es_un_error(self):
        rng = np.random.default_rng(0)
        t = np.arange(20000) / 1000.0
        ruido = np.rint(2048 + rng.normal(0, 3, 20000))
        with self.assertRaises(sma.ErrorAnalisis):
            sma.analizar_registro(t, ruido, 1000.0, V_SCALE)

    def test_registro_nulo_es_un_error(self):
        t = np.arange(20000) / 1000.0
        with self.assertRaises(sma.ErrorAnalisis):
            sma.analizar_registro(t, np.zeros(20000), 1000.0, V_SCALE)

    def test_registro_con_pocos_ciclos_es_un_error(self):
        t, c, fs, _ = sma.simular_registro(n_muestras=2500)
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.analizar_registro(t, c, fs, V_SCALE)
        self.assertIn("ciclos", str(e.exception))

    def test_casi_todo_ilegible_es_un_error(self):
        t = np.arange(1000) / 1000.0
        with self.assertRaises(sma.ErrorAnalisis):
            sma.analizar_registro(t, np.full(1000, np.nan), 1000.0, V_SCALE)


class TestResolver(unittest.TestCase):
    w0 = math.sqrt((2 * math.pi * 1.45) ** 2 + 0.022 ** 2)

    def valores(self, **montaje):
        return sma.resolver_parametros(medido_de_prueba(), sma.Montaje(**montaje))

    def test_sin_datos_solo_salen_las_magnitudes_de_la_medicion(self):
        r = self.valores()
        v = r["valores"]
        self.assertAlmostEqual(v["omega0_rad_s"], self.w0, places=9)
        self.assertAlmostEqual(v["Q"], self.w0 / 0.044, places=6)
        self.assertAlmostEqual(v["tau_amplitud_s"], 1 / 0.022, places=9)
        self.assertNotIn("k_N_m", v)
        self.assertNotIn("B_T", v)
        textos = " ".join(r["faltantes"])
        self.assertIn("k y b", textos)
        self.assertIn("vueltas", textos)

    def test_con_la_masa(self):
        v = self.valores(masa=0.2)["valores"]
        self.assertAlmostEqual(v["k_N_m"], 0.2 * self.w0 ** 2, places=9)
        self.assertAlmostEqual(v["b_kg_s"], 2 * 0.2 * 0.022, places=12)

    def test_con_la_masa_del_resorte_se_usa_un_tercio(self):
        v = self.valores(masa=0.2, masa_resorte=0.03)["valores"]
        self.assertAlmostEqual(v["masa_efectiva_kg"], 0.21, places=12)
        self.assertAlmostEqual(v["k_N_m"], 0.21 * self.w0 ** 2, places=9)

    def test_con_k(self):
        v = self.valores(k=16.6)["valores"]
        self.assertAlmostEqual(v["masa_efectiva_kg"], 16.6 / self.w0 ** 2, places=12)
        self.assertAlmostEqual(v["b_kg_s"], 2 * v["masa_efectiva_kg"] * 0.022, places=12)

    def test_con_b(self):
        v = self.valores(b=0.0088)["valores"]
        self.assertAlmostEqual(v["masa_efectiva_kg"], 0.0088 / 0.044, places=12)
        self.assertAlmostEqual(v["k_N_m"], v["masa_efectiva_kg"] * self.w0 ** 2, places=9)

    def test_dos_datos_consistentes_concuerdan(self):
        r = self.valores(masa=0.2, k=0.2 * self.w0 ** 2)
        self.assertEqual(len(r["chequeos"]), 1)
        self.assertTrue(r["chequeos"][0]["concuerda"])

    def test_dos_datos_inconsistentes_se_senalan(self):
        r = self.valores(masa=0.2, k=30.0)
        self.assertFalse(r["chequeos"][0]["concuerda"])

    def test_tres_datos_dan_tres_comprobaciones(self):
        r = self.valores(masa=0.2, k=16.6, b=0.0088)
        self.assertEqual(len(r["chequeos"]), 3)

    def test_campo_magnetico(self):
        r = self.valores(vueltas=3400, radio=0.0415)
        esperado = 0.0516 / (3400 * math.pi * 0.0415 ** 2)
        self.assertAlmostEqual(r["valores"]["B_T"], esperado, places=12)

    def test_solo_las_vueltas_dan_el_flujo_por_vuelta(self):
        r = self.valores(vueltas=3400)
        self.assertAlmostEqual(r["valores"]["delta_phi_Wb"], 0.0516 / 3400, places=15)
        self.assertNotIn("B_T", r["valores"])
        self.assertTrue(any("radio" in f for f in r["faltantes"]))

    def test_solo_el_radio_pide_las_vueltas(self):
        r = self.valores(radio=0.0415)
        self.assertTrue(any("número de vueltas" in f for f in r["faltantes"]))

    def test_gamma_no_positivo_no_da_b_ni_q(self):
        r = sma.resolver_parametros(medido_de_prueba(gamma_1_s=-0.001), sma.Montaje(masa=0.2))
        self.assertNotIn("Q", r["valores"])
        self.assertNotIn("b_kg_s", r["valores"])
        self.assertIn("k_N_m", r["valores"])


class TestRegistro(unittest.TestCase):
    def test_guardar_y_cargar_conserva_los_datos(self):
        with tempfile.TemporaryDirectory() as d:
            cuentas = np.array([0.0, 1.0, np.nan, 250.0, 3.0])
            ruta_raw, ruta_meta = sma.guardar_registro(d, "20260101_000000", np.arange(1, 6), cuentas, 998.5)
            t, v, fs, idx = sma.cargar_registro(ruta_raw)
            self.assertEqual(fs, 998.5)
            self.assertTrue(np.array_equal(np.isnan(v), np.isnan(cuentas)))
            self.assertTrue(np.array_equal(v[~np.isnan(v)], cuentas[~np.isnan(cuentas)]))
            self.assertAlmostEqual(t[-1], 4 / 998.5, places=12)
            self.assertEqual(list(idx), [1, 2, 3, 4, 5])

    def test_sin_meta_hace_falta_la_frecuencia(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "datos.csv"
            ruta.write_text("index,adc\n1,0\n2,5\n3,9\n", encoding="utf-8")
            with self.assertRaises(sma.ErrorAnalisis) as e:
                sma.cargar_registro(str(ruta))
            self.assertIn("fuente.fs_Hz", str(e.exception))
            t, _, fs, _ = sma.cargar_registro(str(ruta), fs=500.0)
            self.assertEqual(fs, 500.0)
            self.assertAlmostEqual(t[1], 0.002)

    def test_archivo_con_columnas_de_mas_es_un_error(self):
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "medicion_proc.csv"
            ruta.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
            with self.assertRaises(sma.ErrorAnalisis):
                sma.cargar_registro(str(ruta), fs=1000.0)

    def test_archivo_inexistente(self):
        with self.assertRaises(sma.ErrorAnalisis):
            sma.cargar_registro("no_existe.csv", fs=1000.0)

    def test_se_lee_un_csv_con_decimales_en_notacion_cientifica(self):
        # np.savetxt escribe por defecto 18 decimales en notación científica
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / "medicion_raw_20260204_151040.csv"
            ruta.write_text("index,adc\n1.000000000000000000e+00,0.000000000000000000e+00\n"
                            "2.000000000000000000e+00,1.200000000000000000e+01\n", encoding="utf-8")
            (Path(d) / "meta_20260204_151040.json").write_text(json.dumps({"fs_est_Hz": 999.9}), encoding="utf-8")
            _, v, fs, _ = sma.cargar_registro(str(ruta))
            self.assertEqual(fs, 999.9)
            self.assertEqual(list(v), [0.0, 12.0])


class FakeSerial:
    cola = []
    cerrado = False

    def __init__(self, port, baud, timeout=None):
        self.lineas = list(FakeSerial.cola)
        FakeSerial.cerrado = False

    def reset_input_buffer(self):
        pass

    def readline(self):
        return self.lineas.pop(0) if self.lineas else b""

    def close(self):
        FakeSerial.cerrado = True


class TestAdquisicion(unittest.TestCase):
    def setUp(self):
        self.parche = mock.patch.object(sma, "serial", types.SimpleNamespace(Serial=FakeSerial))
        self.parche.start()

    def tearDown(self):
        self.parche.stop()

    def test_captura_con_linea_truncada_corrupta_y_vacia(self):
        lineas = [b"45\r\n"] + [f"{100 + k}\r\n".encode() for k in range(200)]
        lineas[11] = b"ab#c\r\n"
        lineas[21] = b""
        FakeSerial.cola = lineas
        with silencio():
            t, v, fs, idx = sma.acquire_trial(n_samples=200, wait_before=0.0)
        self.assertEqual((len(t), len(v), len(idx)), (200, 200, 200))
        self.assertEqual(v[0], 100.0)   # La primera línea, posiblemente truncada, se descartó
        self.assertEqual(list(np.flatnonzero(np.isnan(v))), [10, 20])
        self.assertAlmostEqual(t[-1] * fs, 199.0, places=6)
        self.assertTrue(FakeSerial.cerrado)

    def test_si_el_esp32_deja_de_transmitir_se_aborta_y_se_cierra_el_puerto(self):
        FakeSerial.cola = [b"1\r\n"] + [b"7\r\n"] * 50
        with silencio(), self.assertRaises(sma.ErrorAnalisis) as e:
            sma.acquire_trial(n_samples=200, wait_before=0.0)
        self.assertIn("dejó de enviar", str(e.exception))
        self.assertTrue(FakeSerial.cerrado)

    def test_sin_pyserial(self):
        with mock.patch.object(sma, "serial", None):
            with self.assertRaises(sma.ErrorAnalisis) as e:
                sma.acquire_trial(n_samples=10, wait_before=0.0)
        self.assertIn("pyserial", str(e.exception))

    def test_puerto_que_no_abre(self):
        def falla(*a, **k):
            raise OSError("acceso denegado")
        with mock.patch.object(sma, "serial", types.SimpleNamespace(Serial=falla)):
            with self.assertRaises(sma.ErrorAnalisis) as e:
                sma.acquire_trial(n_samples=10, wait_before=0.0, puerto="COM99")
        self.assertIn("COM99", str(e.exception))


class TestConfiguracion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cargar(self, **secciones):
        return sma.cargar_configuracion(escribir_config(self.dir, **secciones))

    def assertErrorContiene(self, fragmento, **secciones):
        with self.assertRaises(sma.ErrorAnalisis) as e:
            self.cargar(**secciones)
        self.assertIn(fragmento, str(e.exception))

    def test_la_plantilla_es_valida_y_sus_comentarios_se_ignoran(self):
        ruta = sma.crear_plantilla(self.dir / "plantilla.json")
        cfg = sma.cargar_configuracion(ruta)
        self.assertEqual(cfg.fuente, "demo")
        self.assertEqual(cfg.montaje.vueltas, 3400)
        self.assertAlmostEqual(cfg.montaje.radio, 0.0415)
        self.assertAlmostEqual(cfg.montaje.masa, 0.2)
        self.assertIsNone(cfg.montaje.k)

    def test_la_plantilla_no_sobrescribe(self):
        ruta = sma.crear_plantilla(self.dir / "c.json")
        with self.assertRaises(sma.ErrorAnalisis):
            sma.crear_plantilla(ruta)

    def test_clave_mal_escrita_se_sugiere(self):
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(self._con_clave("montaje", "vuletas", 3400))
        self.assertIn("montaje.vuletas", str(e.exception))
        self.assertIn("vueltas", str(e.exception))

    def _con_clave(self, seccion, clave, valor):
        cfg = sma.plantilla_config()
        cfg[seccion][clave] = valor
        ruta = self.dir / "c.json"
        ruta.write_text(json.dumps(cfg), encoding="utf-8")
        return ruta

    def test_seccion_desconocida(self):
        cfg = sma.plantilla_config()
        cfg["montage"] = {}
        ruta = self.dir / "c.json"
        ruta.write_text(json.dumps(cfg), encoding="utf-8")
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(ruta)
        self.assertIn("montaje", str(e.exception))

    def test_json_con_coma_sobrante(self):
        ruta = self.dir / "c.json"
        ruta.write_text('{\n  "montaje": {\n    "vueltas": 3400,\n  }\n}', encoding="utf-8")
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(ruta)
        mensaje = str(e.exception)
        self.assertIn("línea 4", mensaje)
        self.assertIn("coma", mensaje)

    def test_ruta_de_windows_con_barras_invertidas(self):
        ruta = self.dir / "c.json"
        ruta.write_text('{"fuente": {"tipo": "archivo", "archivo": "C:\\datos\\ensayo.csv"}}', encoding="utf-8")
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(ruta)
        self.assertIn("barras", str(e.exception))

    def test_coma_decimal(self):
        ruta = self.dir / "c.json"
        ruta.write_text('{"montaje": {"radio_mm": 41,5}}', encoding="utf-8")
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(ruta)
        self.assertIn("punto", str(e.exception))

    def test_numero_entre_comillas(self):
        self.assertErrorContiene("montaje.radio_mm", montaje={"radio_mm": "41.5"})

    def test_valor_negativo_o_cero(self):
        self.assertErrorContiene("mayor que 0", montaje={"masa_g": -5})
        self.assertErrorContiene("mayor que 0", montaje={"vueltas": 0})

    def test_booleano_no_vale_como_numero(self):
        self.assertErrorContiene("montaje.k_N_m", montaje={"k_N_m": True})

    def test_tipo_de_fuente_invalido(self):
        self.assertErrorContiene("fuente.tipo", fuente={"tipo": "internet"})

    def test_tipo_archivo_exige_ruta_y_que_exista(self):
        self.assertErrorContiene("fuente.archivo", fuente={"tipo": "archivo", "archivo": None})
        self.assertErrorContiene("no existe el archivo", fuente={"tipo": "archivo", "archivo": "nada.csv"})

    def test_ruta_relativa_se_cuenta_desde_la_carpeta_del_config(self):
        (self.dir / "datos").mkdir()
        (self.dir / "datos" / "r.csv").write_text("index,adc\n1,0\n", encoding="utf-8")
        cfg = self.cargar(fuente={"tipo": "archivo", "archivo": "datos/r.csv"})
        self.assertEqual(cfg.archivo, (self.dir.resolve() / "datos" / "r.csv"))

    def test_calibracion_a_medias(self):
        self.assertErrorContiene("a la vez", adc={"v_cal_V": 0.5, "adc_cal_cuentas": None})

    def test_amplitud_simulada_mayor_que_la_altura(self):
        self.assertErrorContiene("demo.amplitud_mm", demo={"amplitud_mm": 40, "altura_mm": 35})

    def test_prominencia_fuera_de_rango(self):
        self.assertErrorContiene("frac_prominencia", analisis={"frac_prominencia": 1.5})

    def test_masa_del_resorte_sin_masa_avisa(self):
        cfg = self.cargar(montaje={"masa_g": None, "masa_resorte_g": 30})
        self.assertTrue(any("masa_resorte_g" in a for a in cfg.avisos))

    def test_acepta_bom_de_windows(self):
        ruta = self.dir / "c.json"
        ruta.write_text(json.dumps(sma.plantilla_config()), encoding="utf-8-sig")
        self.assertEqual(sma.cargar_configuracion(ruta).fuente, "demo")

    def test_conversion_de_unidades(self):
        cfg = self.cargar(montaje={"radio_mm": 41.5, "masa_g": 250, "masa_resorte_g": 30})
        self.assertAlmostEqual(cfg.montaje.radio, 0.0415)
        self.assertAlmostEqual(cfg.montaje.masa, 0.25)
        self.assertAlmostEqual(cfg.montaje.masa_resorte, 0.03)

    def test_archivo_inexistente_de_configuracion(self):
        with self.assertRaises(sma.ErrorAnalisis) as e:
            sma.cargar_configuracion(self.dir / "no_hay.json")
        self.assertIn("--crear-config", str(e.exception))


class TestEjecucion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def correr(self, argv):
        salida, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(error):
            codigo = sma.main(argv)
        return codigo, salida.getvalue(), error.getvalue()

    def carpeta_de_resultados(self):
        subs = [p for p in (self.dir / "resultados").iterdir() if p.is_dir()]
        self.assertEqual(len(subs), 1)
        return subs[0]

    def test_demo_completa_con_graficas(self):
        ruta = escribir_config(self.dir, salida={"graficas": True})
        os.chdir(tempfile.gettempdir())   # Los resultados deben quedar junto al config, no en el directorio actual
        codigo, salida, error = self.correr(["--config", str(ruta)])
        self.assertEqual(codigo, 0, error)
        carpeta = self.carpeta_de_resultados()
        nombres = sorted(p.name for p in carpeta.iterdir())
        for prefijo in ("01_senal.png", "02_espectro.png", "03_decaimiento.png", "04_lobulo.png", "config_usada.json"):
            self.assertIn(prefijo, nombres)
        for prefijo in ("informe_", "resultados_", "medicion_raw_", "meta_"):
            self.assertTrue(any(n.startswith(prefijo) for n in nombres), prefijo)
        self.assertIn("Comparación con los valores verdaderos del simulador", salida)
        resultados = json.loads(next(carpeta.glob("resultados_*.json")).read_text(encoding="utf-8"))
        self.assertEqual(resultados["fuente"], "simulación")
        self.assertIn("B_T", resultados["deducido"])
        self.assertAlmostEqual(resultados["medido"]["f0_Hz"], 1.45, delta=0.01)

    def test_el_json_de_resultados_no_expone_rutas_absolutas(self):
        self.correr(["--config", str(escribir_config(self.dir))])
        archivo = next(self.carpeta_de_resultados().glob("resultados_*.json"))
        texto = archivo.read_text(encoding="utf-8")
        datos = json.loads(texto)
        self.assertTrue(datos["archivo_raw"].startswith("medicion_raw_"))
        self.assertNotIn(str(self.dir), texto)
        self.assertEqual(sorted(datos["montaje_dado"]),
                         ["b_kg_s", "k_N_m", "masa_kg", "masa_resorte_kg", "radio_m", "vueltas"])
        self.assertAlmostEqual(datos["montaje_dado"]["radio_m"], 0.0415)

    def test_el_informe_cabe_en_una_consola_de_100_columnas(self):
        _, salida, _ = self.correr(["--config", str(escribir_config(self.dir))])
        self.assertLessEqual(max(len(linea) for linea in salida.splitlines()), 100)

    def test_el_informe_con_advertencias_largas_tambien_cabe(self):
        ruta = escribir_config(self.dir, demo={"zona_muerta_mV": 100})
        _, salida, _ = self.correr(["--config", str(ruta)])
        self.assertIn("Advertencias", salida)
        self.assertLessEqual(max(len(linea) for linea in salida.splitlines()), 100)

    def test_el_config_usado_queda_junto_a_los_resultados(self):
        ruta = escribir_config(self.dir, montaje={"masa_g": 250})
        self.correr(["--config", str(ruta)])
        copia = json.loads((self.carpeta_de_resultados() / "config_usada.json").read_text(encoding="utf-8"))
        self.assertEqual(copia["montaje"]["masa_g"], 250)

    def test_sin_argumentos_lee_el_config_junto_al_script(self):
        escribir_config(self.dir)
        with mock.patch.object(sma, "DIRECTORIO_SCRIPT", self.dir):
            codigo, salida, _ = self.correr([])
        self.assertEqual(codigo, 0)
        self.assertIn("RESULTADOS", salida)

    def test_reanalizar_un_registro_guardado_da_lo_mismo(self):
        self.correr(["--config", str(escribir_config(self.dir))])
        raw = next(self.carpeta_de_resultados().glob("medicion_raw_*.csv"))
        primero = json.loads(next(self.carpeta_de_resultados().glob("resultados_*.json")).read_text(encoding="utf-8"))
        relativa = raw.relative_to(self.dir).as_posix()
        ruta2 = escribir_config(self.dir, "otro.json", fuente={"tipo": "archivo", "archivo": relativa},
                                salida={"carpeta": "reanalisis"})
        codigo, _, error = self.correr(["--config", str(ruta2)])
        self.assertEqual(codigo, 0, error)
        segundo = json.loads(next((self.dir / "reanalisis").glob("*/resultados_*.json")).read_text(encoding="utf-8"))
        for clave in ("f0_Hz", "gamma_1_s", "integral_emf_Vs"):
            self.assertAlmostEqual(primero["medido"][clave], segundo["medido"][clave], places=9)

    def test_la_calibracion_cambia_la_escala(self):
        self.correr(["--config", str(escribir_config(self.dir))])
        raw = next(self.carpeta_de_resultados().glob("medicion_raw_*.csv"))
        ruta = escribir_config(self.dir, "cal.json",
                               fuente={"tipo": "archivo", "archivo": str(raw)},
                               adc={"v_cal_V": 0.5, "adc_cal_cuentas": 600.0}, salida={"carpeta": "cal"})
        self.correr(["--config", str(ruta)])
        r = json.loads(next((self.dir / "cal").glob("*/resultados_*.json")).read_text(encoding="utf-8"))
        self.assertAlmostEqual(r["medido"]["V_scale_V_per_count"], 0.5 / 600.0, places=12)
        self.assertTrue(r["calibracion_adc"])

    def test_crear_config(self):
        ruta = self.dir / "nuevo.json"
        codigo, salida, _ = self.correr(["--config", str(ruta), "--crear-config"])
        self.assertEqual(codigo, 0)
        self.assertTrue(ruta.exists())
        codigo, _, error = self.correr(["--config", str(ruta), "--crear-config"])
        self.assertEqual(codigo, 1)
        self.assertIn("ya existe", error)

    def test_config_inexistente_explica_como_crearlo(self):
        codigo, _, error = self.correr(["--config", str(self.dir / "nada.json")])
        self.assertEqual(codigo, 1)
        self.assertIn("--crear-config", error)

    def test_un_error_de_analisis_se_explica_sin_traceback(self):
        raw = self.dir / "medicion_raw_x.csv"
        raw.write_text("index,adc\n" + "\n".join(f"{i},0" for i in range(1, 5001)), encoding="utf-8")
        ruta = escribir_config(self.dir, fuente={"tipo": "archivo", "archivo": "medicion_raw_x.csv", "fs_Hz": 1000})
        codigo, _, error = self.correr(["--config", str(ruta)])
        self.assertEqual(codigo, 1)
        self.assertTrue(error.startswith("Error:"))
        self.assertNotIn("Traceback", error)

    def test_cada_ejecucion_crea_su_carpeta(self):
        ruta = escribir_config(self.dir)
        with mock.patch("time.strftime", side_effect=["20260101_000001", "20260101_000002"]):
            self.correr(["--config", str(ruta)])
            self.correr(["--config", str(ruta)])
        self.assertEqual(sorted(p.name for p in (self.dir / "resultados").iterdir()),
                         ["20260101_000001", "20260101_000002"])


if __name__ == "__main__":
    unittest.main()
