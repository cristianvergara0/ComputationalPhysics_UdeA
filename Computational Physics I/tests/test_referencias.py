"""Compara el paquete con valores de referencia guardados en ``referencias.json``.

Las cifras se calcularon con implementaciones independientes de cada rutina (el trazador de rayos,
el lazo de réplica, el motor común con y sin adhesión lateral y con corte por relieve, las métricas y el
punto de un barrido), sin compilar y con semillas fijas. El trazador independiente recorre el tramo
recto por debajo de la capa de gas enumerando, en orden de descenso, los cruces de bordes de celda en x
y en y. Las referencias detectan cualquier cambio involuntario del comportamiento numérico: los enteros
y las superficies deben coincidir exactamente y los reales con una tolerancia relativa de 1e-9, con o sin
Numba. Las semillas se fijan con ``motor.sembrar_generador``, que siembra el generador que usan las
funciones compiladas.
"""

import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thinfilm_mc import analisis, barridos, motor

REFERENCIAS = json.loads((Path(__file__).with_name("referencias.json")).read_text(encoding="utf-8"))


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def a_float(valor):
    return np.inf if valor == "inf" else float(valor)


def superficie_de_prueba(semilla):
    """Relieve rugoso con columnas aisladas, para que haya contactos laterales."""
    g = np.random.default_rng(semilla)
    sup = g.integers(0, 6, size=(24, 24))
    sup += 8 * (g.random((24, 24)) < 0.15)
    return sup.astype(np.int32)


class TestReferencias(unittest.TestCase):
    def test_trazador_de_rayos(self):
        contactos = 0
        for caso in REFERENCIAS["trazador"]:
            sup = superficie_de_prueba(caso["seed_sup"])
            motor.sembrar_generador(caso["np_seed"])
            salida = motor.trazar_vuelo(caso["xt"], caso["yt"], 24, 24, sup, caso["sigma"], 20.0, a_float(caso["mfp"]),
                                        int(sup.max()), 40.0, caso["som"], 1_000_000, 0, 0.0)
            esperado = caso["out"]
            enteros = [0, 1, 2, 5, 6, 7, 8]
            self.assertEqual([int(salida[i]) for i in enteros], [esperado[i] for i in enteros], caso)
            np.testing.assert_allclose(salida[3:5], esperado[3:5], rtol=1e-9, atol=1e-9)
            contactos += esperado[8] >= 0
        self.assertGreater(contactos, 10)

    def test_lazo_de_replica(self):
        for ref in REFERENCIAS["replica"]:
            motor.sembrar_generador(7)
            sup, n_col, t2, t2r, trunc = motor.ejecutar_replica(
                20, 20, 400, ref["temp"], 500.0, 5, ref["sigma"], 20.0, a_float(ref["mfp"]), 40.0, ref["som"],
                1_000_000, 0, 0.0
            )
            self.assertEqual(int(sup.sum()), ref["suma"])
            self.assertEqual((int(n_col), int(trunc)), (ref["n_col"], ref["trunc"]))
            np.testing.assert_allclose([sup.std(), t2, t2r], [ref["rugosidad"], ref["t2"], ref["t2r"]], rtol=1e-9)
            self.assertEqual(md5(sup), ref["md5"])

    def test_motor_comun(self):
        eventos = np.array([100, 300, 600, 1200], dtype=np.int64)
        marcas = np.array([False, True, False, True])
        for ref in REFERENCIAS["motor"]:
            with self.subTest(flags=ref["flags"], lateral=ref["lateral"], relieve=ref["relieve"]):
                motor.sembrar_generador(11)
                sup, esp, rug, maxi, hue, inst, ncol, trunc, dep = motor.ejecutar_motor_muestreado(
                    20, 20, 1200, ref["temp"], 500.0, 5, 15.0, 20.0, 5.0, 40.0, *ref["flags"], eventos, marcas,
                    1_000_000, a_float(ref["relieve"]), 0, 0.0, ref["lateral"]
                )
                self.assertEqual((int(ncol), int(trunc), int(dep), int(sup.sum())),
                                 (ref["n_col"], 0, ref["depositados"], ref["suma"]))
                self.assertEqual(md5(sup), ref["md5"])
                k = len(ref["rugosidad"])
                for clave, serie in (("espesor", esp), ("rugosidad", rug), ("maximo", maxi), ("huecos", hue)):
                    np.testing.assert_allclose(serie[:k], ref[clave], rtol=1e-9, err_msg=clave)
                    self.assertTrue(np.all(np.isnan(serie[k:])), clave)
                guardadas = [md5(inst[i]) for i in range(len(ref["md5_instantaneas"]))]
                self.assertEqual(guardadas, ref["md5_instantaneas"])

    def test_metricas_y_observables(self):
        for ref in REFERENCIAS["analisis"]:
            sup = np.random.default_rng(ref["seed"]).poisson(8, size=(32, 32)).astype(np.int32)
            espesor, rugosidad = analisis.calcular_metricas(sup, 32, 32)
            self.assertAlmostEqual(espesor, ref["espesor"], places=12)
            self.assertAlmostEqual(rugosidad, ref["rugosidad"], places=12)
            self.assertAlmostEqual(analisis.correlacion_alturas_radial(sup, True)[2], ref["xi"], places=9)
            k, potencia = analisis.espectro_potencia_radial(sup, True)[:2]
            self.assertAlmostEqual(float(k[int(np.argmax(potencia[1:])) + 1]), ref["pico_k"], places=12)
            obs = analisis.calcular_observables(sup)
            for clave, valor in ref["observables"].items():
                self.assertAlmostEqual(float(obs[clave]), valor, places=9, msg=clave)

    def test_punto_de_un_barrido(self):
        ref = REFERENCIAS["punto"]
        params = barridos.ParametrosMotor(nx=16, ny=16, t_escala=500, max_pasos_difusion=5, sigma_dispersion=20.0,
                                          distancia_fuente=40.0, max_pasos_rayo=1_000_000, semilla=42)
        id_exp, i, j, temp, sigma, mfp, som, n_rep, n_ev = ref["args"]
        tabla, primera = barridos.simular_punto(params, id_exp, i, j, temp, sigma, mfp, n_rep, n_ev, som)
        np.testing.assert_allclose(tabla["rugosidad"], ref["rugosidad"], rtol=1e-9)
        np.testing.assert_allclose(tabla["colisiones"], ref["colisiones"], rtol=1e-9)
        self.assertEqual(md5(primera), ref["md5_primera"])


if __name__ == "__main__":
    unittest.main()
