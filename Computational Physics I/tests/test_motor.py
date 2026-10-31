"""Pruebas de los núcleos físicos: relajación, trazado de rayos, ley del coseno, adhesión lateral y lazos de deposición."""

import itertools
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from thinfilm_mc import motor
from thinfilm_mc.barridos import sigma_equivalente_coseno, theta_max_para_sigma


def superficie_aleatoria(semilla, nx=24, ny=24, alto=6):
    return np.random.default_rng(semilla).integers(0, alto, size=(nx, ny)).astype(np.int32)


class TestRelajacion(unittest.TestCase):
    def test_sin_temperatura_la_particula_no_se_mueve(self):
        h = superficie_aleatoria(0)
        motor.sembrar_generador(1)
        self.assertEqual(tuple(motor.difundir_particula_pbc(h, 5, 7, 24, 24, 0.0, 500.0, 5)), (5, 7))

    def test_la_relajacion_solo_desciende_y_devuelve_celdas_validas(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            h = superficie_aleatoria(int(rng.integers(0, 10 ** 6)))
            x, y = int(rng.integers(0, 24)), int(rng.integers(0, 24))
            motor.sembrar_generador(int(rng.integers(0, 2 ** 31 - 1)))
            cx, cy = motor.difundir_particula_pbc(h, x, y, 24, 24, float(rng.choice([100.0, 300.0, 800.0])), 500.0, 5)
            self.assertTrue(0 <= cx < 24 and 0 <= cy < 24)
            self.assertLessEqual(h[cx, cy], h[x, y])

    def test_con_probabilidad_uno_y_pasos_suficientes_termina_en_un_minimo_local(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            h = superficie_aleatoria(int(rng.integers(0, 10 ** 6)), 16, 16)
            motor.sembrar_generador(int(rng.integers(0, 2 ** 31 - 1)))
            cx, cy = motor.difundir_particula_pbc(h, 3, 4, 16, 16, 1000.0, 500.0, 200)
            vecinos = [h[(cx + dx) % 16, (cy + dy) % 16] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            self.assertLessEqual(h[cx, cy], min(vecinos))

    def test_las_fronteras_son_periodicas(self):
        h = np.full((10, 10), 5, dtype=np.int32)
        h[0, 0] = 0
        motor.sembrar_generador(0)
        self.assertEqual(tuple(motor.difundir_particula_pbc(h, 9, 0, 10, 10, 1000.0, 500.0, 5)), (0, 0))
        motor.sembrar_generador(0)
        self.assertEqual(tuple(motor.difundir_particula_pbc(h, 0, 9, 10, 10, 1000.0, 500.0, 5)), (0, 0))


class TestTrazador(unittest.TestCase):
    def trazar(self, h, xt, yt, sigma, sombreado, mfp=np.inf, sigma_disp=20.0, max_pasos=1_000_000, ley=0, theta_max=0.0):
        return motor.trazar_vuelo(xt, yt, 24, 24, h, sigma, sigma_disp, mfp, int(h.max()), 40.0, sombreado, max_pasos,
                                  ley, theta_max)

    def test_un_rayo_vertical_sin_gas_llega_al_punto_de_mira(self):
        rng = np.random.default_rng(4)
        for _ in range(60):
            h = superficie_aleatoria(int(rng.integers(0, 10 ** 6)))
            xt, yt = int(rng.integers(0, 24)), int(rng.integers(0, 24))
            motor.sembrar_generador(int(rng.integers(0, 2 ** 31 - 1)))
            resultado = self.trazar(h, xt, yt, 0.0, 1)
            self.assertEqual(tuple(resultado), (xt, yt, 0, 0.0, 0.0, 0, -1, -1, -1))

    def test_sin_sombreado_ni_gas_el_impacto_es_el_punto_de_la_recta_bajo_la_capa_de_gas(self):
        # Sin sombreado, la partícula llega a la celda en que su recta sale de la capa de gas (z = altura máxima).
        h = superficie_aleatoria(5)
        for semilla in range(40):
            motor.sembrar_generador(semilla)
            cx, cy, n_col, tx, ty, *_, nivel = self.trazar(h, 7, 9, 35.0, 0)
            esperado = [int(np.floor(m + h.max() * np.tan(np.radians(t)) + 0.5)) % 24 for m, t in ((7, tx), (9, ty))]
            self.assertEqual((cx, cy, n_col, nivel), (*esperado, 0, -1))

    def test_por_debajo_de_la_columna_mas_alta_no_hay_colisiones(self):
        # Con gas, el número de colisiones no depende del relieve: un muro alto no cambia los vuelos.
        plana = np.zeros((24, 24), dtype=np.int32)
        muro = plana.copy()
        muro[5, :] = 30
        for semilla in range(30):
            motor.sembrar_generador(semilla)
            a = motor.trazar_vuelo(12, 12, 24, 24, plana, 15.0, 20.0, 6.0, 30, 40.0, 1, 1_000_000, 0, 0.0)
            motor.sembrar_generador(semilla)
            b = motor.trazar_vuelo(12, 12, 24, 24, muro, 15.0, 20.0, 6.0, 30, 40.0, 1, 1_000_000, 0, 0.0)
            self.assertEqual(a[2], b[2])
            np.testing.assert_allclose(a[3:5], b[3:5])

    def test_el_relieve_intercepta_rayos_inclinados_solo_con_sombreado(self):
        h = np.zeros((24, 24), dtype=np.int32)
        h[10, 10] = 12                      # columna alta junto a la celda de mira
        interceptados = {0: 0, 1: 0}
        for sombreado in (0, 1):
            for semilla in range(300):
                motor.sembrar_generador(semilla)
                cx, cy, *_ = self.trazar(h, 11, 10, 30.0, sombreado)
                interceptados[sombreado] += int((cx, cy) == (10, 10))
        self.assertEqual(interceptados[0], 0)
        self.assertGreater(interceptados[1], 20)

    def test_geometria_exacta_frente_a_un_muro(self):
        # Muro de H capas en la fila x = 30 de una superficie plana y rayos dirigidos a celdas tras él. Con la
        # dirección de emisión se predice, para cada rayo, si alcanza la pared (y a qué nivel), si llega a la cima
        # del muro o si llega a su celda de mira.
        H, L = 20, 200
        superficie = np.zeros((L, L), dtype=np.int32)
        superficie[30, :] = H
        sorteo = np.random.default_rng(1)
        motor.sembrar_generador(1)
        casos = {"pared": 0, "cima": 0, "mira": 0}
        for _ in range(1500):
            xt, yt = int(sorteo.integers(31, 41)), int(sorteo.integers(0, L))
            cx, cy, n_col, tx, ty, truncado, ax, ay, nivel = motor.trazar_vuelo(
                xt, yt, L, L, superficie, 25.0, 0.0, np.inf, H, 40.0, 1, 1_000_000, 0, 0.0)
            a = -np.tan(np.radians(tx))
            caso = "mira"
            if a > 0 and (xt - 29.5) / a < H:
                caso = "pared"
            elif a > 0 and (xt - 30.5) / a <= H:
                caso = "cima"
            casos[caso] += 1
            if caso == "pared":
                self.assertEqual((cx, ax, ay, nivel), (30, 29, cy, int(np.floor((xt - 29.5) / a))))
            elif caso == "cima":
                self.assertEqual((cx, nivel), (30, -1))
            else:
                self.assertEqual((cx, cy, nivel), (xt, yt, -1))
        self.assertTrue(all(n > 20 for n in casos.values()), casos)

    def test_el_contacto_lateral_es_coherente_con_el_relieve(self):
        # Cuando hay contacto lateral, la columna de llegada es vecina de von Neumann de la columna de contacto,
        # el nivel es una posición libre de esta y una ocupada de aquella.
        rng = np.random.default_rng(9)
        contactos = 0
        for k in range(400):
            h = superficie_aleatoria(int(rng.integers(0, 10 ** 6)), alto=14)
            motor.sembrar_generador(k)
            cx, cy, _, _, _, _, ax, ay, nivel = self.trazar(h, int(rng.integers(0, 24)), int(rng.integers(0, 24)), 30.0, 1,
                                                            mfp=float(rng.choice([np.inf, 6.0])))
            if nivel < 0:
                self.assertEqual((ax, ay), (-1, -1))
                continue
            contactos += 1
            distancia = (min((ax - cx) % 24, (cx - ax) % 24), min((ay - cy) % 24, (cy - ay) % 24))
            self.assertIn(distancia, ((1, 0), (0, 1)))
            self.assertTrue(h[ax, ay] <= nivel < h[cx, cy])
        self.assertGreater(contactos, 50)

    def test_la_salvaguarda_trunca_los_vuelos_demasiado_largos(self):
        h = superficie_aleatoria(6)
        truncados = 0
        for semilla in range(60):
            motor.sembrar_generador(semilla)
            truncados += self.trazar(h, 3, 3, 60.0, 1, max_pasos=2)[5]
        self.assertGreater(truncados, 20)
        motor.sembrar_generador(0)
        self.assertEqual(self.trazar(h, 3, 3, 0.0, 1, max_pasos=2)[5], 0)   # un vuelo vertical no cruza celdas


class TestLeyDelCoseno(unittest.TestCase):
    THETA_MAX = 30.0

    @classmethod
    def setUpClass(cls):
        n = 5000
        cls.n_col, cls.tx, cls.ty = motor.muestrear_vuelos_libres(
            n, 20, 20, 0.0, 20.0, np.inf, 10.0, 321, 1_000_000, motor.EMISION_COSENO, cls.THETA_MAX
        )

    def test_el_angulo_polar_esta_acotado_y_sigue_la_ley_del_coseno(self):
        tan_theta = np.hypot(np.tan(np.radians(self.tx)), np.tan(np.radians(self.ty)))
        theta = np.arctan(tan_theta)
        self.assertLessEqual(np.degrees(theta.max()), self.THETA_MAX + 1e-9)
        # sen^2(theta) / sen^2(theta_max) es uniforme en (0, 1)
        u = np.sin(theta) ** 2 / np.sin(np.radians(self.THETA_MAX)) ** 2
        self.assertGreater(stats.kstest(u, "uniform").pvalue, 0.001)

    def test_el_acimut_es_uniforme(self):
        acimut = np.arctan2(np.tan(np.radians(self.ty)), np.tan(np.radians(self.tx)))
        self.assertGreater(stats.kstest(acimut, "uniform", args=(-np.pi, 2 * np.pi)).pvalue, 0.001)

    def test_sin_gas_no_hay_colisiones(self):
        self.assertTrue(np.all(self.n_col == 0))

    def test_la_dispersion_por_eje_coincide_con_la_formula(self):
        medida = np.sqrt(0.5 * np.mean(self.tx ** 2 + self.ty ** 2))
        self.assertAlmostEqual(medida / sigma_equivalente_coseno(self.THETA_MAX), 1.0, delta=0.03)

    def test_la_dispersion_equivalente_tiende_a_la_mitad_del_angulo_maximo_para_angulos_pequenos(self):
        self.assertAlmostEqual(sigma_equivalente_coseno(2.0), 1.0, places=3)
        self.assertGreater(sigma_equivalente_coseno(60.0), sigma_equivalente_coseno(30.0))

    def test_la_inversa_recupera_el_angulo_maximo(self):
        for theta_max in (5.0, 30.0, 60.0):
            self.assertAlmostEqual(theta_max_para_sigma(sigma_equivalente_coseno(theta_max)), theta_max, places=6)
        with self.assertRaises(ValueError):
            sigma_equivalente_coseno(95.0)
        with self.assertRaises(ValueError):
            theta_max_para_sigma(80.0)


class TestIdentidadAngular(unittest.TestCase):
    def test_identidad_de_wald_en_vuelo_libre(self):
        # E[(theta_x^2 + theta_y^2) / 2 al llegar] = sigma_ang^2 + E[n_col] sigma_disp^2, para cualquier
        # camino libre medio: la diferencia por átomo tiene esperanza cero.
        sigma_ang, sigma_disp = 15.0, 20.0
        n_col, tx, ty = motor.muestrear_vuelos_libres(
            4000, 20, 20, sigma_ang, sigma_disp, 10.0, 10.0, 99, 10 ** 9, motor.EMISION_GAUSSIANA, 0.0
        )
        d = 0.5 * (tx ** 2 + ty ** 2) - sigma_ang ** 2 - n_col * sigma_disp ** 2
        z = d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
        self.assertLess(abs(z), 4.0)
        self.assertGreater(n_col.mean(), 0.5)


class TestLazoDeReplica(unittest.TestCase):
    CONFIGURACIONES = [
        dict(temp=400.0, sigma_ang=15.0, mfp=10.0, sombreado=1),
        dict(temp=400.0, sigma_ang=15.0, mfp=np.inf, sombreado=1),
        dict(temp=0.0, sigma_ang=35.0, mfp=5.0, sombreado=0),
        dict(temp=250.0, sigma_ang=0.0, mfp=np.inf, sombreado=1),
    ]

    def ejecutar(self, semilla, n_eventos=400, nx=20, ny=20, ley=0, theta_max=0.0, **cfg):
        motor.sembrar_generador(semilla)
        return motor.ejecutar_replica(nx, ny, n_eventos, cfg["temp"], 500.0, 5, cfg["sigma_ang"], 20.0, cfg["mfp"],
                                      40.0, cfg["sombreado"], 1_000_000, ley, theta_max)

    def test_se_conserva_el_numero_de_particulas(self):
        for cfg in self.CONFIGURACIONES:
            superficie, n_col, suma_t2, suma_t2_red, truncados = self.ejecutar(7, **cfg)
            self.assertEqual(int(superficie.sum()), 400, cfg)
            self.assertEqual(superficie.shape, (20, 20))
            self.assertGreaterEqual(superficie.min(), 0)
            if not np.isfinite(cfg["mfp"]):
                self.assertEqual(n_col, 0)

    def test_la_misma_semilla_da_la_misma_superficie_y_otra_semilla_da_otra(self):
        cfg = self.CONFIGURACIONES[0]
        a, b, c = self.ejecutar(11, **cfg)[0], self.ejecutar(11, **cfg)[0], self.ejecutar(12, **cfg)[0]
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_el_limite_de_deposicion_aleatoria(self):
        # Sin relajación, emisión vertical y sin gas, el modelo es una deposición aleatoria:
        # W^2 es, en promedio, el espesor medio en capas (varianza de una Poisson).
        nx = ny = 24
        n_eventos = 1152                      # 2 capas
        valores = []
        for semilla in range(5):
            motor.sembrar_generador(100 + semilla)
            superficie = motor.ejecutar_replica(nx, ny, n_eventos, 0.0, 500.0, 5, 0.0, 20.0, np.inf, 40.0, 1, 1_000_000, 0, 0.0)[0]
            valores.append(superficie.std())
        esperado = np.sqrt(n_eventos / (nx * ny))
        error = np.std(valores, ddof=1) / np.sqrt(len(valores))
        self.assertLess(abs(np.mean(valores) - esperado), max(4 * error, 0.03))

    def test_la_relajacion_reduce_la_rugosidad(self):
        sin = np.mean([self.ejecutar(s, n_eventos=500, temp=0.0, sigma_ang=0.0, mfp=np.inf, sombreado=1)[0].std() for s in range(3)])
        con = np.mean([self.ejecutar(s, n_eventos=500, temp=500.0, sigma_ang=0.0, mfp=np.inf, sombreado=1)[0].std() for s in range(3)])
        self.assertLess(con, 0.9 * sin)

    def test_el_sombreado_aumenta_la_rugosidad_con_incidencia_oblicua(self):
        sin = np.mean([self.ejecutar(s, n_eventos=500, temp=0.0, sigma_ang=35.0, mfp=np.inf, sombreado=0)[0].std() for s in range(3)])
        con = np.mean([self.ejecutar(s, n_eventos=500, temp=0.0, sigma_ang=35.0, mfp=np.inf, sombreado=1)[0].std() for s in range(3)])
        self.assertGreater(con, 1.1 * sin)

    def test_la_ley_del_coseno_acotada_funciona_en_el_lazo(self):
        cfg = dict(temp=400.0, sigma_ang=0.0, mfp=10.0, sombreado=1)
        superficie, n_col, suma_t2, _, truncados = self.ejecutar(5, ley=motor.EMISION_COSENO, theta_max=30.0, **cfg)
        self.assertEqual(int(superficie.sum()), 400)
        self.assertGreater(n_col, 0)
        self.assertGreater(suma_t2 / 400, sigma_equivalente_coseno(30.0) ** 2 * 0.5)


class TestMotorComun(unittest.TestCase):
    ARGS = (20, 20, 400, 400.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0)

    def test_conservacion_e_instantaneas_en_todas_las_combinaciones_de_banderas(self):
        eventos = np.array([100, 200, 300, 400], dtype=np.int64)
        marcas = np.array([False, True, False, True])
        for angulo, gas, sombra, difusion in itertools.product([False, True], repeat=4):
            banderas = (angulo, gas, sombra, difusion)
            with self.subTest(banderas=banderas):
                motor.sembrar_generador(3)
                sup, esp, rug, maxi, hue, inst, col, trunc, dep = motor.ejecutar_motor_muestreado(
                    *self.ARGS, *banderas, eventos, marcas, 1_000_000, np.inf)
                self.assertEqual((int(sup.sum()), dep, trunc), (400, 400, 0))
                np.testing.assert_allclose(esp, eventos / 400, rtol=0, atol=1e-12)
                self.assertAlmostEqual(rug[-1], sup.std(), places=12)
                self.assertEqual(inst.shape, (2, 20, 20))
                np.testing.assert_array_equal(inst.reshape(2, -1).sum(axis=1), [200, 400])
                np.testing.assert_array_equal(inst[1], sup)
                self.assertEqual(maxi[-1], sup.max())
                np.testing.assert_array_equal(hue, 0)
                if not gas:
                    self.assertEqual(col, 0)
                else:
                    self.assertGreater(col, 0)

    def test_sin_sombreado_el_gas_no_cambia_la_estadistica_de_la_deposicion_aleatoria(self):
        # Sin sombreado ni relajación, el gas solo desplaza el impacto: la película es una deposición aleatoria.
        eventos = np.array([1152], dtype=np.int64)
        valores = []
        for semilla in range(6):
            motor.sembrar_generador(200 + semilla)
            sup = motor.ejecutar_motor_muestreado(24, 24, 1152, 0.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0, True, True, False,
                                                  False, eventos, np.zeros(1, dtype=np.bool_), 1_000_000, np.inf)[0]
            valores.append(sup.var())
        self.assertLess(abs(np.mean(valores) / 2.0 - 1.0), 4 * np.std(valores, ddof=1) / np.sqrt(6) / 2.0 + 0.02)

    def test_los_instantes_pueden_ser_irregulares(self):
        eventos = np.array([1, 7, 64, 333, 400], dtype=np.int64)
        motor.sembrar_generador(5)
        sup, esp, rug, maxi, _, inst, _, _, _ = motor.ejecutar_motor_muestreado(
            *self.ARGS, True, True, True, True, eventos, np.array([True, False, False, True, False]), 1_000_000, np.inf)
        np.testing.assert_allclose(esp, eventos / 400)
        self.assertAlmostEqual(rug[-1], sup.std(), places=12)
        self.assertEqual(int(inst[0].sum()), 1)
        self.assertEqual(int(inst[1].sum()), 333)
        self.assertEqual(maxi[3], inst[1].max())
        self.assertTrue(np.all(np.diff(maxi) >= 0))

    def test_la_realizacion_no_depende_de_los_instantes_registrados(self):
        motor.sembrar_generador(7)
        a = motor.ejecutar_motor_muestreado(*self.ARGS, True, True, True, True, np.array([400], dtype=np.int64),
                                            np.zeros(1, dtype=np.bool_), 1_000_000, np.inf)[0]
        motor.sembrar_generador(7)
        b = motor.ejecutar_motor_muestreado(*self.ARGS, True, True, True, True, np.arange(10, 401, 10, dtype=np.int64),
                                            np.zeros(40, dtype=np.bool_), 1_000_000, np.inf)[0]
        np.testing.assert_array_equal(a, b)

    def test_la_salvaguarda_cuenta_los_vuelos_truncados(self):
        eventos = np.array([400], dtype=np.int64)
        motor.sembrar_generador(4)
        sup, _, _, _, _, _, _, trunc, _ = motor.ejecutar_motor_muestreado(
            *self.ARGS, True, True, True, True, eventos, np.array([False]), 2, np.inf)
        self.assertGreater(trunc, 0)
        self.assertEqual(int(sup.sum()), 400)

    def test_el_corte_por_relieve_detiene_la_simulacion(self):
        # Sin relajación ni sombreado (deposición aleatoria), con un corte de 2 capas sobre la media.
        eventos = np.array([50, 100, 200, 400, 800], dtype=np.int64)
        marcas = np.array([True, False, False, True, True])
        args = (20, 20, 800, 0.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0)
        motor.sembrar_generador(6)
        sup, esp, rug, maxi, _, inst, _, _, dep = motor.ejecutar_motor_muestreado(
            *args, False, False, False, False, eventos, marcas, 1_000_000, 2.0)
        self.assertLess(dep, 800)
        self.assertEqual(int(sup.sum()), dep)
        self.assertGreater(sup.max() - dep / 400, 2.0)
        alcanzados = eventos <= dep
        self.assertTrue(np.all(np.isfinite(rug[alcanzados])))
        self.assertTrue(np.all(np.isnan(rug[~alcanzados])))
        self.assertTrue(np.all(np.isnan(esp[~alcanzados])))
        self.assertTrue(np.all(np.isnan(maxi[~alcanzados])))
        self.assertTrue(np.all(maxi[alcanzados] - esp[alcanzados] <= 2.0))
        # Hasta el corte, la realización es la misma que sin corte
        motor.sembrar_generador(6)
        _, _, rug_sin, _, _, inst_sin, _, _, dep_sin = motor.ejecutar_motor_muestreado(
            *args, False, False, False, False, eventos, marcas, 1_000_000, np.inf)
        self.assertEqual(dep_sin, 800)
        np.testing.assert_array_equal(rug[alcanzados], rug_sin[alcanzados])
        np.testing.assert_array_equal(inst[0], inst_sin[0])

    def test_la_suite_reproduce_las_replicas_sembradas_una_a_una(self):
        semillas = np.array([1, 2, 3], dtype=np.uint32)
        eventos = np.array([100, 200, 300, 400], dtype=np.int64)
        marcas = np.zeros(4, dtype=np.bool_)
        sup, esp, rug, maxi, hue, inst, col, trunc, dep = motor.ejecutar_suite_muestreada(
            semillas, *self.ARGS, True, True, True, True, eventos, marcas, 1_000_000, np.inf)
        for r, semilla in enumerate(semillas):
            motor.sembrar_generador(int(semilla))
            una = motor.ejecutar_motor_muestreado(*self.ARGS, True, True, True, True, eventos, marcas, 1_000_000, np.inf)
            np.testing.assert_array_equal(sup[r], una[0])
            np.testing.assert_allclose(rug[r], una[2], rtol=0, atol=1e-12)
            self.assertAlmostEqual(col[r], una[6] / 400, places=12)
        self.assertEqual(inst.shape, (3, 0, 20, 20))
        self.assertFalse(np.array_equal(sup[0], sup[1]))
        np.testing.assert_array_equal(dep, 400)

    def test_la_relajacion_del_motor_comun_respeta_la_bandera(self):
        h = superficie_aleatoria(8)
        motor.sembrar_generador(0)
        self.assertEqual(tuple(motor.modulo_relajacion_superficial(h, 4, 5, 24, 24, 800.0, 500.0, 5, False)), (4, 5))


class TestMotoresEquivalentes(unittest.TestCase):
    """Con sombreado e incidencia angular, el motor de réplicas de los barridos y el motor común consumen los
    mismos números aleatorios: con la misma semilla dan las mismas superficies."""

    def comparar(self, gas, ley, theta_max, sigma):
        semillas = np.array([3, 9], dtype=np.uint32)
        mfp = 10.0 if gas else np.inf
        barrido = motor.ejecutar_replicas(semillas, 20, 20, 400, 400.0, 500.0, 5, sigma, 20.0, mfp, 40.0, 1,
                                          1_000_000, ley, theta_max)
        comun = motor.ejecutar_suite_muestreada(semillas, 20, 20, 400, 400.0, 500.0, 5, sigma, 20.0, 10.0, 40.0, True,
                                                gas, True, True, np.array([200, 400], dtype=np.int64),
                                                np.zeros(2, dtype=np.bool_), 1_000_000, np.inf, ley, theta_max)
        np.testing.assert_array_equal(barrido[0], comun[0])
        np.testing.assert_allclose(barrido[1], comun[6], rtol=0, atol=1e-15)

    def test_emision_gaussiana_sin_y_con_gas(self):
        self.comparar(False, motor.EMISION_GAUSSIANA, 0.0, 15.0)
        self.comparar(True, motor.EMISION_GAUSSIANA, 0.0, 15.0)

    def test_ley_del_coseno(self):
        self.comparar(False, motor.EMISION_COSENO, 30.0, 0.0)
        self.comparar(True, motor.EMISION_COSENO, 30.0, 0.0)

    def test_ley_del_coseno_sin_sombreado_desplaza_el_impacto(self):
        motor.sembrar_generador(2)
        superficie = np.zeros((30, 30), dtype=np.int32)
        desplazamientos = []
        for _ in range(2000):
            cx, cy, n_col, trunc, ax, ay, nivel = motor.transporte_con_salvaguarda(
                15, 15, 30, 30, superficie, 0, 0.0, 20.0, 10.0, 40.0, True, False, False, 1_000_000,
                motor.EMISION_COSENO, 20.0)
            desplazamientos.append(((cx - 15 + 15) % 30 - 15, (cy - 15 + 15) % 30 - 15))
            self.assertEqual((n_col, trunc, ax, ay, nivel), (0, 0, -1, -1, -1))
        d = np.array(desplazamientos, dtype=float)
        # El desplazamiento máximo es 40 tan(20 grados) = 14.6 celdas por eje
        self.assertLessEqual(np.abs(d).max(), 15)
        self.assertGreater(d.std(), 2.0)



class TestAdhesionLateral(unittest.TestCase):
    """Contacto lateral del trazador y adhesión lateral del motor común."""

    ARGS = (20, 20, 1200, 400.0, 500.0, 5, 35.0, 20.0, 10.0, 40.0)

    def ejecutar(self, lateral, banderas=(True, True, True, True), relieve=np.inf, volumen=None):
        eventos = np.array([300, 600, 1200], dtype=np.int64)
        marcas = np.array([False, True, True])
        motor.sembrar_generador(12)
        if volumen is None:
            return motor.ejecutar_motor_muestreado(*self.ARGS, *banderas, eventos, marcas, 1_000_000, relieve, 0, 0.0, lateral)
        return motor.ejecutar_motor_volumen(*self.ARGS, *banderas, eventos, marcas, 1_000_000, relieve, 0, 0.0, lateral, volumen)

    def test_sin_adhesion_lateral_no_hay_huecos(self):
        sup, _, _, _, hue, _, _, _, dep = self.ejecutar(False)
        np.testing.assert_array_equal(hue, 0)
        self.assertEqual(int(sup.sum()), dep)

    def test_la_adhesion_lateral_crea_huecos_y_conserva_las_particulas(self):
        sup, esp, rug, maxi, hue, inst, _, _, dep = self.ejecutar(True)
        self.assertEqual(dep, 1200)
        self.assertGreater(hue[-1], 0)
        self.assertTrue(np.all(np.diff(hue) >= 0))
        self.assertEqual(int(sup.sum()), 1200 + int(hue[-1]))
        np.testing.assert_array_equal(inst.reshape(2, -1).sum(axis=1), [600 + hue[1], 1200 + hue[2]])
        np.testing.assert_allclose(esp, (np.array([300, 600, 1200]) + hue) / 400)

    def test_con_incidencia_vertical_no_hay_contacto_lateral(self):
        # Sin incidencia angular ni gas, los rayos son verticales: la adhesión lateral no cambia nada.
        sin = self.ejecutar(False, banderas=(False, False, True, True))
        con = self.ejecutar(True, banderas=(False, False, True, True))
        np.testing.assert_array_equal(sin[0], con[0])
        np.testing.assert_array_equal(con[4], 0)

    def test_el_volumen_registra_cada_particula_y_los_huecos(self):
        volumen = np.zeros((20, 20, 64), dtype=np.uint8)
        sup, _, _, _, hue, _, _, _, dep = self.ejecutar(True, volumen=volumen)
        sup_sin_volumen = self.ejecutar(True)[0]
        np.testing.assert_array_equal(sup, sup_sin_volumen)
        self.assertEqual(int(volumen.sum()), dep)
        niveles = np.arange(64)[None, None, :]
        bajo_la_cima = niveles < sup[:, :, None]
        self.assertEqual(int(volumen[~bajo_la_cima].sum()), 0)
        self.assertEqual(int((bajo_la_cima & (volumen == 0)).sum()), int(hue[-1]))
        # La posición más alta de cada columna está ocupada
        i, j = np.nonzero(sup > 0)
        np.testing.assert_array_equal(volumen[i, j, sup[i, j] - 1], 1)

    def test_el_corte_por_relieve_usa_la_altura_media_de_las_columnas(self):
        sup, esp, _, maxi, hue, _, _, _, dep = self.ejecutar(True, relieve=3.0)
        alcanzados = np.isfinite(esp)
        self.assertTrue(np.all(maxi[alcanzados] - esp[alcanzados] <= 3.0))
        if dep < 1200:
            self.assertGreater(sup.max() - sup.mean(), 3.0)


if __name__ == "__main__":
    unittest.main()
