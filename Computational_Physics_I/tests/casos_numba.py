"""Casos de cálculo con los que ``test_numba.py`` compara las rutinas compiladas con las no compiladas.

``calcular()`` ejecuta el trazador de rayos (también con el contacto lateral), el lazo de réplica, las réplicas
en paralelo, el motor común con muestreo en instantes arbitrarios (con y sin adhesión lateral), el motor con
registro del volumen y los vuelos libres con semillas fijas y devuelve sus resultados. La prueba lo llama dentro
de su propio proceso (con Numba, compilado) y lo ejecuta como programa en otro proceso con
``NUMBA_DISABLE_JIT=1`` (las mismas funciones, sin compilar):

    NUMBA_DISABLE_JIT=1 python tests/casos_numba.py salida.npz
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from thinfilm_mc import motor


def calcular():
    """Resultados de todos los casos, como diccionario de arreglos de NumPy."""
    r = {}
    rng = np.random.default_rng(1)

    # Trazador de rayos: 30 casos con relieve, emisión, gas y sombreado variados
    filas = []
    for _ in range(30):
        sup = rng.integers(0, 6, size=(24, 24)).astype(np.int32)
        args = (int(rng.integers(0, 24)), int(rng.integers(0, 24)), 24, 24, sup, float(rng.choice([0.0, 15.0, 35.0])), 20.0,
                float(rng.choice([5.0, 15.0, np.inf])), int(sup.max()), 40.0, int(rng.integers(0, 2)), 1_000_000,
                int(rng.integers(0, 2)), 30.0)
        motor.sembrar_generador(int(rng.integers(0, 2 ** 31 - 1)))
        filas.append([float(v) for v in motor.trazar_vuelo(*args)])
    r["trazador"] = np.array(filas)

    # Lazo de réplica en varias configuraciones físicas
    for k, (temp, sigma, mfp, som, ley) in enumerate([(400.0, 15.0, 10.0, 1, 0), (0.0, 35.0, np.inf, 1, 0),
                                                       (250.0, 25.0, 5.0, 0, 0), (400.0, 0.0, 10.0, 1, 1)]):
        motor.sembrar_generador(3 + k)
        sup, n_col, t2, t2r, trunc = motor.ejecutar_replica(20, 20, 400, temp, 500.0, 5, sigma, 20.0, mfp, 40.0, som,
                                                            1_000_000, ley, 30.0)
        r[f"replica_{k}"] = sup
        r[f"replica_{k}_escalares"] = np.array([n_col, t2, t2r, trunc], dtype=float)

    # Réplicas en paralelo (cada una siembra su propio generador)
    semillas = np.array([5, 17, 123456], dtype=np.uint32)
    sup, n_col, t2, t2r, trunc = motor.ejecutar_replicas(semillas, 20, 20, 400, 400.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0,
                                                         1, 1_000_000, 0, 0.0)
    r["replicas"] = sup
    r["replicas_escalares"] = np.vstack([n_col, t2, t2r, trunc]).astype(float)

    # Motor con instantes de registro irregulares, instantáneas, salvaguarda y corte por relieve (en paralelo)
    eventos = np.array([1, 9, 80, 250, 400], dtype=np.int64)
    marcas = np.array([False, True, False, True, False])
    for k, (banderas, max_pasos_rayo, relieve, ley, lateral) in enumerate([
            ((True, True, True, True), 1_000_000, np.inf, 0, False),
            ((True, False, True, True), 85, np.inf, 0, False),
            ((True, False, False, True), 1_000_000, np.inf, 0, False),
            ((False, False, False, False), 1_000_000, 2.0, 0, False),
            ((True, True, True, True), 1_000_000, np.inf, 1, False),
            ((True, False, False, True), 1_000_000, np.inf, 1, False),
            ((True, True, True, True), 1_000_000, np.inf, 0, True),
            ((True, False, True, False), 1_000_000, 6.0, 0, True),
            ((True, True, False, True), 1_000_000, np.inf, 0, True),
            ((False, True, False, False), 1_000_000, np.inf, 0, False)]):
        sup, esp, rug, maxi, hue, inst, col, trunc, dep = motor.ejecutar_suite_muestreada(
            np.array([21, 22], dtype=np.uint32), 20, 20, 400, 400.0, 500.0, 5, 15.0, 20.0, 10.0, 40.0, *banderas,
            eventos, marcas, max_pasos_rayo, relieve, ley, 30.0, lateral
        )
        r[f"muestreado_{k}"] = sup
        r[f"muestreado_{k}_rugosidad"] = np.nan_to_num(rug, nan=-1.0)
        r[f"muestreado_{k}_maximo"] = np.nan_to_num(maxi, nan=-1.0)
        r[f"muestreado_{k}_huecos"] = np.nan_to_num(hue, nan=-1.0)
        r[f"muestreado_{k}_instantaneas"] = inst
        r[f"muestreado_{k}_contadores"] = np.concatenate([col, trunc.astype(float), dep.astype(float)])

    # Motor con registro del volumen y adhesión lateral (una réplica, sembrada desde Python)
    volumen = np.zeros((20, 20, 40), dtype=np.uint8)
    motor.sembrar_generador(23)
    sup, _, _, _, hue, _, _, _, _ = motor.ejecutar_motor_volumen(
        20, 20, 400, 200.0, 500.0, 5, 35.0, 20.0, 10.0, 40.0, True, True, True, True, eventos, marcas, 1_000_000, np.inf,
        0, 0.0, True, volumen
    )
    r["volumen_superficie"] = sup
    r["volumen_ocupacion"] = volumen
    r["volumen_huecos"] = np.nan_to_num(hue, nan=-1.0)

    # Contacto lateral del trazador sobre una superficie rugosa
    sup_rugosa = np.random.default_rng(5).integers(0, 12, size=(24, 24)).astype(np.int32)
    filas = []
    for k in range(40):
        motor.sembrar_generador(100 + k)
        filas.append([float(v) for v in motor.trazar_vuelo(k % 24, (3 * k) % 24, 24, 24, sup_rugosa, 30.0, 20.0, 6.0, 11, 40.0,
                                                           1, 1_000_000, 0, 0.0)])
    r["contacto_lateral"] = np.array(filas)

    # Vuelos libres con emisión gaussiana y con la ley del coseno
    for k, (sigma, mfp, ley) in enumerate([(15.0, 10.0, 0), (0.0, 10.0, 1)]):
        n_col, tx, ty = motor.muestrear_vuelos_libres(300, 20, 20, sigma, 20.0, mfp, 40.0, 9, 1_000_000, ley, 30.0)
        r[f"vuelos_{k}"] = np.vstack([n_col, tx, ty]).astype(float)
    return r


if __name__ == "__main__":
    np.savez(sys.argv[1], **calcular())
