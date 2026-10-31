"""
Núcleos numéricos: algoritmo de Metropolis y muestreo de observables.

Todas las funciones de este módulo se compilan con Numba cuando está disponible
(``config.njit``); sin Numba se ejecutan como Python puro con el mismo resultado.

Generador aleatorio
-------------------
Dentro del código compilado, ``np.random`` usa un generador propio de Numba, uno
por hilo y por proceso, independiente del generador global de NumPy. Por eso cada
réplica llama a ``seed_rng(seed)`` antes de su primer paso: así la secuencia de
sitios propuestos y de decisiones de aceptación queda fijada por la semilla, sin
importar en qué proceso o hilo se ejecute la réplica.

Observables por medición
------------------------
``sample_temperature`` acumula, para cada bloque de mediciones consecutivas, el
promedio de las 16 magnitudes de ``OBS_NAMES``. Guardar promedios por bloque (y
no solo por corrida) permite calcular después tanto el estimador sobre la serie
completa como errores jackknife por bloques, sin almacenar las series.
"""

import numpy as np

from .config import njit

OBS_NAMES = (
    "e", "e2", "e4",            # e = E/N (E incluye el término -h M)
    "mu", "mu2", "emu", "amu",  # mu = M/N, magnetización uniforme por espín
    "ms", "ms2", "ms4",         # ms = |S_A - S_B| / (2N), magnetización de subred (checkerboard)
    "m", "m2", "m4",            # m = sqrt(mx^2 + my^2), parámetro de orden stripe
    "skl", "skt",               # |psi(Q + k)|^2 / N^2 con k paralelo y perpendicular a Q (factor de estructura)
    "mx2",                      # mx^2: componente del orden stripe con franjas a lo largo de x
)
OBS = {name: i for i, name in enumerate(OBS_NAMES)}
NOBS = len(OBS_NAMES)


@njit
def seed_rng(seed):
    """Siembra el generador de ``np.random`` usado dentro de los núcleos (el de Numba si está activo)."""
    np.random.seed(seed)


@njit
def compute_energy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn):
    """
    Energía total E = -J1 sum_NN S_i S_j - J2 sum_NNN S_i S_j - h sum S_i y
    magnetización total M. Cada enlace aparece dos veces al recorrer todos los
    sitios, de ahí el factor 1/2 de los términos de interacción.
    """
    N = spins.size
    s_nn = 0.0
    s_nnn = 0.0
    mag = 0.0
    for i in range(N):
        si = spins[i]
        for k in range(n_nn[i]):
            s_nn += si * spins[nn[i, k]]
        for k in range(n_nnn[i]):
            s_nnn += si * spins[nnn[i, k]]
        mag += si
    return -0.5 * J1 * s_nn - 0.5 * J2 * s_nnn - h * mag, mag


@njit
def order_sums(spins, L):
    """
    Sumas que definen los parámetros de orden, calculadas desde cero:
    M = sum S, S_A - S_B (subredes por paridad de x + y), SX = sum (-1)^x S y
    SY = sum (-1)^y S. Durante el muestreo se actualizan de forma incremental.
    """
    M = 0.0
    SAB = 0.0
    SX = 0.0
    SY = 0.0
    for x in range(L):
        for y in range(L):
            s = spins[x * L + y]
            M += s
            SAB += s if (x + y) % 2 == 0 else -s
            SX += s if x % 2 == 0 else -s
            SY += s if y % 2 == 0 else -s
    return M, SAB, SX, SY


@njit
def build_accept_table(J1, J2, h, beta):
    """
    Probabilidades de aceptación min(1, exp(-beta dE)) para todos los cambios de
    energía posibles, indexadas por (signo de S_i, suma NN + 4, suma NNN + 4).

    Con dE = 2 S_i (J1 sum_NN S_j + J2 sum_NNN S_j + h), las sumas de vecinos van
    de -4 a 4 tanto en PBC como en FBC, de modo que 2 x 9 x 9 entradas cubren
    todos los casos. Se calcula una vez por temperatura y evita evaluar exp() en
    cada intento.
    """
    table = np.empty((2, 9, 9))
    for a in range(2):
        si = 1.0 if a == 1 else -1.0
        for b in range(9):
            for c in range(9):
                dE = 2.0 * si * (J1 * (b - 4) + J2 * (c - 4) + h)
                table[a, b, c] = 1.0 if dE <= 0.0 else np.exp(-beta * dE)
    return table


@njit
def metropolis_mcs(spins, L, table, J1, J2, h, nn, n_nn, nnn, n_nnn, E, M, SAB, SX, SY):
    """
    Un paso Monte Carlo (MCS): N intentos de volteo en sitios elegidos al azar.

    La configuración se modifica en el lugar. E, M y las sumas de los parámetros
    de orden se actualizan de forma incremental con el cambio producido por cada
    volteo aceptado, de modo que medirlos no exige recorrer la red.

    Returns
    -------
    E, M, SAB, SX, SY : float
        Valores actualizados.
    accepted : int
        Número de volteos aceptados en este MCS.
    """
    N = spins.size
    accepted = 0
    for _ in range(N):
        i = np.random.randint(0, N)
        si = spins[i]
        snn = 0
        for k in range(n_nn[i]):
            snn += spins[nn[i, k]]
        snnn = 0
        for k in range(n_nnn[i]):
            snnn += spins[nnn[i, k]]
        p = table[1 if si > 0 else 0, snn + 4, snnn + 4]
        if p >= 1.0 or np.random.random() < p:
            spins[i] = -si
            E += 2.0 * si * (J1 * snn + J2 * snnn + h)
            d = -2.0 * si
            M += d
            x = i // L
            y = i - x * L
            SAB += d if (x + y) % 2 == 0 else -d
            SX += d if x % 2 == 0 else -d
            SY += d if y % 2 == 0 else -d
            accepted += 1
    return E, M, SAB, SX, SY, accepted


@njit
def stripe_fourier(spins, L, cos_tab, sin_tab):
    """
    |psi(Q + k)|^2 / N^2 del orden stripe, sumando los dos canales Q_x = (pi, 0)
    y Q_y = (0, pi), para el menor vector de onda no nulo |k| = 2 pi / L:

    skl : k paralelo a Q (fase exp(i 2 pi x / L) en el canal x; en y para el canal y)
    skt : k perpendicular a Q (fase en y para el canal x; en x para el canal y)

    Con S(q) = |sum_i S_i exp(i q r_i)|^2 / N, el cociente S(Q)/S(Q + k) que
    entra en la longitud de correlación es m^2 / sk con las normalizaciones
    usadas aquí. Costo O(N) por medición.
    """
    alc = 0.0
    als = 0.0
    atc = 0.0
    ats = 0.0
    blc = 0.0
    bls = 0.0
    btc = 0.0
    bts = 0.0
    for x in range(L):
        sgx = 1.0 if x % 2 == 0 else -1.0
        cx = cos_tab[x]
        sx = sin_tab[x]
        for y in range(L):
            sgy = 1.0 if y % 2 == 0 else -1.0
            s = spins[x * L + y]
            ux = s * sgx
            uy = s * sgy
            cy = cos_tab[y]
            sy = sin_tab[y]
            alc += ux * cx
            als += ux * sx
            atc += ux * cy
            ats += ux * sy
            blc += uy * cy
            bls += uy * sy
            btc += uy * cx
            bts += uy * sx
    N2 = float(L * L) ** 2
    skl = (alc * alc + als * als + blc * blc + bls * bls) / N2
    skt = (atc * atc + ats * ats + btc * btc + bts * bts) / N2
    return skl, skt


@njit
def sample_temperature(spins, L, T, J1, J2, h, nn, n_nn, nnn, n_nnn,
                       nterm, n_meas, stride, block_means, measure_sq,
                       series_e, series_m2, keep_series):
    """
    Termaliza y muestrea a una temperatura, partiendo de la configuración recibida.

    Se ejecutan ``nterm`` MCS que se descartan y luego ``n_meas`` mediciones, una
    cada ``stride`` MCS. Las mediciones se reparten en ``nb = block_means.shape[0]``
    bloques consecutivos de igual longitud (n_meas debe ser múltiplo de nb) y en
    ``block_means[b, k]`` se guarda el promedio de la magnitud OBS_NAMES[k] en el
    bloque b. ``spins`` se modifica en el lugar: una sucesión de llamadas con
    temperaturas crecientes o decrecientes forma un barrido con memoria.

    Parameters
    ----------
    measure_sq : bool
        Si es True mide el factor de estructura en Q + k (costo O(N) extra por medición).
    series_e, series_m2 : ndarray (n_meas,)
        Si ``keep_series`` es True se guardan ahí las series de e y de m^2.

    Returns
    -------
    acceptance : float
        Fracción de volteos aceptados durante el muestreo.
    """
    N = spins.size
    nb = block_means.shape[0]
    per_block = n_meas // nb
    table = build_accept_table(J1, J2, h, 1.0 / T)
    E, M = compute_energy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn)
    M, SAB, SX, SY = order_sums(spins, L)

    cos_tab = np.empty(L)
    sin_tab = np.empty(L)
    for a in range(L):
        cos_tab[a] = np.cos(2.0 * np.pi * a / L)
        sin_tab[a] = np.sin(2.0 * np.pi * a / L)

    for _ in range(nterm):
        E, M, SAB, SX, SY, acc = metropolis_mcs(spins, L, table, J1, J2, h, nn, n_nn, nnn, n_nnn,
                                                E, M, SAB, SX, SY)

    for b in range(nb):
        for k in range(block_means.shape[1]):
            block_means[b, k] = 0.0

    accepted = 0
    for t in range(n_meas):
        for _ in range(stride):
            E, M, SAB, SX, SY, acc = metropolis_mcs(spins, L, table, J1, J2, h, nn, n_nn, nnn, n_nnn,
                                                    E, M, SAB, SX, SY)
            accepted += acc
        e = E / N
        mu = M / N
        ms = abs(SAB) / (2.0 * N)
        mx = SX / N
        my = SY / N
        m2 = mx * mx + my * my
        skl = 0.0
        skt = 0.0
        if measure_sq:
            skl, skt = stripe_fourier(spins, L, cos_tab, sin_tab)
        b = t // per_block
        row = block_means[b]
        row[0] += e
        row[1] += e * e
        row[2] += e * e * e * e
        row[3] += mu
        row[4] += mu * mu
        row[5] += e * mu
        row[6] += abs(mu)
        row[7] += ms
        row[8] += ms * ms
        row[9] += ms * ms * ms * ms
        row[10] += np.sqrt(m2)
        row[11] += m2
        row[12] += m2 * m2
        row[13] += skl
        row[14] += skt
        row[15] += mx * mx
        if keep_series:
            series_e[t] = e
            series_m2[t] = m2

    for b in range(nb):
        for k in range(block_means.shape[1]):
            block_means[b, k] /= per_block
    return accepted / (n_meas * stride * N)


@njit
def thermalization_trace(spins, L, T, J1, J2, h, nn, n_nn, nnn, n_nnn, n_mcs):
    """
    Evolución, MCS a MCS, de e = E/N, de la magnetización de subred ms y del
    orden stripe m desde la configuración recibida (diagnóstico de termalización).
    """
    N = spins.size
    table = build_accept_table(J1, J2, h, 1.0 / T)
    E, M = compute_energy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn)
    M, SAB, SX, SY = order_sums(spins, L)
    out = np.empty((n_mcs, 3))
    for t in range(n_mcs):
        E, M, SAB, SX, SY, acc = metropolis_mcs(spins, L, table, J1, J2, h, nn, n_nn, nnn, n_nnn,
                                                E, M, SAB, SX, SY)
        out[t, 0] = E / N
        out[t, 1] = abs(SAB) / (2.0 * N)
        out[t, 2] = np.sqrt((SX * SX + SY * SY)) / N
    return out


@njit
def energy_after_mcs(spins, L, T, J1, J2, h, nn, n_nn, nnn, n_nnn, n_mcs):
    """
    Ejecuta ``n_mcs`` MCS y devuelve la energía acumulada de forma incremental y
    las sumas de orden incrementales (para compararlas con un recálculo directo).
    """
    table = build_accept_table(J1, J2, h, 1.0 / T)
    E, M = compute_energy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn)
    M, SAB, SX, SY = order_sums(spins, L)
    for _ in range(n_mcs):
        E, M, SAB, SX, SY, acc = metropolis_mcs(spins, L, table, J1, J2, h, nn, n_nn, nnn, n_nnn,
                                                E, M, SAB, SX, SY)
    return E, M, SAB, SX, SY
