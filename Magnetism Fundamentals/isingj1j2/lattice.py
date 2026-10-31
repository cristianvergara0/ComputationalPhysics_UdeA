"""
Geometría de la red cuadrada L x L, configuraciones de referencia y enumeración exacta.

La red se guarda aplanada: el sitio (x, y) ocupa la posición i = x L + y de un
arreglo 1D de N = L^2 espines S_i = +-1 (tipo int8).
"""

import numpy as np

# Desplazamientos de primeros vecinos (NN) y de segundos vecinos (NNN, diagonales).
NN_SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1))
NNN_SHIFTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def build_neighbor_tables(L, periodic=True):
    """
    Tablas de vecinos de la red L x L.

    Parameters
    ----------
    L : int
        Lado de la red.
    periodic : bool
        True para fronteras periódicas (PBC), False para fronteras libres (FBC).

    Returns
    -------
    nn, n_nn, nnn, n_nnn : ndarray
        ``nn[i, k]`` es el k-ésimo primer vecino del sitio i y ``n_nn[i]`` cuántos
        tiene (4 en PBC; 2, 3 o 4 en FBC según el sitio sea esquina, borde o
        interior). Lo mismo para los segundos vecinos. Las posiciones sin vecino
        se rellenan con -1 y nunca se leen, porque los bucles llegan hasta n_nn[i].
    """
    if L < 3 and periodic:
        # Con L = 2 y PBC, los vecinos (x+1) y (x-1) coinciden y cada enlace se
        # contaría dos veces; las fórmulas de energía suponen enlaces distintos.
        raise ValueError("Con fronteras periódicas se requiere L >= 3.")
    N = L * L
    tables = []
    for shifts in (NN_SHIFTS, NNN_SHIFTS):
        table = np.full((N, 4), -1, dtype=np.int32)
        count = np.zeros(N, dtype=np.int32)
        for x in range(L):
            for y in range(L):
                i = x * L + y
                for dx, dy in shifts:
                    xx, yy = x + dx, y + dy
                    if periodic:
                        xx %= L
                        yy %= L
                    elif not (0 <= xx < L and 0 <= yy < L):
                        continue
                    table[i, count[i]] = xx * L + yy
                    count[i] += 1
        tables.extend([table, count])
    return tables[0], tables[1], tables[2], tables[3]


# ----------------------------------------------------------------------------
# Configuraciones de referencia
# ----------------------------------------------------------------------------
def random_configuration(L, rng):
    """Configuración aleatoria (S_i = +-1 con igual probabilidad) generada con ``rng`` (np.random.Generator)."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=L * L)


def ferromagnetic_configuration(L, sign=1):
    """Todos los espines iguales: estado fundamental para h > h_c."""
    return np.full(L * L, sign, dtype=np.int8)


def checkerboard_configuration(L):
    """Orden de Néel: S = (-1)^(x+y)."""
    x, y = np.divmod(np.arange(L * L), L)
    return np.where((x + y) % 2 == 0, 1, -1).astype(np.int8)


def stripe_configuration(L, direction="x"):
    """
    Orden en franjas (stripe): S = (-1)^x (franjas alternadas a lo largo de x,
    espines iguales a lo largo de y) o S = (-1)^y. Es el estado fundamental del
    modelo J1-J2 a campo nulo cuando |J2| > J1/2.
    """
    x, y = np.divmod(np.arange(L * L), L)
    coord = x if direction == "x" else y
    return np.where(coord % 2 == 0, 1, -1).astype(np.int8)


def initial_configuration(L, init, rng):
    """Configuración inicial de un barrido: 'random', 'stripe' o 'ferro'."""
    if init == "random":
        return random_configuration(L, rng)
    if init == "stripe":
        return stripe_configuration(L, "x")
    if init == "ferro":
        return ferromagnetic_configuration(L)
    raise ValueError("init debe ser 'random', 'stripe' o 'ferro'")


# ----------------------------------------------------------------------------
# Energías y campo crítico a T = 0
# ----------------------------------------------------------------------------
def energy_numpy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn):
    """
    Energía total de una configuración, calculada de forma independiente del
    núcleo de Numba (sirve para validarlo). Al recorrer todos los sitios cada
    enlace aparece dos veces; de ahí el factor 1/2.
    """
    s = spins.astype(np.float64)
    mask_nn = np.arange(4)[None, :] < n_nn[:, None]
    mask_nnn = np.arange(4)[None, :] < n_nnn[:, None]
    e_nn = (s[:, None] * np.where(mask_nn, s[np.maximum(nn, 0)], 0.0)).sum()
    e_nnn = (s[:, None] * np.where(mask_nnn, s[np.maximum(nnn, 0)], 0.0)).sum()
    return -0.5 * J1 * e_nn - 0.5 * J2 * e_nnn - h * s.sum()


def ground_state_energies(J1, J2, h=0.0):
    """
    Energía por espín (PBC, L par) de los tres órdenes simples:

    ferromagnético  e = -2 J1 - 2 J2 - h
    checkerboard    e = +2 J1 - 2 J2
    stripe          e = +2 J2            (los enlaces NN se cancelan: dos paralelos y dos antiparalelos)
    """
    return {"ferro": -2.0 * J1 - 2.0 * J2 - h, "checkerboard": 2.0 * J1 - 2.0 * J2, "stripe": 2.0 * J2}


def critical_field(J1, J2):
    """
    Campo en que, a T = 0, el estado saturado iguala en energía al stripe:
    -2 J1 - 2 J2 - h = 2 J2  =>  h_c = -4 J2 - 2 J1 = 4|J2| - 2 J1 (0.4 para J2 = -0.6).
    """
    return -4.0 * J2 - 2.0 * J1


# ----------------------------------------------------------------------------
# Enumeración exacta (redes pequeñas)
# ----------------------------------------------------------------------------
def enumerate_states(Lx, Ly, J1, J2, chunk=1 << 20):
    """
    Recorre las 2^(Lx Ly) configuraciones de un toro Lx x Ly (PBC) y devuelve,
    para cada una, la energía de interacción, la magnetización total y las sumas
    stripe SX = sum (-1)^x S y SY = sum (-1)^y S. Pensado para Lx Ly <= 24.

    Returns
    -------
    E_int, M, SX, SY : ndarray (2^N,)
    """
    N = Lx * Ly
    if N > 24:
        raise ValueError("enumeración exacta limitada a N <= 24 espines")
    x, y = np.divmod(np.arange(N), Ly)
    bonds = []
    for shifts, J in ((((1, 0), (0, 1)), J1), (((1, 1), (1, -1)), J2)):
        for dx, dy in shifts:
            j = ((x + dx) % Lx) * Ly + (y + dy) % Ly
            bonds.append((J, j))
    sign_x = np.where(x % 2 == 0, 1, -1)
    sign_y = np.where(y % 2 == 0, 1, -1)
    out = [np.empty(1 << N) for _ in range(4)]
    for start in range(0, 1 << N, chunk):
        codes = np.arange(start, min(start + chunk, 1 << N), dtype=np.int64)
        s = (((codes[:, None] >> np.arange(N)) & 1) * 2 - 1).astype(np.int8)
        E = np.zeros(len(codes))
        for J, j in bonds:
            E -= J * (s * s[:, j]).sum(axis=1)
        sl = slice(start, start + len(codes))
        out[0][sl] = E
        out[1][sl] = s.sum(axis=1)
        out[2][sl] = s @ sign_x
        out[3][sl] = s @ sign_y
    return tuple(out)


def exact_thermal_moments(E_int, M, SX, SY, N, T, h):
    """
    Promedios térmicos exactos a temperatura T y campo h a partir de la
    enumeración, con las mismas definiciones que el muestreador:
    e = E/N (E incluye -h M), mu = M/N y m^2 = (SX^2 + SY^2)/N^2.
    """
    E = E_int - h * M
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    e = E / N
    mu = M / N
    m2 = (SX ** 2 + SY ** 2) / N ** 2
    avg = lambda a: float((w * a).sum())
    return {"e": avg(e), "e2": avg(e * e), "e4": avg(e ** 4), "mu": avg(mu), "mu2": avg(mu * mu),
            "emu": avg(e * mu), "m": avg(np.sqrt(m2)), "m2": avg(m2), "m4": avg(m2 * m2)}


def minimum_energy_states(Lx, Ly, J1, J2, fields):
    """
    Estado de mínima energía en cada campo de ``fields`` sobre un toro Lx x Ly.
    Devuelve {h: (e_min por espín, mu del estado)}. Sirve para comprobar que no
    existen fases intermedias (mesetas de magnetización) entre el stripe y el
    estado saturado en toros pequeños.
    """
    E_int, M, _, _ = enumerate_states(Lx, Ly, J1, J2)
    N = Lx * Ly
    out = {}
    for h in fields:
        E = E_int - h * M
        k = int(np.argmin(E))
        out[h] = (E[k] / N, M[k] / N)
    return out
