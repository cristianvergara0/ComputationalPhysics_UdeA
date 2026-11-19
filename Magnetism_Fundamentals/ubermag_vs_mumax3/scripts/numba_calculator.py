"""Calculador micromagnético en Numba para sistemas definidos con Ubermag.

Ubermag describe el problema (malla, energías, magnetización) con objetos de Python y, para cada
cálculo, lanza OOMMF como proceso externo. Este módulo lee esos mismos objetos y hace el cálculo
dentro del proceso de Python, con núcleos compilados por Numba, sin archivos ni procesos externos:

    import numba_calculator as nc
    md = nc.MinDriver()
    md.drive(system)                                         # como oommfc.MinDriver().drive(system)
    w = nc.compute(system.energy.zeeman.density, system)     # como oommfc.compute(...)
    tau = nc.torque(system)                                  # mu0 <M> x H, como mag2exp.magnetometry.torque

Términos de energía admitidos (los de los dos notebooks del proyecto), con las mismas convenciones
de discretización que OOMMF:

    Exchange             intercambio de 6 vecinos; en un borde libre el vecino que falta no aporta
    UniaxialAnisotropy   w = K |m x u|^2
    Zeeman               campo uniforme
    DMI (clase T)        w = D m . (rot m) con diferencias centrales; el vecino que falta no aporta
    Demag                tensor de Newell entre celdas y convolución con FFT (malla sin periodicidad)

La minimización usa descenso de gradiente con paso de Barzilai-Borwein sobre la esfera (Exl et
al., 2014), el método de Minimize() de MuMax3, y se detiene con el criterio de MinDriver de OOMMF:
máx |m x H x m| < stopping_mxHxm (0.1 A/m por defecto).

Las funciones de bajo nivel (Problem, effective_field, minimize, newell_tensor) trabajan con
arreglos de numpy y no necesitan Ubermag; se usan en tests/ y en benchmark_scaling.

Requiere numpy, scipy y numba; discretisedfield y micromagneticmodel solo para la interfaz con
Ubermag.
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.fft
from numba import njit, prange

MU0 = 4e-7 * np.pi
PARALLEL_MIN_CELLS = 20000      # por debajo, lanzar hilos cuesta más que el cálculo
FFT_WORKERS = os.cpu_count() or 1
DIPOLE_FROM = 32.0              # distancia (en celdas) desde la que el tensor usa la forma dipolar


def _jit(func):
    """Compila dos variantes del mismo núcleo: en serie y con prange en paralelo."""
    return njit(nogil=True)(func), njit(nogil=True, parallel=True)(func)


# ---------------------------------------------------------------------------------------------
# Núcleos locales: cada uno suma su campo (en A/m) al arreglo h, de forma (nx, ny, nz, 3)
# ---------------------------------------------------------------------------------------------
@njit(inline="always")
def _neighbor(i, step, n, periodic):
    """Índice del vecino i + step, o -1 si no existe (borde libre o una sola celda)."""
    j = i + step
    if j < 0:
        return n - 1 if (periodic and n > 1) else -1
    if j >= n:
        return 0 if (periodic and n > 1) else -1
    return j


def _exchange(m, h, c, pbc):
    """h += c_a * (m_vecino - m) por cada vecino existente; c = 2A/(mu0 Ms) / d_a^2."""
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                for step in (-1, 1):
                    ii = _neighbor(i, step, nx, pbc[0])
                    if ii >= 0:
                        for a in range(3):
                            h[i, j, k, a] += c[0] * (m[ii, j, k, a] - m[i, j, k, a])
                    jj = _neighbor(j, step, ny, pbc[1])
                    if jj >= 0:
                        for a in range(3):
                            h[i, j, k, a] += c[1] * (m[i, jj, k, a] - m[i, j, k, a])
                    kk = _neighbor(k, step, nz, pbc[2])
                    if kk >= 0:
                        for a in range(3):
                            h[i, j, k, a] += c[2] * (m[i, j, kk, a] - m[i, j, k, a])


def _dmi_bulk(m, h, c, pbc):
    """DMI de clase T: h += -c_a * (r_ij x m_j) por cada vecino j; c = D/(mu0 Ms) / d_a.

    Equivale a -2D/(mu0 Ms) rot m con diferencias centrales.
    """
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                for step in (-1, 1):
                    ii = _neighbor(i, step, nx, pbc[0])
                    if ii >= 0:              # x^ x m = (0, -mz, my)
                        s = step * c[0]
                        h[i, j, k, 1] += s * m[ii, j, k, 2]
                        h[i, j, k, 2] -= s * m[ii, j, k, 1]
                    jj = _neighbor(j, step, ny, pbc[1])
                    if jj >= 0:              # y^ x m = (mz, 0, -mx)
                        s = step * c[1]
                        h[i, j, k, 0] -= s * m[i, jj, k, 2]
                        h[i, j, k, 2] += s * m[i, jj, k, 0]
                    kk = _neighbor(k, step, nz, pbc[2])
                    if kk >= 0:              # z^ x m = (-my, mx, 0)
                        s = step * c[2]
                        h[i, j, k, 0] += s * m[i, j, kk, 1]
                        h[i, j, k, 1] -= s * m[i, j, kk, 0]


def _anisotropy_zeeman(m, h, ck, u, hz):
    """h += ck (m.u) u + H_aplicado; ck = 2K/(mu0 Ms)."""
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                p = ck * (m[i, j, k, 0] * u[0] + m[i, j, k, 1] * u[1] + m[i, j, k, 2] * u[2])
                for a in range(3):
                    h[i, j, k, a] += p * u[a] + hz[a]


def _torque(m, h, d):
    """d = m x (m x h) en cada celda; devuelve máx |m x h| (igual a máx |m x h x m|)."""
    nx, ny, nz, _ = m.shape
    best = np.zeros(nx)
    for i in prange(nx):
        local = 0.0
        for j in range(ny):
            for k in range(nz):
                mx = m[i, j, k, 0]
                my = m[i, j, k, 1]
                mz = m[i, j, k, 2]
                tx = my * h[i, j, k, 2] - mz * h[i, j, k, 1]
                ty = mz * h[i, j, k, 0] - mx * h[i, j, k, 2]
                tz = mx * h[i, j, k, 1] - my * h[i, j, k, 0]
                d[i, j, k, 0] = my * tz - mz * ty
                d[i, j, k, 1] = mz * tx - mx * tz
                d[i, j, k, 2] = mx * ty - my * tx
                local = max(local, tx * tx + ty * ty + tz * tz)
        best[i] = local
    return np.sqrt(best.max())


def _bb_update(m, d, m_old, d_old, tau):
    """Paso sobre la esfera (transformada de Cayley) y sumas s.s, s.y, y.y de Barzilai-Borwein.

    m' = [(1 - tau^2 |d|^2/4) m - tau d] / (1 + tau^2 |d|^2/4), con d = m x (m x h) perpendicular a m.
    Guarda en m_old y d_old los valores anteriores y deja en m el nuevo estado.
    """
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                d2 = d[i, j, k, 0] ** 2 + d[i, j, k, 1] ** 2 + d[i, j, k, 2] ** 2
                q = 0.25 * tau * tau * d2
                for a in range(3):
                    m_old[i, j, k, a] = m[i, j, k, a]
                    d_old[i, j, k, a] = d[i, j, k, a]
                    m[i, j, k, a] = ((1.0 - q) * m[i, j, k, a] - tau * d[i, j, k, a]) / (1.0 + q)


def _bb_sums(m, d, m_old, d_old):
    nx, ny, nz, _ = m.shape
    acc = np.zeros((nx, 3))
    for i in prange(nx):
        ss = 0.0
        sy = 0.0
        yy = 0.0
        for j in range(ny):
            for k in range(nz):
                for a in range(3):
                    s = m[i, j, k, a] - m_old[i, j, k, a]
                    y = d[i, j, k, a] - d_old[i, j, k, a]
                    ss += s * s
                    sy += s * y
                    yy += y * y
        acc[i, 0] = ss
        acc[i, 1] = sy
        acc[i, 2] = yy
    return acc[:, 0].sum(), acc[:, 1].sum(), acc[:, 2].sum()


def _demag_multiply(fm, kern, out):
    """out_a = -sum_b N_ab(k) M_b(k) en el espacio de Fourier; kern = (xx, yy, zz, xy, xz, yz)."""
    n0, n1, n2 = fm.shape[1], fm.shape[2], fm.shape[3]
    for i in prange(n0):
        for j in range(n1):
            for k in range(n2):
                a = fm[0, i, j, k]
                b = fm[1, i, j, k]
                c = fm[2, i, j, k]
                out[0, i, j, k] = -(kern[0, i, j, k] * a + kern[3, i, j, k] * b + kern[4, i, j, k] * c)
                out[1, i, j, k] = -(kern[3, i, j, k] * a + kern[1, i, j, k] * b + kern[5, i, j, k] * c)
                out[2, i, j, k] = -(kern[4, i, j, k] * a + kern[5, i, j, k] * b + kern[2, i, j, k] * c)


EXCHANGE = _jit(_exchange)
DMI_BULK = _jit(_dmi_bulk)
ANISOTROPY_ZEEMAN = _jit(_anisotropy_zeeman)
TORQUE = _jit(_torque)
BB_UPDATE = _jit(_bb_update)
BB_SUMS = _jit(_bb_sums)
DEMAG_MULTIPLY = _jit(_demag_multiply)


# ---------------------------------------------------------------------------------------------
# Tensor desmagnetizante entre celdas (Newell, Williams y Dunlop, 1993)
# ---------------------------------------------------------------------------------------------
def _newell_f(x, y, z):
    x, y, z = np.abs(x), np.abs(y), np.abs(z)        # f es par en las tres variables
    r = np.sqrt(x * x + y * y + z * z)
    out = (2 * x * x - y * y - z * z) * r / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 0.5 * y * (z * z - x * x) * np.arcsinh(y / np.sqrt(x * x + z * z))
        out += np.where((x * x + z * z) > 0, t, 0.0)
        t = 0.5 * z * (y * y - x * x) * np.arcsinh(z / np.sqrt(x * x + y * y))
        out += np.where((x * x + y * y) > 0, t, 0.0)
        t = x * y * z * np.arctan(y * z / (x * r))
        out -= np.where(x > 0, t, 0.0)
    return out


def _newell_g(x, y, z):
    sign = np.sign(x) * np.sign(y)                    # g es impar en x y en y, y par en z
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    r = np.sqrt(x * x + y * y + z * z)
    out = -x * y * r / 3.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = x * y * z * np.arcsinh(z / np.sqrt(x * x + y * y))
        out += np.where((x * x + y * y) > 0, t, 0.0)
        t = y / 6.0 * (3 * z * z - y * y) * np.arcsinh(x / np.sqrt(y * y + z * z))
        out += np.where((y * y + z * z) > 0, t, 0.0)
        t = x / 6.0 * (3 * z * z - x * x) * np.arcsinh(y / np.sqrt(x * x + z * z))
        out += np.where((x * x + z * z) > 0, t, 0.0)
        t = z ** 3 / 6.0 * np.arctan(x * y / (z * r))
        out -= np.where(z > 0, t, 0.0)
        t = z * y * y / 2.0 * np.arctan(x * z / (y * r))
        out -= np.where(y > 0, t, 0.0)
        t = z * x * x / 2.0 * np.arctan(y * z / (x * r))
        out -= np.where(x > 0, t, 0.0)
    return sign * out


def _second_differences(func, n, d):
    """-(1/(4 pi V)) D2x D2y D2z func en los desplazamientos (i, j, k) d, con 0 <= i < n_a."""
    grids = [np.arange(-1, n[a] + 1) * d[a] for a in range(3)]
    val = func(*np.meshgrid(*grids, indexing="ij"))
    for axis in range(3):
        val = np.diff(val, 2, axis=axis)
    return -val / (4 * np.pi * d[0] * d[1] * d[2])


def newell_tensor(n, cell):
    """Tensor N_ab entre celdas separadas (i dx, j dy, k dz), con 0 <= i < nx, etc.

    Devuelve un arreglo (6, nx, ny, nz) con las componentes (xx, yy, zz, xy, xz, yz). Las
    componentes para desplazamientos negativos salen por simetría: las diagonales son pares en
    las tres direcciones y N_ab (a != b) es impar en a y en b y par en la tercera.

    Las fórmulas analíticas de Newell pierden precisión lejos del origen por cancelación; a partir
    de DIPOLE_FROM celdas se usa la interacción dipolar entre los centros de las celdas, que para
    celdas de lados parecidos difiere de la exacta en menos de 1e-5 relativo a esa distancia.
    """
    n = tuple(int(v) for v in n)
    scale = min(cell)
    d = tuple(float(v) / scale for v in cell)         # longitudes en unidades de la celda menor
    out = np.empty((6,) + n)
    perm = lambda arr, p: np.transpose(arr, np.argsort(p))   # noqa: E731
    # N_yy(X,Y,Z) = N_xx(Y,X,Z), N_zz(X,Y,Z) = N_xx(Z,Y,X), y análogos para las no diagonales
    for idx, p in ((0, (0, 1, 2)), (1, (1, 0, 2)), (2, (2, 1, 0))):
        out[idx] = perm(_second_differences(_newell_f, [n[q] for q in p], [d[q] for q in p]), p)
    for idx, p in ((3, (0, 1, 2)), (4, (0, 2, 1)), (5, (1, 2, 0))):
        out[idx] = perm(_second_differences(_newell_g, [n[q] for q in p], [d[q] for q in p]), p)
    x, y, z = np.meshgrid(*[np.arange(n[a]) * d[a] for a in range(3)], indexing="ij")
    r2 = x * x + y * y + z * z
    far = r2 > (DIPOLE_FROM * max(d)) ** 2
    if far.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            pref = d[0] * d[1] * d[2] / (4 * np.pi * r2 ** 2.5)
            dip = (r2 - 3 * x * x, r2 - 3 * y * y, r2 - 3 * z * z, -3 * x * y, -3 * x * z, -3 * y * z)
            for idx in range(6):
                out[idx][far] = (pref * dip[idx])[far]
    return out


_KERNEL_CACHE = {}


def demag_kernel(n, cell):
    """Transformada del tensor en la malla extendida (2nx, 2ny, 2nz); se calcula una vez por malla."""
    key = (tuple(int(v) for v in n), tuple(float(v) for v in cell))
    if key not in _KERNEL_CACHE:
        nn = key[0]
        t = newell_tensor(nn, cell)
        big = np.zeros((6,) + tuple(2 * v for v in nn))
        parity = ((1, 1, 1), (1, 1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
        for idx in range(6):
            for sx in (0, 1):
                for sy in (0, 1):
                    for sz in (0, 1):
                        src = [slice(None)] * 3
                        dst = [slice(0, nn[0]), slice(0, nn[1]), slice(0, nn[2])]
                        sign = 1.0
                        for a, s in enumerate((sx, sy, sz)):
                            if s:             # desplazamientos negativos: -i queda en el índice 2n - i
                                if nn[a] == 1:
                                    break
                                src[a] = slice(1, nn[a])
                                dst[a] = slice(2 * nn[a] - 1, nn[a], -1)
                                sign *= parity[idx][a]
                        else:
                            big[(idx,) + tuple(dst)] = sign * t[(idx,) + tuple(src)]
        _KERNEL_CACHE[key] = scipy.fft.rfftn(big, axes=(1, 2, 3), workers=FFT_WORKERS)
    return _KERNEL_CACHE[key]


# ---------------------------------------------------------------------------------------------
# Problema, campo efectivo, energías y minimización (solo arreglos de numpy)
# ---------------------------------------------------------------------------------------------
@dataclass
class Problem:
    """Parámetros de un sistema: malla, material y términos de energía activos (None = ausente)."""

    n: tuple
    cell: tuple
    Ms: float
    pbc: tuple = (False, False, False)
    A: float = None
    K: float = None
    u: tuple = (0.0, 0.0, 1.0)
    D: float = None
    H: tuple = None
    demag: bool = False

    def __post_init__(self):
        self.n = tuple(int(v) for v in self.n)
        self.cell = tuple(float(v) for v in self.cell)
        self.pbc = np.array([bool(v) for v in self.pbc])
        self.u = np.asarray(self.u, dtype=float) / np.linalg.norm(self.u)
        if self.demag and self.pbc.any():
            raise NotImplementedError("El campo desmagnetizante con condiciones periódicas no está implementado.")
        self.variant = 1 if int(np.prod(self.n)) >= PARALLEL_MIN_CELLS else 0
        self.cell_volume = float(np.prod(self.cell))

    @property
    def terms(self):
        active = {"exchange": self.A, "uniaxialanisotropy": self.K, "dmi": self.D, "zeeman": self.H}
        return [k for k, v in active.items() if v is not None] + (["demag"] if self.demag else [])


def demag_field(p, m):
    """Campo desmagnetizante en A/m: H = -sum_r' N(r - r') Ms m(r'), por convolución con FFT."""
    kern = demag_kernel(p.n, p.cell)
    big = tuple(2 * v for v in p.n)
    fm = scipy.fft.rfftn(np.moveaxis(m, -1, 0), s=big, axes=(1, 2, 3), workers=FFT_WORKERS)
    out = np.empty_like(fm)
    DEMAG_MULTIPLY[p.variant](fm, kern, out)
    h = scipy.fft.irfftn(out, s=big, axes=(1, 2, 3), workers=FFT_WORKERS)
    return p.Ms * np.moveaxis(h[:, :p.n[0], :p.n[1], :p.n[2]], 0, -1)


def term_field(p, m, term):
    """Campo efectivo (A/m) de un solo término."""
    h = np.zeros_like(m)
    v = p.variant
    if term == "exchange":
        c = 2 * p.A / (MU0 * p.Ms) / np.array(p.cell) ** 2
        EXCHANGE[v](m, h, c, p.pbc)
    elif term == "dmi":
        c = p.D / (MU0 * p.Ms) / np.array(p.cell)
        DMI_BULK[v](m, h, c, p.pbc)
    elif term == "uniaxialanisotropy":
        ANISOTROPY_ZEEMAN[v](m, h, 2 * p.K / (MU0 * p.Ms), p.u, np.zeros(3))
    elif term == "zeeman":
        h += np.asarray(p.H, dtype=float)
    elif term == "demag":
        h = demag_field(p, m)
    else:
        raise ValueError(f"Término desconocido: {term}")
    return h


def effective_field(p, m):
    """Campo efectivo total en A/m."""
    v = p.variant
    h = demag_field(p, m) if p.demag else np.zeros_like(m)
    if p.A is not None:
        EXCHANGE[v](m, h, 2 * p.A / (MU0 * p.Ms) / np.array(p.cell) ** 2, p.pbc)
    if p.D is not None:
        DMI_BULK[v](m, h, p.D / (MU0 * p.Ms) / np.array(p.cell), p.pbc)
    ck = 2 * p.K / (MU0 * p.Ms) if p.K is not None else 0.0
    hz = np.asarray(p.H, dtype=float) if p.H is not None else np.zeros(3)
    ANISOTROPY_ZEEMAN[v](m, h, ck, p.u, hz)
    return h


def term_density(p, m, term):
    """Densidad de energía (J/m^3) de un término, con las convenciones de Ubermag y OOMMF."""
    mh = np.einsum("...a,...a->...", m, term_field(p, m, term))
    if term == "zeeman":
        return -MU0 * p.Ms * mh
    if term == "uniaxialanisotropy":
        return p.K - 0.5 * MU0 * p.Ms * mh            # K |m x u|^2 = K - K (m.u)^2
    return -0.5 * MU0 * p.Ms * mh


def term_energy(p, m, term):
    return float(term_density(p, m, term).sum() * p.cell_volume)


def total_energy(p, m):
    return sum(term_energy(p, m, t) for t in p.terms)


def max_mxHxm(p, m):
    """Par residual máx |m x H x m| en A/m, el criterio de parada de MinDriver de OOMMF."""
    return float(TORQUE[p.variant](m, effective_field(p, m), np.empty_like(m)))


def minimize(p, m0, stopping_mxHxm=0.1, max_iter=200000, max_angle=0.1):
    """Minimiza la energía desde m0 (vectores unitarios). Devuelve (m, información).

    Descenso de gradiente sobre la esfera con el paso de Barzilai-Borwein, alternando sus dos
    fórmulas en iteraciones pares e impares. El campo se escala con Ms, de modo que el paso tau es
    adimensional. El paso se limita para que ninguna celda gire más de max_angle radianes por
    iteración: lejos del equilibrio el método avanza como un descenso de gradiente con pasos cortos,
    que sigue el mismo camino que la dinámica amortiguada, y cerca del equilibrio, donde el par es
    pequeño, el límite deja de actuar y el paso de Barzilai-Borwein acelera la convergencia.
    """
    v = p.variant
    m = np.ascontiguousarray(m0, dtype=float).copy()
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    d, m_old, d_old = np.empty_like(m), np.empty_like(m), np.empty_like(m)
    it, start = 0, time.perf_counter()
    h = effective_field(p, m) / p.Ms
    res = TORQUE[v](m, h, d)                          # máx |m x h|, adimensional
    tau = np.inf
    while res * p.Ms >= stopping_mxHxm and it < max_iter:
        tau = min(tau, max_angle / res)
        BB_UPDATE[v](m, d, m_old, d_old, tau)
        h = effective_field(p, m) / p.Ms
        res = TORQUE[v](m, h, d)
        ss, sy, yy = BB_SUMS[v](m, d, m_old, d_old)
        it += 1
        if sy > 0 and yy > 0:
            tau = ss / sy if it % 2 else sy / yy
        else:                                         # curvatura no positiva: solo actúa el límite de giro
            tau = np.inf
        if it % 512 == 0:                             # corrige la deriva de redondeo de |m|
            m /= np.linalg.norm(m, axis=-1, keepdims=True)
    res *= p.Ms
    info = {"iterations": it, "max_mxHxm": float(res), "converged": bool(res < stopping_mxHxm),
            "time_s": time.perf_counter() - start}
    return m, info


def warm_up():
    """Compila los núcleos en sus dos variantes con un problema mínimo; devuelve el tiempo en s."""
    start = time.perf_counter()
    for n in ((4, 4, 2), (32, 32, 20)):
        p = Problem(n=n, cell=(5e-9,) * 3, Ms=8e5, A=1e-11, K=1e5, D=1e-3, H=(0, 1e4, 1e5), demag=n[0] > 4)
        if p.demag:
            p.D = None
        m0 = np.zeros(n + (3,))
        m0[..., 2] = 1.0
        m0[..., 0] = 0.1
        minimize(p, m0, max_iter=3)
    _KERNEL_CACHE.clear()
    return time.perf_counter() - start


# ---------------------------------------------------------------------------------------------
# Interfaz con Ubermag: el mismo objeto system que se le pasa a oommfc
# ---------------------------------------------------------------------------------------------
def problem_from_system(system):
    """Lee la malla, Ms y los términos de energía de un micromagneticmodel.System."""
    mesh = system.m.mesh
    norm = np.linalg.norm(system.m.array, axis=-1)
    Ms = float(norm.max())
    if Ms == 0 or not np.allclose(norm, Ms, rtol=1e-9):
        raise NotImplementedError("Se requiere una magnetización de norma uniforme (Ms constante y no nula).")
    kw = {"n": tuple(mesh.n), "cell": tuple(mesh.cell), "Ms": Ms, "pbc": tuple(a in mesh.bc for a in "xyz")}
    for term in system.energy:
        name = type(term).__name__
        if name == "Exchange":
            kw["A"] = float(term.A)
        elif name == "UniaxialAnisotropy":
            kw["K"], kw["u"] = float(term.K), tuple(term.u)
        elif name == "Zeeman":
            kw["H"] = tuple(float(v) for v in term.H)
        elif name == "DMI":
            if term.crystalclass != "T":
                raise NotImplementedError("Solo está implementada la DMI de clase T.")
            kw["D"] = float(term.D)
        elif name == "Demag":
            kw["demag"] = True
        else:
            raise NotImplementedError(f"Término de energía no implementado: {name}")
    return Problem(**kw), np.ascontiguousarray(system.m.array / Ms)


class MinDriver:
    """Minimizador con la interfaz de oommfc.MinDriver: drive(system) actualiza system.m."""

    def __init__(self, stopping_mxHxm=0.1, max_iter=200000):
        self.stopping_mxHxm = stopping_mxHxm
        self.max_iter = max_iter
        self.info = None

    def drive(self, system, verbose=0):
        import discretisedfield as df

        p, m = problem_from_system(system)
        m, self.info = minimize(p, m, self.stopping_mxHxm, self.max_iter)
        system.m = df.Field(system.m.mesh, nvdim=3, value=m * p.Ms)
        if verbose:
            print(f"numba_calculator: {self.info['iterations']} iteraciones, "
                  f"máx |m x H x m| = {self.info['max_mxHxm']:.3g} A/m, {self.info['time_s']:.3f} s")
        if not self.info["converged"]:
            raise RuntimeError(f"La minimización no convergió en {self.max_iter} iteraciones.")


def compute(func, system):
    """Como oommfc.compute: func es term.density, term.energy o term.effective_field."""
    import discretisedfield as df

    term, what = func.__self__, func.__name__
    p, m = problem_from_system(system)
    name = type(term).__name__.lower()
    if name not in p.terms:
        raise ValueError(f"El término {name} no está en la energía del sistema.")
    mesh = system.m.mesh
    if what == "energy":
        return term_energy(p, m, name)
    if what == "density":
        return df.Field(mesh, nvdim=1, value=term_density(p, m, name)[..., np.newaxis])
    if what == "effective_field":
        return df.Field(mesh, nvdim=3, value=term_field(p, m, name))
    raise ValueError(f"Magnitud no admitida: {what} (density, energy o effective_field).")


def torque(system):
    """Torque por unidad de volumen mu0 <M> x H (N/m^2), como mag2exp.magnetometry.torque."""
    H = np.zeros(3)
    for term in system.energy:
        if type(term).__name__ == "Zeeman":
            H = np.asarray(term.H, dtype=float)
    return MU0 * np.cross(system.m.array.reshape(-1, 3).mean(axis=0), H)
