"""
Estimadores estadísticos: momentos, observables derivados, errores y autocorrelación.

Convenciones
------------
Los resultados de una réplica se guardan como promedios por bloque
``blocks[..., nT, nblocks, NOBS]`` (ver ``kernels.OBS_NAMES``). Como los bloques
tienen la misma longitud, el promedio de los bloques es el promedio sobre la
serie completa.

Las magnitudes de fluctuación (calor específico, susceptibilidades) se calculan
SIEMPRE con la varianza de la serie completa, ``<x^2> - <x>^2`` con el promedio
global. Promediar varianzas calculadas dentro de cada bloque subestima la
varianza cuando el bloque no es mucho más largo que el tiempo de autocorrelación,
porque cada bloque mide las fluctuaciones respecto de su propia media y pierde las
más lentas que él (``block_variance_estimator`` lo implementa solo para
cuantificar ese efecto).

Con varias réplicas, cada magnitud derivada se evalúa con los momentos promediados
sobre todas las réplicas (equivale a concatenar sus series) y su error se estima
por jackknife eliminando una réplica cada vez; con una sola réplica, por jackknife
eliminando un bloque cada vez.
"""

import warnings

import numpy as np

from .kernels import OBS

# ----------------------------------------------------------------------------
# Observables derivados de los momentos
# ----------------------------------------------------------------------------
QUANTITIES = ("e", "Cv", "ms", "chi_s", "U_s", "m", "chi", "U", "V", "xi_L", "xi_L_par", "xi_L_perp",
              "mu", "amu", "chi_u", "dMdT", "mx2_frac")


def derived(mom, T, L):
    """
    Magnitudes físicas a partir de momentos ``mom[..., nT, NOBS]`` (T con forma (nT,)).

    e        <E>/N                                         energía por espín
    Cv       N (<e^2> - <e>^2) / T^2                       calor específico por espín (a campo fijo)
    ms       <M_s>, M_s = |S_A - S_B| / (2N)               magnetización de subred (definición del enunciado)
    chi_s    N (<M_s^2> - <M_s>^2) / T                     susceptibilidad de subred
    U_s      1 - <M_s^4> / (3 <M_s^2>^2)                   cumulante de Binder de M_s
    m        <m>, m = sqrt(mx^2 + my^2)                    parámetro de orden stripe
    chi      N (<m^2> - <m>^2) / T                         susceptibilidad stripe
    U        1 - <m^4> / (3 <m^2>^2)                       cumulante de Binder stripe (2/3 ordenado, 1/3 desordenado)
    V        1 - <e^4> / (3 <e^2>^2)                       cumulante de la energía (Challa et al., 1986)
    xi_L     xi_2nd / L con S(Q)/S(Q+k), k promedio de las dos orientaciones
    mu, amu  <M>/N y <|M|>/N                               magnetización uniforme por espín
    chi_u    N (<mu^2> - <mu>^2) / T                       susceptibilidad uniforme por espín
    dMdT     N^2 (<e mu> - <e><mu>) / T^2                  d<M>/dT de la magnetización TOTAL a h fijo
    mx2_frac <mx^2> / <m^2>                                fracción del orden en el canal x (orientación de las franjas)
    """
    mom = np.asarray(mom, dtype=float)
    T = np.asarray(T, dtype=float)
    N = L * L
    g = lambda k: mom[..., OBS[k]]
    k_fac = 2.0 * L * np.sin(np.pi / L)          # xi/L = sqrt(S(Q)/S(Q+k) - 1) / (2 L sin(pi/L))
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = 0.5 * (g("skl") + g("skt"))
        out = {
            "e": g("e"),
            "Cv": N * (g("e2") - g("e") ** 2) / T ** 2,
            "ms": g("ms"),
            "chi_s": N * (g("ms2") - g("ms") ** 2) / T,
            "U_s": 1.0 - g("ms4") / (3.0 * g("ms2") ** 2),
            "m": g("m"),
            "chi": N * (g("m2") - g("m") ** 2) / T,
            "U": 1.0 - g("m4") / (3.0 * g("m2") ** 2),
            "V": 1.0 - g("e4") / (3.0 * g("e2") ** 2),
            "xi_L": np.sqrt(np.maximum(g("m2") / sk - 1.0, 0.0)) / k_fac,
            "xi_L_par": np.sqrt(np.maximum(g("m2") / g("skl") - 1.0, 0.0)) / k_fac,
            "xi_L_perp": np.sqrt(np.maximum(g("m2") / g("skt") - 1.0, 0.0)) / k_fac,
            "mu": g("mu"),
            "amu": g("amu"),
            "chi_u": N * (g("mu2") - g("mu") ** 2) / T,
            "dMdT": N ** 2 * (g("emu") - g("e") * g("mu")) / T ** 2,
            "mx2_frac": g("mx2") / g("m2"),
        }
    return out


# Magnitudes que restan el cuadrado de una media. Con varias réplicas se calculan con la
# serie completa de CADA réplica y se promedian: si se usaran los momentos de todas las
# réplicas juntas, la dispersión de las medias entre réplicas se sumaría a la varianza, lo
# que es correcto en equilibrio pero amplifica cualquier réplica que no haya equilibrado
# (por ejemplo, atrapada con dominios de franjas a baja temperatura). El estimador por
# réplica tiene un sesgo relativo de orden 2 tau_int / n, menor que el 5 % con las
# longitudes de muestreo usadas aquí.
VARIANCE_QUANTITIES = ("Cv", "chi_s", "chi", "chi_u", "dMdT")


def combined(mom_R, T, L):
    """
    Magnitudes de un conjunto de réplicas, mom_R (R, nT, NOBS): las de VARIANCE_QUANTITIES
    como promedio de los valores de cada réplica y las demás (medias y cocientes de
    momentos como U, V y xi/L) con los momentos promediados sobre réplicas.
    """
    mom_R = np.asarray(mom_R, dtype=float)
    out = derived(mom_R.mean(axis=0), T, L)
    per_rep = derived(mom_R, T, L)
    for key in VARIANCE_QUANTITIES:
        out[key] = per_rep[key].mean(axis=0)
    return out


def replica_moments(blocks):
    """Momentos sobre la serie completa de cada réplica: (R, nT, nb, NOBS) -> (R, nT, NOBS)."""
    return np.asarray(blocks).mean(axis=-2)


def jackknife(samples, fn):
    """
    Jackknife "eliminando uno" sobre el primer eje de ``samples`` (réplicas o bloques).

    ``fn`` recibe el promedio sobre el primer eje y devuelve un dict de arreglos.
    Devuelve {clave: (valor con todas las muestras, error jackknife)}.
    """
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    full = fn(samples.mean(axis=0))
    if n < 2:
        return {k: (v, np.full_like(np.asarray(v, dtype=float), np.nan)) for k, v in full.items()}
    total = samples.sum(axis=0)
    loo = [fn((total - samples[j]) / (n - 1)) for j in range(n)]
    out = {}
    with warnings.catch_warnings():
        # Magnitudes indefinidas en algunos puntos (por ejemplo U_s cuando M_s es nula en todas
        # las mediciones) dan NaN; no es un error del estimador.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for k, v in full.items():
            arr = np.array([d[k] for d in loo], dtype=float)
            mean = np.nanmean(arr, axis=0)
            err = np.sqrt((n - 1) / n * np.nansum((arr - mean) ** 2, axis=0))
            err = np.where(np.all(np.isfinite(arr), axis=0), err, np.nan)
            out[k] = (v, err)
    return out


def curves(blocks, T, L, quantities=QUANTITIES):
    """
    Curvas de las magnitudes derivadas con su error.

    Con R >= 2 réplicas (blocks de forma (R, nT, nb, NOBS)) se usa
    ``curves_from_moments``; con una sola réplica, el valor usa la serie completa y el
    error es el jackknife eliminando un bloque.

    Returns
    -------
    dict {magnitud: (valor (nT,), error (nT,))}
    """
    blocks = np.asarray(blocks, dtype=float)
    if blocks.ndim == 3:
        blocks = blocks[None]
    T = np.asarray(T, dtype=float)
    def fn(mom):
        d = derived(mom, T, L)
        return {q: d[q] for q in quantities}
    if blocks.shape[0] >= 2:
        return curves_from_moments(replica_moments(blocks), T, L, quantities)
    samples = np.moveaxis(blocks[0], 1, 0)                      # (nb, nT, NOBS)
    return jackknife(samples, fn)


def curves_from_moments(mom, T, L, quantities=QUANTITIES):
    """
    Curvas con su error a partir de momentos por réplica ``mom`` (R, nT, NOBS), con R >= 2.

    Las magnitudes de VARIANCE_QUANTITIES son el promedio de los valores por réplica y su
    error es el error estándar entre réplicas; las demás se calculan con los momentos
    promediados sobre réplicas y su error es el jackknife eliminando una réplica.
    """
    T = np.asarray(T, dtype=float)
    mom = np.asarray(mom, dtype=float)

    def fn(m):
        d = derived(m, T, L)
        return {q: d[q] for q in quantities if q not in VARIANCE_QUANTITIES}
    out = jackknife(mom, fn)
    per_rep = derived(mom, T, L)
    for q in quantities:
        if q in VARIANCE_QUANTITIES:
            out[q] = mean_sem(per_rep[q], axis=0)
    return out


def block_variance_estimator(blocks, T, L, key="m"):
    """
    Estimador por bloques de una susceptibilidad: promedio, sobre bloques, de la
    varianza calculada DENTRO de cada bloque, N (<x^2>_b - <x>_b^2) / T (con
    x = m, ms o e; para e devuelve el calor específico con 1/T^2). Se usa solo
    para compararlo con el de la serie completa; su valor esperado es menor que
    la varianza verdadera por una fracción del orden de 2 tau_int / b, con b el
    tamaño del bloque.

    blocks : (..., nT, nb, NOBS)
    """
    blocks = np.asarray(blocks, dtype=float)
    N = L * L
    T = np.asarray(T, dtype=float)
    k1, k2, power = {"m": ("m", "m2", 1), "ms": ("ms", "ms2", 1), "e": ("e", "e2", 2)}[key]
    var_b = blocks[..., OBS[k2]] - blocks[..., OBS[k1]] ** 2          # (..., nT, nb)
    return N * var_b.mean(axis=-1) / T ** power


def mean_sem(a, axis=0):
    """Media y error estándar de la media a lo largo de ``axis`` (ddof = 1)."""
    a = np.asarray(a, dtype=float)
    n = a.shape[axis]
    return a.mean(axis=axis), a.std(axis=axis, ddof=1) / np.sqrt(n)


# ----------------------------------------------------------------------------
# Bootstrap entre réplicas
# ----------------------------------------------------------------------------
def bootstrap(groups, fn, n_boot=400, seed=2026, min_valid=0.5):
    """
    Error bootstrap de estadísticos derivados que dependen de varios grupos de
    réplicas independientes (por ejemplo, un grupo por tamaño L o por campo h).

    ``groups`` es un dict {clave: arreglo (R_clave, ...)} de muestras por réplica;
    en cada remuestreo se sortean con reposición las réplicas de cada grupo por
    separado (son corridas independientes) y se llama ``fn(grupos_remuestreados)``,
    que debe devolver un dict de escalares. Se recalcula TODO en cada remuestreo
    (posición de picos y cruces, ajustes), de modo que el error incluye también la
    incertidumbre de ubicar esas magnitudes. Se usa bootstrap y no jackknife porque
    varios estadísticos (máximo de una curva ruidosa, mediana de cruces) no son
    funciones suaves de los datos, y para ellos el jackknife no es fiable.

    El error es el semiancho del intervalo central del 68 % de los remuestreos
    (coincide con la desviación estándar si la distribución es gaussiana y no lo
    arrastran los pocos remuestreos en que falla la detección de un cruce).

    Returns
    -------
    dict {nombre: (valor con todas las réplicas, error)}
    """
    rng = np.random.default_rng(seed)
    full = fn(groups)
    boots = {k: [] for k in full}
    for _ in range(n_boot):
        sub = {}
        for key, arr in groups.items():
            R = len(arr)
            sub[key] = arr[rng.integers(0, R, size=R)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = fn(sub)
            except Exception:
                res = {}
        for k in full:
            boots[k].append(res.get(k, np.nan))
    out = {}
    for k, v in full.items():
        arr = np.asarray(boots[k], dtype=float)
        ok = np.isfinite(arr)
        if ok.sum() >= max(20, min_valid * n_boot):
            p16, p84 = np.percentile(arr[ok], [16, 84])
            out[k] = (float(v), float((p84 - p16) / 2.0))
        else:
            out[k] = (float(v), np.nan)
    return out


# ----------------------------------------------------------------------------
# Autocorrelación
# ----------------------------------------------------------------------------
def autocorrelation(x, max_lag=None):
    """Función de autocorrelación normalizada rho(t), t = 0..max_lag, calculada con FFT."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    acf /= np.arange(n, n - max_lag - 1, -1)
    return acf / acf[0] if acf[0] > 0 else np.full(max_lag + 1, np.nan)


def tau_int(x, c=6.0):
    """
    Tiempo de autocorrelación integrado con ventana automática (Madras y Sokal, 1988):

        tau(W) = 1/2 + sum_{t=1}^{W} rho(t),   W = menor ventana con W >= c tau(W).

    La ventana corta la suma antes de que domine el ruido de rho(t) a tiempos
    largos y, a la vez, la deja crecer con tau. Devuelve (tau, error, W) en
    unidades del intervalo entre mediciones; el error es la aproximación
    tau sqrt(2 (2W + 1) / n).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    rho = autocorrelation(x, max_lag=n // 2)
    if not np.isfinite(rho[0]):
        return np.nan, np.nan, 0
    taus = 0.5 + np.cumsum(rho[1:])
    W = np.arange(1, len(rho))
    ok = W >= c * taus
    if not ok.any():
        return np.nan, np.nan, int(W[-1])
    w = int(np.argmax(ok))
    tau = float(taus[w])
    return tau, tau * np.sqrt(2.0 * (2 * W[w] + 1) / n), int(W[w])
