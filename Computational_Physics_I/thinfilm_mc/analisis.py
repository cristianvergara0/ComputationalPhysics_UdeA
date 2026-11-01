"""Métricas morfológicas de superficies, análisis espectral y de correlación espacial.

Las superficies son arreglos enteros ``h[i, j]`` con la altura, en capas, de cada columna de
la malla (aproximación *solid-on-solid*, sin huecos ni voladizos). Ninguna función de este
módulo lee parámetros globales: todo lo que necesitan llega como argumento.
"""

import numpy as np
from scipy import ndimage, stats

from .compat import njit


@njit
def calcular_metricas(superficie, nx, ny):
    """Calcula el espesor medio y la rugosidad RMS de una superficie.

    Parámetros
    ----------
    superficie : ndarray de enteros, forma (nx, ny)
        Altura de cada celda, en capas.
    nx, ny : int
        Número de celdas en cada dirección.

    Devuelve
    --------
    espesor : float
        Altura media, en capas.
    rugosidad : float
        Desviación estándar poblacional de las alturas (divisor nx * ny), en capas.
    """
    # Se calcula primero la media y después las desviaciones respecto a ella
    # (dos pasadas). Restar <h^2> - <h>^2 en una sola pasada perdería precisión
    # cuando la media es grande frente a la dispersión.
    suma_alturas = 0.0
    for i in range(nx):
        for j in range(ny):
            suma_alturas += superficie[i, j]
    espesor = suma_alturas / (nx * ny)

    suma_varianza = 0.0
    for i in range(nx):
        for j in range(ny):
            diferencia = superficie[i, j] - espesor
            suma_varianza += diferencia * diferencia
    rugosidad = np.sqrt(suma_varianza / (nx * ny))

    return espesor, rugosidad


def espectro_potencia_radial(superficie, periodica=True, n_bins=None):
    """PSD 2D de una superficie de alturas, promediada radialmente en k.

    Devuelve (k, P(k)) con k en ciclos por celda (frecuencia espacial).
    Si periodica=False, se aplica una ventana de Hann en ambos ejes antes
    de la FFT para reducir la fuga espectral que introducen los bordes
    abiertos (la superficie no es periodica en ese caso).

    Parametros
    ----------
    superficie : ndarray de enteros, forma (nx, ny)
        Altura de cada celda, en capas.
    periodica : bool
        Si la malla tiene fronteras periodicas (True) o abiertas (False).
    n_bins : int, opcional
        Numero de bins radiales en k. Por defecto, min(nx, ny) // 2.

    Devuelve
    --------
    k_centros : ndarray
        Centros de los bins de numero de onda radial.
    psd_radial : ndarray
        Espectro de potencia promediado en cada bin.
    """
    nx, ny = superficie.shape
    h = superficie.astype(np.float64)
    h = h - h.mean()
    if not periodica:
        wx = np.hanning(nx)
        wy = np.hanning(ny)
        h = h * np.outer(wx, wy)
    F = np.fft.fft2(h)
    psd2d = (np.abs(F) ** 2) / (nx * ny)
    psd2d = np.fft.fftshift(psd2d)
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    if n_bins is None:
        n_bins = min(nx, ny) // 2
    k_bordes = np.linspace(0, K.max(), n_bins + 1)
    k_centros = 0.5 * (k_bordes[1:] + k_bordes[:-1])
    psd_radial = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mascara = (K >= k_bordes[i]) & (K < k_bordes[i + 1])
        if mascara.any():
            psd_radial[i] = psd2d[mascara].mean()
    return k_centros, psd_radial


def correlacion_alturas_radial(superficie, periodica=True, n_bins=None):
    """Funcion de correlacion de alturas G(r) = <h(0) h(r)> - h_medio^2,
    promediada radialmente, y la longitud de correlacion lateral xi (radio
    donde la correlacion normalizada cae a 1/e de su valor en r = 0).

    Con fronteras periodicas se usa la autocorrelacion circular exacta via
    FFT (coincide con el dominio real del modelo). Con fronteras abiertas
    se usa autocorrelacion lineal (relleno de ceros al doble del tamano y
    normalizacion por el numero de solapamientos validos en cada rezago),
    para no contaminar el resultado con el supuesto de periodicidad que
    no se cumple en ese caso.

    Parametros
    ----------
    superficie : ndarray de enteros, forma (nx, ny)
        Altura de cada celda, en capas.
    periodica : bool
        Si la malla tiene fronteras periodicas (True) o abiertas (False).
    n_bins : int, opcional
        Numero de bins radiales en r. Por defecto, min(nx, ny) // 2.

    Devuelve
    --------
    r_centros : ndarray
        Centros de los bins de distancia radial, en celdas.
    g_radial : ndarray
        Funcion de correlacion promediada en cada bin.
    xi : float
        Longitud de correlacion lateral, en celdas (np.nan si la
        correlacion ya cae por debajo de 1/e dentro del primer bin).
    """
    nx, ny = superficie.shape
    h = superficie.astype(np.float64)
    h = h - h.mean()

    if periodica:
        F = np.fft.fft2(h)
        acov = np.fft.ifft2(F * np.conj(F)).real / (nx * ny)
        acov = np.fft.fftshift(acov)
        lag_x = np.arange(-(nx // 2), nx - nx // 2)
        lag_y = np.arange(-(ny // 2), ny - ny // 2)
    else:
        nx2, ny2 = 2 * nx, 2 * ny
        Fh = np.fft.fft2(h, s=(nx2, ny2))
        acov_sum = np.fft.ifft2(Fh * np.conj(Fh)).real
        unos = np.ones((nx, ny))
        Fu = np.fft.fft2(unos, s=(nx2, ny2))
        conteo = np.fft.ifft2(Fu * np.conj(Fu)).real
        conteo = np.clip(conteo, 1, None)
        acov = acov_sum / conteo
        acov = np.fft.fftshift(acov)
        lag_x = np.arange(-nx, nx2 - nx)
        lag_y = np.arange(-ny, ny2 - ny)

    LX, LY = np.meshgrid(lag_x, lag_y, indexing="ij")
    R = np.sqrt(LX ** 2 + LY ** 2)
    r_max = min(nx, ny) // 2
    if n_bins is None:
        n_bins = r_max
    r_bordes = np.linspace(0, r_max, n_bins + 1)
    r_centros = 0.5 * (r_bordes[1:] + r_bordes[:-1])
    g_radial = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mascara = (R >= r_bordes[i]) & (R < r_bordes[i + 1])
        if mascara.any():
            g_radial[i] = acov[mascara].mean()

    g0 = g_radial[0]
    rho = g_radial / g0 if g0 > 0 else np.full_like(g_radial, np.nan)

    xi = np.nan
    for i in range(len(rho) - 1):
        if rho[i] >= 1 / np.e > rho[i + 1]:
            frac = (rho[i] - 1 / np.e) / (rho[i] - rho[i + 1])
            xi = r_centros[i] + frac * (r_centros[i + 1] - r_centros[i])
            break

    return r_centros, g_radial, xi


_CACHE_RADIAL = {}


def _indices_radiales(nx, ny):
    """Bins radiales de la autocorrelación circular, los mismos que usa ``correlacion_alturas_radial``."""
    clave = (nx, ny)
    if clave not in _CACHE_RADIAL:
        lag_x = np.arange(-(nx // 2), nx - nx // 2)
        lag_y = np.arange(-(ny // 2), ny - ny // 2)
        LX, LY = np.meshgrid(lag_x, lag_y, indexing="ij")
        R = np.sqrt(LX ** 2 + LY ** 2)
        r_max = min(nx, ny) // 2
        r_bordes = np.linspace(0, r_max, r_max + 1)
        r_centros = 0.5 * (r_bordes[1:] + r_bordes[:-1])
        indice = np.searchsorted(r_bordes, R.ravel(), side="right") - 1
        dentro = (indice >= 0) & (indice < r_max)
        conteo = np.bincount(indice[dentro], minlength=r_max).astype(np.float64)
        _CACHE_RADIAL[clave] = (r_centros, indice, dentro, conteo, r_max)
    return _CACHE_RADIAL[clave]


def longitud_correlacion(superficie):
    """Longitud de correlación lateral xi de una superficie con fronteras periódicas.

    Es el radio, en celdas, donde la función de correlación de alturas normalizada,
    promediada radialmente, cae a 1/e de su valor en r = 0 (la misma definición de
    ``correlacion_alturas_radial``). El promedio radial se vectoriza con ``np.bincount``
    porque esta función se evalúa miles de veces en los barridos.

    Parámetros
    ----------
    superficie : ndarray de enteros, forma (nx, ny)
        Altura de cada celda, en capas.

    Devuelve
    --------
    float
        xi en celdas; np.nan si la correlación no cruza 1/e dentro del intervalo radial.
    """
    nx, ny = superficie.shape
    h = superficie.astype(np.float64)
    h = h - h.mean()
    F = np.fft.fft2(h)
    acov = np.fft.fftshift(np.fft.ifft2(F * np.conj(F)).real / (nx * ny))
    r_centros, indice, dentro, conteo, r_max = _indices_radiales(nx, ny)
    suma = np.bincount(indice[dentro], weights=acov.ravel()[dentro], minlength=r_max)
    with np.errstate(invalid="ignore", divide="ignore"):
        g_radial = np.where(conteo > 0, suma / conteo, np.nan)
    g0 = g_radial[0]
    if not g0 > 0:
        return np.nan
    rho = g_radial / g0
    for i in range(len(rho) - 1):
        if rho[i] >= 1 / np.e > rho[i + 1]:
            frac = (rho[i] - 1 / np.e) / (rho[i] - rho[i + 1])
            return r_centros[i] + frac * (r_centros[i + 1] - r_centros[i])
    return np.nan


def calcular_observables(superficie):
    """Observables morfológicos de una superficie con fronteras periódicas.

    Devuelve un diccionario con:

    - ``rugosidad``: W, desviación estándar de las alturas, en capas.
    - ``maximo``: altura máxima, en capas.
    - ``vacias``: porcentaje de celdas con altura cero.
    - ``escalones``: porcentaje de enlaces entre vecinos de von Neumann cuyas alturas
      difieren en dos capas o más. Es un indicador de la tensión sobre la aproximación SOS,
      que no admite voladizos ni huecos y falla donde hay paredes verticales altas.
    - ``xi``: longitud de correlación lateral, en celdas (``longitud_correlacion``).
    """
    sup = superficie.astype(np.float64)
    salto_x = np.abs(sup - np.roll(sup, 1, axis=0))
    salto_y = np.abs(sup - np.roll(sup, 1, axis=1))
    return {
        "rugosidad": float(sup.std()),
        "maximo": int(superficie.max()),
        "vacias": float(100.0 * np.mean(superficie == 0)),
        "escalones": float(100.0 * 0.5 * (np.mean(salto_x >= 2.0) + np.mean(salto_y >= 2.0))),
        "xi": float(longitud_correlacion(superficie)),
    }


def resumir_superficie(superficie):
    """Resume una superficie terminada.

    Devuelve un diccionario con el espesor medio y la rugosidad RMS (en capas), las
    alturas máxima y mínima y el porcentaje de celdas sin cubrir (altura cero).
    """
    return {
        "espesor": superficie.mean(),
        "rugosidad": superficie.std(),
        "maximo": superficie.max(),
        "minimo": superficie.min(),
        "vacias": 100 * np.mean(superficie == 0),
    }


def exponente_efectivo(eventos, rugosidad, n_celdas, t_min_ajuste):
    """Pendiente de ln W frente a ln t (t en capas) para t >= t_min_ajuste.

    ``n_celdas`` es el número de celdas de la malla (convierte eventos en capas).
    """
    t_capas = eventos / n_celdas
    tramo = t_capas >= t_min_ajuste
    if np.count_nonzero(tramo) < 3:
        return np.nan      # muy pocas muestras en el tramo: el ajuste no es fiable
    return np.polyfit(np.log(t_capas[tramo]), np.log(rugosidad[tramo]), 1)[0]


def beta_ventana(historia_h, historia_w, h_min, h_max):
    """Pendiente beta de la regresión log(W) frente a log(h) en la ventana [h_min, h_max).

    La ventana se fija de antemano en capas (``h_min`` y ``h_max``), en lugar de buscar la
    posición que maximiza el coeficiente de determinación: maximizar R^2 sobre una
    ventana deslizante favorece intervalos cortos que, por azar en una única
    realización, parecen más lineales, sin relación necesaria con el régimen físico de
    crecimiento, y además puede seleccionar una ventana distinta en cada réplica, lo que
    impide promediar beta de forma consistente entre ellas.
    """
    mascara = (historia_h >= h_min) & (historia_h < h_max)
    if mascara.sum() < 3:
        return np.nan
    pendiente, _, _, _, _ = stats.linregress(np.log(historia_h[mascara]), np.log(historia_w[mascara]))
    return pendiente


def pendiente_local(x, y):
    """Pendiente local d ln y / d ln x de una curva muestreada (diferencias centradas en escala logarítmica).

    Aplicada a W(t), da el exponente de crecimiento efectivo en cada instante. Los extremos usan
    diferencias de un solo lado. ``x`` e ``y`` deben ser positivos y ``x`` estrictamente creciente.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.gradient(np.log(y), np.log(x))


def tiempo_de_cruce(t, w, w_sat, fraccion=0.8):
    """Primer instante en que la rugosidad alcanza una fracción de su valor de saturación.

    Interpola linealmente en ln t entre las dos muestras que rodean el cruce. En la ley de Family-Vicsek,
    el tiempo en que W alcanza una fracción fija de W_sat escala como L^z con cualquier fracción, de modo
    que sirve para estimar el exponente dinámico. Devuelve ``nan`` si la rugosidad no llega a ese valor.
    """
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    objetivo = fraccion * w_sat
    indices = np.nonzero(w >= objetivo)[0]
    if indices.size == 0:
        return float("nan")
    i = int(indices[0])
    if i == 0:
        return float(t[0])
    ln_t = np.log(t[i - 1]) + (objetivo - w[i - 1]) / (w[i] - w[i - 1]) * (np.log(t[i]) - np.log(t[i - 1]))
    return float(np.exp(ln_t))


def islas_periodicas(superficie, umbral):
    """Áreas, en celdas, de las regiones conexas de la superficie que superan una altura umbral.

    Las regiones se definen con vecinos de von Neumann (cuatro vecinos) y con fronteras periódicas: una
    región que cruza un borde de la malla se cuenta una sola vez. Con ``umbral`` igual a la altura media más
    la rugosidad, las regiones son las columnas o montículos que sobresalen de la película.

    Devuelve
    --------
    ndarray de enteros con el área de cada región, ordenadas de mayor a menor (vacío si no hay regiones).
    """
    mascara = np.asarray(superficie) > umbral
    etiquetas, n = ndimage.label(mascara)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    padre = np.arange(n + 1)

    def raiz(a):
        while padre[a] != a:
            padre[a] = padre[padre[a]]
            a = padre[a]
        return a

    def unir(a, b):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)

    # Regiones que continúan al otro lado de cada borde
    for a, b in zip(etiquetas[:, 0], etiquetas[:, -1]):
        if a and b:
            unir(a, b)
    for a, b in zip(etiquetas[0, :], etiquetas[-1, :]):
        if a and b:
            unir(a, b)
    areas = np.bincount(etiquetas.ravel(), minlength=n + 1)
    totales = {}
    for etiqueta in range(1, n + 1):
        r = raiz(etiqueta)
        totales[r] = totales.get(r, 0) + int(areas[etiqueta])
    return np.array(sorted(totales.values(), reverse=True), dtype=np.int64)
