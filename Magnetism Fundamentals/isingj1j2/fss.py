"""
Escalado de tamaño finito (FSS): picos, cruces, pendientes, colapsos y exponentes.

Entrada común: ``data = {L: (T, mom)}``, con T ascendente (nT,) y ``mom`` los
momentos por réplica (R, nT, NOBS). Las magnitudes se evalúan con
``estimators.combined`` y los errores de todo lo que se deriva de ellas se obtienen
con ``estimators.bootstrap`` remuestreando réplicas dentro de cada tamaño.

Relaciones de escala usadas (Binder, 1981; Landau y Binder, 2021)
--------------------------------------------------------------------
Cerca de T_c y para L grande,

    xi/L, U          = f((T - T_c) L^{1/nu})       -> se cruzan en T_c; pendiente ~ L^{1/nu}
    chi_max          ~ L^{gamma/nu}
    Cv_max           ~ L^{alpha/nu}                 (logaritmo si alpha = 0)
    <m>(T_c)         ~ L^{-beta/nu}

Todas son relaciones asintóticas: con tamaños moderados las correcciones al
escalado pueden ser comparables a los errores estadísticos, y los exponentes
obtenidos deben leerse como exponentes efectivos de la ventana de tamaños usada.
"""

import numpy as np

from .estimators import combined

# Referencias para comparar cocientes de exponentes (L -> infinito).
# Ising 2D (Onsager, 1944; Yang, 1952): nu = 1, gamma = 7/4, beta = 1/8, alpha = 0 (log).
# Potts de 4 estados (Baxter, 1982; Wu, 1982): nu = 2/3, alpha = 2/3, gamma = 7/6, beta = 1/12.
# A lo largo de la línea crítica de Ashkin-Teller cambia nu, pero gamma/nu = 7/4 y beta/nu = 1/8
# se mantienen (universalidad débil; Suzuki, 1974). Primer orden en d = 2: exponentes de
# "discontinuidad", 1/nu = alpha/nu = gamma/nu = d y beta/nu = 0 (Fisher y Berker, 1982).
REFERENCE_EXPONENTS = {
    "Ising 2D": {"inv_nu": 1.0, "gamma_nu": 1.75, "alpha_nu": 0.0, "beta_nu": 0.125},
    "Potts 4 estados": {"inv_nu": 1.5, "gamma_nu": 1.75, "alpha_nu": 1.0, "beta_nu": 0.125},
    "Primer orden (d = 2)": {"inv_nu": 2.0, "gamma_nu": 2.0, "alpha_nu": 2.0, "beta_nu": 0.0},
}


# ----------------------------------------------------------------------------
# Herramientas sobre curvas
# ----------------------------------------------------------------------------
def pooled_observables(T, mom, L):
    """Magnitudes de un conjunto de réplicas, mom (R, nT, NOBS) -> dict (nT,) (ver estimators.combined)."""
    return combined(mom, T, L)


def parabolic_peak(T, y, half=1):
    """
    Posición y altura del máximo de y(T): parábola por los 2*half + 1 nodos
    centrados en el máximo discreto. Si el máximo está en el borde, si la
    parábola no es cóncava o si su vértice cae fuera de esos nodos, devuelve el
    nodo discreto (y se marca con ``edge``).

    Returns
    -------
    T_peak, y_peak, edge : float, float, bool
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    i = int(np.nanargmax(y))
    lo, hi = i - half, i + half + 1
    if lo < 0 or hi > len(T):
        return float(T[i]), float(y[i]), True
    a, b, c = np.polyfit(T[lo:hi] - T[i], y[lo:hi], 2)
    if a >= 0:
        return float(T[i]), float(y[i]), False
    t = -b / (2.0 * a)
    if not (T[lo] - T[i] <= t <= T[hi - 1] - T[i]):
        return float(T[i]), float(y[i]), False
    return float(T[i] + t), float(c + b * t + a * t * t), False


def find_crossing(T, y_small, y_large, y_range=None, t_window=None):
    """
    Temperatura en que la curva del tamaño mayor cruza a la del menor, es decir,
    en que y_large - y_small pasa de positiva (fase ordenada: U y xi/L crecen con
    L) a negativa (fase desordenada), con interpolación lineal en el intervalo.

    ``y_range`` limita la búsqueda a nodos donde ambas curvas están en ese rango:
    en la fase ordenada U satura en 2/3 y xi/L crece sin límite, y ahí las curvas
    se cruzan por ruido. ``t_window`` limita además el intervalo de temperaturas.
    Si hay varios cambios de signo se toma el de mayor caída de la diferencia; si
    no hay ninguno, el cero de una recta ajustada a la diferencia (solo si su
    pendiente es negativa y el cero cae en la ventana). NaN si no hay cruce.
    """
    T = np.asarray(T, dtype=float)
    ys = np.asarray(y_small, dtype=float)
    yl = np.asarray(y_large, dtype=float)
    diff = yl - ys
    ok = np.isfinite(diff)
    if t_window is not None:
        ok &= (T >= t_window[0]) & (T <= t_window[1])
    if y_range is not None:
        lo, hi = y_range
        ok &= (ys >= lo) & (ys <= hi) & (yl >= lo) & (yl <= hi)
    best = None
    for i in range(len(T) - 1):
        if ok[i] and ok[i + 1] and diff[i] > 0.0 >= diff[i + 1]:
            drop = diff[i] - diff[i + 1]
            if best is None or drop > best[0]:
                best = (drop, i)
    if best is not None:
        i = best[1]
        return float(T[i] + diff[i] * (T[i + 1] - T[i]) / (diff[i] - diff[i + 1]))
    if ok.sum() >= 3:
        slope, intercept = np.polyfit(T[ok], diff[ok], 1)
        if slope < 0.0:
            t0 = -intercept / slope
            if T[ok].min() <= t0 <= T[ok].max():
                return float(t0)
    return np.nan


def slope_near(T, y, T0, n_side=2):
    """
    dy/dT en T0: parábola ajustada a los 2*n_side + 1 nodos más cercanos a T0
    (una recta sesgaría la pendiente de los tamaños grandes, cuyas curvas son muy
    curvas en la ventana crítica).
    """
    T = np.asarray(T, dtype=float)
    idx = np.argsort(np.abs(T - T0))[: 2 * n_side + 1]
    coef = np.polyfit(T[idx] - T0, np.asarray(y, dtype=float)[idx], 2)
    return float(coef[1])


def loglog_fit(x, y):
    """Ajuste por mínimos cuadrados de log y = a log x + b; devuelve (a, b, R^2)."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    a, b = np.polyfit(lx, ly, 1)
    resid = ly - (a * lx + b)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else np.nan
    return float(a), float(b), float(r2)


def weighted_loglog_fit(x, y, yerr):
    """
    Ajuste ponderado de log y = a log x + b con pesos (y / yerr)^2.

    Devuelve (a, error de a, b, chi^2 por grado de libertad). Si chi^2/gdl > 1, el
    error de la pendiente se multiplica por su raíz: los errores de entrada son
    entonces menores que la dispersión real de los puntos respecto de la recta, y
    el error sin corregir sería optimista.
    """
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    sig = np.asarray(yerr, dtype=float) / np.asarray(y, dtype=float)
    w = 1.0 / sig ** 2
    A = np.vstack([lx, np.ones_like(lx)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, b = cov @ (A.T @ (w * ly))
    dof = len(lx) - 2
    chi2_dof = float(np.sum(w * (ly - (a * lx + b)) ** 2) / dof) if dof > 0 else np.nan
    err = float(np.sqrt(cov[0, 0]) * (np.sqrt(chi2_dof) if np.isfinite(chi2_dof) and chi2_dof > 1 else 1.0))
    return float(a), err, float(b), chi2_dof


def size_pairs(L_list, min_ratio=1.3):
    """Pares (L1, L2) con L2/L1 >= min_ratio: pares de tamaños casi iguales dan cruces muy ruidosos."""
    L_list = sorted(L_list)
    return [(a, b) for i, a in enumerate(L_list) for b in L_list[i + 1:] if b / a >= min_ratio]


# ----------------------------------------------------------------------------
# Colapso
# ----------------------------------------------------------------------------
def collapse_cost(curves_by_L, T, T_c, inv_nu, half_width, deg=4):
    """
    Calidad del colapso de y(T, L) contra x = (T - T_c) L^{1/nu}: error cuadrático
    medio de UN solo polinomio de grado ``deg`` ajustado a todos los tamaños, con
    los mismos nodos (|T - T_c| <= half_width) para cualquier valor de prueba.
    """
    sel = np.abs(T - T_c) <= half_width
    xs = np.concatenate([(T[sel] - T_c) * L ** inv_nu for L in curves_by_L])
    ys = np.concatenate([np.asarray(curves_by_L[L])[sel] for L in curves_by_L])
    if len(xs) <= deg + 1:
        return np.nan
    coef = np.polyfit(xs, ys, deg)
    return float(np.mean((ys - np.polyval(coef, xs)) ** 2))


def best_collapse(curves_by_L, T, T_center, half_width, dT=0.006, inv_nu_grid=None):
    """(costo, T_c, 1/nu) que minimizan ``collapse_cost`` en una malla alrededor de T_center."""
    inv_nu_grid = np.linspace(0.8, 2.4, 33) if inv_nu_grid is None else inv_nu_grid
    best = (np.inf, np.nan, np.nan)
    for T_c in T_center + np.linspace(-dT, dT, 13):
        for inv_nu in inv_nu_grid:
            c = collapse_cost(curves_by_L, T, T_c, inv_nu, half_width)
            if np.isfinite(c) and c < best[0]:
                best = (c, float(T_c), float(inv_nu))
    return best


# ----------------------------------------------------------------------------
# Estadísticos de escalado de tamaño finito
# ----------------------------------------------------------------------------
def fss_statistics(data, L1_min=16, xi_range=(0.15, 1.8), U_range=(0.15, 0.655), U_halfwindow=0.05,
                   collapse_halfwidth=0.035, do_collapse=True, chi_sizes=None, cv_sizes=None):
    """
    Todas las magnitudes de FSS a partir de ``data = {L: (T, mom)}`` (mismo T para todos los L).

    T* es la mediana de los cruces de xi/L entre los pares con L1 >= L1_min (los
    menos afectados por correcciones al escalado). Los cruces de U se buscan solo
    en T* +- U_halfwindow, porque fuera de ahí las curvas de U de distintos L
    coinciden dentro del ruido y pueden cruzarse espuriamente.

    ``chi_sizes`` y ``cv_sizes`` limitan los tamaños de los ajustes de gamma/nu y
    alpha/nu (por defecto, todos): un máximo que queda en el borde de la malla
    térmica no es un máximo y sesgaría la pendiente.

    Returns
    -------
    dict de escalares: posiciones y alturas de los picos de chi y Cv por tamaño,
    cruces por pares, T*, pendientes, exponentes efectivos y parámetros del colapso.
    """
    L_list = sorted(data)
    T = data[L_list[0]][0]
    for L in L_list:
        if len(data[L][0]) != len(T) or not np.allclose(data[L][0], T):
            raise ValueError("fss_statistics requiere la misma malla térmica para todos los tamaños")
    obs = {L: pooled_observables(T, data[L][1], L) for L in L_list}
    pairs = size_pairs(L_list)
    out = {}
    for L in L_list:
        out[f"Tchi_{L}"], out[f"chimax_{L}"], _ = parabolic_peak(T, obs[L]["chi"])
        out[f"TCv_{L}"], out[f"Cvmax_{L}"], _ = parabolic_peak(T, obs[L]["Cv"])
    for a, b in pairs:
        out[f"Txi_{a}_{b}"] = find_crossing(T, obs[a]["xi_L"], obs[b]["xi_L"], y_range=xi_range)
    T_star = np.nanmedian([out[f"Txi_{a}_{b}"] for a, b in pairs if a >= L1_min])
    if not np.isfinite(T_star):
        raise ValueError("no se encontró ningún cruce de xi/L")
    out["T_star"] = float(T_star)
    for a, b in pairs:
        out[f"TU_{a}_{b}"] = find_crossing(T, obs[a]["U"], obs[b]["U"], y_range=U_range,
                                           t_window=(T_star - U_halfwindow, T_star + U_halfwindow))
    out["T_star_U"] = float(np.nanmedian([out[f"TU_{a}_{b}"] for a, b in pairs if a >= L1_min]))
    for L in L_list:
        out[f"dxi_{L}"] = slope_near(T, obs[L]["xi_L"], T_star)
        out[f"dU_{L}"] = slope_near(T, obs[L]["U"], T_star)
        out[f"m_{L}"] = float(np.interp(T_star, T, obs[L]["m"]))
        out[f"xiL_{L}"] = float(np.interp(T_star, T, obs[L]["xi_L"]))
    L_chi = L_list if chi_sizes is None else sorted(chi_sizes)
    L_cv = L_list if cv_sizes is None else sorted(cv_sizes)
    out["gamma_nu"] = loglog_fit(L_chi, [out[f"chimax_{L}"] for L in L_chi])[0]
    out["alpha_nu"] = loglog_fit(L_cv, [out[f"Cvmax_{L}"] for L in L_cv])[0]
    out["inv_nu_xi"] = loglog_fit(L_list, [abs(out[f"dxi_{L}"]) for L in L_list])[0]
    out["inv_nu_U"] = loglog_fit(L_list, [abs(out[f"dU_{L}"]) for L in L_list])[0]
    out["beta_nu"] = -loglog_fit(L_list, [out[f"m_{L}"] for L in L_list])[0]
    if do_collapse:
        xi_curves = {L: obs[L]["xi_L"] for L in L_list}
        _, out["T_collapse"], out["inv_nu_collapse"] = best_collapse(xi_curves, T, T_star, collapse_halfwidth)
    return out


def pair_crossings(data, pairs, xi_range=(0.15, 1.8), U_range=(0.15, 0.664)):
    """
    Cruces de U y de xi/L para los pares de tamaños dados (diagrama de fase): dict de escalares.

    Para g grande el valor de U en el cruce se acerca a 2/3 (límite de dos redes de Ising
    desacopladas), por eso el rango admitido para U llega hasta 0.664; la ventana de
    temperaturas alrededor del cruce de xi/L evita los cruces espurios de la fase ordenada.
    """
    out = {}
    obs = {L: pooled_observables(data[L][0], data[L][1], L) for L in data}
    T = data[sorted(data)[0]][0]
    for a, b in pairs:
        out[f"Txi_{a}_{b}"] = find_crossing(T, obs[a]["xi_L"], obs[b]["xi_L"], y_range=xi_range)
        ref = out[f"Txi_{a}_{b}"]
        # Los cruces de U se buscan cerca del de xi/L (+-4 %): lejos de la transición las
        # curvas de U de dos tamaños coinciden dentro del ruido y pueden cruzarse por azar.
        window = None if not np.isfinite(ref) else (0.96 * ref, 1.04 * ref)
        out[f"TU_{a}_{b}"] = find_crossing(T, obs[a]["U"], obs[b]["U"], y_range=U_range, t_window=window)
    for L in data:
        out[f"Tchi_{L}"], out[f"chimax_{L}"], _ = parabolic_peak(T, obs[L]["chi"])
    return out


def extrapolate_inverse_L(L_values, T_values):
    """Ajuste lineal T(L) = T_inf + a / L; devuelve (T_inf, a). Corrimiento esperado con FBC."""
    x = 1.0 / np.asarray(L_values, dtype=float)
    a, T_inf = np.polyfit(x, np.asarray(T_values, dtype=float), 1)
    return float(T_inf), float(a)


def effective_exponent_fit(T, y, T_c, eps_min, eps_max, side="below"):
    """
    Exponente efectivo de un ajuste directo y ~ |eps|^x en una ventana
    eps_min <= |T - T_c|/T_c <= eps_max por debajo ('below') o por encima ('above')
    de T_c. Devuelve (x, R^2, número de puntos). Sirve para ilustrar por qué los
    ajustes directos en un solo tamaño no dan exponentes fiables: el resultado
    depende de T_c y de la ventana, y cerca de T_c la curva está redondeada por el
    tamaño finito.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    eps = (T - T_c) / T_c
    sel = (np.abs(eps) >= eps_min) & (np.abs(eps) <= eps_max) & (y > 0)
    sel &= (eps < 0) if side == "below" else (eps > 0)
    if sel.sum() < 3:
        return np.nan, np.nan, int(sel.sum())
    a, _, r2 = loglog_fit(np.abs(eps[sel]), y[sel])
    return a, r2, int(sel.sum())
