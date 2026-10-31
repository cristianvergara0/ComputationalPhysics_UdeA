"""
Efecto magnetocalórico (EMC): variación de entropía, cambio de temperatura adiabático y RCP.

Termodinámica (unidades k_B = J1 = 1; M es la magnetización TOTAL, E la energía
total incluido el término -h M, N = L^2)
-----------------------------------------------------------------------------
Variación isotérmica de entropía al aplicar el campo H:

    Delta S_M(T, H) = S(T, H) - S(T, 0)                                        (1)

que puede obtenerse por dos caminos equivalentes en el límite continuo:

(a) Relación de Maxwell, integrando en el campo a T fijo,

    Delta S_M(T, H) = int_0^H (dM/dT)_h dh,   (dM/dT)_h = Cov(M, E) / T^2       (2)

(b) Integración de la entropía en temperatura a campo fijo, dS = dE / T,

    S(T, h) - S(T_ref, h) = - int_T^{T_ref} (1/T') (d<E>/dT') dT'              (3)

    y Delta S_M(T, H) = Delta S_M(T_ref, H) + [S(T,H) - S(T_ref,H)] - [S(T,0) - S(T_ref,0)],

    con T_ref en la fase paramagnética, donde (2) es suave en h y basta una malla
    gruesa de campos para fijar la constante Delta S_M(T_ref, H).

En la fase ordenada el sistema atraviesa, al aumentar h a T fijo, la línea de
transición inducida por el campo; ahí (dM/dT)_h tiene un pico muy agudo en T para
cada h, y la regla del trapecio en (2) con una malla discreta de campos convierte
esos picos en estructuras espurias en T cuya altura crece con L. El camino (3)
solo usa <E>(T) a cada campo, que es una función acotada y monótona, y no depende
de la malla de campos salvo a través de la constante en T_ref. Por eso (3) es el
estimador principal y (2) se usa como comprobación en la fase paramagnética.

Cambio de temperatura adiabático: el proceso adiabático conserva la entropía, de
modo que Delta T_ad(T, H) = T' - T con S(T', H) = S(T, 0) (sin linealizar), y la
aproximación habitual de primer orden es Delta T_ad ~ -T Delta S_M(T, H) / C_H(T),
con C_H = N c_h la capacidad calorífica total al campo H (Pecharsky et al., 2001).

Poder de enfriamiento relativo: RCP = |Delta S_M|_pico x FWHM (Franco et al., 2018).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .estimators import combined
from .fss import loglog_fit, parabolic_peak
from .kernels import OBS


# ----------------------------------------------------------------------------
# Datos por campo
# ----------------------------------------------------------------------------
def field_dataset(result, **criteria):
    """
    Reúne los momentos por réplica de un experimento de campo uniforme.

    Returns
    -------
    H : ndarray (nH,)            campos ordenados (debe incluir h = 0)
    T : ndarray (nT,)            temperaturas en orden ascendente
    groups : dict {h: (R, nT, NOBS)}
    """
    idx = result.select(**criteria)
    if not idx:
        raise ValueError(f"no hay tareas con {criteria}")
    H = np.array(sorted({result.tasks[i]["h"] for i in idx}))
    T0 = result.temperatures(idx[0])
    order = np.argsort(T0)
    groups = {}
    for h in H:
        sel = [i for i in idx if np.isclose(result.tasks[i]["h"], h)]
        groups[float(h)] = np.stack([result.results[i]["blocks"].mean(axis=-2)[order] for i in sel])
    if not np.isclose(H[0], 0.0):
        raise ValueError("el experimento debe incluir h = 0")
    return H, T0[order], groups


def thermodynamics(H, T, groups, L):
    """
    Magnitudes por campo para un conjunto de réplicas (``estimators.combined``).

    Returns
    -------
    dict con arreglos (nH, nT): e, mu, c_h (calor específico por espín), dMdT
    (magnetización total), y E = N e.
    """
    N = L * L
    rows = [combined(groups[float(h)], T, L) for h in H]
    out = {k: np.array([r[k] for r in rows]) for k in ("e", "mu", "Cv", "dMdT", "chi_u")}
    out["dMdT"][np.isclose(H, 0.0)] = 0.0      # <M> = 0 a h = 0 por la simetría S -> -S
    out["E"] = N * out["e"]
    return out


def delta_S_maxwell(H, dMdT):
    """Camino (2): Delta S_M(T, H) acumulado con la regla del trapecio sobre la malla de campos. (nH, nT)."""
    return cumulative_trapezoid(dMdT, H, axis=0, initial=0.0)


def entropy_relative(T, E):
    """
    Camino (3): S(T_k, h) - S(T_max, h) para cada campo, con <E> interpolada
    linealmente entre nodos, de modo que cada intervalo aporta exactamente
    (E_{k+1} - E_k) ln(T_{k+1}/T_k) / (T_{k+1} - T_k).

    T ascendente (nT,), E (nH, nT) energía total. Devuelve (nH, nT); la última
    columna (T_max) es 0.
    """
    T = np.asarray(T, dtype=float)
    seg = (E[:, 1:] - E[:, :-1]) * np.log(T[1:] / T[:-1]) / (T[1:] - T[:-1])
    S = np.zeros_like(E)
    S[:, :-1] = -np.cumsum(seg[:, ::-1], axis=1)[:, ::-1]
    return S


def delta_S_thermal(T, E, dS_maxwell_at_Tref):
    """
    Delta S_M(T, H) por integración de la entropía con T_ref = T_max:
    Delta S_M(T_max, H) (de Maxwell) + [S(T,H) - S(T_max,H)] - [S(T,0) - S(T_max,0)].
    """
    S = entropy_relative(T, E)
    return dS_maxwell_at_Tref[:, None] + S - S[0][None, :]


def delta_T_exact(T, S_tilde):
    """
    Delta T_ad(T, H) sin linealizar: T' tal que S~(T', H) = S~(T, 0), con
    S~(T, h) = S(T, h) - S(T_max, 0) (creciente en T porque C_h > 0). NaN si T'
    queda fuera de la malla.
    """
    out = np.full_like(S_tilde, np.nan)
    S0 = S_tilde[0]
    for k in range(S_tilde.shape[0]):
        Sk = S_tilde[k]
        if not np.all(np.diff(Sk) > 0):
            # Con ruido estadístico S~ podría no ser estrictamente creciente en algún tramo
            # plano; se fuerza la monotonía para poder invertirla (afecta solo a T muy bajas).
            Sk = np.maximum.accumulate(Sk + 1e-12 * np.arange(len(Sk)))
        inside = (S0 >= Sk[0]) & (S0 <= Sk[-1])
        out[k, inside] = np.interp(S0[inside], Sk, T) - T[inside]
    return out


# ----------------------------------------------------------------------------
# Picos por rama, FWHM y RCP
# ----------------------------------------------------------------------------
def _half_width(T, y, i_peak, level):
    """Anchura total a media altura alrededor de y[i_peak] (> 0), con interpolación lineal; NaN si falta un lado."""
    left = np.nan
    for i in range(i_peak, 0, -1):
        if y[i - 1] < level <= y[i]:
            left = T[i - 1] + (level - y[i - 1]) * (T[i] - T[i - 1]) / (y[i] - y[i - 1])
            break
    right = np.nan
    for i in range(i_peak, len(T) - 1):
        if y[i + 1] < level <= y[i]:
            right = T[i] + (y[i] - level) * (T[i + 1] - T[i]) / (y[i] - y[i + 1])
            break
    return right - left


def branch_peaks(T, dS, T_star, inverse_upper_margin=0.03):
    """
    Picos de las dos ramas de Delta S_M(T) a un campo dado.

    rama convencional (Delta S_M < 0): mínimo para T >= T*
    rama inversa      (Delta S_M > 0): máximo para T <= T* + margen

    Returns
    -------
    dict con T y amplitud de cada pico (posición refinada con una parábola por tres
    nodos), su FWHM y RCP = |amplitud| x FWHM, y el máximo global de |Delta S_M|.
    """
    T = np.asarray(T, dtype=float)
    dS = np.asarray(dS, dtype=float)
    out = {}
    conv = T >= T_star
    y = np.where(conv, -dS, -np.inf)
    i = int(np.argmax(y))
    if conv.any() and y[i] > 0:
        Tp, amp, _ = parabolic_peak(T[conv], -dS[conv])
        fw = _half_width(T, -dS, i, 0.5 * amp)
        out.update(T_conv=Tp, dS_conv=-amp, fwhm_conv=fw, rcp_conv=amp * fw)
    else:
        out.update(T_conv=np.nan, dS_conv=np.nan, fwhm_conv=np.nan, rcp_conv=np.nan)
    inv = T <= T_star + inverse_upper_margin
    y = np.where(inv, dS, -np.inf)
    i = int(np.argmax(y))
    if inv.any() and y[i] > 0:
        Tp, amp, _ = parabolic_peak(T[inv], dS[inv])
        fw = _half_width(T, dS, i, 0.5 * amp)
        out.update(T_inv=Tp, dS_inv=amp, fwhm_inv=fw, rcp_inv=amp * fw)
    else:
        out.update(T_inv=np.nan, dS_inv=np.nan, fwhm_inv=np.nan, rcp_inv=np.nan)
    j = int(np.argmax(np.abs(dS)))
    out.update(T_global=float(T[j]), dS_global=float(abs(dS[j])),
               global_branch="inversa" if dS[j] > 0 else "convencional")
    return out


# ----------------------------------------------------------------------------
# Análisis completo de un conjunto de campos
# ----------------------------------------------------------------------------
def analyze(H, T, groups, L, T_star):
    """
    Todo el análisis del EMC para un conjunto de campos (curvas y picos).

    Returns
    -------
    dict con
      curvas (nH, nT): e, mu, Cv, dMdT, dS_maxwell, dS (camino térmico), dTad_exact, dTad_approx
      picos por campo: lista de dicts de ``branch_peaks``
      dS_at_Tstar (nH,): Delta S_M interpolada en T*
    """
    N = L * L
    th = thermodynamics(H, T, groups, L)
    dS_max = delta_S_maxwell(H, th["dMdT"])
    dS = delta_S_thermal(T, th["E"], dS_max[:, -1])
    S_tilde = entropy_relative(T, th["E"]) + dS_max[:, -1][:, None]
    dT_exact = delta_T_exact(T, S_tilde)
    with np.errstate(divide="ignore", invalid="ignore"):
        dT_approx = -T[None, :] * dS / (N * th["Cv"])
    peaks = [branch_peaks(T, dS[k], T_star) for k in range(len(H))]
    for k, pk in enumerate(peaks):
        pk["dTad_exact_conv"] = float(np.interp(pk["T_conv"], T, dT_exact[k])) if np.isfinite(pk["T_conv"]) else np.nan
        pk["dTad_approx_conv"] = float(np.interp(pk["T_conv"], T, dT_approx[k])) if np.isfinite(pk["T_conv"]) else np.nan
        finite = np.isfinite(dT_exact[k])
        if finite.any() and k > 0:
            j = int(np.nanargmax(np.abs(np.where(finite, dT_exact[k], 0.0))))
            pk["T_dTad_global"], pk["dTad_global"] = float(T[j]), float(dT_exact[k, j])
        else:
            pk["T_dTad_global"], pk["dTad_global"] = np.nan, np.nan
    dS_at_Tstar = np.array([np.interp(T_star, T, dS[k]) for k in range(len(H))])
    return {"H": np.asarray(H), "T": T, **th, "dS_maxwell": dS_max, "dS": dS, "S_tilde": S_tilde,
            "dTad_exact": dT_exact, "dTad_approx": dT_approx, "peaks": peaks, "dS_at_Tstar": dS_at_Tstar}


def sign_change_field(H, y):
    """Campo en que y(H) pasa de positivo a negativo (interpolación lineal); NaN si no cambia de signo."""
    for k in range(1, len(H) - 1):
        if y[k] > 0 >= y[k + 1]:
            return float(H[k] + y[k] * (H[k + 1] - H[k]) / (y[k] - y[k + 1]))
    return np.nan


def field_laws(an, H_max_low=0.5, H_min=0.1):
    """
    Leyes de campo de la rama convencional y del máximo global (ajustes log-log
    sobre H >= H_min; con campos menores la rama convencional es del orden del ruido
    estadístico y su FWHM no está definida con precisión):

    n      : |Delta S_M|_conv ~ H^n (todos los campos y H <= H_max_low)
    p      : RCP_conv ~ H^p
    m      : Delta T_ad exacto en el pico convencional ~ H^m (y con la fórmula aproximada)
    n_glob : máximo global de |Delta S_M| ~ H^n (mezcla de ramas: solo como contraste)

    Devuelve dict {nombre: (exponente, R^2)}.
    """
    H = an["H"]
    pk = an["peaks"]
    sel = H >= H_min - 1e-9
    get = lambda key: np.array([abs(p[key]) for p in pk])

    def fit(y, mask):
        ok = mask & np.isfinite(y) & (y > 0)
        if ok.sum() < 3:
            return np.nan, np.nan
        a, _, r2 = loglog_fit(H[ok], y[ok])
        return a, r2

    return {
        "n": fit(get("dS_conv"), sel),
        "n_low": fit(get("dS_conv"), sel & (H <= H_max_low + 1e-9)),
        "p": fit(get("rcp_conv"), sel),
        "m": fit(get("dTad_exact_conv"), sel),
        "m_approx": fit(get("dTad_approx_conv"), sel),
        "n_glob": fit(get("dS_global"), sel),
    }


def local_exponent(H, y):
    """Exponente local d ln|y| / d ln H entre campos consecutivos; devuelve (H medio geométrico, exponente)."""
    H = np.asarray(H, dtype=float)
    y = np.abs(np.asarray(y, dtype=float))
    ok = (H > 0) & (y > 0)
    H, y = H[ok], y[ok]
    return np.sqrt(H[1:] * H[:-1]), np.diff(np.log(y)) / np.diff(np.log(H))


def scalar_summary(H, T, groups, L, T_star, H_report=(0.2, 0.4, 0.5, 1.0)):
    """
    Escalares del EMC para el bootstrap entre réplicas: exponentes de campo,
    picos de cada rama en los campos de ``H_report`` y campo en que Delta S_M(T*, H)
    cambia de signo.
    """
    an = analyze(H, T, groups, L, T_star)
    laws = field_laws(an)
    out = {k: v[0] for k, v in laws.items()}
    out["H_sign_change"] = sign_change_field(H, an["dS_at_Tstar"])
    N = L * L
    for h in H_report:
        k = int(np.argmin(np.abs(H - h)))
        if not np.isclose(H[k], h):
            continue
        pk = an["peaks"][k]
        for key in ("T_conv", "T_inv", "fwhm_conv", "fwhm_inv", "dTad_exact_conv", "dTad_approx_conv"):
            out[f"{key}@{h:g}"] = pk[key]
        for key in ("dS_conv", "dS_inv", "rcp_conv", "rcp_inv"):
            out[f"{key}@{h:g}"] = pk[key] / N
    inv = np.array([p["dS_inv"] for p in an["peaks"]])
    if np.isfinite(inv).any():
        k = int(np.nanargmax(inv))
        out["H_inv_max"] = float(H[k])
        out["dS_inv_max"] = float(inv[k] / N)
    return out


def uniform_isotherm(H, T, groups, L, T_target, H_max=0.5):
    """
    Isoterma M(h) en el nodo de la malla más cercano a T_target y ajuste
    |M|/N ~ h^{1/delta_u} con 0 < h <= H_max (con campos mayores M satura).
    Devuelve (T del nodo, mu(h), 1/delta_u, R^2).
    """
    j = int(np.argmin(np.abs(T - T_target)))
    mu = np.array([groups[float(h)].mean(axis=0)[j, OBS["mu"]] for h in H])
    ok = (H > 0) & (H <= H_max + 1e-9)
    a, _, r2 = loglog_fit(H[ok], np.abs(mu[ok]))
    return float(T[j]), mu, a, r2
