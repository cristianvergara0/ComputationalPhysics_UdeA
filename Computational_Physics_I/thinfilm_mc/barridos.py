"""Barridos de parámetros, orquestación de réplicas y análisis angular del transporte con gas.

Este módulo conecta los núcleos de ``motor`` con los cuadernos: fija los tipos de los argumentos de
las rutinas compiladas, deriva las semillas de cada punto de un barrido, resume cada réplica con
``analisis.calcular_observables`` y ofrece el análisis de la composición angular y del colapso de
la rugosidad con el ancho angular. Los parámetros del sistema se pasan siempre de forma explícita en
un objeto (``ParametrosMotor`` o ``ParametrosUnificado``); ninguna función lee variables del cuaderno.
"""

import time
import zlib
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator

from .analisis import calcular_observables
from .motor import EMISION_GAUSSIANA, ejecutar_replicas, ejecutar_suite_muestreada


@dataclass(frozen=True)
class ParametrosMotor:
    """Parámetros del sistema del motor de réplicas (``motor.ejecutar_replicas``).

    Atributos
    ---------
    nx, ny : número de celdas de la malla en cada dirección.
    t_escala : temperatura de escala de la probabilidad de salto P = min(1, T / t_escala), en kelvin.
    max_pasos_difusion : número máximo de saltos de relajación por partícula.
    sigma_dispersion : desviación angular gaussiana por colisión con el gas, en grados.
    distancia_fuente : distancia entre el techo y la altura máxima de la película (espesor de la capa de gas), en capas.
    max_pasos_rayo : salvaguarda del trazador (máximo de etapas por átomo).
    semilla : semilla de la que se derivan las semillas de todas las simulaciones de un barrido.
    """
    nx: int
    ny: int
    t_escala: float
    max_pasos_difusion: int
    sigma_dispersion: float
    distancia_fuente: float
    max_pasos_rayo: int
    semilla: int


@dataclass(frozen=True)
class ParametrosUnificado:
    """Parámetros del sistema del motor común (``ejecutar_modelo_muestreado``).

    Atributos
    ---------
    nx, ny : número de celdas de la malla en cada dirección.
    t_escala : temperatura de escala de la probabilidad de salto, en kelvin.
    max_pasos_difusion : número máximo de saltos de relajación por partícula.
    sigma_incidencia : dispersión angular de la emisión, en grados.
    sigma_dispersion : desviación angular gaussiana por colisión con el gas, en grados.
    mfp : camino libre medio del gas, en celdas.
    distancia_fuente : distancia entre el techo de vuelo y la altura máxima de la película (espesor de la capa de gas), en capas.
    """
    nx: int
    ny: int
    t_escala: float
    max_pasos_difusion: int
    sigma_incidencia: float
    sigma_dispersion: float
    mfp: float
    distancia_fuente: float


def correr_con_semillas(params, semillas, temp, sigma_ang, mfp, n_eventos, sombreado=1,
                        ley_emision=EMISION_GAUSSIANA, theta_max=0.0):
    """Ejecuta una réplica por semilla con los parámetros del sistema y devuelve los resultados del motor.

    Es la única puerta de entrada al motor compilado: fija los tipos de los argumentos (enteros y reales)
    para que Numba compile una sola vez y no una por cada combinación de tipos.

    Parámetros
    ----------
    params : ParametrosMotor
        Parámetros del sistema.
    semillas : array_like de enteros, forma (n_replicas,)
        Semilla de cada réplica.
    temp : float
        Temperatura del sustrato, en kelvin.
    sigma_ang : float
        Dispersión angular de la emisión gaussiana, en grados.
    mfp : float
        Camino libre medio, en celdas (``np.inf``: sin gas).
    n_eventos : int
        Partículas depositadas por réplica.
    sombreado : int
        1 si el relieve intercepta los rayos y 0 si no.
    ley_emision, theta_max : int, float
        Ley de emisión (``motor.EMISION_GAUSSIANA`` o ``motor.EMISION_COSENO``) y ángulo polar máximo de
        la ley del coseno, en grados.

    Devuelve
    --------
    superficies : ndarray int32, forma (n_replicas, nx, ny)
    colisiones : ndarray, forma (n_replicas,)
        Colisiones con el gas por átomo depositado.
    theta2_medio : ndarray, forma (n_replicas,)
        Promedio de (theta_x^2 + theta_y^2) / 2 en el impacto, en grados al cuadrado.
    theta2_red_medio : ndarray, forma (n_replicas,)
        Lo mismo con los ángulos reducidos al intervalo (-90, 90] grados.
    truncados : ndarray, forma (n_replicas,)
        Átomos de cada réplica cuyo vuelo detuvo la salvaguarda del trazador.
    """
    return ejecutar_replicas(
        np.asarray(semillas, dtype=np.uint32), int(params.nx), int(params.ny), int(n_eventos), float(temp),
        float(params.t_escala), int(params.max_pasos_difusion), float(sigma_ang), float(params.sigma_dispersion),
        float(mfp), float(params.distancia_fuente), int(sombreado),
        int(params.max_pasos_rayo), int(ley_emision), float(theta_max)
    )


def simular_punto(params, id_exp, i, j, temp, sigma_ang, mfp, n_replicas, n_eventos, sombreado=1,
                  ley_emision=EMISION_GAUSSIANA, theta_max=0.0):
    """Simula las réplicas de un punto del espacio de parámetros y resume cada una.

    Las semillas de las réplicas se derivan de ``params.semilla`` con la clave (id_exp, i, j), de modo
    que cada punto tiene sus propias semillas, independientes de las demás y reproducibles por
    separado: añadir o quitar puntos de un barrido no cambia los resultados de los restantes.

    Parámetros
    ----------
    params : ParametrosMotor
        Parámetros del sistema.
    id_exp : int
        Identificador del experimento (parte de la clave de semillas).
    i, j : int
        Posición del punto en la malla de parámetros del experimento (parte de la clave).
    temp : float
        Temperatura del sustrato, en kelvin.
    sigma_ang : float
        Dispersión angular de la emisión, en grados.
    mfp : float
        Camino libre medio, en celdas (np.inf: sin gas).
    n_replicas, n_eventos : int
        Número de réplicas y de partículas por réplica.
    sombreado : int
        1 si el relieve intercepta los rayos; 0 si no.
    ley_emision, theta_max : int, float
        Ley de emisión y ángulo polar máximo de la ley del coseno (ver ``correr_con_semillas``).

    Devuelve
    --------
    tabla : DataFrame
        Una fila por réplica, con los observables de ``calcular_observables`` y las columnas
        ``colisiones`` (colisiones con el gas por átomo), ``sigma_eff_med`` (raíz del promedio
        de (theta_x^2 + theta_y^2) / 2 en el impacto, en grados), ``sigma_red_med`` (lo mismo con los
        ángulos reducidos a (-90, 90] grados) y ``truncados`` (átomos que detuvo la salvaguarda).
    primera_superficie : ndarray
        Superficie de la primera réplica, para las figuras topográficas.
    """
    semillas = np.random.SeedSequence(params.semilla, spawn_key=(id_exp, i, j)).generate_state(n_replicas)
    superficies, colisiones, theta2_medio, theta2_red_medio, truncados = correr_con_semillas(
        params, semillas, temp, sigma_ang, mfp, n_eventos, sombreado, ley_emision, theta_max
    )
    # Cada evento añade exactamente una capa: si la suma no coincide con n_eventos, el resultado
    # es incorrecto aunque no se lance ninguna excepción.
    sumas = superficies.reshape(n_replicas, -1).sum(axis=1, dtype=np.int64)
    if np.any(sumas != n_eventos):
        raise RuntimeError(f"La suma de alturas no coincide con n_eventos (T = {temp}, sigma = {sigma_ang}, lambda = {mfp}, sombreado = {sombreado}).")
    filas = []
    for r in range(n_replicas):
        fila = calcular_observables(superficies[r])
        fila["replica"] = r
        fila["colisiones"] = float(colisiones[r])
        fila["sigma_eff_med"] = float(np.sqrt(theta2_medio[r]))
        fila["sigma_red_med"] = float(np.sqrt(theta2_red_medio[r]))
        fila["truncados"] = int(truncados[r])
        filas.append(fila)
    return pd.DataFrame(filas), superficies[0].copy()


def ejecutar_barrido(params, id_exp, puntos, n_replicas, n_eventos, etiqueta):
    """Ejecuta ``simular_punto`` en cada punto de un barrido y reúne los resultados en formato largo.

    Cada elemento de ``puntos`` es un diccionario con las claves ``i``, ``j``, ``temp``, ``sigma`` y ``mfp``
    y, opcionalmente, ``sombreado`` (por defecto 1), ``ley_emision`` (0), ``theta_max`` (0.0) y ``sigma_disp``
    (dispersión angular por colisión del punto, en grados; por defecto la de ``params``); todas las claves,
    también las etiquetas de los ejes, se copian a las filas del resultado.

    Devuelve
    --------
    tabla : DataFrame
        Una fila por réplica de cada punto.
    superficies : dict
        Primera superficie de cada punto, con clave (i, j).
    """
    tablas, superficies = [], {}
    t_ini = time.time()
    for k, punto in enumerate(puntos, start=1):
        params_punto = params if "sigma_disp" not in punto else replace(params, sigma_dispersion=float(punto["sigma_disp"]))
        tabla, primera = simular_punto(
            params_punto, id_exp, punto["i"], punto["j"], punto["temp"], punto["sigma"], punto["mfp"], n_replicas, n_eventos,
            punto.get("sombreado", 1), punto.get("ley_emision", EMISION_GAUSSIANA), punto.get("theta_max", 0.0)
        )
        for clave, valor in punto.items():
            tabla[clave] = valor
        tablas.append(tabla)
        superficies[(punto["i"], punto["j"])] = primera
        if k % 10 == 0 or k == len(puntos):
            print(f"  {etiqueta}: {k}/{len(puntos)} puntos ({time.time() - t_ini:.1f} s)")
    tabla_total = pd.concat(tablas, ignore_index=True)
    truncados = int(tabla_total["truncados"].sum())
    print(f"  {etiqueta}: átomos truncados por la salvaguarda del trazador: {truncados} de {len(tabla_total) * n_eventos}")
    return tabla_total, superficies


def sigma_reducida_gaussiana(sigma):
    """Valor cuadrático medio, en grados, de una gaussiana N(0, sigma^2) reducida al intervalo (-90, 90].

    Es la dispersión angular física que tendría un flujo de direcciones gaussianas de ancho ``sigma``
    una vez que se identifican los ángulos que difieren en 180 grados. Se integra numéricamente la
    densidad de la gaussiana envuelta. Crece con ``sigma`` y se satura en 90 / raíz(3), unos 52 grados,
    el valor de un flujo uniforme (difuso).
    """
    theta = np.linspace(-90.0, 90.0, 20001)
    k = np.arange(-12, 13)[:, None]
    peso = stats.norm.pdf(theta[None, :] + 180.0 * k, 0.0, sigma).sum(axis=0)
    peso = peso / peso.sum()
    return float(np.sqrt(np.sum(peso * theta ** 2)))


def resumir_vuelos(n_col, theta_x, theta_y, sigma_ang, sigma_disp):
    """Compara la dirección de llegada de vuelos libres con la ley de composición angular.

    Para cada vuelo, d = (theta_x^2 + theta_y^2) / 2 - sigma_ang^2 - sigma_disp^2 * n tiene
    esperanza cero (identidad de Wald). El estadístico z contrasta su media con cero.
    Además se calcula la raíz del promedio cuadrático de los ángulos reducidos al intervalo
    (-90, 90] grados, que son los que determinan la dirección física.
    """
    n_vuelos = len(n_col)
    cuad = 0.5 * (theta_x ** 2 + theta_y ** 2)
    d = cuad - sigma_ang ** 2 - sigma_disp ** 2 * n_col
    z = d.mean() / (d.std(ddof=1) / np.sqrt(n_vuelos))
    reducido_x = (theta_x + 90.0) % 180.0 - 90.0
    reducido_y = (theta_y + 90.0) % 180.0 - 90.0
    sigma_eff_regla = float(np.sqrt(sigma_ang ** 2 + sigma_disp ** 2 * n_col.mean()))
    return {
        "n_medio": float(n_col.mean()),
        "sigma_eff_regla": sigma_eff_regla,
        "sigma_eff_med": float(np.sqrt(cuad.mean())),
        "z_wald": float(z),
        "sigma_reducido": float(np.sqrt(0.5 * np.mean(reducido_x ** 2 + reducido_y ** 2))),
        "sigma_reducido_gauss": sigma_reducida_gaussiana(sigma_eff_regla),
    }


def descriptores_llegada(theta_x, theta_y, umbral_rasante=60.0):
    """Descriptores de la distribución angular de llegada de un conjunto de vuelos (ángulos en grados).

    La dirección de vuelo solo depende de los ángulos módulo 180 grados, de modo que los descriptores
    usan los ángulos reducidos a (-90, 90]. Con tan(polar) = sqrt(tan(theta_x)^2 + tan(theta_y)^2) se
    obtiene el ángulo polar de la dirección con respecto a la normal de la película.

    Devuelve
    --------
    dict con ``sigma_red`` (raíz del promedio de (theta_x^2 + theta_y^2) / 2 con los ángulos reducidos),
    ``polar_mediano`` (mediana del ángulo polar, en grados) y ``f_rasante`` (fracción de vuelos cuyo ángulo
    polar supera ``umbral_rasante``, es decir, el peso de las direcciones rasantes, que son las que más
    sombrean).
    """
    tx = np.asarray(theta_x, dtype=float)
    ty = np.asarray(theta_y, dtype=float)
    red_x = (tx + 90.0) % 180.0 - 90.0
    red_y = (ty + 90.0) % 180.0 - 90.0
    polar = np.degrees(np.arctan(np.hypot(np.tan(np.radians(tx)), np.tan(np.radians(ty)))))
    return {
        "sigma_red": float(np.sqrt(0.5 * np.mean(red_x ** 2 + red_y ** 2))),
        "polar_mediano": float(np.median(polar)),
        "f_rasante": float(np.mean(polar > umbral_rasante)),
    }


def curva_balistica(res_ii, res_iii, angulos_ii, angulos_iii, n_cols_ii):
    """Rugosidad del transporte balístico en función de sigma_theta.

    Reúne la columna lambda = infinito del mapa II y el experimento III. Devuelve una tabla ordenada por
    sigma con la rugosidad, su error estándar y la dispersión angular reducida medida en el impacto.
    """
    col_inf = res_ii[res_ii["j"] == n_cols_ii - 1].sort_values("i")
    ext = res_iii.sort_values("i")
    ref = pd.DataFrame({
        "sigma": np.concatenate([np.asarray(angulos_ii, dtype=float), np.asarray(angulos_iii, dtype=float)]),
        "W": np.concatenate([col_inf["rugosidad"].to_numpy(), ext["rugosidad"].to_numpy()]),
        "W_se": np.concatenate([col_inf["rugosidad_se"].to_numpy(), ext["rugosidad_se"].to_numpy()]),
        "sigma_red": np.concatenate([col_inf["sigma_red_med"].to_numpy(), ext["sigma_red_med"].to_numpy()]),
    })
    return ref.sort_values("sigma").reset_index(drop=True)


def referencia_balistica(resumen):
    """Curva balística de referencia W(sigma_theta) a partir de las simulaciones sin gas.

    Parámetros
    ----------
    resumen : DataFrame
        Una fila por punto balístico (``resumir_por_punto``) con las columnas ``sigma_theta`` (dispersión
        angular de la emisión por eje, en grados), ``rugosidad``, ``rugosidad_se`` y ``sigma_red_med``.

    Devuelve
    --------
    DataFrame ordenado por ``sigma`` con las columnas ``sigma``, ``W``, ``W_se`` y ``sigma_red``.
    """
    ref = pd.DataFrame({
        "sigma": resumen["sigma_theta"].to_numpy(dtype=float),
        "W": resumen["rugosidad"].to_numpy(),
        "W_se": resumen["rugosidad_se"].to_numpy(),
        "sigma_red": resumen["sigma_red_med"].to_numpy(),
    })
    return ref.sort_values("sigma").reset_index(drop=True)


def colapso_por_punto(puntos, ref):
    """Predice la rugosidad con gas a partir de una curva balística, con dos descriptores del ancho angular.

    Para cada punto con gas se calculan dos predicciones, sin parámetros ajustados:

    1. ``W_pred``: la curva balística evaluada en sigma_eff = sqrt(sigma_theta^2 + n * sigma_disp^2), con n las
       colisiones por átomo medidas en ese punto (la regla de composición angular). Se interpola de forma monótona
       (PCHIP; Fritsch & Carlson, 1980) y no se extrapola: fuera del intervalo de la curva la predicción es NaN.
    2. ``W_pred_red``: la rugosidad balística correspondiente a la dispersión angular reducida medida en el
       impacto, ``sigma_red_med``, interpolando de forma lineal entre los puntos balísticos (que también la miden).

    Parámetros
    ----------
    puntos : DataFrame
        Una fila por punto (``resumir_por_punto``) con las columnas ``i``, ``j``, ``lambda`` (camino libre medio),
        ``sigma_theta`` y ``sigma_disp`` (dispersión angular de la emisión y por colisión, en grados), ``colisiones``,
        ``sigma_eff_med``, ``sigma_red_med``, ``rugosidad`` y ``rugosidad_se``. Se omiten los puntos con
        ``lambda`` infinito (sin gas).
    ref : DataFrame
        Curva balística de ``referencia_balistica``.

    Devuelve
    --------
    DataFrame con una fila por punto con gas: la rugosidad medida (``W``, ``W_se``), cada predicción, su desviación
    relativa (``desv_rel`` = (W - W_pred) / W_pred) y su diferencia en unidades de error estándar (``z``).
    """
    interpolador = PchipInterpolator(ref["sigma"].to_numpy(), ref["W"].to_numpy(), extrapolate=False)
    ref_red = ref.sort_values("sigma_red")
    filas = []
    for _, r in puntos.iterrows():
        lam = r["lambda"]
        if np.isinf(lam):
            continue
        sigma = float(r["sigma_theta"])
        sigma_eff = float(np.sqrt(sigma ** 2 + float(r["sigma_disp"]) ** 2 * r["colisiones"]))
        W_pred = float(interpolador(sigma_eff))
        se_pred = float(np.interp(sigma_eff, ref["sigma"], ref["W_se"]))
        s_red = float(r["sigma_red_med"])
        dentro_red = ref_red["sigma_red"].min() <= s_red <= ref_red["sigma_red"].max()
        W_pred_red = float(np.interp(s_red, ref_red["sigma_red"], ref_red["W"])) if dentro_red else np.nan
        se_pred_red = float(np.interp(s_red, ref_red["sigma_red"], ref_red["W_se"])) if dentro_red else np.nan
        W, W_se = r["rugosidad"], r["rugosidad_se"]
        filas.append({
            "i": int(r["i"]), "j": int(r["j"]), "sigma_theta": sigma, "lambda": lam, "colisiones": r["colisiones"],
            "sigma_eff": sigma_eff, "sigma_eff_med": r["sigma_eff_med"], "sigma_red": s_red,
            "W": W, "W_se": W_se,
            "W_pred": W_pred, "desv_rel": (W - W_pred) / W_pred if np.isfinite(W_pred) else np.nan,
            "z": (W - W_pred) / np.sqrt(W_se ** 2 + se_pred ** 2) if np.isfinite(W_pred) else np.nan,
            "W_pred_red": W_pred_red, "desv_rel_red": (W - W_pred_red) / W_pred_red if np.isfinite(W_pred_red) else np.nan,
            "z_red": (W - W_pred_red) / np.sqrt(W_se ** 2 + se_pred_red ** 2) if np.isfinite(W_pred_red) else np.nan,
        })
    return pd.DataFrame(filas)


def tabla_colapso(res_ii, ref, angulos, caminos, sigma_disp):
    """Predice la rugosidad con gas de un mapa (sigma_theta, lambda) con una dispersión por colisión fija.

    Es ``colapso_por_punto`` para un mapa cuyas filas son los ángulos ``angulos`` y cuyas columnas son los
    caminos libres ``caminos``, con la misma ``sigma_disp`` en todos los puntos.
    """
    puntos = res_ii.copy()
    puntos["lambda"] = [caminos[int(j)] for j in puntos["j"]]
    puntos["sigma_theta"] = [float(angulos[int(i)]) for i in puntos["i"]]
    puntos["sigma_disp"] = float(sigma_disp)
    return colapso_por_punto(puntos, ref)


def contraste_identidad_por_punto(tabla):
    """Contrasta, punto a punto, la identidad angular en las simulaciones de depósito.

    Por réplica, d = sigma_eff_med^2 - sigma_theta^2 - sigma_disp^2 * (colisiones por átomo) tiene
    esperanza cero. Con las réplicas de cada punto (agrupadas por ``i`` y ``j``) se aplica una prueba t de
    Student de una muestra contra cero. ``tabla`` es la tabla larga de réplicas, con las columnas
    ``sigma_theta`` y ``sigma_disp`` de cada punto.
    """
    filas = []
    for (i, j), g in tabla.groupby(["i", "j"]):
        d = (g["sigma_eff_med"].to_numpy() ** 2 - g["sigma_theta"].to_numpy() ** 2
             - g["sigma_disp"].to_numpy() ** 2 * g["colisiones"].to_numpy())
        if np.isclose(d.std(ddof=1), 0.0):
            t, p = 0.0, 1.0           # sin dispersión (por ejemplo, sigma_theta = 0 sin gas)
        else:
            t, p = stats.ttest_1samp(d, 0.0)
        filas.append({"i": i, "j": j, "d_medio": d.mean(), "t": t, "p": p})
    return pd.DataFrame(filas)


def contraste_identidad(tabla_ii, angulos, sigma_disp):
    """Contrasta la identidad angular en un mapa (sigma_theta, lambda) con una dispersión por colisión fija.

    Es ``contraste_identidad_por_punto`` para un mapa cuyas filas son los ángulos ``angulos``, con la misma
    ``sigma_disp`` en todos los puntos.
    """
    tabla = tabla_ii.copy()
    tabla["sigma_theta"] = [float(angulos[int(i)]) for i in tabla["i"]]
    tabla["sigma_disp"] = float(sigma_disp)
    return contraste_identidad_por_punto(tabla)


def sigma_equivalente_coseno(theta_max, n_theta=4001, n_acimut=1440):
    """Dispersión angular por eje, en grados, de la ley del coseno acotada en [0, theta_max].

    Con el ángulo polar de densidad 2 sen(theta) cos(theta) / sen^2(theta_max) y el acimut uniforme,
    devuelve la raíz de E[(theta_x^2 + theta_y^2) / 2], donde theta_x = arctan(tan(theta) cos(phi)) y
    theta_y = arctan(tan(theta) sen(phi)) son los ángulos que usa el trazador. Es la ``sigma_ang`` de la
    emisión gaussiana con el mismo valor cuadrático medio por eje, y permite comparar las dos leyes con el
    mismo ancho angular. La integral se evalúa con la regla del punto medio.

    Parámetros
    ----------
    theta_max : float
        Ángulo polar máximo, en grados, en (0, 90).
    """
    if not 0.0 < theta_max < 90.0:
        raise ValueError("theta_max debe estar en (0, 90) grados.")
    sen_max = np.sin(np.radians(theta_max))
    # Variable u = sen^2(theta) / sen^2(theta_max), uniforme en (0, 1)
    u = (np.arange(n_theta) + 0.5) / n_theta
    theta = np.arcsin(sen_max * np.sqrt(u))
    acimut = (np.arange(n_acimut) + 0.5) * (2.0 * np.pi / n_acimut)
    pendiente = np.tan(theta)[:, None]
    tx = np.arctan(pendiente * np.cos(acimut)[None, :])
    ty = np.arctan(pendiente * np.sin(acimut)[None, :])
    return float(np.degrees(np.sqrt(0.5 * np.mean(tx ** 2 + ty ** 2))))


def theta_max_para_sigma(sigma):
    """Ángulo polar máximo, en grados, de la ley del coseno con dispersión por eje ``sigma`` (inversa de ``sigma_equivalente_coseno``)."""
    from scipy.optimize import brentq

    if not 0.0 < sigma < sigma_equivalente_coseno(89.9):
        raise ValueError("sigma fuera del intervalo alcanzable por la ley del coseno acotada.")
    return float(brentq(lambda t: sigma_equivalente_coseno(t) - sigma, 1e-3, 89.9, xtol=1e-10))


def semillas_de_modelo(semilla_raiz, nombre, n_replicas):
    """Semillas de las réplicas de un modelo, derivadas de la semilla raíz y del nombre del modelo.

    Usa ``SeedSequence([semilla_raiz, crc32(nombre)])``: ``zlib.crc32`` es determinista (a diferencia de
    ``hash``, que cambia en cada sesión de Python), de modo que el mismo nombre da las mismas semillas en
    cualquier cuaderno y en cualquier sesión. Las primeras n semillas no dependen de cuántas se pidan.
    """
    return np.random.SeedSequence([semilla_raiz, zlib.crc32(nombre.encode("utf-8"))]).generate_state(n_replicas)


def malla_de_muestreo(n_celdas, espesor_maximo, n_logaritmicos, espesores_instantaneas=(), desde_lineal=None, n_lineales=0,
                      espesor_minimo=0.1):
    """Instantes de registro, en eventos, para ``ejecutar_modelo_muestreado``.

    Combina ``n_logaritmicos`` instantes espaciados de forma logarítmica entre ``espesor_minimo`` y
    ``espesor_maximo`` (en capas), los espesores de ``espesores_instantaneas`` (que además guardan la
    superficie) y, si se indica ``desde_lineal``, ``n_lineales`` instantes equiespaciados entre ese espesor y
    el máximo, útiles para promediar en el tiempo una rugosidad saturada. Cada espesor se convierte en un
    número entero de eventos, ``round(espesor * n_celdas)``. Los instantes logarítmicos que quedan a menos de
    medio paso logarítmico de una instantánea se omiten: dos instantes casi iguales amplificarían el ruido de
    las derivadas numéricas, como el exponente de crecimiento efectivo.

    Devuelve
    --------
    eventos : ndarray int64
        Instantes de registro, crecientes y sin repetir; el último es ``round(espesor_maximo * n_celdas)``.
    es_instantanea : ndarray bool
        Qué instantes guardan una copia de la superficie.
    """
    n_max = int(round(espesor_maximo * n_celdas))
    logaritmicos = np.round(np.geomspace(espesor_minimo * n_celdas, n_max, n_logaritmicos))
    instantaneas = np.array([int(round(e * n_celdas)) for e in espesores_instantaneas], dtype=np.int64)
    if np.any(instantaneas < 1) or np.any(instantaneas > n_max):
        raise ValueError("Cada espesor de instantánea debe estar entre una partícula y el espesor máximo.")
    if instantaneas.size and n_logaritmicos > 1:
        medio_paso = 0.5 * np.log(n_max / (espesor_minimo * n_celdas)) / (n_logaritmicos - 1)
        distancia = np.abs(np.log(np.maximum(logaritmicos, 1.0))[:, None] - np.log(instantaneas)[None, :])
        logaritmicos = logaritmicos[(distancia >= medio_paso).all(axis=1)]
    candidatos = [logaritmicos, instantaneas]
    if desde_lineal is not None and n_lineales > 0:
        candidatos.append(np.round(np.linspace(desde_lineal * n_celdas, n_max, n_lineales)))
    eventos = np.unique(np.concatenate(candidatos).astype(np.int64))
    eventos = eventos[(eventos >= 1) & (eventos <= n_max)]
    if eventos[-1] != n_max:
        eventos = np.append(eventos, n_max)
    return eventos, np.isin(eventos, instantaneas)


def ejecutar_modelo_muestreado(params, nombre, flags, n_replicas, semilla_raiz, eventos, es_instantanea,
                               max_pasos_rayo=1_000_000, relieve_maximo=np.inf):
    """Ejecuta un modelo del motor común registrando la superficie en los instantes ``eventos``.

    Las semillas se derivan de ``semilla_raiz`` y del nombre del modelo (``semillas_de_modelo``), de
    modo que, con el mismo nombre y la misma semilla raíz, las réplicas son las mismas: mientras los
    parámetros físicos coincidan, la superficie tras un número dado de eventos es idéntica. Como las semillas
    se obtienen con ``generate_state``, las primeras n réplicas de una ejecución con más réplicas son las
    mismas que las de una ejecución con n. El número de partículas de cada réplica es el último instante de
    ``eventos``, salvo que actúe el corte por relieve (ver ``motor.ejecutar_motor_muestreado``).

    Parámetros
    ----------
    params : ParametrosUnificado
        Parámetros del sistema.
    nombre : str
        Identificador del modelo; entra en la derivación de las semillas.
    flags : dict
        Banderas físicas (``enable_angle``, ``enable_scattering``, ``enable_shadowing``, ``enable_diffusion``)
        y temperatura ``temp``. De forma opcional, ``ley_emision`` (``EMISION_GAUSSIANA``, por defecto, o
        ``EMISION_COSENO``), ``theta_max`` (ángulo máximo de la ley del coseno, en grados) y ``enable_lateral``
        (adhesión lateral, por defecto desactivada; ver ``motor.ejecutar_motor_volumen``).
    n_replicas, semilla_raiz : int
        Número de réplicas y semilla raíz.
    eventos, es_instantanea : ndarray
        Instantes de registro y cuáles guardan la superficie (ver ``malla_de_muestreo``).
    max_pasos_rayo : int
        Salvaguarda del trazador de rayos.
    relieve_maximo : float
        Corte por relieve, en capas: cada réplica se detiene cuando su columna más alta supera la altura
        media de las columnas en más de este valor. Por defecto no hay corte.

    Devuelve
    --------
    dict con ``eventos``, ``espesor`` (altura media de las columnas), ``rugosidad``, ``relieve`` (altura de la
    columna más alta sobre la altura media), ``huecos`` (posiciones vacías bajo la cima de las columnas) y
    ``porosidad`` (fracción de huecos en el volumen bajo la cima), de forma (n_replicas, k) y con ``nan`` tras
    el corte,
    ``instantaneas`` (forma (n_replicas, m, nx, ny)), ``eventos_instantaneas``, ``instantanea_valida`` (forma
    (n_replicas, m): si la réplica llegó a ese instante), ``superficies`` al terminar, ``depositados`` y
    ``espesor_final`` de cada réplica, ``detenida`` (si actuó el corte), ``colisiones_por_atomo``, ``truncados``
    y ``segundos``.
    """
    eventos = np.asarray(eventos, dtype=np.int64)
    es_instantanea = np.asarray(es_instantanea, dtype=np.bool_)
    if np.any(np.diff(eventos) <= 0) or eventos[0] < 1:
        raise ValueError("Los instantes de registro deben ser crecientes, sin repetir y positivos.")
    if not relieve_maximo > 0:
        raise ValueError("El corte por relieve debe ser positivo.")
    n_eventos = int(eventos[-1])
    n_celdas = params.nx * params.ny
    semillas = semillas_de_modelo(semilla_raiz, nombre, n_replicas)

    t_i = time.time()
    lateral = bool(flags.get("enable_lateral", False))
    superficies, espesor, rugosidad, maximo, huecos, instantaneas, colisiones, truncados, depositados = ejecutar_suite_muestreada(
        semillas, params.nx, params.ny, n_eventos, float(flags["temp"]), float(params.t_escala),
        int(params.max_pasos_difusion), float(params.sigma_incidencia), float(params.sigma_dispersion), float(params.mfp),
        float(params.distancia_fuente), flags["enable_angle"], flags["enable_scattering"],
        flags["enable_shadowing"], flags["enable_diffusion"], eventos, es_instantanea, int(max_pasos_rayo),
        float(relieve_maximo), int(flags.get("ley_emision", EMISION_GAUSSIANA)), float(flags.get("theta_max", 0.0)),
        lateral
    )
    segundos = time.time() - t_i

    # Conservación del número de partículas, exacta, al terminar y en cada instantánea alcanzada: la suma de
    # alturas es el número de partículas más el de huecos (que solo existen con adhesión lateral).
    eventos_inst = eventos[es_instantanea]
    valida = eventos_inst[None, :] <= depositados[:, None]
    sumas_finales = superficies.reshape(n_replicas, -1).sum(axis=1)
    huecos_finales = sumas_finales - depositados
    if np.any(huecos_finales < 0) or (not lateral and np.any(huecos_finales != 0)):
        raise RuntimeError(f"{nombre}: la suma de alturas final no coincide con el número de partículas depositadas.")
    sumas = instantaneas.reshape(n_replicas, eventos_inst.size, n_celdas).sum(axis=2)
    huecos_inst = huecos[:, es_instantanea]
    esperado = np.broadcast_to(eventos_inst, valida.shape) + np.where(valida, huecos_inst, 0)
    if np.any(sumas[valida] != esperado[valida]):
        raise RuntimeError(f"{nombre}: la suma de alturas de alguna instantánea no coincide con el número de partículas.")

    detenida = depositados < n_eventos
    texto_corte = ""
    if np.isfinite(relieve_maximo):
        texto_corte = f"; detenidas por el corte de relieve ({relieve_maximo:g} capas): {int(detenida.sum())}"
    print(f" -> {nombre} ({params.nx} x {params.ny}): {n_replicas} réplicas de {n_eventos / n_celdas:g} capas "
          f"en {segundos:.1f} s; átomos truncados por el trazador: {int(truncados.sum())}{texto_corte}")
    with np.errstate(invalid="ignore", divide="ignore"):
        porosidad = huecos / (eventos[None, :] + huecos)
    return {"eventos": eventos, "espesor": espesor, "rugosidad": rugosidad, "relieve": maximo - espesor,
            "huecos": huecos, "porosidad": porosidad, "huecos_finales": huecos_finales,
            "instantaneas": instantaneas,
            "eventos_instantaneas": eventos_inst, "instantanea_valida": valida, "superficies": superficies,
            "depositados": depositados, "espesor_final": depositados / n_celdas, "detenida": detenida,
            "colisiones_por_atomo": colisiones, "truncados": truncados, "segundos": segundos}
