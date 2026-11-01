"""Estadística de las réplicas: bondad de ajuste, intervalos de confianza, interacciones y contrastes.

Reúne el análisis estadístico que comparten los cuadernos: la prueba chi-cuadrado frente a una
distribución de Poisson, los intervalos de confianza de las réplicas, los residuos de interacción y
el análisis de varianza de los mapas de dos parámetros, los efectos de un diseño factorial, el
contraste de Welch con su comprobación de normalidad y el ajuste de una recta a medias con errores
conocidos.
"""

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson


def estadistico_chi2_poisson(superficie):
    """Estadistico chi-cuadrado entre la distribucion empirica de alturas de una superficie y una
    Poisson cuya media es la altura media de esa superficie.

    Los conteos se agrupan por la derecha hasta que cada intervalo tenga una frecuencia esperada
    >= 5 (regla de Cochran). Devuelve (estadistico, n_intervalos).
    """
    alturas = superficie.ravel().astype(int)
    n = alturas.size
    conteos_obs = np.bincount(alturas)
    probs = poisson.pmf(np.arange(len(conteos_obs)), alturas.mean())
    prob_cola = 1.0 - probs.sum()
    esperado = probs * n

    obs_agrup, esp_agrup = [], []
    acum_obs = acum_esp = 0.0
    for o, e in zip(conteos_obs, esperado):
        acum_obs += o
        acum_esp += e
        if acum_esp >= 5:
            obs_agrup.append(acum_obs)
            esp_agrup.append(acum_esp)
            acum_obs = acum_esp = 0.0
    acum_esp += prob_cola * n
    if acum_esp > 0 or acum_obs > 0:
        if obs_agrup and acum_esp < 5:
            obs_agrup[-1] += acum_obs
            esp_agrup[-1] += acum_esp
        else:
            obs_agrup.append(acum_obs)
            esp_agrup.append(acum_esp)
    obs_agrup = np.array(obs_agrup, dtype=float)
    esp_agrup = np.array(esp_agrup, dtype=float) * (obs_agrup.sum() / np.sum(esp_agrup))
    return float(np.sum((obs_agrup - esp_agrup) ** 2 / esp_agrup)), len(obs_agrup)


def prueba_bondad_ajuste_poisson(superficie):
    """Bondad de ajuste chi-cuadrado de una superficie frente a Poisson.

    Grados de libertad = n_intervalos - 2: uno por la normalizacion y otro porque la media de la
    Poisson sale de los propios datos. En una malla completa esa media no es aleatoria -- el numero
    total de particulas esta fijado, porque cada evento anade exactamente una --, y fijarla resta
    un grado de libertad igual que estimarla; en una region interior, en cambio, la media si
    fluctua entre replicas. En ambos casos n_intervalos - 2 es lo correcto.

    Devuelve (estadistico, valor_p, grados_libertad, n_intervalos).
    """
    estadistico, n_intervalos = estadistico_chi2_poisson(superficie)
    gl = n_intervalos - 2
    return estadistico, float(stats.chi2.sf(estadistico, gl)), gl, n_intervalos


def resumen_bondad_ajuste(superficies):
    """Aplica la prueba a cada superficie y resume: fraccion de rechazos al 5 %, prueba conjunta
    (los estadisticos de replicas independientes suman chi-cuadrado con la suma de sus grados de
    libertad) y uniformidad de los valores p (bajo H0 son uniformes en [0, 1])."""
    resultados = [prueba_bondad_ajuste_poisson(s) for s in superficies]
    estadisticos = np.array([r[0] for r in resultados])
    valores_p = np.array([r[1] for r in resultados])
    grados = np.array([r[2] for r in resultados])
    n = len(valores_p)
    n_rechazos = int(np.sum(valores_p < 0.05))
    return {
        "n": n, "valores_p": valores_p, "n_rechazos": n_rechazos,
        "p_binomial": stats.binom.sf(n_rechazos - 1, n, 0.05),   # P(>= n_rechazos | modelo correcto)
        "chi2_conjunto": estadisticos.sum(), "gl_conjunto": int(grados.sum()),
        "p_conjunto": stats.chi2.sf(estadisticos.sum(), grados.sum()),
        "p_uniformidad": stats.kstest(valores_p, "uniform").pvalue,
    }


def imprimir_resumen_bondad(titulo, r, ancho, fila):
    """Imprime el resumen de ``resumen_bondad_ajuste``.

    ``ancho`` es el ancho de la tabla en caracteres y ``fila`` la función que imprime cada línea
    (por ejemplo, la que devuelve ``entorno.crear_fila``).
    """
    print("=" * ancho)
    print(f"{titulo} ({r['n']} réplicas)".center(ancho))
    print("=" * ancho)
    p = r["valores_p"]
    fila("Valor p por réplica: mínimo / mediana / máximo",
         f"{p.min():.3f} / {np.median(p):.3f} / {p.max():.3f}")
    fila("Réplicas que rechazan Poisson al 5%",
         f"{r['n_rechazos']} de {r['n']} (esperado por azar: {0.05 * r['n']:.2f})")
    fila("P(al menos ese número de rechazos | H0)", f"{r['p_binomial']:.3f}")
    fila("Chi-cuadrado conjunto / grados de libertad", f"{r['chi2_conjunto']:.1f} / {r['gl_conjunto']}")
    fila("Valor p conjunto (réplicas independientes)", f"{r['p_conjunto']:.3f}")
    fila("Uniformidad de los valores p (Kolmogorov-Smirnov)", f"p = {r['p_uniformidad']:.3f}")
    print("=" * ancho)


def media_ic95(valores):
    """Media de las réplicas y semiancho de su intervalo de confianza del 95 % (t de Student).

    Ignora los valores no finitos. Con una sola réplica el semiancho es ``nan``.
    """
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    n = valores.size
    media = valores.mean()
    ic = stats.t.ppf(0.975, n - 1) * valores.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return media, ic


def contraste_welch(x, y):
    """Prueba t de Welch entre dos muestras de réplicas y comprobación de su normalidad.

    La prueba de Welch no supone varianzas iguales, pero sí que la media de cada muestra se
    distribuya de forma aproximadamente normal. Por eso se devuelve también el valor p de la
    prueba de Shapiro-Wilk de cada muestra. Con menos de unas ocho réplicas esa prueba tiene muy
    poca potencia: un valor p alto no demuestra normalidad, y un valor p bajo con pocas réplicas
    indica una desviación grande.

    Parámetros
    ----------
    x, y : array_like
        Valores de la misma magnitud (por ejemplo, la rugosidad final) en cada réplica de dos
        condiciones. Cada muestra necesita al menos dos valores.

    Devuelve
    --------
    dict con la media de cada muestra, su diferencia, el estadístico ``t``, el valor ``p`` de dos
    colas, los grados de libertad de Welch-Satterthwaite (``gl``), los tamaños ``n_x`` y ``n_y`` y
    los valores p de Shapiro-Wilk (``p_shapiro_x`` y ``p_shapiro_y``; ``nan`` si hay menos de tres
    valores o si todos son iguales).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x, n_y = x.size, y.size
    if n_x < 2 or n_y < 2:
        raise ValueError("Cada muestra necesita al menos dos valores.")
    v_x, v_y = x.var(ddof=1), y.var(ddof=1)
    error_x, error_y = v_x / n_x, v_y / n_y
    if error_x + error_y == 0.0:
        t, p, gl = 0.0, 1.0, float(n_x + n_y - 2)
    else:
        t, p = stats.ttest_ind(x, y, equal_var=False)
        gl = (error_x + error_y) ** 2 / (error_x ** 2 / (n_x - 1) + error_y ** 2 / (n_y - 1))

    def p_shapiro(valores):
        if valores.size < 3 or np.ptp(valores) == 0.0:
            return float("nan")
        return float(stats.shapiro(valores).pvalue)

    return {
        "media_x": float(x.mean()), "media_y": float(y.mean()), "diferencia": float(x.mean() - y.mean()),
        "t": float(t), "p": float(p), "gl": float(gl), "n_x": int(n_x), "n_y": int(n_y),
        "p_shapiro_x": p_shapiro(x), "p_shapiro_y": p_shapiro(y),
    }


def ajuste_lineal_multiple(columnas, y):
    """Mínimos cuadrados ordinarios de ``y`` sobre una constante y las variables de ``columnas``.

    Parámetros
    ----------
    columnas : dict
        Variables explicativas: ``{nombre: valores}``, con un valor por observación.
    y : array_like
        Variable respuesta, con una observación por elemento. Debe haber más observaciones que coeficientes.

    Devuelve
    --------
    tabla : DataFrame
        Una fila por coeficiente (``constante`` y cada variable), con las columnas ``coef``, ``se`` (error
        estándar), ``t``, ``p`` (de dos colas, con n - k grados de libertad) y los límites ``ic95_inf`` e
        ``ic95_sup`` del intervalo de confianza del 95 %.
    r2 : float
        Coeficiente de determinación.
    """
    nombres = list(columnas)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(y)] + [np.asarray(columnas[nombre], dtype=float) for nombre in nombres])
    n, k = X.shape
    if n <= k:
        raise ValueError("Hacen falta más observaciones que coeficientes.")
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residuos = y - X @ coef
    gl = n - k
    covarianza = (residuos @ residuos / gl) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(covarianza))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = coef / se
    p = 2.0 * stats.t.sf(np.abs(t), gl)
    critico = stats.t.ppf(0.975, gl)
    tabla = pd.DataFrame({"coef": coef, "se": se, "t": t, "p": p, "ic95_inf": coef - critico * se,
                          "ic95_sup": coef + critico * se}, index=["constante"] + nombres)
    suma_total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - (residuos @ residuos) / suma_total) if suma_total > 0 else float("nan")
    return tabla, r2


def ajuste_ponderado(x, y, se):
    """Recta y = a + b x ajustada a medias con errores estándar conocidos (mínimos cuadrados ponderados).

    Cada punto pesa 1 / se^2. Los errores de los coeficientes se calculan con esos errores, sin
    reescalarlos por el residuo, y la bondad del ajuste se mide con chi^2 = sum((y - a - b x) / se)^2,
    que, si la recta describe los datos y los errores son correctos, sigue una distribución chi-cuadrado
    con n - 2 grados de libertad.

    Parámetros
    ----------
    x, y, se : array_like
        Abscisas, medias y errores estándar de las medias (positivos); al menos tres puntos.

    Devuelve
    --------
    dict con ``pendiente``, ``se_pendiente``, ``ordenada``, ``se_ordenada``, ``covarianza`` (entre
    ordenada y pendiente), ``chi2``, ``gl`` y ``p`` (probabilidad de un chi^2 igual o mayor).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    se = np.asarray(se, dtype=float)
    if x.size < 3:
        raise ValueError("Hacen falta al menos tres puntos para contrastar la recta.")
    if np.any(se <= 0):
        raise ValueError("Los errores estándar deben ser positivos.")
    X = np.column_stack([np.ones_like(x), x])
    pesos = 1.0 / se ** 2
    covarianza = np.linalg.inv(X.T @ (X * pesos[:, None]))
    coef = covarianza @ (X.T @ (pesos * y))
    chi2 = float(np.sum(((y - X @ coef) / se) ** 2))
    gl = x.size - 2
    return {"pendiente": float(coef[1]), "se_pendiente": float(np.sqrt(covarianza[1, 1])), "ordenada": float(coef[0]),
            "se_ordenada": float(np.sqrt(covarianza[0, 0])), "covarianza": float(covarianza[0, 1]), "chi2": chi2,
            "gl": gl, "p": float(stats.chi2.sf(chi2, gl))}


def a_matriz(resumen, columna, n_filas, n_cols):
    """Reorganiza una columna del resumen en una matriz [i, j]."""
    matriz = np.full((n_filas, n_cols), np.nan)
    matriz[resumen["i"].to_numpy(), resumen["j"].to_numpy()] = resumen[columna].to_numpy()
    return matriz


def a_cubo(tabla, columna, n_i, n_j):
    """Reorganiza las réplicas de un observable en un arreglo (n_i, n_j, n_r)."""
    ordenada = tabla.sort_values(["i", "j", "replica"])
    return ordenada[columna].to_numpy().reshape(n_i, n_j, -1)


def cociente_con_error(num, se_num, den, se_den):
    """Cociente de dos medias independientes y su error estándar (método delta)."""
    c = num / den
    return c, np.abs(c) * np.sqrt((se_num / num) ** 2 + (se_den / den) ** 2)


def residuos_interaccion(media, var_media):
    """Residuos del modelo aditivo y su error estándar.

    Para una matriz de medias M[i, j] de celdas independientes, el modelo aditivo es
    M[i, j] = mu + a_i + b_j. El residuo (interacción) es
    gamma[i, j] = M[i, j] - media_fila_i - media_columna_j + media_global,
    combinación lineal de las medias de todas las celdas, de modo que su varianza es la suma
    de los cuadrados de los coeficientes por la varianza de la media de cada celda.

    Parámetros
    ----------
    media : ndarray, forma (n_i, n_j)
        Media de cada celda.
    var_media : ndarray, forma (n_i, n_j)
        Varianza de la media de cada celda (error estándar al cuadrado).

    Devuelve
    --------
    gamma, se_gamma : ndarray, forma (n_i, n_j)
    """
    n_i, n_j = media.shape
    gamma = media - media.mean(axis=1, keepdims=True) - media.mean(axis=0, keepdims=True) + media.mean()
    se_gamma = np.zeros_like(media)
    for i in range(n_i):
        for j in range(n_j):
            w = np.full((n_i, n_j), 1.0 / (n_i * n_j))
            w[i, :] -= 1.0 / n_j
            w[:, j] -= 1.0 / n_i
            w[i, j] += 1.0
            se_gamma[i, j] = np.sqrt(np.sum(w ** 2 * var_media))
    return gamma, se_gamma


def anova_dos_factores(y):
    """ANOVA de dos factores cruzados con réplicas.

    Parámetros
    ----------
    y : ndarray, forma (n_i, n_j, n_r)
        Observable de cada réplica r en la celda (i, j). Diseño balanceado.

    Devuelve
    --------
    dict
        ``eta2_a``, ``eta2_b``, ``eta2_ab``, ``eta2_e``: fracción de la suma de cuadrados total que
        explican el factor de las filas, el de las columnas, su interacción y el ruido entre réplicas.
        ``F_ab`` y ``p_ab``: estadístico F de la interacción y su valor p, que supone varianzas
        iguales en todas las celdas (aquí no se cumple, de modo que es indicativo).
    """
    n_i, n_j, n_r = y.shape
    gran = y.mean()
    m_i = y.mean(axis=(1, 2))
    m_j = y.mean(axis=(0, 2))
    m_ij = y.mean(axis=2)
    ss_a = n_j * n_r * np.sum((m_i - gran) ** 2)
    ss_b = n_i * n_r * np.sum((m_j - gran) ** 2)
    ss_ab = n_r * np.sum((m_ij - m_i[:, None] - m_j[None, :] + gran) ** 2)
    ss_e = np.sum((y - m_ij[:, :, None]) ** 2)
    ss_t = np.sum((y - gran) ** 2)
    gl_ab = (n_i - 1) * (n_j - 1)
    gl_e = n_i * n_j * (n_r - 1)
    f_ab = (ss_ab / gl_ab) / (ss_e / gl_e)
    return {
        "eta2_a": ss_a / ss_t, "eta2_b": ss_b / ss_t, "eta2_ab": ss_ab / ss_t, "eta2_e": ss_e / ss_t,
        "F_ab": f_ab, "p_ab": stats.f.sf(f_ab, gl_ab, gl_e), "gl_ab": gl_ab, "gl_e": gl_e,
    }


def resumir_por_punto(tabla, claves=("i", "j")):
    """Media, desviación estándar muestral y error estándar de cada observable por punto del barrido."""
    observables = ["rugosidad", "xi", "escalones", "vacias", "maximo", "colisiones", "sigma_eff_med", "sigma_red_med"]
    agrupado = tabla.groupby(list(claves), sort=True)
    medias = agrupado[observables].mean()
    desv = agrupado[observables].std(ddof=1)
    n = agrupado["rugosidad"].count()
    resumen = medias.copy()
    for obs in observables:
        resumen[obs + "_sd"] = desv[obs]
        resumen[obs + "_se"] = desv[obs] / np.sqrt(n)
    resumen["n_replicas"] = n
    return resumen.reset_index()


def analizar_mapa(tabla, n_i, n_j, referencia):
    """Efectos principales, interacción y amplificación de la rugosidad en un mapa de dos parámetros.

    Parámetros
    ----------
    tabla : DataFrame
        Réplicas del mapa, con las columnas ``i``, ``j``, ``replica`` y los observables.
    n_i, n_j : int
        Número de filas y de columnas del mapa.
    referencia : tuple
        ('fila', k): el mecanismo desactivado es la fila k, y el factor de amplificación divide cada
        columna por su valor en esa fila. ('columna', k): análogo, dividiendo cada fila por su valor
        en la columna k.

    Devuelve
    --------
    dict con el resumen por punto, las matrices de W y de su error estándar, los residuos de
    interacción (y su error estándar y z) para W y para ln W, los factores de amplificación y el
    análisis de varianza de W y de ln W.
    """
    resumen = resumir_por_punto(tabla)
    W = a_matriz(resumen, "rugosidad", n_i, n_j)
    W_se = a_matriz(resumen, "rugosidad_se", n_i, n_j)
    gamma, se_gamma = residuos_interaccion(W, W_se ** 2)
    # Modelo multiplicativo: aditivo en ln W (la varianza de ln W sale del método delta)
    gamma_ln, se_gamma_ln = residuos_interaccion(np.log(W), (W_se / W) ** 2)

    eje, k = referencia
    es_ref = np.zeros((n_i, n_j), dtype=bool)
    if eje == "fila":
        W_ref = np.repeat(W[k:k + 1, :], n_i, axis=0)
        se_ref = np.repeat(W_se[k:k + 1, :], n_i, axis=0)
        es_ref[k, :] = True
    else:
        W_ref = np.repeat(W[:, k:k + 1], n_j, axis=1)
        se_ref = np.repeat(W_se[:, k:k + 1], n_j, axis=1)
        es_ref[:, k] = True
    A, A_se = cociente_con_error(W, W_se, W_ref, se_ref)
    A_se = np.where(es_ref, 0.0, A_se)       # la referencia vale 1 por definición

    cubo = a_cubo(tabla, "rugosidad", n_i, n_j)
    return {
        "resumen": resumen, "W": W, "W_se": W_se,
        "gamma": gamma, "se_gamma": se_gamma, "z_gamma": gamma / se_gamma,
        "gamma_ln": gamma_ln, "se_gamma_ln": se_gamma_ln, "z_gamma_ln": gamma_ln / se_gamma_ln,
        "A": A, "A_se": A_se, "es_ref": es_ref,
        "anova_W": anova_dos_factores(cubo), "anova_lnW": anova_dos_factores(np.log(cubo)),
    }


def efectos_factorial(tabla, factores, respuesta="rugosidad", transformar=None):
    """Efectos de un diseño factorial 2^k completo con réplicas, estimados con contrastes.

    Para cada subconjunto no vacío de factores, el efecto es el contraste
    E = sum_celdas c * media_celda, con c = prod_{f en el subconjunto} (2 * nivel_f - 1) / 2^(k-1).
    Un efecto principal es la diferencia entre la media de la respuesta con el factor alto y con el factor
    bajo, promediada sobre los demás factores. Una interacción de dos factores es la mitad de la diferencia
    entre el efecto de uno con el otro alto y con el otro bajo, y así sucesivamente (Montgomery, 2017).
    La varianza de cada efecto se calcula con la varianza muestral de cada celda, sin suponer que sea igual
    en todas, y los grados de libertad con la aproximación de Satterthwaite.

    Parámetros
    ----------
    tabla : DataFrame
        Una fila por réplica, con una columna por factor (nivel codificado: 0 bajo, 1 alto) y la de la respuesta.
    factores : lista de str
        Nombres de las columnas de los factores, en el orden en que se rotulan los efectos.
    respuesta : str
        Columna de la respuesta.
    transformar : callable, opcional
        Transformación de la respuesta (por ejemplo, np.log): con np.log, los efectos son logaritmos de cocientes.

    Devuelve
    --------
    DataFrame con una fila por efecto: ``efecto`` (por ejemplo, 'D', 'A×S'), ``orden``, ``valor``, ``se``, ``t``,
    ``gl`` y ``p``. Con np.log, exp(valor) es el cociente entre la respuesta con el factor alto y con el bajo.
    """
    k = len(factores)
    y = tabla[respuesta].to_numpy(dtype=float)
    if transformar is not None:
        y = transformar(y)
    celdas = tabla.assign(_y=y).groupby(list(factores))["_y"].agg(["mean", "var", "count"]).reset_index()
    n_rep = int(celdas["count"].iloc[0])
    if len(celdas) != 2 ** k or not (celdas["count"] == n_rep).all():
        raise ValueError("El diseño debe ser un factorial 2^k completo y balanceado.")
    filas = []
    for orden in range(1, k + 1):
        for sub in itertools.combinations(range(k), orden):
            signo = np.ones(len(celdas))
            for f in sub:
                signo = signo * (2.0 * celdas[factores[f]].to_numpy() - 1.0)
            c = signo / 2 ** (k - 1)
            valor = float(np.sum(c * celdas["mean"].to_numpy()))
            terminos = c ** 2 * celdas["var"].to_numpy() / n_rep
            se = float(np.sqrt(terminos.sum()))
            gl = se ** 4 / float(np.sum(terminos ** 2 / (n_rep - 1)))
            t = valor / se
            filas.append({"efecto": "×".join(factores[f] for f in sub), "orden": orden, "valor": valor,
                          "se": se, "t": t, "gl": gl, "p": float(2 * stats.t.sf(abs(t), gl))})
    return pd.DataFrame(filas)
