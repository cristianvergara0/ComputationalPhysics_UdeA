"""Simulación y análisis del crecimiento registrado en instantes arbitrarios.

Los cuadernos de la serie siguen la rugosidad de cada modelo desde una fracción de capa hasta cientos
de capas. Este módulo envuelve ``barridos.ejecutar_modelo_muestreado`` (instantes de registro y
superficies guardadas a espesores fijados) y reúne el análisis que comparten: estadística entre réplicas
en cada instante, observables de las superficies guardadas, ajustes réplica a réplica, rugosidad saturada
en mallas de distinto tamaño, el inicio de la inestabilidad de columnas y, con adhesión lateral, la
porosidad y el perfil de densidad de la película. Las mallas son cuadradas.

Con adhesión lateral la película tiene huecos: la altura de cada columna es la de su posición ocupada más
alta, y la altura media de las columnas supera el espesor de material (partículas por celda). El instante de
registro se mide siempre con el espesor de material; la rugosidad y el relieve, con la altura de las columnas.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .analisis import calcular_observables, islas_periodicas, tiempo_de_cruce
from .barridos import ejecutar_modelo_muestreado, malla_de_muestreo, semillas_de_modelo
from .motor import EMISION_GAUSSIANA, ejecutar_motor_volumen, sembrar_generador


def simular_modelo(params, nombre, flags, n_replicas, semilla_raiz, espesor_final, espesores_instantaneas=(), n_log=80,
                   desde_lineal=None, n_lineales=0, max_pasos_rayo=1_000_000, relieve_maximo=np.inf, espesor_minimo=0.1):
    """Ejecuta un modelo hasta ``espesor_final`` capas y guarda la superficie en ``espesores_instantaneas``.

    Construye los instantes de registro con ``malla_de_muestreo`` (``n_log`` espaciados de forma logarítmica
    desde ``espesor_minimo`` capas y, si se indica ``desde_lineal``, ``n_lineales`` equiespaciados hasta el
    final) y ejecuta ``ejecutar_modelo_muestreado``. Las semillas dependen solo de ``semilla_raiz`` y de
    ``nombre``, de modo que dos llamadas con el mismo nombre y los mismos parámetros físicos dan las mismas
    réplicas aunque registren instantes distintos.

    Devuelve
    --------
    El diccionario de ``ejecutar_modelo_muestreado`` con, además, ``objetivos`` (espesores de las
    instantáneas, crecientes), ``L`` (lado de la malla) y ``nombre``.
    """
    if params.nx != params.ny:
        raise ValueError("Las mallas de la serie son cuadradas.")
    objetivos = sorted(float(e) for e in espesores_instantaneas)
    n_celdas = params.nx * params.ny
    eventos, marcas = malla_de_muestreo(n_celdas, espesor_final, n_log, objetivos, desde_lineal, n_lineales, espesor_minimo)
    if int(marcas.sum()) != len(objetivos):
        raise ValueError("Dos espesores de instantánea corresponden al mismo número de partículas.")
    datos = ejecutar_modelo_muestreado(params, nombre, flags, n_replicas, semilla_raiz, eventos, marcas,
                                       max_pasos_rayo, relieve_maximo)
    datos.update(objetivos=np.array(objetivos, dtype=float), L=params.nx, nombre=nombre)
    return datos


def capas(datos):
    """Espesor nominal, en capas, de cada instante de registro de una simulación."""
    return datos["eventos"] / datos["L"] ** 2


def estadistica_por_instante(datos, valores=None):
    """Media, error estándar y número de réplicas activas en cada instante de registro.

    ``valores`` es un arreglo de forma (n_replicas, k); por defecto, la rugosidad. Los instantes
    posteriores al corte por relieve de una réplica (``nan``) no cuentan.

    Devuelve
    --------
    h, media, se, n : ndarray de forma (k,)
    """
    x = datos["rugosidad"] if valores is None else np.asarray(valores, dtype=float)
    n = np.isfinite(x).sum(axis=0)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        media = np.nanmean(x, axis=0)
        se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(n)
    return capas(datos), media, se, n


def observables_de_instantaneas(datos, etiqueta):
    """Observables de cada réplica en cada instantánea alcanzada (una fila por réplica e instantánea).

    Columnas: ``modelo`` (la etiqueta), ``L``, ``objetivo`` (espesor pedido), ``espesor`` (de material),
    ``replica``, ``rugosidad``, ``xi`` (longitud de correlación), ``escalones`` (porcentaje de enlaces con
    desniveles de dos capas o más), ``relieve`` (altura máxima menos altura media de las columnas),
    ``columnas_por_mil`` y ``area_media`` (regiones que superan la altura media más la rugosidad, con fronteras
    periódicas) y ``porosidad`` (fracción de posiciones vacías bajo la cima de las columnas).
    """
    L = datos["L"]
    filas = []
    for k, objetivo in enumerate(datos["objetivos"]):
        espesor = datos["eventos_instantaneas"][k] / L ** 2
        for r in range(datos["instantaneas"].shape[0]):
            if not datos["instantanea_valida"][r, k]:
                continue
            superficie = datos["instantaneas"][r, k]
            obs = calcular_observables(superficie)
            altura_media = float(superficie.mean())
            areas = islas_periodicas(superficie, altura_media + obs["rugosidad"])
            suma = float(superficie.sum(dtype=np.int64))
            filas.append({"modelo": etiqueta, "L": L, "objetivo": float(objetivo), "espesor": espesor, "replica": r,
                          "rugosidad": obs["rugosidad"], "xi": obs["xi"], "escalones": obs["escalones"],
                          "relieve": obs["maximo"] - altura_media, "columnas_por_mil": 1000 * areas.size / L ** 2,
                          "area_media": float(areas.mean()) if areas.size else 0.0,
                          "porosidad": (suma - datos["eventos_instantaneas"][k]) / suma})
    return pd.DataFrame(filas)


def fraccion_elevada(datos, umbrales=(10.0, 20.0)):
    """Fracción de columnas que sobresalen de la altura media, en cada réplica y cada instantánea alcanzada.

    Para cada umbral ``u`` (en capas), la columna ``fraccion_<u>`` es la fracción de columnas cuya altura supera
    la altura media de su superficie en ``u`` capas o más. A diferencia del relieve, que depende de la columna
    más alta y por tanto del número de columnas de la malla, esta fracción es una propiedad local de la
    superficie y no depende del tamaño de la malla mientras las columnas elevadas sean independientes.
    Devuelve una fila por réplica e instantánea alcanzada, con las columnas ``L``, ``objetivo``, ``replica`` y
    una ``fraccion_<u>`` por umbral.
    """
    L = datos["L"]
    filas = []
    for k, objetivo in enumerate(datos["objetivos"]):
        for r in range(datos["instantaneas"].shape[0]):
            if not datos["instantanea_valida"][r, k]:
                continue
            superficie = datos["instantaneas"][r, k].astype(np.float64)
            exceso = superficie - superficie.mean()
            fila = {"L": L, "objetivo": float(objetivo), "replica": r}
            for u in umbrales:
                fila[f"fraccion_{u:g}"] = float(np.mean(exceso >= u))
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_de_historias(resultados):
    """Rugosidad y relieve de cada réplica en cada instante anterior al corte (tabla larga).

    ``resultados`` es un diccionario {etiqueta: datos de ``simular_modelo``}.
    """
    partes = []
    for etiqueta, datos in resultados.items():
        h = capas(datos)
        for r in range(datos["rugosidad"].shape[0]):
            valido = np.isfinite(datos["rugosidad"][r])
            partes.append(pd.DataFrame({"modelo": etiqueta, "L": datos["L"], "replica": r, "espesor": h[valido],
                                        "rugosidad": datos["rugosidad"][r, valido],
                                        "relieve": datos["relieve"][r, valido],
                                        "porosidad": datos["porosidad"][r, valido]}))
    return pd.concat(partes, ignore_index=True)


def media_y_se(valores):
    """Media y error estándar de los valores finitos (``nan`` con menos de dos)."""
    v = np.asarray(valores, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def ajuste_por_replica(x, y, x_min, x_max):
    """Pendiente y ordenada de la recta y(x) en [x_min, x_max], ajustada a cada fila de ``y``.

    Devuelve un arreglo de forma (n_filas, 2) con (pendiente, ordenada); ``nan`` si la ventana tiene menos de
    tres puntos o si la fila tiene algún valor no finito en ella.
    """
    x = np.asarray(x, dtype=float)
    m = (x >= x_min) & (x <= x_max)
    resultado = []
    for fila in np.atleast_2d(np.asarray(y, dtype=float)):
        if m.sum() < 3 or not np.all(np.isfinite(fila[m])):
            resultado.append((np.nan, np.nan))
        else:
            resultado.append(tuple(np.polyfit(x[m], fila[m], 1)))
    return np.array(resultado)


def saturacion_por_tamano(resultados, ventana):
    """Rugosidad cuadrática saturada en cada malla y comprobación de que está saturada.

    Para cada malla de ``resultados`` ({L: datos}), promedia W^2 de cada réplica en los instantes con
    espesor entre ``ventana[0] * L^2`` y ``ventana[1] * L^2`` capas, y compara las dos mitades de la
    ventana con una prueba t pareada (las mitades salen de las mismas réplicas).

    Devuelve
    --------
    DataFrame con ``L``, ``replicas``, ``W_sat^2``, ``SE``, ``primera mitad``, ``segunda mitad`` y
    ``p (t pareada)``.
    """
    a, b = ventana
    centro = 0.5 * (a + b)
    filas = []
    for L, datos in sorted(resultados.items()):
        u = capas(datos) / L ** 2
        w2 = datos["rugosidad"] ** 2
        por_replica = w2[:, (u >= a) & (u <= b)].mean(axis=1)
        primera = w2[:, (u >= a) & (u < centro)].mean(axis=1)
        segunda = w2[:, (u >= centro) & (u <= b)].mean(axis=1)
        p = stats.ttest_rel(primera, segunda).pvalue if np.any(primera != segunda) else 1.0
        filas.append({"L": L, "replicas": por_replica.size, "W_sat^2": por_replica.mean(),
                      "SE": por_replica.std(ddof=1) / np.sqrt(por_replica.size),
                      "primera mitad": primera.mean(), "segunda mitad": segunda.mean(), "p (t pareada)": p})
    return pd.DataFrame(filas)


def espesor_de_inicio(datos, umbral, hasta=np.inf):
    """Espesor en que el relieve de cada réplica alcanza ``umbral`` por primera vez, sin pasar de ``hasta``.

    Interpola en el logaritmo del espesor entre los dos instantes que rodean el cruce. Si la réplica se
    detuvo por el corte de relieve antes de que un instante registrado alcanzara el umbral, se toma el
    espesor del corte (el relieve ya superaba entonces el corte, mayor que el umbral). Es ``nan`` si el
    relieve no alcanza el umbral.
    """
    h = capas(datos)
    salida = []
    for r, relieve in enumerate(datos["relieve"]):
        valido = np.isfinite(relieve) & (h <= hasta)
        hc = tiempo_de_cruce(h[valido], relieve[valido], umbral, 1.0) if valido.any() else np.nan
        if np.isnan(hc) and datos["detenida"][r] and datos["espesor_final"][r] <= hasta:
            hc = float(datos["espesor_final"][r])
        salida.append(hc)
    return np.array(salida)


def ritmo_del_relieve(datos, inferior, superior):
    """Pendiente de ln(relieve) frente al espesor mientras el relieve está entre dos valores, por réplica (1/capa)."""
    h = capas(datos)
    ritmos = []
    for relieve in datos["relieve"]:
        m = np.isfinite(relieve) & (relieve >= inferior) & (relieve <= superior)
        ritmos.append(np.polyfit(h[m], np.log(relieve[m]), 1)[0] if m.sum() >= 3 else np.nan)
    return np.array(ritmos)


def codigo_fuente(funcion):
    """Código fuente de una función del paquete, compilada o no (para mostrarlo en los cuadernos)."""
    return inspect.getsource(getattr(funcion, "py_func", funcion))


def mediana_censurada(valores):
    """Mediana de valores en los que ``nan`` significa que el evento no ocurrió dentro de lo simulado.

    Los ``nan`` cuentan como infinito, de modo que la mediana es finita solo si al menos la mitad de los valores
    lo son; si no, es ``inf``. Es la estadística adecuada para un espesor de inicio que algunas réplicas no
    alcanzan dentro del espesor simulado. Con un número par de valores y la mitad infinitos, también es ``inf``.
    """
    v = np.where(np.isfinite(np.asarray(valores, dtype=float)), valores, np.inf)
    if v.size == 0:
        return float("nan")
    return float(np.median(v))


def resumir_simulacion(datos, umbral_relieve, hasta=np.inf):
    """Resumen de una simulación con instantáneas: rugosidad en cada espesor guardado y espesores de inicio.

    Devuelve un diccionario con el número de réplicas, ``W_<e>``, ``W_<e>_se``, ``p_<e>`` y ``p_<e>_se`` (rugosidad
    y porosidad) para cada espesor guardado ``e`` que alcanzan todas las réplicas (``nan`` si alguna se detuvo
    antes), ``h_umbral`` (mediana del espesor
    en que el relieve alcanza ``umbral_relieve``, con ``mediana_censurada``), ``h_corte`` (mediana del espesor
    del corte por relieve; infinito si menos de la mitad de las réplicas se detuvo) y ``detenidas``.
    """
    fila = {"replicas": datos["rugosidad"].shape[0]}
    for k, objetivo in enumerate(datos["objetivos"]):
        if datos["instantanea_valida"][:, k].all():
            superficies = datos["instantaneas"][:, k].reshape(fila["replicas"], -1)
            suma = superficies.sum(axis=1, dtype=np.int64).astype(float)
            fila[f"W_{objetivo:g}"], fila[f"W_{objetivo:g}_se"] = media_y_se(superficies.std(axis=1))
            fila[f"p_{objetivo:g}"], fila[f"p_{objetivo:g}_se"] = media_y_se((suma - datos["eventos_instantaneas"][k]) / suma)
        else:
            fila[f"W_{objetivo:g}"], fila[f"W_{objetivo:g}_se"] = np.nan, np.nan
            fila[f"p_{objetivo:g}"], fila[f"p_{objetivo:g}_se"] = np.nan, np.nan
    fila["h_umbral"] = mediana_censurada(espesor_de_inicio(datos, umbral_relieve, hasta))
    corte = np.where(datos["detenida"], datos["espesor_final"], np.nan)
    fila["h_corte"] = mediana_censurada(corte)
    fila["detenidas"] = int(datos["detenida"].sum())
    return fila


def simular_volumen(params, nombre, flags, semilla_raiz, espesor_final, n_niveles, replica=0,
                    max_pasos_rayo=1_000_000, relieve_maximo=np.inf):
    """Repite una réplica de ``simular_modelo`` registrando la ocupación de cada posición de la película.

    Usa la semilla de la réplica ``replica`` del modelo ``nombre`` (la misma que le asigna ``simular_modelo``),
    de modo que la película es la de esa réplica: la realización no depende de los instantes de registro.

    Parámetros
    ----------
    n_niveles : int
        Número de niveles del arreglo de ocupación; debe superar la altura de la columna más alta.

    Devuelve
    --------
    dict con ``volumen`` (ndarray uint8 de forma (L, L, n_niveles): 1 en las posiciones ocupadas),
    ``superficie`` (altura de cada columna), ``depositados`` y ``huecos`` (posiciones vacías bajo la cima).
    """
    n_celdas = params.nx * params.ny
    n_eventos = int(round(espesor_final * n_celdas))
    semilla = int(semillas_de_modelo(semilla_raiz, nombre, replica + 1)[replica])
    volumen = np.zeros((params.nx, params.ny, int(n_niveles)), dtype=np.uint8)
    eventos = np.array([n_eventos], dtype=np.int64)
    sembrar_generador(semilla)
    superficie, _, _, _, _, _, _, _, depositados = ejecutar_motor_volumen(
        params.nx, params.ny, n_eventos, float(flags["temp"]), float(params.t_escala), int(params.max_pasos_difusion),
        float(params.sigma_incidencia), float(params.sigma_dispersion), float(params.mfp), float(params.distancia_fuente), flags["enable_angle"], flags["enable_scattering"], flags["enable_shadowing"],
        flags["enable_diffusion"], eventos, np.zeros(1, dtype=np.bool_), int(max_pasos_rayo), float(relieve_maximo),
        int(flags.get("ley_emision", EMISION_GAUSSIANA)), float(flags.get("theta_max", 0.0)),
        bool(flags.get("enable_lateral", False)), volumen
    )
    if superficie.max() > n_niveles:
        raise ValueError(f"{nombre}: la columna más alta ({superficie.max()}) supera los {n_niveles} niveles registrados.")
    ocupadas = int(volumen.sum(dtype=np.int64))
    if ocupadas != depositados:
        raise RuntimeError(f"{nombre}: el volumen registra {ocupadas} posiciones ocupadas y se depositaron {depositados} partículas.")
    return {"volumen": volumen, "superficie": superficie, "depositados": int(depositados),
            "huecos": int(superficie.sum(dtype=np.int64)) - int(depositados)}


def perfil_de_densidad(volumen, superficie):
    """Perfil vertical de la película a partir de su ocupación.

    Para cada nivel z devuelve, como fracción de las celdas de la malla, las posiciones ocupadas (``densidad``)
    y las columnas cuya cima está por encima de z (``envolvente``), y la fracción de huecos dentro de la
    envolvente (``porosidad``, ``nan`` donde la envolvente está vacía).

    Devuelve
    --------
    DataFrame con ``nivel``, ``densidad``, ``envolvente`` y ``porosidad``, hasta el nivel de la columna más alta.
    """
    n_celdas = superficie.size
    n = int(superficie.max())
    niveles = np.arange(n)
    densidad = volumen[:, :, :n].reshape(n_celdas, n).sum(axis=0, dtype=np.int64) / n_celdas
    envolvente = (superficie.reshape(-1)[:, None] > niveles[None, :]).sum(axis=0) / n_celdas
    with np.errstate(invalid="ignore", divide="ignore"):
        porosidad = np.where(envolvente > 0, 1.0 - densidad / envolvente, np.nan)
    return pd.DataFrame({"nivel": niveles, "densidad": densidad, "envolvente": envolvente, "porosidad": porosidad})
