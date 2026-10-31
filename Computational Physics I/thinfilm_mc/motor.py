"""Núcleos físicos compartidos del modelo de crecimiento: relajación, transporte y lazos de deposición.

La película se describe con una altura por columna sobre una malla ``nx`` x ``ny`` con fronteras
periódicas: ``superficie[i, j]`` es la altura, en capas, de la columna ``(i, j)``, y cada evento de Monte
Carlo deposita una partícula. Los mecanismos son:

* **Transporte**: la partícula vuela en línea recta desde la fuente, con una dirección de emisión angular,
  y las colisiones con una capa de gas situada por encima de la película la desvían. Con sombreado, el vuelo
  se recorre celda a celda por debajo de la columna más alta y termina en la primera columna cuya cima o
  cuya pared lateral alcanza (``trazar_vuelo``).
* **Relajación**: una vez depositada, la partícula salta hasta ``max_pasos`` veces a un vecino estrictamente
  más bajo, con probabilidad ``P = min(1, T / T_escala)`` (``difundir_particula_pbc``).
* **Adhesión lateral**: si la partícula alcanza la pared lateral de una columna, puede quedar adherida a ella,
  a la altura del contacto, en la columna desde la que llegaba. Las posiciones que quedan vacías bajo ella son
  huecos de la película, que deja de ser compacta: la altura de cada columna es entonces la de su posición
  ocupada más alta (``ejecutar_motor_volumen``).

Hay dos lazos de deposición: el de réplicas con registro de los ángulos de llegada (``ejecutar_replica`` y
``ejecutar_replicas``), que usan los barridos de la composición angular, y el motor con una bandera por
mecanismo, instantes de registro arbitrarios, instantáneas de la superficie, corte por relieve, adhesión
lateral y registro opcional de la ocupación (``ejecutar_motor_volumen``, ``ejecutar_motor_muestreado`` y
``ejecutar_suite_muestreada``). Con sombreado e incidencia angular, los dos consumen los mismos números
aleatorios y, con la misma semilla, dan la misma superficie.

Generador pseudoaleatorio. Con Numba, las funciones compiladas usan un generador propio, distinto
del de NumPy: ``np.random.seed`` llamado desde Python no lo siembra. Por eso las rutinas de réplicas
(``ejecutar_replicas``, ``ejecutar_suite_muestreada`` y ``muestrear_vuelos_libres``) siembran dentro de
la parte compilada, una semilla por réplica, y para llamar con una semilla a las demás funciones
desde Python se usa ``sembrar_generador``. Ambos generadores siguen el mismo algoritmo (Mersenne
Twister), de modo que, con la misma semilla, el resultado compilado y el no compilado coinciden.
"""

import numpy as np

from .analisis import calcular_metricas
from .compat import njit, prange

# Valores de ``ley_emision``
EMISION_GAUSSIANA = 0
EMISION_COSENO = 1


@njit
def sembrar_generador(semilla):
    """Siembra el generador que usan las funciones compiladas de este módulo.

    Con Numba siembra el generador de Numba del hilo que llama; sin Numba, el de NumPy. Es la forma
    de fijar la semilla antes de llamar directamente, desde Python, a ``trazar_vuelo``,
    ``difundir_particula_pbc``, ``ejecutar_replica`` o ``ejecutar_motor_volumen``.
    """
    np.random.seed(semilla)


@njit
def difundir_particula_pbc(superficie, x, y, nx, ny, temp, t_escala, max_pasos):
    """Aplica la difusión de corto alcance por descenso con fronteras periódicas.

    En cada paso, la partícula salta con probabilidad P = min(1, temp / t_escala)
    a uno de sus vecinos de von Neumann estrictamente más bajos, elegido con igual
    probabilidad. Se detiene si no hay vecinos más bajos, si el salto falla o si
    alcanza max_pasos saltos.

    Parámetros
    ----------
    superficie : ndarray de enteros, forma (nx, ny)
        Alturas actuales, en capas. La función no las modifica.
    x, y : int
        Celda de colisión.
    nx, ny : int
        Número de celdas en cada dirección.
    temp : float
        Temperatura del sustrato, en kelvin.
    t_escala : float
        Temperatura de escala de la probabilidad de salto, en kelvin.
    max_pasos : int
        Número máximo de saltos.

    Devuelve
    --------
    curr_x, curr_y : int
        Índices de la celda que recibe la partícula.
    """
    curr_x = x
    curr_y = y

    # Desplazamientos de von Neumann: izquierda, derecha, abajo, arriba
    dx = (-1, 1, 0, 0)
    dy = (0, 0, -1, 1)

    # Probabilidad de salto según la temperatura
    prob_difusion = temp / t_escala
    if prob_difusion > 1.0:
        prob_difusion = 1.0

    for _ in range(max_pasos):
        curr_h = superficie[curr_x, curr_y]

        # Primer recorrido: número de vecinos estrictamente más bajos, con fronteras periódicas.
        conteo_validos = 0
        for i in range(4):
            nx_c = (curr_x + dx[i]) % nx
            ny_c = (curr_y + dy[i]) % ny
            if nx_c < 0:
                nx_c += nx
            if ny_c < 0:
                ny_c += ny
            if superficie[nx_c, ny_c] < curr_h:
                conteo_validos += 1

        # Sin microvalles en la vecindad, la partícula se deposita.
        if conteo_validos == 0:
            break

        # Intento de salto. Un segundo recorrido localiza al vecino elegido sin
        # crear arreglos temporales en cada paso.
        if np.random.random() < prob_difusion:
            idx = np.random.randint(0, conteo_validos)
            next_x = curr_x
            next_y = curr_y
            contador = 0
            for i in range(4):
                nx_c = (curr_x + dx[i]) % nx
                ny_c = (curr_y + dy[i]) % ny
                if nx_c < 0:
                    nx_c += nx
                if ny_c < 0:
                    ny_c += ny
                if superficie[nx_c, ny_c] < curr_h:
                    if contador == idx:
                        next_x = nx_c
                        next_y = ny_c
                        break
                    contador += 1
            curr_x = next_x
            curr_y = next_y
        else:
            break

    return curr_x, curr_y


@njit
def direccion_de_emision(sigma_ang, ley_emision, theta_max):
    """Ángulos de emisión por eje (theta_x, theta_y), en grados, de una partícula que sale de la fuente.

    Con ``ley_emision = 0`` cada eje sigue una normal de media 0 y desviación ``sigma_ang``. Con
    ``ley_emision = 1`` la dirección sigue la ley del coseno acotada a ``theta_max``: la función de
    distribución del ángulo polar es sen^2(theta) / sen^2(theta_max), que se invierte con una uniforme, y el
    acimut es uniforme. Los ángulos por eje son los de las proyecciones de la dirección sobre los planos xz e yz.
    """
    if ley_emision == 1:
        theta_polar = np.arcsin(np.sin(theta_max * np.pi / 180.0) * np.sqrt(np.random.random()))
        acimut = 2.0 * np.pi * np.random.random()
        pendiente_emision = np.tan(theta_polar)
        theta_x = np.arctan(pendiente_emision * np.cos(acimut)) * 180.0 / np.pi
        theta_y = np.arctan(pendiente_emision * np.sin(acimut)) * 180.0 / np.pi
    else:
        theta_x = np.random.normal(0.0, sigma_ang)
        theta_y = np.random.normal(0.0, sigma_ang)
    return theta_x, theta_y


@njit
def trazar_vuelo(x_target, y_target, nx, ny, superficie, sigma_ang, sigma_disp, mfp, max_h_actual,
                 distancia_fuente, sombreado, max_pasos_rayo, ley_emision, theta_max):
    """Determina la celda de colisión de una partícula, la dirección con que llega y el contacto lateral.

    La partícula se lanza desde un techo situado ``distancia_fuente`` capas por encima de la altura máxima
    de la película, con una posición lateral tal que, sin colisiones, su recta pasa por el punto de mira
    (``x_target``, ``y_target``) en z = 0. Vuela en línea recta entre colisiones con el gas. La celda
    (i, j) ocupa el cuadrado [i - 1/2, i + 1/2) x [j - 1/2, j + 1/2), con fronteras periódicas, y su columna
    es sólida entre z = 0 y su altura.

    Capa de gas. El gas ocupa una capa de espesor ``distancia_fuente`` entre el techo y la columna más alta, y
    la partícula la atraviesa de colisión en colisión. Por debajo de la columna más alta no hay colisiones:
    la partícula cruza el relieve en línea recta con la dirección que le dejó la última colisión. Es lo que
    ocurre en una cámara de evaporación, donde el camino libre medio (milímetros o más) es mucho mayor que
    el relieve de la película. Así, el recorrido por el gas tiene siempre el mismo espesor y no depende del
    relieve, y el gas solo cambia la distribución de direcciones con que llegan las partículas.

    Trazado exacto. Por debajo de la columna más alta, con sombreado, el vuelo se recorre celda a celda: en
    cada una se calcula cuánto debe descender la partícula para llegar a la cima de su columna y para cruzar
    el borde de la celda en x o en y, y ocurre lo que requiere menos descenso. Si llega a la cima de la
    columna, se detiene sobre ella. Si cruza a una celda cuya columna supera su altura, ha alcanzado la pared
    lateral de esa columna y se detiene en ella. El resultado no depende de ningún paso de discretización.
    Con ``sombreado = 0`` el relieve no intercepta nada y la partícula llega a la celda en que se encuentra al
    salir de la capa de gas, de modo que la emisión angular y las colisiones solo desplazan el punto de
    llegada.

    Dirección de emisión. Con ``ley_emision = 0`` los ángulos ``theta_x`` y ``theta_y`` se sortean de una
    distribución normal de media cero y desviación ``sigma_ang`` (en grados). Con ``ley_emision = 1`` la
    dirección sigue la ley del coseno de Knudsen acotada: el ángulo polar tiene densidad proporcional a
    sen(theta) cos(theta) en [0, ``theta_max``] y el acimut es uniforme; ``theta_x`` y ``theta_y`` son los
    ángulos de las proyecciones de esa dirección sobre los planos xz e yz (tan theta_x = tan(theta) cos(phi),
    tan theta_y = tan(theta) sen(phi)). La partícula avanza -tan(theta_x) y -tan(theta_y) celdas por cada
    capa que desciende.

    Colisiones. Dentro de la capa de gas, las distancias libres, medidas sobre la trayectoria, siguen una
    distribución exponencial de media ``mfp`` y, como el gas no tiene memoria, se vuelven a sortear tras cada
    colisión. En cada colisión se suma a ``theta_x`` y a ``theta_y`` un ruido normal de desviación
    ``sigma_disp``.

    Contacto lateral. Si la partícula se detiene al cruzar de una celda A a una celda B cuya columna supera
    la altura z_e del cruce, ha alcanzado la pared lateral de B. El contacto se describe con la columna A y
    el nivel ``floor(z_e)``, una posición libre de A (la partícula volaba por encima de ella) contigua a una
    posición ocupada de B. Si llega a la cima de una columna, no hay contacto lateral. El trazado no cambia
    por ello: el contacto solo se informa, y es el lazo de deposición quien decide si la partícula queda
    adherida a la pared (``ejecutar_motor_volumen``).

    Salvaguarda. Un vuelo casi rasante podría cruzar un número enorme de celdas o de colisiones antes de
    detenerse. Después de ``max_pasos_rayo`` etapas (tramos entre colisiones y cruces de celda) la partícula
    se deposita en la celda en que está y se contabiliza como truncada.

    Parámetros
    ----------
    x_target, y_target : int
        Punto de mira en el plano z = 0.
    nx, ny : int
        Número de celdas en cada dirección.
    superficie : ndarray de enteros, forma (nx, ny)
        Alturas actuales, en capas. La función no las modifica.
    sigma_ang : float
        Desviación estándar de los ángulos de emisión, en grados (solo con ``ley_emision = 0``).
    sigma_disp : float
        Desviación estándar de la desviación angular por colisión, en grados.
    mfp : float
        Camino libre medio, en celdas. Con ``np.inf`` no hay colisiones.
    max_h_actual : int
        Altura máxima actual de la superficie, en capas.
    distancia_fuente : float
        Distancia entre el techo y la altura máxima de la película, en capas: es el espesor de la capa de gas.
    sombreado : int
        1 para que el relieve intercepte los vuelos; 0 para desactivarlo.
    max_pasos_rayo : int
        Número máximo de etapas del vuelo de un átomo.
    ley_emision : int
        0 para emisión gaussiana; 1 para la ley del coseno acotada.
    theta_max : float
        Ángulo polar máximo de la ley del coseno, en grados (solo con ``ley_emision = 1``).

    Devuelve
    --------
    cx, cy : int
        Índices de la celda de colisión.
    n_col : int
        Número de colisiones con el gas sufridas durante el vuelo.
    theta_x, theta_y : float
        Ángulos de la última dirección de vuelo, en grados. Son la suma del ángulo de emisión y de las
        desviaciones de las colisiones, sin reducirlos módulo 180 grados.
    truncado : int
        1 si la salvaguarda detuvo el vuelo; 0 en caso contrario.
    ax, ay : int
        Columna desde la que la partícula alcanzó la pared lateral (``-1`` si no hubo contacto lateral).
    nivel : int
        Nivel del contacto lateral, una posición libre de la columna (``ax``, ``ay``); ``-1`` si no hubo
        contacto lateral.
    """
    # Dirección de emisión (gaussiana por eje o ley del coseno acotada)
    theta_x, theta_y = direccion_de_emision(sigma_ang, ley_emision, theta_max)

    # Techo de vuelo, medido desde la parte más alta de la película: así la geometría
    # del vuelo no depende del espesor acumulado.
    z = float(max_h_actual) + distancia_fuente

    # Posición de lanzamiento: la recta inicial pasa por el punto de mira en z = 0.
    tan_x = np.tan(theta_x * np.pi / 180.0)
    tan_y = np.tan(theta_y * np.pi / 180.0)
    x = x_target + z * tan_x
    y = y_target + z * tan_y

    # Distancia de trayectoria hasta la primera colisión (transformada inversa de la exponencial;
    # 1 - random() está en (0, 1], de modo que el logaritmo es finito).
    if np.isinf(mfp):
        s_libre = np.inf
    else:
        s_libre = mfp * (-np.log(1.0 - np.random.random()))

    n_col = 0
    pasos = 0
    truncado = 0
    ax = -1
    ay = -1
    nivel = -1

    # Tramo 1: la capa de gas, entre el techo y la columna más alta. Ningún relieve puede interceptar ahí
    # la partícula, y el vuelo se recorre de colisión en colisión.
    altura_libre = float(max_h_actual)
    while z > altura_libre:
        pasos += 1
        longitud = np.sqrt(1.0 + tan_x * tan_x + tan_y * tan_y)     # trayectoria por capa de descenso
        dz = z - altura_libre
        if s_libre < dz * longitud:
            # Colisión antes de llegar: se desvía la dirección y se sortea la siguiente distancia libre.
            dz = s_libre / longitud
            z -= dz
            x -= dz * tan_x
            y -= dz * tan_y
            theta_x += np.random.normal(0.0, sigma_disp)
            theta_y += np.random.normal(0.0, sigma_disp)
            tan_x = np.tan(theta_x * np.pi / 180.0)
            tan_y = np.tan(theta_y * np.pi / 180.0)
            n_col += 1
            s_libre = mfp * (-np.log(1.0 - np.random.random()))
        else:
            s_libre -= dz * longitud
            x -= dz * tan_x
            y -= dz * tan_y
            z = altura_libre
        if z > altura_libre and pasos >= max_pasos_rayo:
            truncado = 1
            break

    # Celda en que se encuentra la partícula (índices sin reducir; las fronteras son periódicas).
    ix = int(np.floor(x + 0.5))
    iy = int(np.floor(y + 0.5))
    cx = ix % nx
    cy = iy % ny
    if cx < 0:
        cx += nx
    if cy < 0:
        cy += ny
    if truncado == 1 or sombreado == 0 or z <= superficie[cx, cy]:
        return cx, cy, n_col, theta_x, theta_y, truncado, ax, ay, nivel

    # Tramo 2: recorrido celda a celda por debajo de la columna más alta, en línea recta (sin gas).
    while True:
        pasos += 1
        dz_cima = z - superficie[cx, cy]
        # Descenso hasta el borde siguiente en x y en y (avance horizontal -tan por capa descendida)
        dz_x = np.inf
        if tan_x < 0.0:
            dz_x = (ix + 0.5 - x) / (-tan_x)
        elif tan_x > 0.0:
            dz_x = (x - (ix - 0.5)) / tan_x
        dz_y = np.inf
        if tan_y < 0.0:
            dz_y = (iy + 0.5 - y) / (-tan_y)
        elif tan_y > 0.0:
            dz_y = (y - (iy - 0.5)) / tan_y
        if dz_x < 0.0:
            dz_x = 0.0
        if dz_y < 0.0:
            dz_y = 0.0

        if dz_cima <= dz_x and dz_cima <= dz_y:
            # Llega a la cima de la columna de esta celda.
            break
        else:
            # Cruce del borde de la celda (en x si llega antes o a la vez; el otro eje, en la etapa siguiente).
            previa_x = cx
            previa_y = cy
            if dz_x <= dz_y:
                dz = dz_x
                z -= dz
                y -= dz * tan_y
                if tan_x < 0.0:
                    x = ix + 0.5
                    ix += 1
                else:
                    x = ix - 0.5
                    ix -= 1
            else:
                dz = dz_y
                z -= dz
                x -= dz * tan_x
                if tan_y < 0.0:
                    y = iy + 0.5
                    iy += 1
                else:
                    y = iy - 0.5
                    iy -= 1
            cx = ix % nx
            cy = iy % ny
            if cx < 0:
                cx += nx
            if cy < 0:
                cy += ny
            if z < superficie[cx, cy]:
                # Alcanza la pared lateral de la columna de la nueva celda.
                ax = previa_x
                ay = previa_y
                nivel = int(np.floor(z))
                break
            if z == superficie[cx, cy]:
                # Llega justo al borde superior de la pared: queda sobre la cima.
                break
        if pasos >= max_pasos_rayo:
            truncado = 1
            break

    return cx, cy, n_col, theta_x, theta_y, truncado, ax, ay, nivel


@njit
def ejecutar_replica(nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_ang, sigma_disp, mfp,
                     distancia_fuente, sombreado, max_pasos_rayo, ley_emision, theta_max):
    """Lazo secuencial de deposición de una réplica, con sombreado, gas y relajación térmica.

    Cada evento elige un punto de mira uniforme, traza la partícula hasta su celda de colisión
    (``trazar_vuelo``), la relaja (``difundir_particula_pbc``) y la deposita. Es el lazo sin historial
    temporal, con el acumulado del cuadrado del ángulo de llegada. Los argumentos ``sombreado``,
    ``max_pasos_rayo``, ``ley_emision`` y ``theta_max`` son los de ``trazar_vuelo``. El generador pseudoaleatorio debe haberse sembrado antes de
    llamar a esta función (desde Python, con ``sembrar_generador``).

    Devuelve
    --------
    superficie : ndarray int32, forma (nx, ny)
        Alturas finales, en capas.
    n_colisiones : int
        Número total de colisiones con el gas de todas las partículas.
    suma_theta2 : float
        Suma, sobre todas las partículas, de (theta_x^2 + theta_y^2) / 2 en el instante del
        impacto, en grados al cuadrado.
    suma_theta2_red : float
        Lo mismo con los ángulos reducidos al intervalo (-90, 90] grados, que son los que determinan
        la dirección física de vuelo (la tangente tiene período de 180 grados).
    n_truncados : int
        Número de partículas cuyo vuelo detuvo la salvaguarda del trazador.
    """
    superficie = np.zeros((nx, ny), dtype=np.int32)
    max_h_actual = 0
    n_colisiones = 0
    suma_theta2 = 0.0
    suma_theta2_red = 0.0
    n_truncados = 0
    for _ in range(n_eventos):
        # Punto de mira, equiprobable e independiente de la altura actual.
        x_i = np.random.randint(0, nx)
        y_i = np.random.randint(0, ny)

        # El transporte y la relajación dependen del relieve que dejaron los eventos anteriores,
        # por lo que los eventos deben procesarse en orden.
        x_col, y_col, n_col, theta_x, theta_y, truncado, _ax, _ay, _nivel = trazar_vuelo(
            x_i, y_i, nx, ny, superficie, sigma_ang, sigma_disp, mfp, max_h_actual,
            distancia_fuente, sombreado, max_pasos_rayo, ley_emision, theta_max
        )
        n_colisiones += n_col
        n_truncados += truncado
        suma_theta2 += 0.5 * (theta_x * theta_x + theta_y * theta_y)
        red_x = (theta_x + 90.0) % 180.0 - 90.0
        red_y = (theta_y + 90.0) % 180.0 - 90.0
        suma_theta2_red += 0.5 * (red_x * red_x + red_y * red_y)
        x_f, y_f = difundir_particula_pbc(superficie, x_col, y_col, nx, ny, temp, t_escala, max_pasos)
        superficie[x_f, y_f] += 1

        # Las alturas solo crecen: basta comparar con la celda que acaba de recibir la partícula.
        if superficie[x_f, y_f] > max_h_actual:
            max_h_actual = superficie[x_f, y_f]

    return superficie, n_colisiones, suma_theta2, suma_theta2_red, n_truncados


@njit(parallel=True)
def ejecutar_replicas(semillas, nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_ang, sigma_disp,
                      mfp, distancia_fuente, sombreado, max_pasos_rayo, ley_emision, theta_max):
    """Ejecuta en paralelo réplicas independientes de un mismo punto del espacio de parámetros.

    Las réplicas no comparten estado (cada una tiene su propia superficie y su propia semilla),
    de modo que el bucle es genuinamente paralelizable. Cada réplica siembra el generador del
    hilo que la ejecuta con ``semillas[rep]`` antes de correr, por lo que su resultado depende
    únicamente de esa semilla y no del orden ni del hilo. Dentro de cada réplica los eventos son
    secuenciales.

    Parámetros
    ----------
    semillas : ndarray de enteros, forma (n_replicas,)
        Semilla de cada réplica.

    El resto de los parámetros son los de ``ejecutar_replica``.

    Devuelve
    --------
    superficies : ndarray int32, forma (n_replicas, nx, ny)
    colisiones_por_atomo : ndarray float64, forma (n_replicas,)
        Número medio de colisiones con el gas por partícula depositada.
    theta2_medio : ndarray float64, forma (n_replicas,)
        Promedio, sobre las partículas, de (theta_x^2 + theta_y^2) / 2 en el impacto, en grados
        al cuadrado.
    theta2_red_medio : ndarray float64, forma (n_replicas,)
        Lo mismo con los ángulos reducidos al intervalo (-90, 90] grados.
    truncados : ndarray int64, forma (n_replicas,)
        Número de partículas de cada réplica cuyo vuelo detuvo la salvaguarda del trazador.
    """
    n_replicas = semillas.shape[0]
    superficies = np.zeros((n_replicas, nx, ny), dtype=np.int32)
    colisiones_por_atomo = np.zeros(n_replicas, dtype=np.float64)
    theta2_medio = np.zeros(n_replicas, dtype=np.float64)
    theta2_red_medio = np.zeros(n_replicas, dtype=np.float64)
    truncados = np.zeros(n_replicas, dtype=np.int64)

    for rep in prange(n_replicas):
        np.random.seed(semillas[rep])
        superficie, n_col, suma_theta2, suma_theta2_red, n_trunc = ejecutar_replica(
            nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_ang, sigma_disp, mfp,
            distancia_fuente, sombreado, max_pasos_rayo, ley_emision, theta_max
        )
        superficies[rep] = superficie
        colisiones_por_atomo[rep] = n_col / n_eventos
        theta2_medio[rep] = suma_theta2 / n_eventos
        theta2_red_medio[rep] = suma_theta2_red / n_eventos
        truncados[rep] = n_trunc

    return superficies, colisiones_por_atomo, theta2_medio, theta2_red_medio, truncados


@njit
def muestrear_vuelos_libres(n_vuelos, nx, ny, sigma_ang, sigma_disp, mfp, distancia_fuente, semilla,
                            max_pasos_rayo, ley_emision, theta_max):
    """Muestrea vuelos de átomos sobre una superficie plana y vacía.

    Sin relieve, cada átomo atraviesa la capa de gas de espesor ``distancia_fuente`` hasta el sustrato, y las
    colisiones son la única fuente de variación de la dirección. Devuelve, por átomo, el número
    de colisiones y los ángulos ``theta_x`` y ``theta_y`` de la dirección de llegada (grados, sin
    reducir módulo 180 grados). Los argumentos ``max_pasos_rayo``, ``ley_emision`` y ``theta_max`` son
    los de ``trazar_vuelo``.
    """
    np.random.seed(semilla)
    superficie = np.zeros((nx, ny), dtype=np.int32)
    n_col = np.zeros(n_vuelos, dtype=np.int64)
    theta_x = np.zeros(n_vuelos, dtype=np.float64)
    theta_y = np.zeros(n_vuelos, dtype=np.float64)
    for k in range(n_vuelos):
        x_i = np.random.randint(0, nx)
        y_i = np.random.randint(0, ny)
        _, _, n, tx, ty, _, _, _, _ = trazar_vuelo(
            x_i, y_i, nx, ny, superficie, sigma_ang, sigma_disp, mfp, 0, distancia_fuente, 1,
            max_pasos_rayo, ley_emision, theta_max
        )
        n_col[k] = n
        theta_x[k] = tx
        theta_y[k] = ty
    return n_col, theta_x, theta_y


@njit
def modulo_relajacion_superficial(superficie, x, y, nx, ny, temp, t_escala, max_pasos, enable_diffusion):
    """Relajación del motor común: ``difundir_particula_pbc`` con la bandera ``enable_diffusion``.

    Si ``enable_diffusion`` es False, la partícula queda en la celda de colisión ``(x, y)`` sin
    consumir números aleatorios. En caso contrario aplica ``difundir_particula_pbc``.

    Devuelve
    --------
    curr_x, curr_y : int
        Índices de la celda que recibe la partícula.
    """
    if not enable_diffusion:
        return x, y
    return difundir_particula_pbc(superficie, x, y, nx, ny, temp, t_escala, max_pasos)


@njit
def transporte_con_salvaguarda(x_target, y_target, nx, ny, superficie, max_h_actual, sigma_init, sigma_scat, mfp,
                               distancia_fuente, enable_angle, enable_scattering, enable_shadowing,
                               max_pasos_rayo, ley_emision, theta_max):
    """Transporte del motor común: lleva la partícula del punto de mira a su celda de colisión.

    * Sin sombreado ni gas, la trayectoria no interactúa con la película. Con incidencia angular, el impacto
      se desplaza ``distancia_fuente * tan(theta)`` en cada eje respecto del punto de mira (con fronteras
      periódicas); sin ella, coincide con el punto de mira.
    * Con gas y sin sombreado, el vuelo se traza a través de la capa de gas, sin que el relieve lo intercepte.
    * Con sombreado pero sin incidencia angular ni gas, el vuelo es vertical y ningún relieve lo intercepta
      antes de la columna de mira, de modo que no hace falta trazarlo.
    * En el resto de los casos el vuelo se traza con ``trazar_vuelo`` y un máximo de ``max_pasos_rayo``
      etapas por átomo.

    La emisión puede ser gaussiana (``ley_emision = 0``, desviación ``sigma_init`` por eje) o seguir la ley del
    coseno acotada a ``theta_max`` grados (``ley_emision = 1``); ver ``direccion_de_emision``.

    Devuelve
    --------
    x, y : int
        Celda de colisión.
    n_col : int
        Colisiones con el gas durante el vuelo.
    truncado : int
        1 si la salvaguarda detuvo el vuelo.
    ax, ay, nivel : int
        Contacto lateral con la pared de una columna (ver ``trazar_vuelo``); ``-1`` si no lo hubo. Sin
        trazado de rayos no hay contacto lateral.
    """
    if not enable_shadowing:
        if enable_scattering:
            # Sin sombreado, el relieve no intercepta el vuelo: se traza con las colisiones hasta el sustrato.
            sigma_ang = sigma_init if enable_angle else 0.0
            cx, cy, n_col, _theta_x, _theta_y, truncado, _ax, _ay, _nivel = trazar_vuelo(
                x_target, y_target, nx, ny, superficie, sigma_ang, sigma_scat, mfp, max_h_actual,
                distancia_fuente, 0, max_pasos_rayo, ley_emision if enable_angle else EMISION_GAUSSIANA, theta_max
            )
            return cx, cy, n_col, truncado, -1, -1, -1
        if enable_angle:
            theta_x, theta_y = direccion_de_emision(sigma_init, ley_emision, theta_max)
            idx_x = int(np.round(x_target + distancia_fuente * np.tan(theta_x * np.pi / 180.0))) % nx
            idx_y = int(np.round(y_target + distancia_fuente * np.tan(theta_y * np.pi / 180.0))) % ny
            return idx_x, idx_y, 0, 0, -1, -1, -1
        return x_target, y_target, 0, 0, -1, -1, -1

    if not enable_angle and not enable_scattering:
        return x_target, y_target, 0, 0, -1, -1, -1

    sigma_ang = sigma_init if enable_angle else 0.0
    mfp_efectivo = mfp if enable_scattering else np.inf
    cx, cy, n_col, _theta_x, _theta_y, truncado, ax, ay, nivel = trazar_vuelo(
        x_target, y_target, nx, ny, superficie, sigma_ang, sigma_scat, mfp_efectivo, max_h_actual,
        distancia_fuente, 1, max_pasos_rayo, ley_emision if enable_angle else EMISION_GAUSSIANA, theta_max
    )
    return cx, cy, n_col, truncado, ax, ay, nivel


@njit
def ejecutar_motor_volumen(nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_init, sigma_scat, mfp,
                           distancia_fuente, enable_angle, enable_scattering, enable_shadowing, enable_diffusion,
                           eventos_muestreo, es_instantanea, max_pasos_rayo, relieve_maximo, ley_emision, theta_max,
                           enable_lateral, volumen):
    """Motor común: deposición con muestreo en instantes arbitrarios, adhesión lateral y registro del volumen.

    Deposita ``n_eventos`` partículas. Cada una elige un punto de mira uniforme, viaja hasta su celda de
    colisión (``transporte_con_salvaguarda``), se relaja (``modulo_relajacion_superficial``) y se deposita.
    Cada mecanismo se activa con una bandera: ``enable_angle`` (incidencia angular, con dispersión ``sigma_init``
    por eje), ``enable_scattering`` (colisiones con un gas de camino libre medio ``mfp`` y desviación
    ``sigma_scat`` por colisión), ``enable_shadowing`` (sombreado), ``enable_diffusion`` (relajación con
    probabilidad de salto ``min(1, temp / t_escala)`` y ``max_pasos`` saltos como máximo) y ``enable_lateral``
    (adhesión lateral). La rugosidad y el espesor medio se calculan después
    de los eventos de ``eventos_muestreo``, que pueden estar espaciados de forma logarítmica (las leyes de
    escalamiento abarcan varias décadas de tiempo), y en los instantes marcados con ``es_instantanea`` se
    guarda una copia de la superficie. El generador debe haberse sembrado antes de llamar a esta función
    (desde Python, con ``sembrar_generador``).

    Adhesión lateral (``enable_lateral``). Cuando el trazador informa de que la partícula alcanzó la pared
    lateral de una columna desde la columna A, al nivel ``nivel`` (ver ``trazar_vuelo``), la partícula queda
    apoyada en A a ese nivel y se relaja desde allí con la regla común, como si A tuviera esa altura: salta a
    un vecino de A cuya altura sea menor que ``nivel``. Si no salta, queda adherida a la pared: ocupa la
    posición ``nivel`` de A, la altura de A pasa a ser ``nivel + 1`` y las posiciones de A entre su altura
    anterior y ``nivel`` quedan vacías. Son huecos cerrados: el trazador trata cada columna como sólida
    hasta su cima, de modo que ningún vuelo posterior puede alcanzarlos. Si salta, la partícula se deposita
    sobre el vecino como cualquier otra y A conserva su altura. Sin la bandera, el contacto lateral se
    ignora y la partícula se coloca sobre la columna que la interceptó (aproximación *solid-on-solid*).

    La simulación se detiene antes de ``n_eventos`` si la columna más alta supera la altura media de las
    columnas en más de ``relieve_maximo`` capas. Con sombreado, las columnas que sobresalen interceptan más
    flujo cuanto más altas son y pueden crecer sin límite; como el vuelo se recorre celda a celda por debajo
    de la columna más alta, el coste del trazado crece con ellas. El corte acota ese coste y marca el espesor
    a partir del cual la superficie ya está dominada por esas columnas. Con ``relieve_maximo = np.inf`` no hay
    corte.

    Parámetros
    ----------
    eventos_muestreo : ndarray int64, forma (k,)
        Números de eventos, crecientes, sin repetir y entre 1 y ``n_eventos``, tras los que se registra
        la superficie.
    es_instantanea : ndarray bool, forma (k,)
        Qué instantes de ``eventos_muestreo`` guardan una copia de la superficie.
    max_pasos_rayo : int
        Salvaguarda del trazador (ver ``trazar_vuelo``).
    distancia_fuente : float
        Distancia entre el techo de vuelo y la columna más alta (espesor de la capa de gas), en capas.
    relieve_maximo : float
        Diferencia máxima, en capas, entre la columna más alta y la altura media de las columnas.
    ley_emision, theta_max : int, float
        Ley de emisión (0 gaussiana, 1 coseno acotada) y ángulo máximo de la ley del coseno, en grados.
    enable_lateral : bool
        Activa la adhesión lateral.
    volumen : ndarray uint8, forma (nx, ny, n_niveles)
        Ocupación de la película: la función pone a 1 la posición (i, j, nivel) de cada partícula depositada
        con nivel menor que ``n_niveles``. Con ``n_niveles = 1`` no se registra (basta un arreglo de forma
        (1, 1, 1)).

    Devuelve
    --------
    superficie : ndarray int32, forma (nx, ny)
        Altura de cada columna al terminar (o al detenerse): su posición ocupada más alta más uno.
    espesor, rugosidad : ndarray, forma (k,)
        Altura media de las columnas y su desviación estándar en cada instante de ``eventos_muestreo``;
        ``nan`` en los instantes posteriores al corte. Sin huecos, la altura media es el espesor medio de
        la película.
    maximo : ndarray, forma (k,)
        Altura de la columna más alta en cada instante (``nan`` tras el corte).
    huecos : ndarray, forma (k,)
        Número de posiciones vacías bajo la cima de las columnas en cada instante (``nan`` tras el corte).
    instantaneas : ndarray int32, forma (m, nx, ny)
        Superficies en los m instantes marcados, en orden; las posteriores al corte quedan en cero.
    n_colisiones : int
        Colisiones con el gas en toda la simulación.
    truncados : int
        Átomos cuyo vuelo detuvo la salvaguarda del trazador.
    depositados : int
        Partículas depositadas: ``n_eventos``, o el evento en que actuó el corte.
    """
    superficie = np.zeros((nx, ny), dtype=np.int32)
    k = eventos_muestreo.shape[0]
    espesor = np.zeros(k, dtype=np.float64)
    rugosidad = np.zeros(k, dtype=np.float64)
    maximo = np.zeros(k, dtype=np.float64)
    huecos_registro = np.zeros(k, dtype=np.float64)
    n_inst = 0
    for i in range(k):
        if es_instantanea[i]:
            n_inst += 1
    instantaneas = np.zeros((n_inst, nx, ny), dtype=np.int32)
    registra_volumen = volumen.shape[2] > 1
    n_niveles = volumen.shape[2]

    idx_m = 0
    idx_i = 0
    max_h_actual = 0
    n_colisiones = 0
    truncados = 0
    huecos = 0
    depositados = n_eventos
    n_celdas = nx * ny
    for h in range(1, n_eventos + 1):
        x_i = np.random.randint(0, nx)
        y_i = np.random.randint(0, ny)
        x_col, y_col, n_col, trunc, x_lat, y_lat, nivel = transporte_con_salvaguarda(
            x_i, y_i, nx, ny, superficie, max_h_actual, sigma_init, sigma_scat, mfp,
            distancia_fuente, enable_angle, enable_scattering, enable_shadowing, max_pasos_rayo, ley_emision, theta_max
        )
        n_colisiones += n_col
        truncados += trunc
        if enable_lateral and nivel >= 0:
            # La partícula está apoyada en la pared, en la columna (x_lat, y_lat) al nivel ``nivel``: se
            # relaja desde allí como si esa columna tuviera esa altura.
            altura_previa = superficie[x_lat, y_lat]
            superficie[x_lat, y_lat] = nivel
            x_f, y_f = modulo_relajacion_superficial(
                superficie, x_lat, y_lat, nx, ny, temp, t_escala, max_pasos, enable_diffusion
            )
            if x_f == x_lat and y_f == y_lat:
                huecos += nivel - altura_previa          # queda adherida: posiciones vacías bajo ella
            else:
                superficie[x_lat, y_lat] = altura_previa  # saltó a un vecino: la columna no cambia
        else:
            x_f, y_f = modulo_relajacion_superficial(
                superficie, x_col, y_col, nx, ny, temp, t_escala, max_pasos, enable_diffusion
            )
        if registra_volumen and superficie[x_f, y_f] < n_niveles:
            volumen[x_f, y_f, superficie[x_f, y_f]] = 1
        superficie[x_f, y_f] += 1
        if superficie[x_f, y_f] > max_h_actual:
            max_h_actual = superficie[x_f, y_f]

        if idx_m < k and h == eventos_muestreo[idx_m]:
            esp, rug = calcular_metricas(superficie, nx, ny)
            espesor[idx_m] = esp
            rugosidad[idx_m] = rug
            maximo[idx_m] = max_h_actual
            huecos_registro[idx_m] = huecos
            if es_instantanea[idx_m]:
                instantaneas[idx_i, :, :] = superficie
                idx_i += 1
            idx_m += 1

        # Tras h eventos, la suma de alturas es exactamente h más los huecos (conservación de partículas).
        if max_h_actual - (h + huecos) / n_celdas > relieve_maximo:
            depositados = h
            break

    for i in range(idx_m, k):
        espesor[i] = np.nan
        rugosidad[i] = np.nan
        maximo[i] = np.nan
        huecos_registro[i] = np.nan
    return superficie, espesor, rugosidad, maximo, huecos_registro, instantaneas, n_colisiones, truncados, depositados


@njit
def ejecutar_motor_muestreado(nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_init, sigma_scat, mfp,
                              distancia_fuente, enable_angle, enable_scattering, enable_shadowing, enable_diffusion,
                              eventos_muestreo, es_instantanea, max_pasos_rayo, relieve_maximo, ley_emision=0,
                              theta_max=0.0, enable_lateral=False):
    """``ejecutar_motor_volumen`` sin registro del volumen.

    Devuelve ``superficie, espesor, rugosidad, maximo, huecos, instantaneas, n_colisiones, truncados,
    depositados`` (ver ``ejecutar_motor_volumen``).
    """
    volumen = np.zeros((1, 1, 1), dtype=np.uint8)
    return ejecutar_motor_volumen(nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_init, sigma_scat, mfp,
                                  distancia_fuente, enable_angle, enable_scattering, enable_shadowing, enable_diffusion,
                                  eventos_muestreo, es_instantanea, max_pasos_rayo, relieve_maximo, ley_emision, theta_max,
                                  enable_lateral, volumen)


@njit(parallel=True)
def ejecutar_suite_muestreada(semillas, nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_init, sigma_scat, mfp,
                              distancia_fuente, enable_angle, enable_scattering, enable_shadowing,
                              enable_diffusion, eventos_muestreo, es_instantanea, max_pasos_rayo, relieve_maximo,
                              ley_emision=0, theta_max=0.0, enable_lateral=False):
    """Réplicas independientes de ``ejecutar_motor_muestreado``, en paralelo, una semilla por réplica.

    Devuelve
    --------
    superficies : ndarray int32, forma (n_replicas, nx, ny)
    espesores, rugosidades, maximos, huecos : ndarray, forma (n_replicas, k)
    instantaneas : ndarray int32, forma (n_replicas, m, nx, ny)
    colisiones_por_atomo : ndarray, forma (n_replicas,)
        Colisiones con el gas por partícula depositada.
    truncados : ndarray int64, forma (n_replicas,)
    depositados : ndarray int64, forma (n_replicas,)
        Partículas depositadas en cada réplica (menos que ``n_eventos`` si actuó el corte por relieve).
    """
    n_replicas = semillas.shape[0]
    k = eventos_muestreo.shape[0]
    n_inst = 0
    for i in range(k):
        if es_instantanea[i]:
            n_inst += 1
    superficies = np.zeros((n_replicas, nx, ny), dtype=np.int32)
    espesores = np.zeros((n_replicas, k), dtype=np.float64)
    rugosidades = np.zeros((n_replicas, k), dtype=np.float64)
    maximos = np.zeros((n_replicas, k), dtype=np.float64)
    huecos = np.zeros((n_replicas, k), dtype=np.float64)
    instantaneas = np.zeros((n_replicas, n_inst, nx, ny), dtype=np.int32)
    colisiones_por_atomo = np.zeros(n_replicas, dtype=np.float64)
    truncados = np.zeros(n_replicas, dtype=np.int64)
    depositados = np.zeros(n_replicas, dtype=np.int64)

    for rep in prange(n_replicas):
        np.random.seed(semillas[rep])
        sup, esp, rug, maxi, hue, inst, n_col, trunc, n_dep = ejecutar_motor_muestreado(
            nx, ny, n_eventos, temp, t_escala, max_pasos, sigma_init, sigma_scat, mfp, distancia_fuente,
            enable_angle, enable_scattering, enable_shadowing, enable_diffusion, eventos_muestreo, es_instantanea,
            max_pasos_rayo, relieve_maximo, ley_emision, theta_max, enable_lateral
        )
        superficies[rep] = sup
        espesores[rep, :] = esp
        rugosidades[rep, :] = rug
        maximos[rep, :] = maxi
        huecos[rep, :] = hue
        instantaneas[rep] = inst
        colisiones_por_atomo[rep] = n_col / n_dep
        truncados[rep] = trunc
        depositados[rep] = n_dep

    return superficies, espesores, rugosidades, maximos, huecos, instantaneas, colisiones_por_atomo, truncados, depositados
