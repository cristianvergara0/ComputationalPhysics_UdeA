"""Código compartido de la serie de cuadernos sobre el crecimiento de películas delgadas por Monte Carlo.

Módulos
-------
compat       Numba como dependencia opcional (``njit`` y ``prange`` con respaldo en Python puro).
entorno      Rutas del proyecto, guardado de tablas y registro de las versiones del entorno.
analisis     Métricas de superficie, espectro de potencia, correlación de alturas y exponentes efectivos.
estadistica  Bondad de ajuste, intervalos de confianza, contraste de Welch, interacciones, ANOVA y regresión.
motor        Núcleos físicos: relajación, trazador de rayos exacto, adhesión lateral y lazos de deposición.
barridos     Parámetros del sistema, barridos, referencias balísticas, colapso angular y ley del coseno.
crecimiento  Simulación con instantes de registro y análisis del crecimiento (saturación, inestabilidad,
             porosidad y perfil de densidad).
graficos     Figuras reutilizables.

Cada cuaderno importa de forma explícita lo que necesita de estos módulos, por ejemplo
``from thinfilm_mc.analisis import calcular_metricas``.
"""

__version__ = "1.0.0"
