"""
isingj1j2: simulación Monte Carlo del modelo de Ising J1-J2 en la red cuadrada.

El paquete reúne el código numérico que usa el notebook
``antiferromagnetic_ising2d_magnetocaloric_effect.ipynb``:

- ``lattice``     : tablas de vecinos, configuraciones de referencia y enumeración exacta.
- ``kernels``     : algoritmo de Metropolis y muestreo de observables (Numba, con respaldo en Python).
- ``simulation``  : definición de tareas, ejecución en paralelo, semillas y caché en disco.
- ``estimators``  : momentos, observables derivados, jackknife, bootstrap y tiempo de autocorrelación.
- ``fss``         : escalado de tamaño finito (picos, cruces, pendientes, colapso, exponentes).
- ``mce``         : efecto magnetocalórico (relación de Maxwell, integración de la entropía, Delta T_ad).
- ``plotting``    : estilo gráfico común.

Las funciones de análisis trabajan sobre arreglos de NumPy y no dependen del
estado del notebook, de modo que pueden probarse por separado (carpeta ``tests``).
"""

from .config import NUMBA_AVAILABLE, J1_DEFAULT, J2_DEFAULT, BASE_SEED

__all__ = ["NUMBA_AVAILABLE", "J1_DEFAULT", "J2_DEFAULT", "BASE_SEED"]
__version__ = "1.0.0"
