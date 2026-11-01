"""
Configuración común: constantes del modelo, rutas, aceleración opcional y escala de ejecución.

Unidades reducidas en todo el proyecto: k_B = 1 y J1 = 1. Las temperaturas se
expresan en unidades de J1/k_B, el campo h en unidades de J1 y las entropías en
unidades de k_B.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
J1_DEFAULT = 1.0      # acoplamiento a primeros vecinos (J1 > 0: ferromagnético con H = -J1 sum S_i S_j)
J2_DEFAULT = -0.6     # acoplamiento a segundos vecinos (J2 < 0: antiferromagnético)
BASE_SEED = 12345     # semilla base de la que se derivan todas las demás (simulation.task_seed)

# ----------------------------------------------------------------------------
# Aceleración opcional con Numba
# ----------------------------------------------------------------------------
# Si Numba no está instalado, los núcleos numéricos se ejecutan como Python puro.
# Los resultados son los mismos (mismo algoritmo, mismo generador de NumPy), pero
# cientos de veces más lentos: solo es razonable para pruebas con redes pequeñas.
# ISINGJ1J2_DISABLE_NUMBA=1 fuerza el modo Python puro aunque Numba esté instalado.
try:
    if os.environ.get("ISINGJ1J2_DISABLE_NUMBA", "0") == "1":
        raise ImportError
    from numba import njit as _numba_njit
    NUMBA_AVAILABLE = True
except ImportError:
    _numba_njit = None
    NUMBA_AVAILABLE = False


def njit(*args, **kwargs):
    """
    Decorador equivalente a ``numba.njit`` cuando Numba está disponible.

    Por defecto compila con ``cache=True`` (el código máquina se guarda en
    ``__pycache__`` y los procesos de trabajo no recompilan) y ``nogil=True``
    (las funciones liberan el GIL, de modo que también pueden ejecutarse en
    paralelo con hilos). Sin Numba devuelve la función sin modificar.
    """
    decorating_directly = len(args) == 1 and callable(args[0])
    if _numba_njit is None:
        return args[0] if decorating_directly else (lambda f: f)
    kwargs.setdefault("cache", True)
    kwargs.setdefault("nogil", True)
    if decorating_directly:
        return _numba_njit(**kwargs)(args[0])
    return _numba_njit(*args, **kwargs)


# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Factor de escala del número de pasos Monte Carlo. En producción vale 1. Un valor
# menor (por ejemplo ISINGJ1J2_SCALE=0.05) ejecuta el notebook completo con
# corridas cortas, útil para comprobar la instalación; esos resultados se guardan
# en una carpeta aparte y no tienen valor físico.
MC_SCALE = float(os.environ.get("ISINGJ1J2_SCALE", "1"))


def data_dir():
    """Carpeta de resultados crudos de simulación (caché). Se puede cambiar con ISINGJ1J2_DATA_DIR."""
    default = PROJECT_ROOT / ("data" if MC_SCALE == 1.0 else f"data_scale_{MC_SCALE:g}")
    path = Path(os.environ.get("ISINGJ1J2_DATA_DIR", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir():
    """Carpeta de figuras exportadas por el notebook."""
    default = PROJECT_ROOT / ("figures" if MC_SCALE == 1.0 else f"figures_scale_{MC_SCALE:g}")
    path = Path(os.environ.get("ISINGJ1J2_FIGURES_DIR", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir():
    """Carpeta de tablas de resultados (CSV) exportadas por el notebook."""
    default = PROJECT_ROOT / ("results" if MC_SCALE == 1.0 else f"results_scale_{MC_SCALE:g}")
    path = Path(os.environ.get("ISINGJ1J2_RESULTS_DIR", default))
    path.mkdir(parents=True, exist_ok=True)
    return path
