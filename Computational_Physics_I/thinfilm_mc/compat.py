"""Compatibilidad con Numba.

Numba es una dependencia opcional de aceleración. Si está instalada, ``njit`` y ``prange`` son los
de Numba y las funciones numéricas se compilan; si no lo está, ``njit`` deja la función sin compilar
y ``prange`` es ``range``. El código es el mismo en ambos casos. Los generadores pseudoaleatorios de
NumPy y de Numba son independientes y cada uno se siembra por su vía (ver ``motor.sembrar_generador``).
Con Numba instalada, ``tests/test_numba.py`` comprueba que las rutinas compiladas producen las mismas
superficies que las mismas rutinas ejecutadas con ``NUMBA_DISABLE_JIT=1``.
"""

try:
    from numba import njit, prange
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    prange = range

    def njit(*args, **kwargs):
        """Sustituto de ``numba.njit`` que deja la función sin compilar.

        Admite tanto la forma ``@njit`` como la forma ``@njit(...)``.
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorador(funcion):
            return funcion

        return decorador

__all__ = ["NUMBA_DISPONIBLE", "njit", "prange"]
