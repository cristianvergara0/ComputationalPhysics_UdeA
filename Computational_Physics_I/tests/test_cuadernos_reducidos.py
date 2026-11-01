"""Ejecuta cada cuaderno de punta a punta a escala reducida (prueba de integración).

Es lenta en Python puro (varios minutos en total), de modo que solo se ejecuta si se define
``TFMC_PRUEBAS_LENTAS=1``. Con Numba tarda menos, aunque la compilación domina:

    TFMC_PRUEBAS_LENTAS=1 python -m unittest discover -s tests -p "test_cuadernos_reducidos.py" -v

Cada cuaderno se ejecuta en un proceso aparte, con las salidas en una carpeta temporal.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LENTAS = os.environ.get("TFMC_PRUEBAS_LENTAS") == "1"
NUMEROS = sorted(p.name[:2] for p in RAIZ.glob("[0-9][0-9]_*.ipynb"))


@unittest.skipUnless(LENTAS, "prueba lenta: defina TFMC_PRUEBAS_LENTAS=1")
class TestCuadernosAEscalaReducida(unittest.TestCase):
    def test_cada_cuaderno_se_ejecuta_sin_errores(self):
        for numero in NUMEROS:
            with self.subTest(cuaderno=numero), tempfile.TemporaryDirectory() as tmp:
                entorno = dict(os.environ, TFMC_RESULTADOS=tmp, MPLBACKEND="Agg")
                resultado = subprocess.run([sys.executable, str(RAIZ / "tests" / "ejecutar_reducido.py"), numero],
                                           capture_output=True, text=True, env=entorno, timeout=3600)
                self.assertEqual(resultado.returncode, 0, resultado.stdout + resultado.stderr)
                self.assertIn("sin errores", resultado.stdout)


if __name__ == "__main__":
    unittest.main()
