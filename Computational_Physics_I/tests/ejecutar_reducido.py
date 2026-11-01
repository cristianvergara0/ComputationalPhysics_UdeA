"""Ejecuta un cuaderno de punta a punta a escala reducida para comprobar que funciona.

Reduce el tamaño de la malla, el número de partículas y de réplicas (y acorta algunas listas de
parámetros) sustituyendo las asignaciones de nivel superior de las celdas de código, y ejecuta todas las
celdas, en orden, en un único espacio de nombres. No sustituye a la ejecución científica del cuaderno:
solo comprueba que las importaciones, las funciones, las tablas y las figuras funcionan juntas.

    python tests/ejecutar_reducido.py 07            # un cuaderno, por su número
    python tests/ejecutar_reducido.py 01 02 03      # varios

Las tablas y las figuras se escriben en una carpeta temporal (variable de entorno ``TFMC_RESULTADOS``,
que se crea si no está definida), no en ``resultados/``.
"""

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

ESCALA_REDUCIDA = {
    "01": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20,
           "REPLICAS": 2, "VENTANAS_BETA": "[(1, 4), (4, 16)]", "TAMANOS_MALLA": "[12, 24, 48]", "REPLICAS_MALLA": "[4, 2, 2]",
           "ESPESOR_MALLA": 4, "L_CALIBRACION": 12, "REPLICAS_CALIBRACION": 40, "ESPESORES_CALIBRACION": "[1, 4]"},
    "02": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 2,
           "VENTANA_EW": "(2, 16)", "TAMANOS_FSS": "[6, 8, 12]", "REPLICAS_FSS": "[6, 4, 3]", "N_MUESTRAS_LINEALES_FSS": 20,
           "TAMANO_MINIMO_AJUSTE": 6, "MAXIMOS_SALTOS": "[1, 5, 10]", "ESPESOR_SALTOS": 4, "ESPESORES_SALTOS": "[1, 4]"},
    "03": {"NX": 24, "NY": 24, "TEMPERATURAS": "[0, 100, 400, 500]", "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]",
           "N_MUESTRAS_LOG": 20, "REPLICAS": 2, "VENTANA_EW": "(2, 16)", "TEMPERATURAS_FSS": "[100, 400]",
           "TAMANOS_FSS": "[6, 8, 12]", "REPLICAS_FSS": "[6, 4, 3]", "N_MUESTRAS_LINEALES_FSS": 20, "TAMANO_MINIMO_AJUSTE": 6,
           "MAXIMOS_SALTOS": "[1, 5, 10]", "ESPESOR_SALTOS": 4, "ESPESORES_SALTOS": "[1, 4]", "T_GRUESA": 100, "L_GRUESA": 24,
           "REPLICAS_GRUESA": 2, "ESPESOR_GRUESA": 6, "VENTANA_TARDIA": "(2, 6)"},
    "04": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 3,
           "DISPERSIONES_GAUSS": "[0.0, 15.0, 60.0]", "DISPERSIONES_COSENO": "[35.0]", "ESPESOR_EMISION": 4,
           "ESPESOR_UNIFORMIDAD": 4, "N_VUELOS": 2000},
    "05": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 3,
           "RELIEVE_INESTABLE": 4.0, "RELIEVE_MAXIMO": 8.0, "L_MAPA": 12, "REPLICAS_MAPA": 2, "ESPESOR_MAPA": 8,
           "ESPESORES_MAPA": "[4, 8]", "DISPERSIONES_MAPA": "[0.0, 15.0, 35.0]", "PROBABILIDADES_MAPA": "[0.0, 0.8]",
           "DISPERSIONES_LEYES": "[15.0, 35.0]", "TAMANOS_MALLA": "[12, 24]", "REPLICAS_MALLA": "[3, 3]", "ESPESOR_MALLA": 8,
           "ESPESORES_MALLA": "[1, 4]", "ESPESOR_ROBUSTEZ": 8, "ESPESORES_ROBUSTEZ": "[4]", "L_MURO, X_MURO, N_RAYOS": "200, 30, 60",
           "L_CONTENCION, REPLICAS_CONTENCION, ESPESOR_CONTENCION": "30, 2, 8", "RELIEVE_COLUMNA": 2.0,
           "ESPESORES_FRACCION": "[2, 3, 4, 6, 8]"},
    "06": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 2, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 3,
           "RELIEVE_INESTABLE": 4.0, "RELIEVE_MAXIMO": 8.0, "CAMINOS_LIBRES": "[10.0, 40.0, np.inf]", "ESPESOR_CAMINOS": 8,
           "ESPESORES_CAMINOS": "[1, 2]", "ESPESOR_ANGULAR": 1, "REPLICAS_ANGULAR": 3, "DISPERSIONES_COLISION": "[10.0, 20.0, 45.0]",
           "CAMINOS_LIBRES_PLANO": "[5.0, 10.0]", "DISPERSIONES_BALISTICAS": "[0.0, 15.0, 45.0, 180.0]", "L_MAPA": 12,
           "REPLICAS_MAPA": 2, "ESPESOR_MAPA": 8, "ESPESORES_MAPA": "[2, 8]", "PROBABILIDADES_MAPA": "[0.0, 0.8]",
           "CAMINOS_LIBRES_MAPA": "[10.0, np.inf]", "TAMANOS_MALLA": "[12, 24]", "REPLICAS_MALLA": "[3, 3]", "ESPESOR_MALLA": 8,
           "ESPESORES_MALLA": "[1, 2]", "ESPESOR_ROBUSTEZ": 8, "ESPESORES_ROBUSTEZ": "[1, 2]", "N_VUELOS": 300,
           "L_CONTENCION, REPLICAS_CONTENCION, ESPESOR_CONTENCION": "20, 2, 4", "RELIEVE_COLUMNA": 2.0,
           "ESPESORES_FRACCION": "[1, 1.5, 2, 3, 4, 6, 8]"},
    "07": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 3,
           "INTERVALOS_POROSIDAD": "[(1, 4), (4, 16)]", "L_MAPA": 12, "REPLICAS_MAPA": 2, "ESPESOR_MAPA": 8, "ESPESORES_MAPA": "[4, 8]",
           "PROBABILIDADES_MAPA": "[0.0, 0.8]", "CAMINOS_LIBRES_MAPA": "[10.0, np.inf]", "DISPERSIONES_ANGULARES": "[0.0, 35.0]",
           "PROBABILIDADES_ANGULARES": "[0.0, 0.8]", "TAMANOS_MALLA": "[12, 24]", "REPLICAS_MALLA": "[3, 3]", "ESPESOR_MALLA": 16,
           "ESPESORES_MALLA": "[4, 16]", "ESPESOR_ROBUSTEZ": 16, "ESPESORES_ROBUSTEZ": "[4, 16]",
           "L_CONTENCION, REPLICAS_CONTENCION, ESPESOR_CONTENCION": "20, 2, 4", "L_MURO, X_MURO, ALTURA_MURO, N_RAYOS": "200, 30, 30, 80",
           "TAMANOS_GRUESA": "[8, 16, 24]", "REPLICAS_GRUESA": "[6, 4, 3]", "ESPESOR_GRUESA": 64,
           "ESPESORES_ESPECTRO": "[16, 32, 64]", "N_MUESTRAS_GRUESA": 20, "N_LINEALES_GRUESA": 9,
           "VENTANAS_BETA": "[(4, 16), (16, 64)]", "VENTANA_SATURACION": "(32, 64)", "MALLAS_SATURADAS": "[8, 16]", "ESPESORES_MONTICULOS": "(16, 32)",
           "N_REMUESTREOS": 50},
    "08": {"NX": 24, "NY": 24, "ESPESOR_MAXIMO": 16, "ESPESORES_INSTANTANEAS": "[1, 4, 16]", "N_MUESTRAS_LOG": 20, "REPLICAS": 3,
           "VENTANAS_BETA": "[(1, 4), (4, 16)]", "REPLICAS_FACTORIAL": 3,
           "L_CONTENCION, REPLICAS_CONTENCION, ESPESOR_CONTENCION": "16, 2, 3"},
}


def reducir(codigo, cambios, usados):
    """Sustituye la parte derecha de las asignaciones ``NOMBRE = valor`` (o ``A, B = v1, v2``) de nivel superior."""
    claves = {re.sub(r"\s+", "", k): (k, v) for k, v in cambios.items()}
    salida = []
    for linea in codigo.splitlines():
        m = re.match(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=\s*(?!=)", linea)
        if m and re.sub(r"\s+", "", m.group(1)) in claves:
            clave, valor = claves[re.sub(r"\s+", "", m.group(1))]
            salida.append(f"{m.group(1)} = {valor}")
            usados.add(clave)
        else:
            salida.append(linea)
    return "\n".join(salida)


def ejecutar(ruta, cambios):
    """Ejecuta las celdas de código de ``ruta``; devuelve (celdas ejecutadas, lista de errores)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    celdas = json.loads(Path(ruta).read_text(encoding="utf-8"))["cells"]
    espacio = {"__name__": "__main__", "display": print}
    usados, errores, n = set(), [], 0
    cwd = os.getcwd()
    os.chdir(RAIZ)
    try:
        for i, celda in enumerate(celdas):
            if celda["cell_type"] != "code":
                continue
            texto = "".join(celda["source"]) if isinstance(celda["source"], list) else celda["source"]
            texto = "\n".join("pass" if l.lstrip().startswith(("%", "!")) else l for l in texto.splitlines())
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                try:
                    exec(compile(reducir(texto, cambios, usados), f"{Path(ruta).name}:celda {i}", "exec"), espacio)
                    n += 1
                except BaseException as error:        # noqa: BLE001 - se informa de cualquier fallo
                    errores.append((i, f"{type(error).__name__}: {error}", traceback.format_exc()[-1200:]))
                    break
            plt.close("all")
    finally:
        os.chdir(cwd)
    sin_usar = sorted(set(cambios) - usados)
    if sin_usar:
        errores.append((-1, f"parámetros reducidos que el cuaderno ya no define: {sin_usar}", ""))
    return n, errores


def main(argumentos):
    os.environ.setdefault("TFMC_RESULTADOS", tempfile.mkdtemp(prefix="tfmc_resultados_"))
    sys.path.insert(0, str(RAIZ))
    codigo_salida = 0
    for numero in argumentos:
        rutas = sorted(RAIZ.glob(f"{numero}_*.ipynb"))
        if not rutas:
            print(f"{numero}: no hay cuaderno con ese número")
            codigo_salida = 1
            continue
        t0 = time.time()
        n, errores = ejecutar(rutas[0], ESCALA_REDUCIDA[numero])
        estado = "sin errores" if not errores else f"{len(errores)} error(es)"
        print(f"{rutas[0].name}: {n} celdas ejecutadas a escala reducida en {time.time() - t0:.0f} s, {estado}")
        for i, mensaje, traza in errores:
            print(f"   celda {i}: {mensaje}")
            if traza:
                print("   " + traza.replace("\n", "\n   "))
            codigo_salida = 1
    return codigo_salida


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(ESCALA_REDUCIDA)))
