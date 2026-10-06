#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analizador de oscilaciones de un imán en un resorte.

Un imán sujeto a un resorte oscila sobre una bobina. La fem inducida se digitaliza con un ESP32 y este
programa extrae del registro la frecuencia de oscilación, el coeficiente de amortiguamiento y una estimación
del campo magnético. Con los datos del montaje que se conozcan (vueltas y radio de la bobina, masa, constante
del resorte) calcula además el resto de las magnitudes que se pueden deducir.

Uso:
    1. Edita config.json, que está junto al script: fuente de los datos y lo que conozcas del montaje.
    2. Ejecuta:  python spring_magnet_analyzer.py
    Los resultados se guardan en una subcarpeta nueva de resultados/ con la fecha y la hora.

    python spring_magnet_analyzer.py --config otro.json     usa otro archivo de configuración
    python spring_magnet_analyzer.py --crear-config         crea un config.json de ejemplo
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import platform
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import scipy
from scipy.constants import mu_0
from scipy.fft import next_fast_len, rfft, rfftfreq
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, hilbert, medfilt, windows
from scipy.stats import linregress

try:
    import serial  # pyserial: solo se necesita para adquirir datos con el ESP32
except ImportError:
    serial = None

__version__ = "1.0"

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ADC_MAX = 4095                # Fondo de escala del ADC de 12 bits (cuentas)
V_REF_NOMINAL = 3.3           # V. Tensión de referencia nominal del ADC
F_MIN = 0.2                   # Hz. Frecuencia mínima para buscar el pico del espectro
FACTOR_RELLENO = 8            # Relleno de ceros de la FFT: afina la resolución en frecuencia
KERNEL_MEDIANO = 101          # Muestras (impar) del filtro mediano de la envolvente
FRAC_DISTANCIA = 0.85         # Distancia mínima entre picos, en fracción del periodo 1/f0
FRAC_PROMINENCIA = 0.04       # Prominencia mínima de un pico, en fracción del máximo de la envolvente
UMBRAL_SIGMA = 2.5            # Rechazo de valores atípicos en el ajuste de ln A frente a t
MAX_ITER_AJUSTE = 4           # Iteraciones máximas de ese rechazo
MIN_LOBULOS_AREA = 6          # Lóbulos mínimos para estimar gamma a partir de sus áreas
FRAC_AREA_MIN = 0.05          # Se ignoran lóbulos con menos de esta fracción del área máxima
CICLOS_MINIMOS = 5            # Ciclos mínimos que debe contener el registro
PICO_ESPECTRAL_MIN = 10.0     # Cociente mínimo pico / mediana del espectro para aceptar una oscilación
UMBRAL_DISCREPANCIA = 0.10    # Tolerancia del cociente gamma(altura) / gamma(área)
ANCHO_RELATIVO_MIN = 0.75     # Duración mínima del lóbulo, en fracción del medio periodo
UMBRAL_SUBARMONICO = 0.15     # Amplitud relativa en f0/2 a partir de la cual se avisa
TOLERANCIA_CHEQUEO = 0.10     # Diferencia relativa a partir de la cual dos valores "no concuerdan"
PUERTO_DEFECTO = "COM10" if os.name == "nt" else "/dev/ttyUSB0"


class ErrorAnalisis(Exception):
    """Error que se explica al usuario sin mostrar el traceback."""


@dataclass
class Montaje:
    """Datos del montaje que el usuario conoce, en unidades SI. None significa que no se conoce."""
    vueltas: Optional[float] = None        # Vueltas de la bobina
    radio: Optional[float] = None          # m. Radio de la bobina
    masa: Optional[float] = None           # kg. Masa que oscila (imán y soporte)
    masa_resorte: Optional[float] = None   # kg
    k: Optional[float] = None              # N/m. Constante del resorte
    b: Optional[float] = None              # kg/s. Coeficiente de amortiguamiento


@dataclass
class OpcionesAnalisis:
    """Parámetros del análisis; los valores por defecto son los validados con registros simulados."""
    f_min: float = F_MIN                    # Hz
    kernel_mediano: int = KERNEL_MEDIANO    # Muestras
    umbral_sigma: float = UMBRAL_SIGMA
    frac_prominencia: float = FRAC_PROMINENCIA


@dataclass
class Configuracion:
    """Contenido ya validado del archivo de configuración."""
    base: Path                              # Carpeta del archivo de configuración: desde aquí se cuentan las rutas relativas
    origen: Optional[Path] = None           # Archivo de configuración leído
    fuente: str = "demo"                    # demo | archivo | serie
    archivo: Optional[Path] = None
    fs: Optional[float] = None              # Hz. Solo si el archivo no tiene su meta_<fecha>.json
    puerto: str = PUERTO_DEFECTO
    baudrate: int = 115200
    muestras: int = 40000
    v_ref: float = V_REF_NOMINAL
    v_cal: Optional[float] = None           # V. Tensión conocida para calibrar el ADC
    adc_cal: Optional[float] = None         # Cuentas leídas con esa tensión
    montaje: Montaje = field(default_factory=Montaje)
    analisis: OpcionesAnalisis = field(default_factory=OpcionesAnalisis)
    demo_amplitud: float = 0.012            # m
    demo_altura: float = 0.035              # m
    demo_zona_muerta: float = 0.0           # V
    semilla: int = 1
    salida: Path = Path("resultados")
    graficas: bool = True
    mostrar: bool = False
    avisos: list = field(default_factory=list)   # Observaciones sobre el archivo de configuración


# ---------------------------------------------------------------------------
# Registro: adquisición, lectura y guardado
# ---------------------------------------------------------------------------
def listar_puertos():
    """Nombres de los puertos serie disponibles; lista vacía si no hay pyserial."""
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


def acquire_trial(trial_label=None, n_samples=40000, puerto=PUERTO_DEFECTO, baudrate=115200, wait_before=1.0):
    """Adquiere un registro del ADC del ESP32 a través del puerto serie.

    Parámetros
    ----------
    trial_label : str, opcional
        Nombre del ensayo; solo aparece en los mensajes.
    n_samples : int
        Número de líneas (muestras) que se leen.
    puerto, baudrate : str, int
        Puerto serie y velocidad; deben coincidir con los del sketch del ESP32.
    wait_before : float
        Espera (s) tras abrir el puerto. Abrirlo reinicia el ESP32, que necesita unos instantes para transmitir.

    Devuelve
    --------
    t_sec : ndarray, s
        Instante de cada muestra, t_k = k / fs_est con k = 0, 1, ...
    voltajes : ndarray, cuentas del ADC (0 a 4095)
        Lecturas del ADC; NaN en las líneas que no se pudieron interpretar.
    fs_est : float, Hz
        Frecuencia de muestreo efectiva, n_samples / duración de la captura.
    tiempo_idx : ndarray
        Índice de muestra, de 1 a n_samples.

    Se supone que las muestras son equiespaciadas. Si el ESP32 deja de transmitir, la captura se aborta con un
    error en lugar de esperar el tiempo de espera de cada muestra restante.
    """
    if serial is None:
        raise ErrorAnalisis("Para adquirir datos hace falta pyserial: pip install pyserial")
    max_lecturas_vacias = 3   # Tiempos de espera consecutivos tras los que se da por perdida la conexión
    etiqueta = f" '{trial_label}'" if trial_label else ""
    try:
        ser = serial.Serial(puerto, baudrate, timeout=2)
    except Exception as e:
        disponibles = ", ".join(listar_puertos()) or "ninguno detectado"
        raise ErrorAnalisis(f"No se pudo abrir {puerto}: {e}\nPuertos disponibles: {disponibles}") from e
    try:
        print(f"Ensayo{etiqueta}: conectado a {puerto}. Esperando {wait_before} s...")
        time.sleep(wait_before)
        ser.reset_input_buffer()
        ser.readline()   # Tras vaciar el búfer la primera línea puede llegar truncada: se descarta

        voltajes = np.full(n_samples, np.nan)
        vacias_seguidas = 0
        print(f"Iniciando captura de {n_samples} muestras...")
        t_start = time.perf_counter()   # Reloj monótono: no le afectan los ajustes de la hora del sistema
        for k in range(n_samples):
            linea = ser.readline().decode("utf-8", errors="ignore").strip()
            if not linea:
                vacias_seguidas += 1
                if vacias_seguidas >= max_lecturas_vacias:
                    raise ErrorAnalisis("El ESP32 dejó de enviar datos (tiempo de espera agotado).")
                continue
            vacias_seguidas = 0
            try:
                voltajes[k] = float(linea)
            except ValueError:
                pass   # Línea corrupta: la muestra queda como NaN y se interpola más adelante
        duracion = time.perf_counter() - t_start
    finally:
        ser.close()

    fs_est = n_samples / duracion
    print(f"Captura terminada: {duracion:.2f} s, fs estimada = {fs_est:.2f} Hz")
    return np.arange(n_samples) / fs_est, voltajes, fs_est, np.arange(1, n_samples + 1)


def buscar_fs_en_meta(ruta_raw):
    """Frecuencia de muestreo guardada en el meta_<fecha>.json que acompaña a un medicion_raw_<fecha>.csv."""
    coincide = re.match(r"medicion_raw_(.+)\.csv$", os.path.basename(ruta_raw))
    if not coincide:
        return None
    ruta_meta = os.path.join(os.path.dirname(ruta_raw), f"meta_{coincide.group(1)}.json")
    if not os.path.exists(ruta_meta):
        return None
    with open(ruta_meta, encoding="utf-8") as f:
        return json.load(f).get("fs_est_Hz")


def cargar_registro(ruta_raw, fs=None):
    """Carga un registro crudo guardado por este programa (medicion_raw_<fecha>.csv).

    Parámetros
    ----------
    ruta_raw : str
        Archivo CSV con las columnas `index,adc` (índice de muestra y cuentas del ADC).
    fs : float, opcional
        Frecuencia de muestreo (Hz). Si no se da, se lee `fs_est_Hz` de meta_<fecha>.json, que debe estar en la
        misma carpeta que el registro.

    Devuelve
    --------
    Las mismas cuatro magnitudes que `acquire_trial`: t_sec (s), voltajes (cuentas), fs_est (Hz) y tiempo_idx.
    """
    if not os.path.exists(ruta_raw):
        raise ErrorAnalisis(f"No existe el archivo {ruta_raw}")
    datos = np.genfromtxt(ruta_raw, delimiter=",", skip_header=1)
    if datos.ndim != 2 or datos.shape[1] != 2:
        raise ErrorAnalisis(f"{ruta_raw} debe tener dos columnas: index,adc")
    tiempo_idx = datos[:, 0].astype(int)
    voltajes = datos[:, 1]
    if fs is None:
        fs = buscar_fs_en_meta(ruta_raw)
    if fs is None:
        raise ErrorAnalisis("No se encontró el meta_<fecha>.json del registro: escribe la frecuencia de muestreo "
                            "en 'fuente.fs_Hz' del config.json.")
    if not fs > 0:
        raise ErrorAnalisis("La frecuencia de muestreo debe ser positiva.")
    return (tiempo_idx - 1) / fs, voltajes, float(fs), tiempo_idx


def guardar_registro(carpeta, ts, tiempo_idx, voltajes, fs, extra=None):
    """Guarda el registro crudo (medicion_raw_<ts>.csv) y su meta_<ts>.json. Devuelve las dos rutas."""
    os.makedirs(carpeta, exist_ok=True)
    ruta_raw = os.path.join(carpeta, f"medicion_raw_{ts}.csv")
    np.savetxt(ruta_raw, np.column_stack((tiempo_idx, voltajes)), delimiter=",", fmt="%.10g",
               header="index,adc", comments="")
    meta = {"timestamp": ts, "fs_est_Hz": float(fs), "n_muestras": int(len(voltajes))}
    if extra:
        meta.update(extra)
    ruta_meta = os.path.join(carpeta, f"meta_{ts}.json")
    with open(ruta_meta, "w", encoding="utf-8") as f:
        json.dump(a_python(meta), f, indent=2, ensure_ascii=False)
    return ruta_raw, ruta_meta


def ruta_relativa(ruta, base):
    """Ruta de `ruta` relativa a `base`, con barras normales; si no se puede (otra unidad en Windows), la ruta tal cual."""
    try:
        return Path(os.path.relpath(ruta, base)).as_posix()
    except ValueError:
        return str(ruta)


def a_python(obj):
    """Convierte tipos de NumPy a tipos de Python para poder escribir JSON."""
    if isinstance(obj, dict):
        return {str(k): a_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [a_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return a_python(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Simulador del experimento
# ---------------------------------------------------------------------------
def simular_registro(m=0.200, k=16.6, b=0.0088, vueltas=3400, radio=0.0415, altura=0.035, amplitud=0.012,
                     pico_V=0.25, fs=1000.0, n_muestras=40000, ruido_V=5e-4, n_espurios=2, offset_V=0.0,
                     zona_muerta_V=0.0, t_inicio=0.0, v_ref=V_REF_NOMINAL, semilla=1):
    """Simula el registro que entregaría el ADC del ESP32 y devuelve también los valores verdaderos.

    Modelo: el cuerpo de masa m, unido a un resorte de constante k y amortiguamiento viscoso b, se suelta desde
    el reposo en x = amplitud por encima de su posición de equilibrio, que está a `altura` sobre el plano de la
    bobina. El imán se trata como un dipolo magnético sobre el eje de la bobina, de modo que el flujo por vuelta
    es Phi(d) = mu0 p R^2 / (2 (R^2 + d^2)^(3/2)) a la distancia d, y la fem es -N dPhi/dt. El momento dipolar p
    se ajusta para que el pico de la fem sea `pico_V`.

    El ADC se modela como lineal, de 12 bits, con ruido gaussiano, `offset_V` (desplazamiento de la señal) y una
    zona muerta opcional: las tensiones por debajo de `zona_muerta_V` se leen como cero y el resto se desplaza
    esa cantidad. Las tensiones negativas se leen como cero.

    Parámetros
    ----------
    m, k, b : float
        Masa (kg), constante del resorte (N/m) y coeficiente de amortiguamiento (kg/s).
    vueltas, radio : float
        Vueltas y radio (m) de la bobina.
    altura, amplitud : float
        Altura del equilibrio sobre la bobina (m) y amplitud inicial (m).
    pico_V : float
        Pico de la fem (V).
    fs, n_muestras : float, int
        Frecuencia de muestreo (Hz) y número de muestras.
    ruido_V : float
        Desviación típica del ruido (V).
    n_espurios : int
        Picos de interferencia de tres muestras añadidos en posiciones aleatorias.
    offset_V, zona_muerta_V : float
        Desplazamiento de la señal y zona muerta del ADC (V).
    t_inicio : float
        Instante, contado desde que se suelta el cuerpo, en el que empieza el registro (s).
    v_ref : float
        Tensión de referencia del ADC (V).
    semilla : int
        Semilla del generador aleatorio.

    Devuelve
    --------
    t_sec, voltajes, fs, verdad
        Instantes (s), cuentas del ADC, frecuencia de muestreo (Hz) y diccionario con los valores verdaderos.
    """
    w0 = math.sqrt(k / m)
    gamma = b / (2.0 * m)
    if gamma >= w0:
        raise ErrorAnalisis("El simulador necesita un sistema subamortiguado (b < 2*sqrt(k*m)).")
    w1 = math.sqrt(w0 ** 2 - gamma ** 2)
    rng = np.random.default_rng(semilla)

    t_rec = np.arange(n_muestras) / fs
    t = t_inicio + t_rec
    envolvente = amplitud * np.exp(-gamma * t)
    x = envolvente * (np.cos(w1 * t) + (gamma / w1) * np.sin(w1 * t))
    v = -envolvente * (w0 ** 2 / w1) * np.sin(w1 * t)
    d = altura + x

    def flujo(dist, momento):
        return momento * mu_0 * radio ** 2 / (2.0 * (radio ** 2 + dist ** 2) ** 1.5)

    base = 3.0 * vueltas * (mu_0 * radio ** 2 / 2.0) * d * (radio ** 2 + d ** 2) ** -2.5 * v   # fem por A*m^2
    momento = pico_V / base.max()
    fem = momento * base

    v_scale = v_ref / ADC_MAX
    tension = fem + offset_V + rng.normal(0.0, ruido_V, n_muestras)
    for pos in rng.integers(int(0.1 * n_muestras), int(0.9 * n_muestras), size=n_espurios):
        tension[pos:pos + 3] += 0.4 * pico_V
    leida = np.maximum(tension - zona_muerta_V, 0.0)
    cuentas = np.clip(np.rint(leida / v_scale), 0, ADC_MAX)

    # Valores verdaderos: los lóbulos de la fem van entre dos puntos de retorno consecutivos, t_j = j*pi/w1
    area_espira = math.pi * radio ** 2
    lobulos = []
    j = 0
    while (j + 1) * math.pi / w1 <= t_inicio + t_rec[-1]:
        t_a, t_b = j * math.pi / w1, (j + 1) * math.pi / w1
        if t_a >= t_inicio:
            pos_a = altura + amplitud * math.exp(-gamma * t_a) * (-1) ** j
            pos_b = altura + amplitud * math.exp(-gamma * t_b) * (-1) ** (j + 1)
            integral = abs(vueltas * (flujo(pos_b, momento) - flujo(pos_a, momento)))
            positivo = (j % 2 == 1)   # el ascenso da fem positiva
            if positivo or offset_V != 0.0:
                lobulos.append(integral)
        j += 1
    integral_verdadera = max(lobulos)
    verdad = {
        "masa_kg": m, "k_N_m": k, "b_kg_s": b, "gamma_1_s": gamma, "omega0_rad_s": w0, "omega1_rad_s": w1,
        "f0_Hz": w1 / (2.0 * math.pi), "Q": w0 / (2.0 * gamma), "vueltas": vueltas, "radio_m": radio,
        "integral_emf_Vs": integral_verdadera, "B_T": integral_verdadera / (vueltas * area_espira),
        "momento_dipolar_Am2": momento, "altura_m": altura, "amplitud_m": amplitud,
        "V_scale_V_per_count": v_scale, "zona_muerta_V": zona_muerta_V, "offset_V": offset_V,
    }
    return t_rec, cuentas, float(fs), verdad


# ---------------------------------------------------------------------------
# Análisis de la señal
# ---------------------------------------------------------------------------
def preparar_senal(t_sec, voltajes, v_scale):
    """Interpola las lecturas ilegibles y expresa la señal en voltios respecto al nivel de reposo.

    El ADC recorta las tensiones negativas, así que la señal es un tren de pulsos unipolares y su media no es
    el nivel de fem nula. Mientras los pulsos ocupen menos de la mitad del tiempo, la mediana sí lo es.

    Devuelve un diccionario con
        v : señal en V respecto al reposo (para amplitudes e integrales)
        proc : la misma señal con media cero (para el espectro)
        interp, reposo : cuentas con los NaN interpolados y nivel de reposo en cuentas
        valid : máscara de las lecturas legibles
        advertencias : lista de textos
    """
    voltajes = np.asarray(voltajes, dtype=float)
    n = len(voltajes)
    valid = np.isfinite(voltajes)
    n_validas = int(np.count_nonzero(valid))
    if n_validas < 100:
        raise ErrorAnalisis("Hay menos de 100 muestras legibles: no se puede analizar el registro.")
    advertencias = []
    if n_validas < 0.99 * n:
        advertencias.append(f"Se interpolaron {n - n_validas} de {n} muestras ({100 * (n - n_validas) / n:.1f} %).")
    interp = np.interp(t_sec, t_sec[valid], voltajes[valid])
    if np.any(interp >= ADC_MAX):
        advertencias.append("Hay lecturas en el fondo de escala del ADC (saturación): amplitudes e integrales "
                            "quedan subestimadas.")
    reposo = float(np.median(interp))
    v = (interp - reposo) * v_scale
    if np.mean(np.abs(v) > v_scale) >= 0.5:
        advertencias.append("La señal está activa más de la mitad del tiempo: la mediana ya no es el nivel de "
                            "reposo y las amplitudes pueden estar sesgadas.")
    return {"v": v, "proc": v - v.mean(), "interp": interp, "reposo": reposo, "valid": valid,
            "advertencias": advertencias, "n_interpoladas": n - n_validas}


def frecuencia_dominante(x, fs, f_min=F_MIN, relleno=FACTOR_RELLENO):
    """Frecuencia dominante de una señal real y ancho de su pico espectral.

    Se multiplica la señal por una ventana de Hann, se calcula la FFT real con relleno de ceros y se toma el
    máximo del espectro de amplitud para f >= f_min. La incertidumbre es la mitad del ancho a media altura
    (FWHM) del pico, cuyos cruces con la media altura se obtienen por interpolación lineal.

    Parámetros
    ----------
    x : ndarray
        Señal real de media cero, en cualquier unidad.
    fs : float
        Frecuencia de muestreo, Hz.
    f_min : float
        Frecuencia mínima, Hz, desde la que se busca el máximo.
    relleno : int
        Factor de relleno de ceros; no cambia el ancho físico de la línea, solo muestrea mejor el espectro.

    Devuelve
    --------
    dict con
        f0, sigma_f, fwhm, f_izq, f_der : Hz (frecuencia dominante, FWHM / 2, FWHM y cruces a media altura)
        freqs : ndarray, Hz (frecuencias positivas)
        espectro : ndarray (amplitud normalizada a su máximo)
        contraste : cociente entre el pico y la mediana del espectro
        amp_subarmonico : amplitud relativa al pico en f0 / 2
    """
    n = len(x)
    n_fft = next_fast_len(n * relleno, real=True)
    amplitud = np.abs(rfft(x * windows.hann(n), n=n_fft))
    freqs = rfftfreq(n_fft, d=1.0 / fs)
    amplitud, freqs = amplitud[1:], freqs[1:]   # Se excluye f = 0
    if not amplitud.max() > 0:
        raise ErrorAnalisis("El espectro es nulo: el registro no contiene ninguna oscilación.")
    espectro = amplitud / amplitud.max()

    candidatos = np.flatnonzero(freqs >= f_min)
    if candidatos.size == 0:
        raise ErrorAnalisis("f_min supera la frecuencia máxima del espectro.")
    i_pico = candidatos[np.argmax(espectro[candidatos])]
    f0 = freqs[i_pico]
    mitad = espectro[i_pico] / 2.0

    # Último punto bajo la media altura a la izquierda del pico y primero a la derecha; se interpola entre ese
    # punto y su vecino hacia el pico. Si no hay cruce, se toma el borde del espectro.
    izq = np.flatnonzero(espectro[:i_pico] <= mitad)
    if izq.size:
        j = izq[-1]
        f_izq = np.interp(mitad, [espectro[j], espectro[j + 1]], [freqs[j], freqs[j + 1]])
    else:
        f_izq = freqs[0]
    der = np.flatnonzero(espectro[i_pico:] <= mitad)
    if der.size:
        j = i_pico + der[0]
        f_der = np.interp(mitad, [espectro[j], espectro[j - 1]], [freqs[j], freqs[j - 1]])
    else:
        f_der = freqs[-1]
    fwhm = f_der - f_izq

    amp_sub = 0.0
    if f0 / 2.0 >= f_min:
        cerca = np.abs(freqs - f0 / 2.0) <= max(fwhm, 2.0 * (freqs[1] - freqs[0]))
        if cerca.any():
            amp_sub = float(espectro[cerca].max() / espectro[i_pico])
    contraste = float(espectro[i_pico] / max(np.median(espectro), 1e-300))
    return {"f0": f0, "sigma_f": fwhm / 2.0, "fwhm": fwhm, "f_izq": f_izq, "f_der": f_der, "freqs": freqs,
            "espectro": espectro, "contraste": contraste, "amp_subarmonico": amp_sub}


def envolvente_y_picos(x, f0, fs, kernel=KERNEL_MEDIANO, frac_prominencia=FRAC_PROMINENCIA):
    """Envolvente de Hilbert suavizada de una señal y posiciones de sus picos (uno por ciclo).

    Parámetros
    ----------
    x : ndarray
        Señal real, referida a su nivel de fem nula para que la altura de los pulsos sea la amplitud.
    f0 : float
        Frecuencia dominante, Hz; fija la distancia mínima entre picos.
    fs : float
        Frecuencia de muestreo, Hz.
    kernel : int
        Tamaño, en muestras, de la ventana del filtro mediano (si es par se aumenta en uno).
    frac_prominencia : float
        Prominencia mínima de un pico, en fracción del máximo de la envolvente suavizada.

    Devuelve
    --------
    envelope, env_smooth : ndarray, misma unidad que x (envolvente y envolvente suavizada)
    indices_picos : ndarray de int (índices de los picos de env_smooth)
    """
    envelope = np.abs(hilbert(x))
    k = kernel if kernel % 2 == 1 else kernel + 1            # medfilt exige una ventana impar
    env_smooth = medfilt(envelope, kernel_size=k)            # Elimina picos espurios más estrechos que la ventana
    distancia = max(1, int((1.0 / f0) * fs * FRAC_DISTANCIA))   # Muestras: obliga a un solo pico por ciclo
    indices_picos, _ = find_peaks(env_smooth, distance=distancia,
                                  prominence=frac_prominencia * np.max(env_smooth))
    return envelope, env_smooth, indices_picos


def ajustar_decaimiento(t, A, umbral_sigma=UMBRAL_SIGMA, max_iter=MAX_ITER_AJUSTE):
    """Ajuste lineal de ln A frente a t con rechazo iterativo de valores atípicos.

    Modelo: A(t) = A0 exp(-gamma t), es decir, ln A = ln A0 - gamma t. En cada iteración se ajusta con los
    valores conservados y se descartan los que tienen un residuo mayor que umbral_sigma desviaciones típicas.
    El proceso termina cuando el conjunto no cambia o quedarían menos de tres valores, y el ajuste final se
    calcula con el conjunto conservado.

    Parámetros
    ----------
    t : ndarray, s
        Instante de cada valor.
    A : ndarray
        Valores positivos (alturas de pulsos o áreas de lóbulos).
    umbral_sigma, max_iter : float, int
        Criterio de rechazo y número máximo de iteraciones.

    Devuelve
    --------
    ajuste : resultado de scipy.stats.linregress sobre (t, ln A) con los valores conservados; gamma = -ajuste.slope
    mascara : ndarray de bool, True en los valores conservados
    """
    lnA = np.log(A)
    mascara = np.ones(lnA.shape, dtype=bool)
    for _ in range(max_iter):
        ajuste = linregress(t[mascara], lnA[mascara])
        residuos = lnA - (ajuste.intercept + ajuste.slope * t)
        sigma = np.std(residuos[mascara])
        nueva = np.abs(residuos) <= umbral_sigma * sigma   # <= para no descartar nada si el ajuste es exacto
        if np.array_equal(nueva, mascara) or np.count_nonzero(nueva) < 3:
            break
        mascara = nueva
    return linregress(t[mascara], lnA[mascara]), mascara


def lobulos_de_signo_constante(v, t):
    """Tramos contiguos de signo constante de una señal y su integral temporal.

    Parámetros
    ----------
    v : ndarray
        Señal (por ejemplo, tensión en V) referida a su nivel nulo.
    t : ndarray
        Instantes de las muestras (s), de igual longitud que v.

    Devuelve
    --------
    inicios, fines : ndarray de int
        Índices de la primera y la última muestra de cada tramo (ambas incluidas). Los tramos con v == 0 se omiten.
    areas : ndarray
        Integral de v sobre cada tramo con la regla del trapecio (unidad de v por segundo).
    """
    s = np.sign(v)
    cambios = np.flatnonzero(np.diff(s) != 0) + 1
    inicios = np.concatenate(([0], cambios))
    fines = np.concatenate((cambios - 1, [len(v) - 1]))
    # Integral acumulada por trapecios: el área del tramo [i, f] es C[f] - C[i]. Así se evita un bucle por tramo.
    C = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    areas = C[fines] - C[inicios]
    con_senal = s[inicios] != 0
    return inicios[con_senal], fines[con_senal], areas[con_senal]


def sigma_ruido(v, sigma_min):
    """Desviación típica del ruido blanco superpuesto a una señal suave.

    Para ruido blanco de desviación sigma, la segunda diferencia d2[i] = v[i+1] - 2 v[i] + v[i-1] tiene
    desviación sqrt(6) sigma, mientras que la parte suave de la señal aporta poco. Se usa la desviación absoluta
    mediana de d2 porque no se ve afectada por las pocas muestras donde la curvatura es mayor.

    Parámetros
    ----------
    v : ndarray
        Señal en cualquier unidad.
    sigma_min : float
        Cota inferior en la misma unidad: el ruido de cuantización del ADC, V_scale / sqrt(12).

    Devuelve
    --------
    float, en la unidad de v.
    """
    d2 = v[2:] - 2.0 * v[1:-1] + v[:-2]
    mad = np.median(np.abs(d2 - np.median(d2)))
    return max(1.4826 * mad / np.sqrt(6.0), sigma_min)


def analizar_registro(t_sec, voltajes, fs, v_scale, opciones=None):
    """Extrae del registro f0, gamma y la integral de la fem del lóbulo de mayor área.

    `opciones` es una OpcionesAnalisis (por defecto, la validada). Devuelve (medido, series): `medido` es un
    diccionario con números y textos de advertencia, y `series` los arreglos que usan las gráficas.
    """
    opc = opciones or OpcionesAnalisis()
    prep = preparar_senal(t_sec, voltajes, v_scale)
    advertencias = list(prep["advertencias"])
    v, n = prep["v"], len(prep["v"])
    duracion = n / fs

    # 1) Frecuencia dominante
    esp = frecuencia_dominante(prep["proc"], fs, f_min=opc.f_min)
    f0 = esp["f0"]
    if esp["contraste"] < PICO_ESPECTRAL_MIN:
        raise ErrorAnalisis("No se detecta una oscilación clara en el registro (el pico del espectro apenas "
                            f"supera el fondo: contraste {esp['contraste']:.1f}).")
    if duracion * f0 < CICLOS_MINIMOS:
        raise ErrorAnalisis(f"El registro dura {duracion:.1f} s y contiene {duracion * f0:.1f} ciclos; "
                            f"se necesitan al menos {CICLOS_MINIMOS}.")
    if esp["amp_subarmonico"] >= UMBRAL_SUBARMONICO:
        advertencias.append(f"Hay un pico en f0/2 de amplitud {esp['amp_subarmonico']:.2f} veces la de f0: la "
                            "frecuencia de oscilación podría ser la mitad de la detectada.")

    # 2) Decaimiento de la altura de los pulsos
    envelope, env_smooth, indices_picos = envolvente_y_picos(v, f0, fs, opc.kernel_mediano, opc.frac_prominencia)
    ajuste_h = mask_h = None
    if len(indices_picos) >= 3:
        t_picos, A_picos = t_sec[indices_picos], env_smooth[indices_picos]
        ajuste_h, mask_h = ajustar_decaimiento(t_picos, A_picos, opc.umbral_sigma)

    # 3) Lóbulos de la fem: integral de cada uno
    ini, fin, areas = lobulos_de_signo_constante(v, t_sec)
    completos = (ini > 0) & (fin < n - 1)   # Un lóbulo que toca un extremo del registro está cortado
    if not completos.any():
        raise ErrorAnalisis("No hay ningún lóbulo completo en el registro.")
    ini, fin, areas = ini[completos], fin[completos], areas[completos]
    abs_areas = np.abs(areas)
    k_max = int(np.argmax(abs_areas))
    inicio, final = int(ini[k_max]), int(fin[k_max])
    t_pulso, v_pulso = t_sec[inicio:final + 1], v[inicio:final + 1]
    n_lobulo = len(v_pulso)
    if n_lobulo < 10:
        raise ErrorAnalisis(f"El lóbulo de mayor área tiene solo {n_lobulo} muestras: la señal no es fiable.")
    dt = float(np.mean(np.diff(t_pulso)))
    integral = float(trapezoid(v_pulso, t_pulso))   # V*s: N por la variación de flujo, en valor absoluto
    sigma_V = sigma_ruido(v_pulso, v_scale / math.sqrt(12.0))
    sigma_integral = float(sigma_V * math.sqrt(n_lobulo) * dt)   # n muestras independientes con ruido sigma_V
    ancho = float(t_pulso[-1] - t_pulso[0])
    ancho_relativo = ancho * 2.0 * f0   # Cociente entre la duración del lóbulo y medio periodo
    if ancho_relativo < ANCHO_RELATIVO_MIN:
        advertencias.append(
            f"El lóbulo de mayor área dura {100 * ancho_relativo:.0f} % de medio periodo. Si el imán no cruza el "
            "plano de la bobina debería durar casi todo el medio periodo; puede que el ADC no registre la parte "
            "baja de la señal (zona muerta), lo que subestima B y sobrestima gamma. Revisa el rango de tensión "
            "del ADC.")

    # 4) Decaimiento del área de los lóbulos
    usar = abs_areas >= FRAC_AREA_MIN * abs_areas.max()
    ajuste_a = mask_a = None
    t_lobulos = 0.5 * (t_sec[ini] + t_sec[fin])
    if np.count_nonzero(usar) >= MIN_LOBULOS_AREA:
        ajuste_a, mask_a = ajustar_decaimiento(t_lobulos[usar], abs_areas[usar], opc.umbral_sigma)

    if ajuste_a is None and ajuste_h is None:
        raise ErrorAnalisis("No hay suficientes pulsos para ajustar el decaimiento: se necesitan al menos 3 picos "
                            f"(se detectaron {len(indices_picos)}) o {MIN_LOBULOS_AREA} lóbulos.")
    if ajuste_a is not None:
        ajuste, metodo = ajuste_a, "área de los lóbulos"
        n_usados = int(np.count_nonzero(mask_a))
    else:
        ajuste, metodo = ajuste_h, "altura de los pulsos"
        n_usados = int(np.count_nonzero(mask_h))
        advertencias.append(f"Hay menos de {MIN_LOBULOS_AREA} lóbulos completos: gamma se estima solo con la "
                            "altura de los pulsos, que es más sensible a los cambios de forma del pulso.")
    gamma, gamma_err, r2 = -ajuste.slope, ajuste.stderr, ajuste.rvalue ** 2

    if ajuste_a is not None and ajuste_h is not None and gamma > 0 and -ajuste_h.slope > 0:
        cociente = (-ajuste_h.slope) / gamma
        if cociente > 1.0 + UMBRAL_DISCREPANCIA:
            advertencias.append(
                f"La altura de los pulsos decae {100 * (cociente - 1):.0f} % más rápido que su área: la forma de los "
                "pulsos cambia durante el registro (por ejemplo, el imán se acerca menos a la bobina al disminuir la "
                "amplitud). Ambos valores de gamma pueden estar por encima del real. Prueba con una amplitud de "
                "oscilación menor.")
        elif cociente < 1.0 / (1.0 + UMBRAL_DISCREPANCIA):
            advertencias.append(
                f"La altura de los pulsos decae {100 * (1 - cociente):.0f} % más lento que su área. Puede deberse a "
                "que el ADC no registra la parte baja de la señal. Revisa el rango de tensión del ADC.")
    if r2 < 0.95:
        advertencias.append(f"El ajuste exponencial es pobre (R2 = {r2:.3f}): el decaimiento no es exponencial "
                            "limpio o hay interferencias.")
    if gamma <= 0:
        advertencias.append("gamma <= 0: la amplitud no decae en el registro.")
    elif gamma * duracion < 0.1:
        advertencias.append(f"El registro dura {duracion:.0f} s y la amplitud decae solo {100 * (1 - math.exp(-gamma * duracion)):.0f} %: "
                            "gamma tiene mucha incertidumbre. Usa un registro más largo.")

    medido = {
        "n_muestras": n, "fs_Hz": float(fs), "duracion_s": float(duracion), "V_scale_V_per_count": float(v_scale),
        "nivel_reposo_adc": prep["reposo"],
        "f0_Hz": float(f0), "sigma_f_Hz": float(esp["sigma_f"]), "fwhm_Hz": float(esp["fwhm"]),
        "gamma_1_s": float(gamma), "gamma_err_1_s": float(gamma_err), "r2_gamma": float(r2),
        "gamma_metodo": metodo, "gamma_n_usados": n_usados,
        "gamma_altura_1_s": None if ajuste_h is None else float(-ajuste_h.slope),
        "gamma_altura_err_1_s": None if ajuste_h is None else float(ajuste_h.stderr),
        "r2_altura": None if ajuste_h is None else float(ajuste_h.rvalue ** 2),
        "picos_detectados": int(len(indices_picos)),
        "picos_usados": None if mask_h is None else int(np.count_nonzero(mask_h)),
        "gamma_area_1_s": None if ajuste_a is None else float(-ajuste_a.slope),
        "gamma_area_err_1_s": None if ajuste_a is None else float(ajuste_a.stderr),
        "r2_area": None if ajuste_a is None else float(ajuste_a.rvalue ** 2),
        "integral_emf_Vs": abs(integral), "sigma_integral_Vs": sigma_integral,
        "lobulo_t_ini_s": float(t_pulso[0]), "lobulo_t_fin_s": float(t_pulso[-1]), "lobulo_muestras": n_lobulo,
        "lobulo_ancho_relativo": float(ancho_relativo), "adc_negative_clipped": True,
        "advertencias": advertencias,
    }
    series = {
        "t": t_sec, "v": v, "valid": prep["valid"], "n_interpoladas": prep["n_interpoladas"],
        "envelope": envelope, "env_smooth": env_smooth, "indices_picos": indices_picos,
        "ajuste_h": ajuste_h, "mask_h": mask_h,
        "t_lobulos": t_lobulos, "areas_abs": abs_areas, "usar": usar, "ajuste_a": ajuste_a, "mask_a": mask_a,
        "inicio": inicio, "final": final, "integral": integral,
        "freqs": esp["freqs"], "espectro": esp["espectro"], "f_izq": esp["f_izq"], "f_der": esp["f_der"],
    }
    return medido, series


# ---------------------------------------------------------------------------
# Qué se puede calcular con los datos conocidos
# ---------------------------------------------------------------------------
def resolver_parametros(medido, montaje):
    """Deduce el resto de magnitudes a partir de lo medido y de los datos del montaje que se conocen.

    Del registro salen f0, gamma y la integral de la fem. El oscilador subamortiguado m x'' + b x' + k x = 0
    cumple gamma = b / (2 m), omega0^2 = k / m y omega1^2 = omega0^2 - gamma^2, donde omega1 = 2 pi f0 es la
    frecuencia angular medida. De los tres parámetros m, k y b basta conocer uno para obtener los otros dos.
    La integral de la fem es N por la variación de flujo; con N y el radio de la bobina se obtiene B.

    Devuelve un diccionario con
        valores : magnitudes calculadas (omega0, Q, k, b, B, ...)
        filas : lista de (clave, nombre, valor, incertidumbre, unidad, origen) para mostrar
        chequeos : comparaciones entre un dato dado y el valor deducido de la medición
        faltantes : qué no se pudo calcular y qué dato hace falta
    """
    f0, gamma = medido["f0_Hz"], medido["gamma_1_s"]
    gamma_err = medido["gamma_err_1_s"]
    integral = medido["integral_emf_Vs"]
    w1 = 2.0 * math.pi * f0
    gamma_pos = gamma > 0
    w0 = math.sqrt(w1 ** 2 + max(gamma, 0.0) ** 2)

    valores = {"omega1_rad_s": w1, "omega0_rad_s": w0}
    filas = [("omega0", "omega0", w0, None, "rad/s", "sqrt(omega1^2 + gamma^2), con omega1 = 2 pi f0")]
    faltantes, chequeos = [], []

    if gamma_pos:
        rel = gamma_err / gamma
        valores.update(Q=w0 / (2 * gamma), decremento_log=2 * math.pi * gamma / w1, tau_amplitud_s=1 / gamma)
        filas += [
            ("Q", "factor de calidad Q", valores["Q"], valores["Q"] * rel, "", "omega0 / (2 gamma)"),
            ("decremento_log", "decremento logarítmico", valores["decremento_log"],
             valores["decremento_log"] * rel, "", "2 pi gamma / omega1"),
            ("tau", "tiempo de decaimiento de la amplitud", valores["tau_amplitud_s"],
             valores["tau_amplitud_s"] * rel, "s", "1 / gamma"),
        ]
    else:
        faltantes.append("Q, b y el tiempo de decaimiento: gamma no es positivo en este registro.")

    # Parámetros mecánicos: basta uno de m, k, b
    m_dada = montaje.masa
    m_ef = None
    if m_dada:
        m_ef = m_dada + (montaje.masa_resorte / 3.0 if montaje.masa_resorte else 0.0)
    k_dada, b_dada = montaje.k, montaje.b

    if m_ef:
        origen_m = "dada" if not montaje.masa_resorte else "masa + masa del resorte / 3"
        filas.append(("masa_ef", "masa efectiva", m_ef, None, "kg", origen_m))
        valores["masa_efectiva_kg"] = m_ef
        valores["k_N_m"] = m_ef * w0 ** 2
        filas.append(("k", "constante del resorte k", valores["k_N_m"], None, "N/m", "m * omega0^2"))
        if gamma_pos:
            valores["b_kg_s"] = 2 * m_ef * gamma
            filas.append(("b", "coeficiente de amortiguamiento b", valores["b_kg_s"],
                          valores["b_kg_s"] * gamma_err / gamma, "kg/s", "2 m gamma"))
    elif k_dada:
        m_calc = k_dada / w0 ** 2
        valores["masa_efectiva_kg"] = m_calc
        filas.append(("masa_ef", "masa efectiva", m_calc, None, "kg", "k / omega0^2"))
        if gamma_pos:
            valores["b_kg_s"] = 2 * m_calc * gamma
            filas.append(("b", "coeficiente de amortiguamiento b", valores["b_kg_s"],
                          valores["b_kg_s"] * gamma_err / gamma, "kg/s", "2 m gamma"))
    elif b_dada and gamma_pos:
        m_calc = b_dada / (2 * gamma)
        valores["masa_efectiva_kg"] = m_calc
        valores["k_N_m"] = m_calc * w0 ** 2
        filas.append(("masa_ef", "masa efectiva", m_calc, m_calc * gamma_err / gamma, "kg", "b / (2 gamma)"))
        filas.append(("k", "constante del resorte k", valores["k_N_m"], valores["k_N_m"] * gamma_err / gamma,
                      "N/m", "m * omega0^2"))
    else:
        faltantes.append("k y b: hace falta la masa que oscila, o k, o b.")

    # Si se conocen dos o más de m, k, b, se comparan con lo medido
    def chequeo(descripcion, dado, deducido):
        dif = (dado - deducido) / deducido
        chequeos.append({"descripcion": descripcion, "dado": dado, "deducido": deducido, "dif_rel": dif,
                         "concuerda": abs(dif) <= TOLERANCIA_CHEQUEO})
    if m_ef and k_dada:
        chequeo("omega0 con los m y k dados frente a omega0 medida (rad/s)", math.sqrt(k_dada / m_ef), w0)
    if m_ef and b_dada and gamma_pos:
        chequeo("gamma = b / (2 m) con los b y m dados frente a gamma medida (1/s)", b_dada / (2 * m_ef), gamma)
    if k_dada and b_dada and gamma_pos:
        chequeo("masa deducida de b frente a la deducida de k (kg)", b_dada / (2 * gamma), k_dada / w0 ** 2)

    # Parte magnética
    filas.append(("integral", "integral de la fem del lóbulo (N * variación de flujo)", integral,
                  medido["sigma_integral_Vs"], "V*s", "medida"))
    N, R = montaje.vueltas, montaje.radio
    if N:
        valores["delta_phi_Wb"] = integral / N
        filas.append(("delta_phi", "variación de flujo por vuelta", integral / N, medido["sigma_integral_Vs"] / N,
                      "Wb", "integral / N"))
    if N and R:
        area = math.pi * R ** 2
        valores["B_T"] = integral / (N * area)
        valores["B_err_T"] = medido["sigma_integral_Vs"] / (N * area)
        filas.append(("B", "campo magnético medio B", valores["B_T"], valores["B_err_T"], "T",
                      "integral / (N pi R^2); la incertidumbre es solo ruido de lectura"))
    elif N:
        faltantes.append("B: hace falta el radio de la bobina.")
    elif R:
        faltantes.append("B: hace falta el número de vueltas de la bobina.")
    else:
        faltantes.append("B: hacen falta el número de vueltas y el radio de la bobina.")
    return {"valores": valores, "filas": filas, "chequeos": chequeos, "faltantes": faltantes}


def comparar_con_simulador(medido, resolucion, verdad):
    """Compara lo calculado con los valores verdaderos del simulador. Devuelve lista de (nombre, calculado, verdadero)."""
    v = resolucion["valores"]
    pares = [("f0 (Hz)", medido["f0_Hz"], verdad["f0_Hz"]),
             ("gamma (1/s)", medido["gamma_1_s"], verdad["gamma_1_s"]),
             ("omega0 (rad/s)", v.get("omega0_rad_s"), verdad["omega0_rad_s"]),
             ("Q", v.get("Q"), verdad["Q"]),
             ("k (N/m)", v.get("k_N_m"), verdad["k_N_m"]),
             ("b (kg/s)", v.get("b_kg_s"), verdad["b_kg_s"]),
             ("integral de la fem (V*s)", medido["integral_emf_Vs"], verdad["integral_emf_Vs"]),
             ("B (T)", v.get("B_T"), verdad["B_T"])]
    return [(n, c, t) for n, c, t in pares if c is not None]


# ---------------------------------------------------------------------------
# Informe de texto
# ---------------------------------------------------------------------------
def formatear(valor, inc=None, cifras_inc=2):
    """Valor con su incertidumbre redondeada a `cifras_inc` cifras significativas.

    Fuera de [1e-3, 1e5) usa notación científica con el mismo exponente para el valor y la incertidumbre.
    """
    if valor is None or not np.isfinite(valor):
        return "n/d"
    if inc is None or not np.isfinite(inc) or inc <= 0:
        return f"{valor:.4g}"
    exp_inc = math.floor(math.log10(inc)) - (cifras_inc - 1)
    if valor != 0 and (abs(valor) < 1e-3 or abs(valor) >= 1e5):
        e = math.floor(math.log10(abs(valor)))
        dec = max(e - exp_inc, 0)
        return f"({valor / 10 ** e:.{dec}f} +/- {inc / 10 ** e:.{dec}f})e{e:+03d}"
    dec = max(0, -exp_inc)
    return f"{valor:.{dec}f} +/- {inc:.{dec}f}"


ANCHO_INFORME = 78


def _fila(nombre, valor, unidad="", nota=""):
    return f"  {nombre:<38}{valor:>26} {unidad:<6}{nota}".rstrip()


def _parrafo(texto, sangria="  - "):
    """Texto ajustado al ancho del informe, con sangría francesa."""
    return textwrap.fill(texto, width=ANCHO_INFORME, initial_indent=sangria,
                         subsequent_indent=" " * len(sangria))


def construir_informe(medido, resolucion, montaje, fuente, comparacion=None):
    """Texto del informe que se muestra y se guarda."""
    L = ["=" * ANCHO_INFORME, " RESULTADOS", "=" * ANCHO_INFORME,
         f"Registro: {medido['n_muestras']} muestras, {medido['duracion_s']:.1f} s, "
         f"fs = {medido['fs_Hz']:.2f} Hz (fuente: {fuente})", ""]

    L.append("Medido en el registro")
    L.append(_fila("frecuencia f0", formatear(medido["f0_Hz"], medido["sigma_f_Hz"]), "Hz", "FWHM / 2"))
    principal = medido["gamma_metodo"]
    if medido["gamma_area_1_s"] is not None:
        marca = " (*)" if principal == "área de los lóbulos" else ""
        L.append(_fila("gamma, área de los lóbulos" + marca,
                       formatear(medido["gamma_area_1_s"], medido["gamma_area_err_1_s"]), "1/s",
                       f"R2 = {medido['r2_area']:.4f}"))
    if medido["gamma_altura_1_s"] is not None:
        marca = " (*)" if principal == "altura de los pulsos" else ""
        L.append(_fila("gamma, altura de los pulsos" + marca,
                       formatear(medido["gamma_altura_1_s"], medido["gamma_altura_err_1_s"]), "1/s",
                       f"R2 = {medido['r2_altura']:.4f}, {medido['picos_usados']} picos"))
    L.append(_fila("integral de la fem del lóbulo mayor",
                   formatear(medido["integral_emf_Vs"], medido["sigma_integral_Vs"]), "V*s",
                   f"t = {medido['lobulo_t_ini_s']:.2f} a {medido['lobulo_t_fin_s']:.2f} s"))
    L.append("  (*) valor de gamma que se usa en los cálculos siguientes")
    L.append("")

    claves_medicion = ("omega0", "Q", "decremento_log", "tau")
    L.append("Deducido solo de la medición")
    for clave, nombre, valor, inc, unidad, origen in resolucion["filas"]:
        if clave in claves_medicion:
            L.append(_fila(nombre, formatear(valor, inc), unidad))
    L.append("")

    L.append("Con los datos del montaje")
    hay = False
    for clave, nombre, valor, inc, unidad, origen in resolucion["filas"]:
        if clave in claves_medicion or clave == "integral":
            continue
        hay = True
        if clave == "B":
            L.append(_fila(nombre, formatear(valor * 1e3, None if inc is None else inc * 1e3), "mT",
                           "solo ruido de lectura"))
        else:
            L.append(_fila(nombre, formatear(valor, inc), unidad, origen))
    if not hay:
        L.append("  (no diste ningún dato del montaje)")
    if resolucion["chequeos"]:
        L.append("")
        L.append("Comprobaciones entre los datos dados y la medición")
        for c in resolucion["chequeos"]:
            estado = "concuerda" if c["concuerda"] else "NO concuerda"
            L.append(_parrafo(f"{c['descripcion']}: dado {formatear(c['dado'])}, deducido "
                              f"{formatear(c['deducido'])} ({100 * c['dif_rel']:+.1f} %, {estado})"))
    if resolucion["faltantes"]:
        L.append("")
        L.append("No se pudo calcular")
        for texto in resolucion["faltantes"]:
            L.append(_parrafo(texto))
    if medido["advertencias"]:
        L.append("")
        L.append("Advertencias")
        for texto in medido["advertencias"]:
            L.append(_parrafo(texto))
    L.append("")
    L.append(_parrafo("Las incertidumbres son estadísticas. No incluyen la calibración del ADC, los valores de N y "
                      "del radio, la falta de uniformidad del campo sobre la bobina ni un posible cambio de forma "
                      "de los pulsos.", sangria=""))
    if comparacion:
        L.append("")
        L.append("Comparación con los valores verdaderos del simulador")
        L.append(f"  {'magnitud':<26} {'calculado':>13} {'verdadero':>13} {'diferencia':>11}")
        for nombre, calc, verd in comparacion:
            L.append(f"  {nombre:<26} {calc:>13.5g} {verd:>13.5g} {100 * (calc - verd) / verd:>+10.1f} %")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def graficar(medido, series, carpeta, mostrar=False):
    """Guarda las cuatro gráficas del análisis en `carpeta` y devuelve sus rutas."""
    import matplotlib.pyplot as plt
    if not mostrar:
        plt.switch_backend("Agg")
    rutas = []
    t, v = series["t"], series["v"] * 1e3   # mV

    # 1) Señal
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5))
    ax1.plot(t, v, lw=0.8, label="Señal")
    if series["n_interpoladas"]:
        ax1.plot(t[~series["valid"]], v[~series["valid"]], "rx", ms=5, label="Muestras interpoladas")
    ax1.set(xlabel="Tiempo (s)", ylabel="Tensión respecto al reposo (mV)", title="Fem inducida en la bobina")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    t_zoom = min(3.0, t[-1])
    en_zoom = t <= t_zoom
    ax2.plot(t[en_zoom], v[en_zoom], ".-", ms=3, lw=0.6)
    ax2.set(xlabel="Tiempo (s)", ylabel="Tensión respecto al reposo (mV)", title=f"Primeros {t_zoom:.1f} s")
    ax2.grid(True)
    fig.tight_layout()
    rutas.append(os.path.join(carpeta, "01_senal.png"))
    fig.savefig(rutas[-1], dpi=130)

    # 2) Espectro
    f0 = medido["f0_Hz"]
    xlim_max = max(5.0, 3.0 * f0)
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series["freqs"], series["espectro"], label="Espectro normalizado")
    ax.axvline(f0, color="r", ls="--", label=f"f0 = {f0:.4f} Hz")
    ax.axvspan(series["f_izq"], series["f_der"], color="r", alpha=0.2,
               label=f"FWHM ({series['f_izq']:.3f} a {series['f_der']:.3f} Hz)")
    ax.set(xlim=(0, xlim_max), ylim=(0, 1.05), xlabel="Frecuencia (Hz)", ylabel="Amplitud normalizada",
           title=f"Espectro de amplitud (0 a {xlim_max:.1f} Hz)")
    ax.legend()
    ax.grid(True)
    fig2.tight_layout()
    rutas.append(os.path.join(carpeta, "02_espectro.png"))
    fig2.savefig(rutas[-1], dpi=130)

    # 3) Decaimiento: altura de los pulsos y área de los lóbulos
    fig3, (ax_h, ax_a) = plt.subplots(2, 1, figsize=(10, 7.5))
    ax_h.plot(t, v, alpha=0.3, label="Señal")
    ax_h.plot(t, series["env_smooth"] * 1e3, color="tab:green", lw=1, label="Envolvente suavizada")
    ajuste_h = series["ajuste_h"]
    if ajuste_h is not None:
        idx, mh = series["indices_picos"], series["mask_h"]
        ax_h.plot(t[idx][~mh], series["env_smooth"][idx][~mh] * 1e3, "o", color="gray", ms=5, label="Picos descartados")
        ax_h.plot(t[idx][mh], series["env_smooth"][idx][mh] * 1e3, "o", color="blue", ms=5, label="Picos usados")
        tc = np.linspace(t[idx][mh].min(), t[idx][mh].max(), 300)
        ax_h.plot(tc, np.exp(ajuste_h.intercept + ajuste_h.slope * tc) * 1e3, "k--", lw=2,
                  label=f"gamma = {-ajuste_h.slope:.4f} 1/s")
    ax_h.set(xlabel="Tiempo (s)", ylabel="Altura de los pulsos (mV)", title="Decaimiento de la altura de los pulsos")
    ax_h.legend(loc="upper right")
    ax_h.grid(True)
    usar, mask_a, ajuste_a = series["usar"], series["mask_a"], series["ajuste_a"]
    tl, al = series["t_lobulos"][usar], series["areas_abs"][usar] * 1e3   # mV*s
    if ajuste_a is not None:
        ax_a.plot(tl[~mask_a], al[~mask_a], "o", color="gray", ms=5, label="Lóbulos descartados")
        ax_a.plot(tl[mask_a], al[mask_a], "o", color="blue", ms=5, label="Lóbulos usados")
        tc = np.linspace(tl[mask_a].min(), tl[mask_a].max(), 300)
        ax_a.plot(tc, np.exp(ajuste_a.intercept + ajuste_a.slope * tc) * 1e3, "k--", lw=2,
                  label=f"gamma = {-ajuste_a.slope:.4f} 1/s")
        ax_a.legend(loc="upper right")
    else:
        ax_a.plot(tl, al, "o", color="gray", ms=5)
        ax_a.text(0.5, 0.5, "Lóbulos insuficientes para el ajuste", transform=ax_a.transAxes, ha="center")
    ax_a.set(xlabel="Tiempo (s)", ylabel="Integral del lóbulo (mV*s)", title="Decaimiento del área de los lóbulos")
    ax_a.grid(True)
    fig3.tight_layout()
    rutas.append(os.path.join(carpeta, "03_decaimiento.png"))
    fig3.savefig(rutas[-1], dpi=130)

    # 4) Lóbulo usado para la integral
    i0, i1 = series["inicio"], series["final"]
    margen = 0.3
    ventana = (t >= t[i0] - margen) & (t <= t[i1] + margen)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(t[ventana], v[ventana], ".-", ms=3, lw=0.6, label="Señal")
    ax4.fill_between(t[i0:i1 + 1], v[i0:i1 + 1], color="tab:orange", alpha=0.4,
                     label=f"Lóbulo integrado ({series['integral'] * 1e3:.2f} mV*s)")
    ax4.set(xlabel="Tiempo (s)", ylabel="Tensión respecto al reposo (mV)", title="Lóbulo usado para la integral de la fem")
    ax4.legend()
    ax4.grid(True)
    fig4.tight_layout()
    rutas.append(os.path.join(carpeta, "04_lobulo.png"))
    fig4.savefig(rutas[-1], dpi=130)

    if mostrar:
        plt.show()
    for f in (fig, fig2, fig3, fig4):
        plt.close(f)
    return rutas


# ---------------------------------------------------------------------------
# Archivo de configuración
# ---------------------------------------------------------------------------
NOMBRE_CONFIG = "config.json"
DIRECTORIO_SCRIPT = Path(__file__).resolve().parent
TIPOS_FUENTE = ("demo", "archivo", "serie")
CLAVES_PERMITIDAS = {
    "fuente": {"tipo", "archivo", "fs_Hz", "puerto", "baudrate", "muestras"},
    "adc": {"v_ref_V", "v_cal_V", "adc_cal_cuentas"},
    "montaje": {"vueltas", "radio_mm", "masa_g", "masa_resorte_g", "k_N_m", "b_kg_s"},
    "analisis": {"f_min_Hz", "kernel_mediano", "umbral_sigma", "frac_prominencia"},
    "demo": {"amplitud_mm", "altura_mm", "zona_muerta_mV", "semilla"},
    "salida": {"carpeta", "graficas", "mostrar_graficas"},
}


def plantilla_config():
    """Contenido del config.json de ejemplo. Las claves que empiezan con guion bajo son comentarios."""
    return {
        "_ayuda": [
            "Configuración de spring_magnet_analyzer.py. Edita los valores y ejecuta: python spring_magnet_analyzer.py",
            "Escribe null en los datos que no conozcas. Las claves que empiezan con guion bajo son comentarios y se ignoran.",
            "Los números van sin comillas y con punto decimal. Las rutas relativas se cuentan desde la carpeta de este archivo.",
            "En Windows escribe las rutas con barras normales, por ejemplo C:/datos/medicion_raw_20260204_151040.csv",
        ],
        "fuente": {
            "_ayuda": "tipo: demo (datos simulados), archivo (registro medicion_raw_<fecha>.csv ya guardado) o serie (adquirir ahora con el ESP32).",
            "tipo": "demo",
            "archivo": None,
            "fs_Hz": None,
            "puerto": PUERTO_DEFECTO,
            "baudrate": 115200,
            "muestras": 40000,
        },
        "adc": {
            "_ayuda": "v_ref_V es la referencia nominal del ADC. Para calibrar, aplica una tensión continua conocida (v_cal_V, medida con un multímetro) y anota la lectura media del ADC (adc_cal_cuentas).",
            "v_ref_V": V_REF_NOMINAL,
            "v_cal_V": None,
            "adc_cal_cuentas": None,
        },
        "montaje": {
            "_ayuda": "Lo que conoces del montaje; null en lo que no. masa_g es la masa que oscila (imán y soporte), sin el resorte. Con la masa, o con k, o con b, el programa calcula las otras dos. Estos valores son los del simulador: cámbialos por los de tu montaje.",
            "vueltas": 3400,
            "radio_mm": 41.5,
            "masa_g": 200,
            "masa_resorte_g": None,
            "k_N_m": None,
            "b_kg_s": None,
        },
        "analisis": {
            "_ayuda": "Opcional. Son los valores con los que se validó el análisis; cámbialos solo si el registro lo exige.",
            "f_min_Hz": F_MIN,
            "kernel_mediano": KERNEL_MEDIANO,
            "umbral_sigma": UMBRAL_SIGMA,
            "frac_prominencia": FRAC_PROMINENCIA,
        },
        "demo": {
            "_ayuda": "Solo con tipo demo: amplitud inicial y altura del equilibrio sobre la bobina de la oscilación simulada, zona muerta del ADC simulado y semilla del ruido.",
            "amplitud_mm": 12,
            "altura_mm": 35,
            "zona_muerta_mV": 0,
            "semilla": 1,
        },
        "salida": {
            "_ayuda": "Cada ejecución crea una subcarpeta con la fecha y la hora dentro de carpeta.",
            "carpeta": "resultados",
            "graficas": True,
            "mostrar_graficas": False,
        },
    }


def crear_plantilla(ruta):
    """Escribe un config.json de ejemplo. No sobrescribe un archivo existente."""
    ruta = Path(ruta)
    if ruta.exists():
        raise ErrorAnalisis(f"{ruta} ya existe. Bórralo o elige otro nombre para crear una plantilla nueva.")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(plantilla_config(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return ruta


def _sugerir(clave, permitidas):
    cerca = difflib.get_close_matches(clave, sorted(permitidas), n=1, cutoff=0.6)
    if cerca:
        return f"¿Quisiste decir '{cerca[0]}'?"
    return f"Las claves válidas son: {', '.join(sorted(permitidas))}."


def _pista_json(mensaje):
    """Sugerencia para los errores de sintaxis JSON más frecuentes."""
    m = mensaje.lower()
    if "escape" in m:
        return "En JSON las barras invertidas de una ruta se escriben dobles (\\\\) o se reemplazan por barras normales (/)."
    if "property name" in m:
        return "Revisa que no sobre una coma antes de una llave de cierre, que los decimales lleven punto (41.5, no 41,5) y que las claves lleven comillas dobles."
    if "expecting value" in m:
        return "Revisa que los números lleven punto decimal (no coma), que los textos lleven comillas dobles y que no falte ningún valor."
    if "delimiter" in m:
        return "Falta una coma entre dos elementos, o sobra una comilla."
    if "extra data" in m:
        return "Hay texto después del bloque principal."
    return ""


class _Bloque:
    """Una sección del archivo de configuración, con lectura validada de cada clave."""

    def __init__(self, archivo, nombre, valores):
        self.archivo, self.nombre, self.valores = archivo, nombre, valores

    def _error(self, clave, texto):
        return ErrorAnalisis(f"{self.archivo}: '{self.nombre}.{clave}' {texto}")

    def numero(self, clave, defecto=None, minimo=0.0, exclusivo=True, entero=False, maximo=None):
        valor = self.valores.get(clave)
        if valor is None:
            return defecto
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor):
            raise self._error(clave, f"debe ser un número sin comillas, o null (se leyó {valor!r}).")
        if entero and valor != int(valor):
            raise self._error(clave, f"debe ser un entero (se leyó {valor}).")
        if (valor <= minimo) if exclusivo else (valor < minimo):
            raise self._error(clave, f"debe ser {'mayor que' if exclusivo else 'al menos'} {minimo:g} (se leyó {valor}).")
        if maximo is not None and valor >= maximo:
            raise self._error(clave, f"debe ser menor que {maximo:g} (se leyó {valor}).")
        return int(valor) if entero else float(valor)

    def texto(self, clave, defecto=None):
        valor = self.valores.get(clave)
        if valor is None:
            return defecto
        if not isinstance(valor, str) or not valor.strip():
            raise self._error(clave, f"debe ser un texto entre comillas, o null (se leyó {valor!r}).")
        return valor.strip()

    def booleano(self, clave, defecto):
        valor = self.valores.get(clave)
        if valor is None:
            return defecto
        if not isinstance(valor, bool):
            raise self._error(clave, f"debe ser true o false (se leyó {valor!r}).")
        return valor


def cargar_configuracion(ruta):
    """Lee y valida el archivo de configuración. Devuelve una Configuracion.

    Las claves que empiezan con guion bajo son comentarios. Una clave desconocida es un error, porque un nombre
    mal escrito dejaría un dato sin usar sin avisar. Las rutas relativas se cuentan desde la carpeta del archivo.
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ErrorAnalisis(f"No se encontró el archivo de configuración: {ruta}\n"
                            "Para crear uno de ejemplo ejecuta: python spring_magnet_analyzer.py --crear-config")
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        raise ErrorAnalisis(f"{ruta.name}: error de sintaxis en la línea {e.lineno}, columna {e.colno} "
                            f"({e.msg}). {_pista_json(e.msg)}".strip()) from e
    except UnicodeDecodeError as e:
        raise ErrorAnalisis(f"{ruta.name}: el archivo debe estar guardado como UTF-8.") from e
    if not isinstance(datos, dict):
        raise ErrorAnalisis(f"{ruta.name}: el contenido debe ser un bloque con llaves {{ ... }}.")
    for clave in datos:
        if not clave.startswith("_") and clave not in CLAVES_PERMITIDAS:
            raise ErrorAnalisis(f"{ruta.name}: la sección '{clave}' no existe. {_sugerir(clave, CLAVES_PERMITIDAS)}")

    def bloque(nombre):
        crudo = datos.get(nombre)
        if crudo is None:
            crudo = {}
        if not isinstance(crudo, dict):
            raise ErrorAnalisis(f"{ruta.name}: '{nombre}' debe ser un bloque con llaves {{ ... }}.")
        propias = {k: v for k, v in crudo.items() if not k.startswith("_")}
        for clave in propias:
            if clave not in CLAVES_PERMITIDAS[nombre]:
                raise ErrorAnalisis(f"{ruta.name}: la clave '{nombre}.{clave}' no existe. "
                                    f"{_sugerir(clave, CLAVES_PERMITIDAS[nombre])}")
        return _Bloque(ruta.name, nombre, propias)

    base = ruta.resolve().parent

    def resolver(texto):
        p = Path(texto).expanduser()
        return p if p.is_absolute() else base / p

    avisos = []
    fu = bloque("fuente")
    tipo = fu.texto("tipo", "demo")
    if tipo not in TIPOS_FUENTE:
        raise ErrorAnalisis(f"{ruta.name}: 'fuente.tipo' debe ser \"demo\", \"archivo\" o \"serie\" (se leyó {tipo!r}).")
    archivo = None
    if tipo == "archivo":
        texto_archivo = fu.texto("archivo")
        if texto_archivo is None:
            raise ErrorAnalisis(f"{ruta.name}: con tipo \"archivo\" hay que escribir 'fuente.archivo'.")
        archivo = resolver(texto_archivo)
        if not archivo.is_file():
            raise ErrorAnalisis(f"{ruta.name}: no existe el archivo de datos {archivo}")
    fs = fu.numero("fs_Hz")
    puerto = fu.texto("puerto", PUERTO_DEFECTO)
    baudrate = fu.numero("baudrate", 115200, entero=True)
    muestras = fu.numero("muestras", 40000, minimo=100, exclusivo=False, entero=True)

    ad = bloque("adc")
    v_ref = ad.numero("v_ref_V", V_REF_NOMINAL)
    v_cal, adc_cal = ad.numero("v_cal_V"), ad.numero("adc_cal_cuentas")
    if (v_cal is None) != (adc_cal is None):
        raise ErrorAnalisis(f"{ruta.name}: la calibración necesita 'adc.v_cal_V' y 'adc.adc_cal_cuentas' a la vez.")

    mo = bloque("montaje")
    radio_mm, masa_g, masa_resorte_g = mo.numero("radio_mm"), mo.numero("masa_g"), mo.numero("masa_resorte_g")
    montaje = Montaje(
        vueltas=mo.numero("vueltas"),
        radio=None if radio_mm is None else radio_mm * 1e-3,
        masa=None if masa_g is None else masa_g * 1e-3,
        masa_resorte=None if masa_resorte_g is None else masa_resorte_g * 1e-3,
        k=mo.numero("k_N_m"), b=mo.numero("b_kg_s"))
    if montaje.masa_resorte and not montaje.masa:
        avisos.append("montaje.masa_resorte_g se ignora porque no se dio montaje.masa_g.")

    an = bloque("analisis")
    analisis = OpcionesAnalisis(
        f_min=an.numero("f_min_Hz", F_MIN),
        kernel_mediano=an.numero("kernel_mediano", KERNEL_MEDIANO, minimo=1, exclusivo=False, entero=True),
        umbral_sigma=an.numero("umbral_sigma", UMBRAL_SIGMA),
        frac_prominencia=an.numero("frac_prominencia", FRAC_PROMINENCIA, maximo=1.0))

    de = bloque("demo")
    amplitud = de.numero("amplitud_mm", 12.0) * 1e-3
    altura = de.numero("altura_mm", 35.0) * 1e-3
    if tipo == "demo" and amplitud >= altura:
        raise ErrorAnalisis(f"{ruta.name}: 'demo.amplitud_mm' debe ser menor que 'demo.altura_mm' "
                            "(el imán simulado no cruza el plano de la bobina).")
    zona_muerta = de.numero("zona_muerta_mV", 0.0, minimo=0.0, exclusivo=False) * 1e-3
    semilla = de.numero("semilla", 1, minimo=0, exclusivo=False, entero=True)

    sa = bloque("salida")
    return Configuracion(
        base=base, origen=ruta.resolve(), fuente=tipo, archivo=archivo, fs=fs, puerto=puerto, baudrate=baudrate,
        muestras=muestras, v_ref=v_ref, v_cal=v_cal, adc_cal=adc_cal, montaje=montaje, analisis=analisis,
        demo_amplitud=amplitud, demo_altura=altura, demo_zona_muerta=zona_muerta, semilla=semilla,
        salida=Path(sa.texto("carpeta", "resultados")), graficas=sa.booleano("graficas", True),
        mostrar=sa.booleano("mostrar_graficas", False), avisos=avisos)


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def ejecutar(cfg):
    """Obtiene el registro, lo analiza, escribe los archivos de salida y devuelve un diccionario con todo."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    salida = cfg.salida if cfg.salida.is_absolute() else cfg.base / cfg.salida
    carpeta = str(salida / ts)
    os.makedirs(carpeta, exist_ok=True)
    if cfg.origen is not None:
        shutil.copyfile(cfg.origen, os.path.join(carpeta, "config_usada.json"))
    verdad = None

    if cfg.fuente == "demo":
        t_sec, voltajes, fs, verdad = simular_registro(
            amplitud=cfg.demo_amplitud, altura=cfg.demo_altura, zona_muerta_V=cfg.demo_zona_muerta,
            semilla=cfg.semilla, v_ref=cfg.v_ref)
        ruta_raw, _ = guardar_registro(carpeta, ts, np.arange(1, len(voltajes) + 1), voltajes, fs,
                                       extra={"fuente": "simulacion", "verdad_simulador": verdad})
        fuente = "simulación"
    elif cfg.fuente == "serie":
        t_sec, voltajes, fs, idx = acquire_trial("ensayo", cfg.muestras, cfg.puerto, cfg.baudrate)
        ruta_raw, _ = guardar_registro(carpeta, ts, idx, voltajes, fs,
                                       extra={"fuente": "esp32", "puerto": cfg.puerto, "baudrate": cfg.baudrate})
        fuente = "ESP32"
    else:
        t_sec, voltajes, fs, _ = cargar_registro(str(cfg.archivo), cfg.fs)
        ruta_raw, fuente = str(cfg.archivo), "archivo"

    if cfg.fuente != "demo" and cfg.v_cal is not None:
        v_scale = cfg.v_cal / cfg.adc_cal   # Calibración de un punto: pendiente medida con una tensión conocida
    else:
        v_scale = cfg.v_ref / ADC_MAX

    medido, series = analizar_registro(t_sec, voltajes, fs, v_scale, cfg.analisis)
    medido["advertencias"] = list(cfg.avisos) + medido["advertencias"]
    resolucion = resolver_parametros(medido, cfg.montaje)
    comparacion = comparar_con_simulador(medido, resolucion, verdad) if verdad else None
    informe = construir_informe(medido, resolucion, cfg.montaje, fuente, comparacion)

    resultados = {
        "version_programa": __version__, "timestamp": ts, "fuente": fuente,
        "archivo_raw": ruta_relativa(ruta_raw, carpeta),   # Relativa a esta carpeta: no expone rutas personales
        "calibracion_adc": cfg.v_cal is not None and cfg.fuente != "demo",
        "montaje_dado": {"vueltas": cfg.montaje.vueltas, "radio_m": cfg.montaje.radio, "masa_kg": cfg.montaje.masa,
                         "masa_resorte_kg": cfg.montaje.masa_resorte, "k_N_m": cfg.montaje.k,
                         "b_kg_s": cfg.montaje.b},
        "medido": medido, "deducido": resolucion["valores"], "chequeos": resolucion["chequeos"],
        "faltantes": resolucion["faltantes"],
    }
    if comparacion:
        resultados["comparacion_simulador"] = [
            {"magnitud": n, "calculado": c, "verdadero": t} for n, c, t in comparacion]
    ruta_json = os.path.join(carpeta, f"resultados_{ts}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(a_python(resultados), f, indent=2, ensure_ascii=False)
    ruta_txt = os.path.join(carpeta, f"informe_{ts}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(informe + "\n")
    rutas_graficas = graficar(medido, series, carpeta, cfg.mostrar) if cfg.graficas else []
    return {"carpeta": carpeta, "informe": informe, "medido": medido, "resolucion": resolucion,
            "verdad": verdad, "comparacion": comparacion, "rutas": [ruta_json, ruta_txt] + rutas_graficas}


def texto_version():
    """Versión del programa y de las bibliotecas que usa, útil para describir un problema."""
    try:
        import matplotlib
        mpl = f"matplotlib {matplotlib.__version__}"
    except ImportError:
        mpl = "matplotlib no instalado"
    pyserial = f"pyserial {getattr(serial, 'VERSION', '?')}" if serial is not None else "pyserial no instalado"
    return (f"spring_magnet_analyzer.py {__version__} (Python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, {mpl}, {pyserial}; {platform.system()})")


class _AccionVersion(argparse.Action):
    """Imprime texto_version() al pedirlo; se calcula en ese momento para no importar matplotlib en cada ejecución."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(texto_version())
        parser.exit()


class _FormatoEsp(argparse.HelpFormatter):
    """Formato de la ayuda con el encabezado de uso en español."""

    def add_usage(self, usage, actions, groups, prefix=None):
        super().add_usage(usage, actions, groups, "uso: " if prefix is None else prefix)


def crear_parser():
    p = argparse.ArgumentParser(
        prog="spring_magnet_analyzer.py", add_help=False, formatter_class=_FormatoEsp,
        description="Analiza la fem inducida por un imán que oscila en un resorte sobre una bobina: obtiene f0, "
                    "gamma y B, y deduce el resto de magnitudes a partir de los datos del montaje que se conozcan. "
                    f"Lee todo de un archivo de configuración ({NOMBRE_CONFIG}, junto al script).")
    g = p.add_argument_group("opciones")
    g.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    g.add_argument("--config", metavar="RUTA", help=f"archivo de configuración (por defecto {NOMBRE_CONFIG}, junto al script)")
    g.add_argument("--crear-config", action="store_true",
                   help="crea un archivo de configuración de ejemplo y termina; no sobrescribe uno existente")
    g.add_argument("--version", action=_AccionVersion, help="muestra la versión del programa y de las bibliotecas, y termina")
    return p


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = crear_parser().parse_args(argv)
    ruta = Path(args.config) if args.config else DIRECTORIO_SCRIPT / NOMBRE_CONFIG
    try:
        if args.crear_config:
            print(f"Se creó {crear_plantilla(ruta)}. Edítalo y vuelve a ejecutar el programa.")
            return 0
        cfg = cargar_configuracion(ruta)
        print(f"Configuración: {cfg.origen}")
        resultado = ejecutar(cfg)
    except ErrorAnalisis as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrumpido.", file=sys.stderr)
        return 130
    print(resultado["informe"])
    print(f"\nArchivos guardados en: {resultado['carpeta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
