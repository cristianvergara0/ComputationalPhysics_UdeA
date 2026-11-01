"""
Tareas de simulación, ejecución en paralelo, semillas y caché en disco.

Una *tarea* es una réplica independiente: un barrido térmico con memoria (la
configuración final de cada temperatura es la inicial de la siguiente) para un
tamaño, unas condiciones de frontera, unos acoplamientos y un campo dados. Las
tareas de un mismo experimento no dependen unas de otras, así que se ejecutan en
paralelo; el barrido de cada tarea, en cambio, es secuencial por construcción y
nunca se paraleliza.

Reproducibilidad
----------------
Cada tarea lleva su propia semilla, derivada de la semilla base, del nombre del
experimento y de los índices de la tarea con ``numpy.random.SeedSequence``. La
misma semilla fija la configuración inicial (generador de NumPy) y la dinámica de
Metropolis (generador interno de los núcleos), de modo que el resultado de cada
tarea no depende del proceso que la ejecute ni del orden de ejecución.

Caché
-----
Los resultados de cada experimento se guardan en ``data/<nombre>.npz`` junto con
la lista de tareas que los produjo. Si el archivo existe y las tareas coinciden,
se recarga en vez de simular; si los parámetros cambian, se vuelve a simular.
"""

import concurrent.futures
import hashlib
import json
import os
import time
import zlib

import numpy as np

from . import config, kernels, lattice

CODE_VERSION = "1"   # cambia si se modifica el algoritmo de muestreo, para invalidar la caché
MEASURED_RATE = None  # velocidad de un proceso medida con measure_rate [intentos de volteo / s]


# ----------------------------------------------------------------------------
# Semillas
# ----------------------------------------------------------------------------
def task_seed(experiment, *indices, base_seed=config.BASE_SEED):
    """
    Semilla de 32 bits, distinta para cada (experimento, índices), derivada con
    SeedSequence a partir de la semilla base. Los índices deben ser enteros.
    """
    ss = np.random.SeedSequence([int(base_seed), zlib.crc32(experiment.encode()), *[int(i) for i in indices]])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


# ----------------------------------------------------------------------------
# Definición de tareas
# ----------------------------------------------------------------------------
def make_task(L, temperatures, *, periodic=True, J1=config.J1_DEFAULT, J2=config.J2_DEFAULT, h=0.0,
              init="random", nterm, nsamp, stride=1, nblocks=20, measure_sq=False,
              keep_series=False, keep_snapshots=False, seed, **labels):
    """
    Describe una réplica. Las temperaturas se recorren en el orden dado, de modo
    que una lista decreciente es un enfriamiento y una creciente un calentamiento.

    Parameters
    ----------
    nterm, nsamp : int
        MCS de termalización y de muestreo en cada temperatura.
    stride : int
        Se mide una vez cada ``stride`` MCS (nsamp // stride mediciones).
    nblocks : int
        Número de bloques en que se agrupan las mediciones (para errores jackknife
        dentro de una corrida y para el estudio del estimador por bloques).
    labels : dict
        Etiquetas adicionales (por ejemplo ``replica``, ``g``) que se guardan con
        la tarea y sirven para seleccionarla después.
    """
    task = {
        "L": int(L), "periodic": bool(periodic), "J1": float(J1), "J2": float(J2), "h": float(h),
        "init": str(init), "temperatures": [float(t) for t in temperatures],
        "nterm": int(nterm), "nsamp": int(nsamp), "stride": int(stride), "nblocks": int(nblocks),
        "measure_sq": bool(measure_sq), "keep_series": bool(keep_series),
        "keep_snapshots": bool(keep_snapshots), "seed": int(seed),
    }
    for key, value in labels.items():
        task[key] = value.item() if isinstance(value, np.generic) else value
    return task


def effective_counts(task, scale=None):
    """
    MCS de termalización, número de mediciones y MCS de muestreo efectivos, con
    el factor de escala de ``config.MC_SCALE`` aplicado. El número de mediciones
    se redondea a un múltiplo del número de bloques.
    """
    scale = config.MC_SCALE if scale is None else scale
    nterm = max(1, int(round(task["nterm"] * scale)))
    n_meas = max(task["nblocks"], int(round(task["nsamp"] * scale / task["stride"])))
    n_meas -= n_meas % task["nblocks"]
    return nterm, n_meas, n_meas * task["stride"]


def scaled(n_mcs):
    """Número de MCS con el factor de escala de ``config.MC_SCALE`` (igual a ``n_mcs`` en producción)."""
    return max(1, int(round(n_mcs * config.MC_SCALE)))


def task_cost(task):
    """Número de intentos de volteo (actualizaciones de espín) de una tarea."""
    nterm, _, nsamp = effective_counts(task)
    return task["L"] ** 2 * len(task["temperatures"]) * (nterm + nsamp)


# ----------------------------------------------------------------------------
# Ejecución de una tarea
# ----------------------------------------------------------------------------
def run_task(task):
    """
    Ejecuta una réplica y devuelve un dict de arreglos:

    blocks     (nT, nblocks, NOBS) promedios por bloque de las magnitudes de kernels.OBS_NAMES
    acceptance (nT,)               tasa de aceptación
    series     (nT, n_meas, 2)     series de e y m^2 (float32), si keep_series
    snapshots  (nT, N)             configuración al final de cada temperatura (int8), si keep_snapshots
    elapsed    ()                  tiempo de cómputo de la tarea [s]

    Los arreglos siguen el orden de ``task['temperatures']``.
    """
    t0 = time.perf_counter()
    L = task["L"]
    N = L * L
    nn, n_nn, nnn, n_nnn = lattice.build_neighbor_tables(L, task["periodic"])
    rng = np.random.default_rng(task["seed"])
    spins = lattice.initial_configuration(L, task["init"], rng)
    kernels.seed_rng(task["seed"])

    nterm, n_meas, _ = effective_counts(task)
    temps = task["temperatures"]
    nT = len(temps)
    blocks = np.empty((nT, task["nblocks"], kernels.NOBS))
    acceptance = np.empty(nT)
    keep_series = task["keep_series"]
    series = np.empty((nT, n_meas if keep_series else 1, 2), dtype=np.float32)
    snapshots = np.empty((nT, N), dtype=np.int8) if task["keep_snapshots"] else None
    se = np.empty(n_meas if keep_series else 1, dtype=np.float32)
    sm = np.empty(n_meas if keep_series else 1, dtype=np.float32)

    for it, T in enumerate(temps):
        acceptance[it] = kernels.sample_temperature(
            spins, L, float(T), task["J1"], task["J2"], task["h"], nn, n_nn, nnn, n_nnn,
            nterm, n_meas, task["stride"], blocks[it], task["measure_sq"], se, sm, keep_series)
        if keep_series:
            series[it, :, 0] = se
            series[it, :, 1] = sm
        if snapshots is not None:
            snapshots[it] = spins

    out = {"blocks": blocks, "acceptance": acceptance, "elapsed": np.float64(time.perf_counter() - t0)}
    if keep_series:
        out["series"] = series
    if snapshots is not None:
        out["snapshots"] = snapshots
    return out


def warm_up():
    """Compila los núcleos en el proceso actual con una red mínima (con Numba, deja el código en caché)."""
    task = make_task(4, [2.0], nterm=2, nsamp=4, nblocks=2, measure_sq=True, keep_series=True, seed=1)
    run_task(task)


def measure_rate(L=24, n_mcs=200, measure_sq=False):
    """Velocidad de esta máquina con un solo proceso [intentos de volteo por segundo]."""
    warm_up()
    task = make_task(L, [1.0], nterm=n_mcs // 2, nsamp=n_mcs // 2, nblocks=10, measure_sq=measure_sq, seed=2)
    nterm, _, nsamp = effective_counts(task, scale=1.0)
    global MEASURED_RATE
    t0 = time.perf_counter()
    saved = config.MC_SCALE
    config.MC_SCALE = 1.0
    try:
        run_task(task)
    finally:
        config.MC_SCALE = saved
    MEASURED_RATE = L * L * (nterm + nsamp) / (time.perf_counter() - t0)
    return MEASURED_RATE


# ----------------------------------------------------------------------------
# Ejecución en paralelo
# ----------------------------------------------------------------------------
def default_workers():
    """Número de procesos: ISINGJ1J2_WORKERS si está definido; si no, el número de núcleos lógicos."""
    return int(os.environ.get("ISINGJ1J2_WORKERS", os.cpu_count() or 1))


def default_executor():
    """'process' (por defecto), 'thread' o 'serial', según ISINGJ1J2_EXECUTOR."""
    return os.environ.get("ISINGJ1J2_EXECUTOR", "process")


def run_tasks(tasks, workers=None, executor=None, label="", verbose=True):
    """
    Ejecuta una lista de tareas independientes y devuelve sus resultados en el
    mismo orden de ``tasks``.

    Las tareas se envían de la más cara a la más barata para repartir mejor la
    carga. Con 'process' cada tarea corre en un proceso aparte; como las
    funciones viven en un módulo importable, esto funciona igual con el método
    de inicio *fork* (Linux) que con *spawn* (Windows, macOS). Con 'thread' se
    usan hilos, posible porque los núcleos de Numba liberan el GIL.
    """
    workers = default_workers() if workers is None else workers
    executor = default_executor() if executor is None else executor
    order = sorted(range(len(tasks)), key=lambda i: -task_cost(tasks[i]))
    results = [None] * len(tasks)
    t0 = time.perf_counter()
    done = 0
    next_report = 0.1

    def report():
        nonlocal next_report
        frac = done / len(tasks)
        if verbose and (frac >= next_report or done == len(tasks)):
            print(f"  {label}: {done}/{len(tasks)} tareas ({time.perf_counter() - t0:.0f} s)", flush=True)
            while next_report <= frac:
                next_report += 0.1

    if executor == "serial" or workers == 1 or len(tasks) == 1:
        warm_up()
        for i in order:
            results[i] = run_task(tasks[i])
            done += 1
            report()
    else:
        warm_up()       # compila una vez antes de crear los procesos (con Numba, los demás leen la caché)
        pool_cls = (concurrent.futures.ThreadPoolExecutor if executor == "thread"
                    else concurrent.futures.ProcessPoolExecutor)
        with pool_cls(max_workers=min(workers, len(tasks))) as pool:
            futures = {pool.submit(run_task, tasks[i]): i for i in order}
            for fut in concurrent.futures.as_completed(futures):
                results[futures[fut]] = fut.result()
                done += 1
                report()
    return results


# ----------------------------------------------------------------------------
# Resultados de un experimento y caché
# ----------------------------------------------------------------------------
class ExperimentResult:
    """
    Resultados de un experimento: la lista de tareas y, para cada una, sus arreglos.

    ``select(**criterios)`` devuelve los índices de las tareas cuyas claves
    coinciden con los criterios (por ejemplo ``select(L=24, h=0.5)``), y
    ``stack(indices, 'blocks')`` apila los arreglos de esas tareas.
    """

    def __init__(self, name, tasks, results, elapsed=None):
        self.name = name
        self.tasks = tasks
        self.results = results
        self.elapsed = elapsed

    def __len__(self):
        return len(self.tasks)

    def select(self, **criteria):
        def match(task):
            for key, value in criteria.items():
                tv = task.get(key)
                if isinstance(value, float) or isinstance(tv, float):
                    if tv is None or not np.isclose(tv, value):
                        return False
                elif tv != value:
                    return False
            return True
        return [i for i, t in enumerate(self.tasks) if match(t)]

    def values(self, key, **criteria):
        """Valores distintos (ordenados) de la clave ``key`` entre las tareas seleccionadas."""
        return sorted({self.tasks[i][key] for i in self.select(**criteria)})

    def stack(self, indices, key="blocks"):
        return np.stack([self.results[i][key] for i in indices])

    def temperatures(self, index):
        return np.asarray(self.tasks[index]["temperatures"], dtype=float)


def _tasks_hash(tasks):
    payload = json.dumps({"code": CODE_VERSION, "scale": config.MC_SCALE, "tasks": tasks}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def save_experiment(path, result):
    arrays = {"meta": np.array(json.dumps({"name": result.name, "hash": _tasks_hash(result.tasks),
                                            "tasks": result.tasks, "elapsed": result.elapsed}))}
    for i, res in enumerate(result.results):
        for key, value in res.items():
            arrays[f"t{i}__{key}"] = np.asarray(value)
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_experiment(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        results = [dict() for _ in meta["tasks"]]
        for key in data.files:
            if key == "meta":
                continue
            idx, field = key[1:].split("__", 1)
            arr = data[key]
            results[int(idx)][field] = arr if arr.ndim else arr[()]
    return ExperimentResult(meta["name"], meta["tasks"], results, meta.get("elapsed")), meta["hash"]


def run_experiment(name, tasks, workers=None, executor=None, force=False, verbose=True):
    """
    Ejecuta (o recarga de la caché) un experimento completo.

    Antes de simular imprime el costo estimado; al terminar, el tiempo real. Si
    ``data/<name>.npz`` existe y fue producido por exactamente las mismas tareas,
    se recarga sin simular.
    """
    path = config.data_dir() / f"{name}.npz"
    if path.exists() and not force:
        try:
            cached, digest = load_experiment(path)
            if digest == _tasks_hash(tasks):
                if verbose:
                    extra = f", simulado en {cached.elapsed:.0f} s" if cached.elapsed else ""
                    print(f"{name}: {len(tasks)} tareas recargadas de {path.name}{extra}")
                return cached
        except Exception as exc:          # archivo incompleto o de otra versión: se vuelve a simular
            print(f"{name}: no se pudo recargar {path.name} ({exc}); se simula de nuevo")
    total = sum(task_cost(t) for t in tasks)
    if verbose:
        scale_note = f" [escala de MCS = {config.MC_SCALE:g}]" if config.MC_SCALE != 1.0 else ""
        estimate = ""
        if MEASURED_RATE:
            estimate = f"; costo estimado {total / MEASURED_RATE / 60:.1f} min de CPU con un proceso"
        print(f"{name}: {len(tasks)} tareas, {total / 1e9:.2f} x 10^9 intentos de volteo{estimate}{scale_note}")
    t0 = time.perf_counter()
    results = run_tasks(tasks, workers=workers, executor=executor, label=name, verbose=verbose)
    elapsed = time.perf_counter() - t0
    result = ExperimentResult(name, tasks, results, elapsed)
    save_experiment(path, result)
    if verbose:
        cpu = sum(float(r["elapsed"]) for r in results)
        print(f"{name}: terminado en {elapsed:.0f} s de reloj ({cpu / 60:.1f} min de CPU); guardado en {path.name}")
    return result
