#!/usr/bin/env python3
"""Reproducción en numpy (doble precisión) del problema del notebook de energías del skyrmion.

Implementa las mismas operaciones que el notebook de Ubermag sobre la malla de 20 x 20 x 1 celdas
de 5 nm, periódica en x e y (laplaciano de 5 puntos y rotacional con diferencias centrales; no hay
derivadas en z con una sola celda) y comprueba:

  1. el estado relajado del modelo de Ubermag: <m_z>, par residual y las cuatro energías frente a
     las referencias de la documentación de Ubermag (Field operations 2);
  2. el integrador de Euler explícito del notebook (100 pasos de 1 ps) y la variante de 500 pasos
     de 0.2 ps, junto con la estimación de estabilidad lambda * dt < 2;
  3. el término de frontera -D^2/(2A) (m_x^2 + m_y^2) que añade la condición de Neumann por
     defecto de MuMax3 en una película de una celda en z (equivale a K' = K - D^2/(2A)), y el
     efecto de que MuMax3 con DoPrecess=false no incluya el factor alpha/(1+alpha^2) = 1/2;
  4. la comparación con lo que devolvió MuMax3 (log, energías y estados OVF de
     mumax3/skyrmion_energy_terms_mumax3.out/), con la gamma0 que el script usó en GammaLL;
  5. la comparación con los campos exactos de Ubermag (results/, creado con
     benchmark_ubermag_vs_mumax3.py --dump-arrays), frente a numpy y frente a MuMax3;
  6. si existe mumax3/skyrmion_default_settings_mumax3.out/, los estados de MuMax3 con la frontera
     de Neumann y con GammaLL por defecto frente a sus réplicas en numpy.

Uso (por defecto lee mumax3/ y results/):
    python scripts/validate_skyrmion_numpy.py
    python scripts/validate_skyrmion_numpy.py --out-dir ruta/skyrmion_energy_terms_mumax3.out

Requiere solo numpy.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from ovf_io import read_ovf

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
OUT = ROOT / "mumax3" / "skyrmion_energy_terms_mumax3.out"
OUT_DEFAULT = ROOT / "mumax3" / "skyrmion_default_settings_mumax3.out"
MU0 = 4e-7 * np.pi
# Razón giromagnética en m A^-1 s^-1. El Euler del notebook usa mm.consts.gamma0 de micromagneticmodel, igual a
# mu0 * gamma_e con los valores de CODATA. El script de MuMax3 fija GammaLL = 0.5 * gamma0 / mu0 (sección 8.3);
# la gamma0 que usó cada ejecución se lee de su log.txt (gamma0_from_log).
GAMMA0_UBERMAG = 2.21276148e5
GAMMA0_MUMAX3_DEFAULT = 1.7595e11 * MU0   # GammaLL por defecto de MuMax3 (1.7595e11 rad T^-1 s^-1)
L, T_FILM, D_CELL, N = 100e-9, 5e-9, 5e-9, 20
MS, A_EX, D_DMI, K_U, HZ = 3.84e5, 8.78e-12, 1.58e-3, 1e4, 1e5
ALPHA = 1.0
V_CELL = D_CELL * D_CELL * T_FILM
# Referencias de la documentación de Ubermag (Field operations 2), en J
REF = dict(Zeeman=-1.03472186e-18, Anisotropía=2.0433559e-19, Intercambio=1.74383742e-18, DMI=-4.35058832e-18)

_xc = -L / 2 + (np.arange(N) + 0.5) * D_CELL
_X, _Y = np.meshgrid(_xc, _xc, indexing="ij")
INSIDE = _X ** 2 + _Y ** 2 < (L / 4) ** 2        # disco inicial de radio L/4


def initial():
    m = np.zeros((N, N, 3))
    m[..., 2] = 1.0
    m[INSIDE] = (0.0, 0.0, -1.0)
    return m


def ddx(f): return (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * D_CELL)
def ddy(f): return (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * D_CELL)
def lap(f): return (np.roll(f, -1, 0) + np.roll(f, 1, 0) + np.roll(f, -1, 1) + np.roll(f, 1, 1) - 4 * f) / D_CELL ** 2
def laplace(m): return np.stack([lap(m[..., i]) for i in range(3)], axis=-1)
def curl(m): return np.stack([ddy(m[..., 2]), -ddx(m[..., 2]), ddx(m[..., 1]) - ddy(m[..., 0])], axis=-1)


def heff(m, k=K_U):
    """H_eff en A/m: intercambio + DMI (clase T) + anisotropía uniaxial + Zeeman."""
    h = 2 * A_EX / (MU0 * MS) * laplace(m) - 2 * D_DMI / (MU0 * MS) * curl(m)
    h[..., 2] += 2 * k / (MU0 * MS) * m[..., 2] + HZ
    return h


def step(m, dt, factor=1.0, k=K_U, gamma0=GAMMA0_UBERMAG):
    """Un paso de Euler del torque de amortiguamiento puro, con renormalización.

    factor = 1: el esquema del notebook (-gamma0 alpha/(1+alpha^2) m x (m x H)).
    factor = 2: lo que hace MuMax3 con DoPrecess=false y GammaLL sin ajustar (sin el factor 1/2).
    gamma0: GAMMA0_UBERMAG reproduce el Euler del notebook; con DoPrecess=false, MuMax3 usa GammaLL * mu0.
    """
    mxh = np.cross(m, heff(m, k))
    m = m - gamma0 * ALPHA / (1 + ALPHA ** 2) * factor * np.cross(m, mxh) * dt
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def energies(m):
    """Energías en J, con las convenciones de Ubermag (anisotropía K|m x u|^2)."""
    return dict(
        Zeeman=(-MU0 * MS * HZ * m[..., 2]).sum() * V_CELL,
        Anisotropía=(K_U * (1 - m[..., 2] ** 2)).sum() * V_CELL,
        Intercambio=(-A_EX * (m * laplace(m)).sum(-1)).sum() * V_CELL,
        DMI=(D_DMI * (m * curl(m)).sum(-1)).sum() * V_CELL,
    )


def max_torque_T(m, k=K_U):
    return np.linalg.norm(np.cross(m, heff(m, k) * MU0), axis=-1).max()


def relax(k=K_U, dt=0.5e-12, n=40000, tol=1e-13):
    """Equilibrio por el torque de amortiguamiento, hasta que máx |m x B_eff| < tol (en T) o n pasos."""
    m = initial()
    for i in range(1, n + 1):
        m = step(m, dt, 1.0, k)
        if i % 500 == 0 and max_torque_T(m, k) < tol:
            break
    return m


def euler(dt, n, factor=1.0, k=K_U, gamma0=GAMMA0_UBERMAG):
    m = initial()
    for _ in range(n):
        m = step(m, dt, factor, k, gamma0)
    return m


NUM = r"([-+0-9.eE]+)"


def parse_mumax_log(path):
    """Valores que imprimen los scripts de MuMax3 del skyrmion (claves ausentes si no aparecen)."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    pats = {
        "maxtorque_run": r"MaxTorque \(T\) tras Run:\s*" + NUM,
        "maxtorque_relax": r"MaxTorque \(T\) tras Relax:\s*" + NUM,
        "mz_relaxed": r"<m> tras relajar:\s*" + NUM + r"\s+" + NUM + r"\s+" + NUM,
        "mz_euler_1ps": r"Euler 1 ps[^\n]*<m>:\s*" + NUM + r"\s+" + NUM + r"\s+" + NUM,
        "mz_euler_02ps": r"Euler 0\.2 ps[^\n]*<m>:\s*" + NUM + r"\s+" + NUM + r"\s+" + NUM,
    }
    for key, pat in pats.items():
        m = re.search(r"^//" + pat, txt, re.M) or re.search(pat, txt)
        if m:
            out[key] = float(m.group(m.re.groups))   # la última captura (m_z en las líneas de <m>)
    g = gamma0_from_log(txt)
    if g is not None:
        out["gamma0"] = g
    return out


def gamma0_from_log(txt):
    """gamma0 (m A^-1 s^-1) de la línea "GammaLL = 0.5 * <gamma0> / mu0" que MuMax3 copia en su log."""
    m = re.search(r"^GammaLL\s*=\s*0\.5\s*\*\s*" + NUM + r"\s*/\s*mu0", txt, re.M)
    return float(m.group(1)) if m else None


def unit_layer(a):
    """m de la única capa en z, normalizado (Ubermag guarda M en A/m y MuMax3 m adimensional)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 4:
        a = a[:, :, 0, :]
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def max_dm(a, b):
    """máx. |m_a - m_b| celda a celda."""
    return float(np.linalg.norm(unit_layer(a) - unit_layer(b), axis=-1).max())


def neumann_energies(m):
    """Energías en J con la frontera de Neumann de MuMax3: la DMI incluye el término de frontera."""
    e = energies(m)
    e["DMI"] += -D_DMI ** 2 / (2 * A_EX) * (m[..., 0] ** 2 + m[..., 1] ** 2).sum() * V_CELL
    return e


def parse_energies(path):
    d = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, *vals = line.split()
        d[name] = tuple(float(v) for v in vals)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT), help="salida de skyrmion_energy_terms_mumax3.mx3")
    ap.add_argument("--default-out-dir", default=str(OUT_DEFAULT), help="salida de skyrmion_default_settings_mumax3.mx3")
    ap.add_argument("--ubermag-npz", default=str(ROOT / "results" / "ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz"))
    args = ap.parse_args()
    out, out_def = Path(args.out_dir), Path(args.default_out_dir)

    print(f"Celdas dentro del disco inicial: {INSIDE.sum()} de {N * N}")
    m_rel = relax()
    e = energies(m_rel)
    print("\n== 1. Estado relajado del modelo de Ubermag (numpy, doble precisión) ==")
    print(f"  <m_z> = {m_rel[..., 2].mean():.5f}   máx |m x B_eff| = {max_torque_T(m_rel):.1e} T")
    for k, ref in REF.items():
        print(f"  {k:12s} numpy {e[k]: .8e}  referencia Ubermag {ref: .8e}  dif. rel. {abs(e[k] - ref) / abs(ref):.1e}")

    lg = parse_mumax_log(out / "log.txt") if (out / "log.txt").exists() else {}
    g_mx = lg.get("gamma0", GAMMA0_UBERMAG)

    print("\n== 2. Euler explícito del notebook ==")
    m1_ub = euler(1e-12, 100)                                  # gamma0 del notebook
    m1, m2 = euler(1e-12, 100, gamma0=g_mx), euler(0.2e-12, 500, gamma0=g_mx)
    lam = GAMMA0_UBERMAG * ALPHA / (1 + ALPHA ** 2) * 2 * A_EX / (MU0 * MS) * 8 / D_CELL ** 2
    print(f"  gamma0 = {GAMMA0_UBERMAG:.6e} (notebook): 100 pasos de 1 ps -> <m_z> = {m1_ub[..., 2].mean():.4f}")
    print(f"  gamma0 = {g_mx:.6e} (GammaLL del script de MuMax3): 100 x 1 ps -> <m_z> = {m1[..., 2].mean():.4f};"
          f" 500 x 0.2 ps -> {m2[..., 2].mean():.4f}")
    print(f"  lambda = {lam:.3e} 1/s; lambda*dt = {lam * 1e-12:.2f} con el esquema del notebook y {2 * lam * 1e-12:.2f} sin el factor 1/2 (límite de estabilidad: 2)")

    print("\n== 3. Lo que cambia en MuMax3 con los ajustes por defecto ==")
    kp = K_U - D_DMI ** 2 / (2 * A_EX)
    mperp2 = m_rel[..., 0] ** 2 + m_rel[..., 1] ** 2
    e_front = -D_DMI ** 2 / (2 * A_EX) * mperp2.sum() * V_CELL
    print(f"  D^2/(2A) = {D_DMI ** 2 / (2 * A_EX):.3e} J/m3; K' = K - D^2/(2A) = {kp:.3e} J/m3")
    print(f"  sobre el estado relajado: <m_perp^2> = {mperp2.mean():.4f}; E_frontera = {e_front:.4e} J; DMI con Neumann prevista = {e['DMI'] + e_front:.4e} J")
    mn = relax(kp)
    print(f"  relajado con K'                     : <m_z> = {mn[..., 2].mean():.4f}")
    m_f2 = euler(1e-12, 100, 2.0, kp, GAMMA0_MUMAX3_DEFAULT)
    m_f1 = euler(1e-12, 100, 1.0, kp, GAMMA0_MUMAX3_DEFAULT)
    print(f"  Euler 1 ps con K' y factor 1 (MuMax3): <m_z> = {m_f2[..., 2].mean():.4f}; con el factor 1/2: {m_f1[..., 2].mean():.4f}; máx |dm| entre ambos = {np.abs(m_f2 - m_f1).max():.2f}")
    print(f"  Euler 0.2 ps con K' y factor 1       : <m_z> = {euler(0.2e-12, 500, 2.0, kp, GAMMA0_MUMAX3_DEFAULT)[..., 2].mean():.4f}")
    en = neumann_energies(mn)
    print("  energías de ese equilibrio con la frontera de Neumann (J):", {k: float(f"{v:.4e}") for k, v in en.items()})

    states = {name: out / f"{name}.ovf" for name in ("m_relaxed", "m_euler", "m_euler_dt02ps")}
    mx = {k: read_ovf(p) for k, p in states.items() if p.exists()}
    if lg:
        print(f"\n== 4. MuMax3 frente a numpy (gamma0 del script: {g_mx:.6e}) ==")
        for key, label in (("maxtorque_run", "tras Run"), ("maxtorque_relax", "tras Relax")):
            if key in lg:
                print(f"  MaxTorque {label:10s}: MuMax3 {lg[key]:.2e} T | numpy (equilibrio) {max_torque_T(m_rel):.1e} T")
        rows = (("m_relaxed", "mz_relaxed", "relajado    ", m_rel, ".5f"),
                ("m_euler", "mz_euler_1ps", "Euler 1 ps  ", m1, ".4f"),
                ("m_euler_dt02ps", "mz_euler_02ps", "Euler 0.2 ps", m2, ".4f"))
        for state, key, label, ref_np, fmt in rows:
            if key not in lg:
                continue
            line = f"  {label}: <m_z> MuMax3 {lg[key]:{fmt}} | numpy {ref_np[..., 2].mean():{fmt}}"
            if state in mx:
                line += f"; máx |dm| = {max_dm(mx[state], ref_np):.1e}"
            print(line)
    if (out / "energies.txt").exists():
        d = parse_energies(out / "energies.txt")
        names = [("Zeeman", "Zeeman"), ("Anisotropia_conv_Ubermag", "Anisotropía"), ("Intercambio", "Intercambio"), ("DMI_OpenBC_true", "DMI")]
        print("  energías (J): MuMax3 | referencia Ubermag | dif. rel. MuMax3-Ubermag | dif. rel. numpy-Ubermag")
        for key, name in names:
            v, ref = d[key][:2]
            print(f"    {name:12s} {v: .8e} | {ref: .8e} | {abs(v - ref) / abs(ref):.1e} | {abs(e[name] - ref) / abs(ref):.1e}")
        if "DMI_Neumann_informativo" in d:
            print(f"    DMI con Neumann (MuMax3): {d['DMI_Neumann_informativo'][0]:.4e} | prevista con el término de frontera: {e['DMI'] + e_front:.4e}")

    z = Path(args.ubermag_npz)
    if z.exists():
        u = np.load(z)
        mu_rel = unit_layer(u["m_relaxed"])
        mu_eul = unit_layer(u["M_euler"])
        print("\n== 5. Ubermag (campos exactos) frente a numpy y a MuMax3 ==")
        print(f"  relajado  : <m_z> Ubermag {mu_rel[..., 2].mean():.5f} | numpy {m_rel[..., 2].mean():.5f}; máx |dm| numpy-Ubermag = {max_dm(mu_rel, m_rel):.1e}")
        print(f"  Euler 1 ps: <m_z> Ubermag {mu_eul[..., 2].mean():.4f} | numpy {m1_ub[..., 2].mean():.4f}; máx |dm| numpy-Ubermag = {max_dm(mu_eul, m1_ub):.1e}")
        if "m_relaxed" in mx:
            print(f"  relajado  : máx |dm| MuMax3-Ubermag = {max_dm(mx['m_relaxed'], mu_rel):.1e}")
        if "m_euler" in mx:
            print(f"  Euler 1 ps: máx |dm| MuMax3-Ubermag = {max_dm(mx['m_euler'], mu_eul):.1e}"
                  f" (numpy con la gamma0 del script de MuMax3 frente a Ubermag: {max_dm(m1, mu_eul):.1e})")

    if (out_def / "log.txt").exists():
        ld = parse_mumax_log(out_def / "log.txt")
        print("\n== 6. MuMax3 con los ajustes por defecto frente a sus réplicas en numpy ==")
        for key, label in (("maxtorque_run", "tras Run"), ("maxtorque_relax", "tras Relax")):
            if key in ld:
                print(f"  MaxTorque {label:10s}: MuMax3 {ld[key]:.2e} T | numpy (equilibrio con K') {max_torque_T(mn, kp):.1e} T")
        p = out_def / "m_relaxed_neumann.ovf"
        if p.exists():
            mxn = read_ovf(p)
            print(f"  relajado con Neumann: <m_z> MuMax3 {unit_layer(mxn)[..., 2].mean():.5f} | numpy (K') {mn[..., 2].mean():.5f}; máx |dm| = {max_dm(mxn, mn):.1e}")
        if (out_def / "energies_neumann.txt").exists():
            dn = parse_energies(out_def / "energies_neumann.txt")
            names = [("Zeeman", "Zeeman"), ("Anisotropia_conv_Ubermag", "Anisotropía"), ("Intercambio", "Intercambio"), ("DMI_Neumann", "DMI")]
            print("  energías (J): MuMax3 | numpy (K', con el término de frontera en la DMI) | dif. rel.")
            for key, name in names:
                v = dn[key][0]
                print(f"    {name:12s} {v: .6e} | {en[name]: .6e} | {abs(v - en[name]) / abs(en[name]):.1e}")
        p = out_def / "m_euler_default.ovf"
        if p.exists():
            mxe = read_ovf(p)
            print(f"  Euler 1 ps con GammaLL por defecto: <m_z> MuMax3 {unit_layer(mxe)[..., 2].mean():.4f} | numpy {m_f2[..., 2].mean():.4f};"
                  f" máx |dm| = {max_dm(mxe, m_f2):.1e}")


if __name__ == "__main__":
    main()
