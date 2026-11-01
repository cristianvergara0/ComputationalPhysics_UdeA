#!/usr/bin/env python3
"""Estados del skyrmion de Ubermag y de MuMax3 dibujados con la misma función.

Lee los estados que guardó MuMax3 (archivos OVF de skyrmion_energy_terms_mumax3.out/) y los de
Ubermag, y los dibuja con el mismo mapa de colores, la misma escala de m_z, la misma densidad de
flechas y los mismos ejes, junto con el mapa de la diferencia |m_MuMax3 - m_Ubermag| celda a celda.
Así la comparación depende solo de los datos y no de las herramientas de graficación de cada código.

Figuras que escribe:

  fig_skyrmion_ubermag_vs_mumax3.png
      Fila 1: equilibrio (Ubermag | MuMax3 con OpenBC=true | diferencia).
      Fila 2: Euler explícito de 100 pasos de 1 ps (notebook | MuMax3 con GammaLL a la mitad | diferencia).

  fig_skyrmion_mumax3_default_settings.png
      Efecto de los ajustes por defecto de MuMax3 sobre el mismo problema: equilibrio con la
      frontera de Neumann del DMI y Euler de 1 ps sin el factor alpha/(1+alpha^2). Los estados se
      leen de mumax3/skyrmion_default_settings_mumax3.out/ (o de --neumann-ovf y
      --euler-default-ovf); si no existen, se usan las réplicas en numpy de
      validate_skyrmion_numpy.py, y el título de cada panel indica el origen de cada estado.

Datos de Ubermag: si existe ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz (lo crea
benchmark_ubermag_vs_mumax3.py --dump-arrays) se usan los campos exactos del notebook; si no, la
réplica en numpy del modelo de Ubermag (doble precisión), que coincide con sus energías a 7e-8.

Uso (desde cualquier carpeta, después de ejecutar mumax3/skyrmion_energy_terms_mumax3.mx3):
    python scripts/plot_skyrmion_states.py
    python scripts/plot_skyrmion_states.py --out-dir ruta/a/la/salida.out --ubermag-npz ruta/al/archivo.npz

Por defecto lee mumax3/ y results/, y escribe en figures/.

Requiere numpy y matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_skyrmion_numpy as ref  # noqa: E402  (mismo modelo, mismas constantes)
from ovf_io import read_ovf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent   # carpeta del proyecto
EXTENT_NM = (-50, 50, -50, 50)   # coordenadas de la malla de Ubermag, centrada en el origen


def unit(m):
    """m de una sola capa (nx, ny, 3), normalizado: Ubermag guarda M en A/m y MuMax3 m adimensional."""
    m = np.asarray(m, dtype=float)
    if m.ndim == 4:
        m = m[:, :, 0, :]
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------------------------
# Dibujo: una sola función para todos los paneles
# ---------------------------------------------------------------------------------------------
def draw_state(ax, m, title):
    im = ax.imshow(m[..., 2].T, origin="lower", extent=EXTENT_NM, cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    n = m.shape[0]
    c = EXTENT_NM[0] + (np.arange(n) + 0.5) * (EXTENT_NM[1] - EXTENT_NM[0]) / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    ax.quiver(X, Y, m[..., 0], m[..., 1], color="0.15", pivot="mid", scale=26, width=0.005, alpha=0.75)
    ax.set_title(f"{title}\n$\\langle m_z\\rangle$ = {m[..., 2].mean():.4f}", fontsize=9.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    return im


def draw_difference(ax, m_a, m_b, title):
    d = np.linalg.norm(m_a - m_b, axis=-1)
    im = ax.imshow(d.T, origin="lower", extent=EXTENT_NM, cmap="magma", vmin=0, vmax=max(d.max(), 1e-12),
                   interpolation="nearest")
    ax.set_title(f"{title}\nmáx. = {d.max():.1e}", fontsize=9.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()
    return d.max()


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "mumax3" / "skyrmion_energy_terms_mumax3.out"),
                    help="carpeta de salida de MuMax3")
    ap.add_argument("--ubermag-npz", default=str(ROOT / "results" / "ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz"))
    ap.add_argument("--neumann-ovf", default=str(ref.OUT_DEFAULT / "m_relaxed_neumann.ovf"),
                    help="OVF del equilibrio de MuMax3 con la frontera por defecto")
    ap.add_argument("--euler-default-ovf", default=str(ref.OUT_DEFAULT / "m_euler_default.ovf"),
                    help="OVF del Euler de MuMax3 con GammaLL por defecto")
    ap.add_argument("--save-dir", default=str(ROOT / "figures"), help="carpeta donde se guardan las figuras")
    args = ap.parse_args()
    out = Path(args.out_dir)
    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)

    # MuMax3: estados del script final (OpenBC=true; GammaLL a la mitad en el Euler)
    mx_rel = unit(read_ovf(out / "m_relaxed.ovf"))
    mx_eul = unit(read_ovf(out / "m_euler.ovf"))

    # Ubermag: campos exactos del notebook si están disponibles; si no, la réplica en numpy de su modelo
    npz = Path(args.ubermag_npz)
    if npz.exists():
        u = np.load(npz)
        ub_rel, ub_eul = unit(u["m_relaxed"]), unit(u["M_euler"])
        src_ub = "Ubermag"
        print(f"Ubermag: campos del notebook ({npz})")
    else:
        ub_rel, ub_eul = ref.relax(), ref.euler(1e-12, 100, gamma0=ref.GAMMA0_UBERMAG)
        src_ub = "Modelo de Ubermag (numpy)"
        print("Ubermag: no se encontró el .npz; se usa la réplica en numpy del modelo de Ubermag")

    # --- Figura 1: la comparación de igual a igual -----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.4), layout="constrained")
    im = draw_state(axes[0, 0], ub_rel, f"{src_ub}: equilibrio")
    draw_state(axes[0, 1], mx_rel, "MuMax3: equilibrio (OpenBC = true)")
    d_rel = draw_difference(axes[0, 2], mx_rel, ub_rel, r"$|\mathbf{m}_\mathrm{MuMax3}-\mathbf{m}_\mathrm{Ubermag}|$")
    draw_state(axes[1, 0], ub_eul, f"{src_ub}: Euler, 100 × 1 ps")
    draw_state(axes[1, 1], mx_eul, "MuMax3: Euler, 100 × 1 ps (GammaLL / 2)")
    d_eul = draw_difference(axes[1, 2], mx_eul, ub_eul, r"$|\mathbf{m}_\mathrm{MuMax3}-\mathbf{m}_\mathrm{Ubermag}|$")
    fig.colorbar(im, ax=axes[:, 1], fraction=0.05, pad=0.02, label=r"$m_z$")
    fig.suptitle("Skyrmion de Bloch con el mismo modelo en los dos códigos (misma función de dibujo)")
    f1 = save / "fig_skyrmion_ubermag_vs_mumax3.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{f1}: máx |dm| equilibrio = {d_rel:.1e}; Euler = {d_eul:.1e}")

    # --- Figura 2: qué cambia con los ajustes por defecto de MuMax3 ---------------------------
    kp = ref.K_U - ref.D_DMI ** 2 / (2 * ref.A_EX)
    if Path(args.neumann_ovf).exists():
        neu, neu_src = unit(read_ovf(args.neumann_ovf)), "MuMax3"
    else:
        neu, neu_src = ref.relax(kp), "réplica numpy de MuMax3"
    if Path(args.euler_default_ovf).exists():
        eud, eud_src = unit(read_ovf(args.euler_default_ovf)), "MuMax3"
    else:
        eud, eud_src = ref.euler(1e-12, 100, 2.0, kp, ref.GAMMA0_MUMAX3_DEFAULT), "réplica numpy de MuMax3"
    print(f"Ajustes por defecto: equilibrio con Neumann de {neu_src}; Euler de {eud_src}")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), layout="constrained")
    im = draw_state(axes[0], mx_rel, "MuMax3, OpenBC = true\n(mismo modelo que Ubermag)")
    draw_state(axes[1], neu, f"Frontera de Neumann por defecto\n({neu_src})")
    draw_state(axes[2], eud, f"Euler 1 ps sin el factor α/(1+α²)\ny con Neumann ({eud_src})")
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02, label=r"$m_z$")
    fig.suptitle("Efecto de los ajustes por defecto de MuMax3: otro modelo (centro) y un paso inestable (derecha)", y=1.06)
    f2 = save / "fig_skyrmion_mumax3_default_settings.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{f2}")


if __name__ == "__main__":
    main()
