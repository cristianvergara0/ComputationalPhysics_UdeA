#!/usr/bin/env python3
"""Estabilidad del skyrmion circular del notebook frente a deformaciones elípticas.

El estado relajado del notebook (Ubermag) es simétrico: el disco inicial está centrado en la malla y
la relajación en doble precisión conserva esa simetría. Este script comprueba, con la réplica en numpy
del modelo (validate_skyrmion_numpy.py), si ese estado es un mínimo:

  1. relaja el modelo hasta un par residual menor que 1e-14 T (el estado simétrico);
  2. añade una perturbación aleatoria de amplitud 1e-9 y la deja evolucionar con el torque de
     amortiguamiento del notebook; si crece, el estado simétrico es un punto de silla, y la dirección
     en la que crece es su modo inestable (se mide su tasa de crecimiento);
  3. proyecta sobre el modo inestable la diferencia entre el estado relajado de MuMax3 (precisión
     simple) y el estado simétrico, y la diferencia entre el estado de Ubermag y el simétrico;
  4. sigue la relajación desde una perturbación de 1e-6 a lo largo de ese modo hasta el nuevo
     equilibrio y compara energías;
  5. repite la prueba de la perturbación con el equilibrio de la frontera de Neumann de MuMax3
     (anisotropía efectiva K' = K - D^2/(2A), sección 8.4 del notebook).

Escribe figures/fig_skyrmion_elliptical_instability.png e imprime las cifras de la sección 8.2 del
notebook del skyrmion.

Uso:
    python scripts/skyrmion_instability_numpy.py
    python scripts/skyrmion_instability_numpy.py --mumax3-ovf ruta/m_relaxed.ovf --save-dir ruta

Requiere numpy y matplotlib.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_skyrmion_numpy as v  # noqa: E402
from ovf_io import read_ovf  # noqa: E402
from plot_skyrmion_states import EXTENT_NM, draw_state  # noqa: E402

DT = 0.5e-12   # paso de la relajación en numpy (lambda*dt = 0.64, estable)


def total_energy(m):
    return sum(v.energies(m).values())


def evolve(m, n, dt=DT, k=v.K_U):
    for _ in range(n):
        m = v.step(m, dt, 1.0, k)
    return m


def unstable_mode(m0, seed=0, amp=1e-9, n=12000, k=v.K_U):
    """Dirección en la que crece una perturbación aleatoria pequeña (régimen lineal), normalizada."""
    rng = np.random.default_rng(seed)
    x = m0 + amp * rng.standard_normal(m0.shape)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    x = evolve(x, n, k=k)
    d = x - m0
    return d / np.linalg.norm(d), v.max_dm(x, m0)


def growth_rate(m0, mode, amp=1e-9, n=4000):
    """Tasa de crecimiento (1/s) de una perturbación a lo largo del modo, con el torque del notebook."""
    x = m0 + amp * mode
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    a0 = ((x - m0) * mode).sum()
    x = evolve(x, n)
    a1 = ((x - m0) * mode).sum()
    return np.log(a1 / a0) / (n * DT)


def relax_from(m, tol=1e-13, n_max=200000, check=500):
    for i in range(1, n_max + 1):
        m = v.step(m, DT)
        if i % check == 0 and v.max_torque_T(m) < tol:
            break
    return m


def draw_signed(ax, f, title):
    lim = np.abs(f).max()
    im = ax.imshow(f.T, origin="lower", extent=EXTENT_NM, cmap="BrBG_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_aspect("equal")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.formatter.set_powerlimits((-2, 2))
    cb.update_ticks()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mumax3-ovf", default=str(v.OUT / "m_relaxed.ovf"), help="estado relajado de MuMax3")
    ap.add_argument("--ubermag-npz", default=str(v.ROOT / "results" / "ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz"))
    ap.add_argument("--save-dir", default=str(v.ROOT / "figures"))
    args = ap.parse_args()

    m0 = v.relax(tol=1e-14)
    e0 = total_energy(m0)
    print("== 1. Estado simétrico (réplica en numpy del modelo de Ubermag) ==")
    print(f"  <m_z> = {m0[..., 2].mean():.5f}; máx |m x B_eff| = {v.max_torque_T(m0):.1e} T; E = {e0:.6e} J")

    mode, dm_lin = unstable_mode(m0)
    rate = growth_rate(m0, mode)
    print("\n== 2. Perturbación aleatoria de 1e-9 ==")
    print(f"  tras 6 ns de la dinámica del notebook: máx |dm| = {dm_lin:.1e} (crece: el estado simétrico es un punto de silla)")
    print(f"  tasa de crecimiento del modo inestable: {rate:.3e} 1/s (la amplitud se multiplica por e cada {1e9 / rate:.2f} ns)")
    print(f"  con DoPrecess=false y GammaLL por defecto (factor 2): {1e9 / (2 * rate * v.GAMMA0_MUMAX3_DEFAULT / v.GAMMA0_UBERMAG):.2f} ns;"
          f" en Run(5e-9) la perturbación se multiplica por e^{5e-9 * 2 * rate * v.GAMMA0_MUMAX3_DEFAULT / v.GAMMA0_UBERMAG:.1f}")

    mx_dev = None
    p = Path(args.mumax3_ovf)
    if p.exists():
        mx = v.unit_layer(read_ovf(p))
        d = mx - m0
        proj = (d * mode).sum()
        if proj < 0:                       # el signo del modo es arbitrario: se alinea con MuMax3
            mode, proj = -mode, -proj
        mx_dev = d
        print("\n== 3a. Estado relajado de MuMax3 frente al simétrico ==")
        print(f"  máx |dm| = {v.max_dm(mx, m0):.2e}; norma de la diferencia = {np.linalg.norm(d):.3e};"
              f" fracción a lo largo del modo inestable = {proj / np.linalg.norm(d):.4f}")
        print(f"  E(MuMax3) - E(simétrico) = {total_energy(mx) - e0:.2e} J; par residual del estado de MuMax3 en numpy = {v.max_torque_T(mx):.2e} T")
    z = Path(args.ubermag_npz)
    if z.exists():
        ub = v.unit_layer(np.load(z)["m_relaxed"])
        d = ub - m0
        print("\n== 3b. Estado de Ubermag frente al simétrico ==")
        print(f"  máx |dm| = {v.max_dm(ub, m0):.2e}; fracción a lo largo del modo inestable = {abs((d * mode).sum()) / np.linalg.norm(d):.3f}")

    x = m0 + 1e-6 * mode
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    m1 = relax_from(x)
    e1 = total_energy(m1)
    en0, en1 = v.energies(m0), v.energies(m1)
    print("\n== 4. Nuevo equilibrio tras seguir el modo inestable ==")
    print(f"  <m_z> = {m1[..., 2].mean():.4f}; máx |m x B_eff| = {v.max_torque_T(m1):.1e} T; máx |dm| respecto del simétrico = {v.max_dm(m1, m0):.2f}")
    print(f"  E - E_simétrico = {e1 - e0:.4e} J ({100 * (e1 - e0) / abs(e0):.2f} %)")
    for k in en0:
        print(f"    {k:12s} {en0[k]: .4e} -> {en1[k]: .4e} J")

    kp = v.K_U - v.D_DMI ** 2 / (2 * v.A_EX)
    mn = v.relax(kp, tol=1e-14)
    _, dmn = unstable_mode(mn, n=24000, k=kp)
    print("\n== 5. Equilibrio con la frontera de Neumann (K') ==")
    print(f"  perturbación aleatoria de 1e-9 tras 12 ns de la dinámica del notebook: máx |dm| = {dmn:.1e}"
          f" ({'no crece: equilibrio estable' if dmn < 1e-8 else 'crece'})")

    save = Path(args.save_dir)
    save.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.7), layout="constrained")
    for ax, m, title in ((axes[0], m0, "Estado simétrico (Ubermag y numpy):\npunto de silla"),
                         (axes[3], m1, f"Equilibrio al seguir el modo:\nE un {100 * (e0 - e1) / abs(e0):.1f} % menor")):
        im = draw_state(ax, m, title)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label=r"$m_z$")
    if mx_dev is not None:
        draw_signed(axes[1], mx_dev[..., 2], "$m_z$(MuMax3) $-$ $m_z$(simétrico)\n(precisión simple, Run + Relax)")
    else:
        axes[1].set_axis_off()
    draw_signed(axes[2], mode[..., 2] / np.abs(mode[..., 2]).max(),
                f"Modo inestable en numpy, componente z normalizada\n(amplitud × e cada {1e9 / rate:.2f} ns)")
    fig.suptitle("El skyrmion circular del notebook es inestable frente a una deformación elíptica")
    out = save / "fig_skyrmion_elliptical_instability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura: {out}")


if __name__ == "__main__":
    main()
