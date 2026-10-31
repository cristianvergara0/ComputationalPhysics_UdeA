"""
Pruebas del modelo y de los núcleos de Monte Carlo.

Se ejecutan con ``pytest`` desde la carpeta del proyecto. Funcionan con y sin
Numba; sin Numba las corridas de Monte Carlo son cortas para que la prueba dure
pocos segundos.
"""

import numpy as np
import pytest

from isingj1j2 import config, kernels, lattice, simulation
from isingj1j2.estimators import derived

J1, J2 = 1.0, -0.6


@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("h", [0.0, 0.35])
def test_incremental_energy_matches_direct(periodic, h):
    """La energía y las sumas de orden actualizadas en cada volteo coinciden con un recálculo directo."""
    L = 6
    nn, n_nn, nnn, n_nnn = lattice.build_neighbor_tables(L, periodic)
    spins = lattice.random_configuration(L, np.random.default_rng(1))
    kernels.seed_rng(7)
    E, M, SAB, SX, SY = kernels.energy_after_mcs(spins, L, 1.3, J1, J2, h, nn, n_nn, nnn, n_nnn, 25)
    assert E == pytest.approx(lattice.energy_numpy(spins, J1, J2, h, nn, n_nn, nnn, n_nnn), abs=1e-9)
    assert np.allclose((M, SAB, SX, SY), kernels.order_sums(spins, L))


def test_neighbor_counts_free_boundaries():
    """Con FBC, las esquinas tienen 2 NN y 1 NNN, los bordes 3 y 2, y el interior 4 y 4."""
    L = 5
    _, n_nn, _, n_nnn = lattice.build_neighbor_tables(L, periodic=False)
    n_nn = n_nn.reshape(L, L)
    n_nnn = n_nnn.reshape(L, L)
    assert n_nn[0, 0] == 2 and n_nnn[0, 0] == 1
    assert n_nn[0, 2] == 3 and n_nnn[0, 2] == 2
    assert n_nn[2, 2] == 4 and n_nnn[2, 2] == 4


def test_ground_state_energies_and_critical_field():
    """Energías por espín de los órdenes de referencia y h_c = 4|J2| - 2 J1."""
    L = 8
    tables = lattice.build_neighbor_tables(L, True)
    gs = lattice.ground_state_energies(J1, J2)
    for name, conf in (("ferro", lattice.ferromagnetic_configuration(L)),
                       ("checkerboard", lattice.checkerboard_configuration(L)),
                       ("stripe", lattice.stripe_configuration(L))):
        assert lattice.energy_numpy(conf, J1, J2, 0.0, *tables) / L ** 2 == pytest.approx(gs[name])
    assert lattice.critical_field(J1, J2) == pytest.approx(0.4)


def test_no_intermediate_plateau_small_torus():
    """En un toro 4 x 4 el estado de mínima energía pasa directamente de stripe (mu = 0) a saturado (mu = 1) en h_c."""
    states = lattice.minimum_energy_states(4, 4, J1, J2, [0.2, 0.39, 0.41, 0.6])
    assert states[0.2][1] == 0.0 and states[0.39][1] == 0.0
    assert states[0.41][1] == 1.0 and states[0.6][1] == 1.0


def test_structure_factor_matches_fft():
    """|psi(Q + k)|^2 del muestreador coincide con la FFT 2D de la configuración."""
    for L in (6, 8):
        spins = lattice.random_configuration(L, np.random.default_rng(L))
        c = np.cos(2 * np.pi * np.arange(L) / L)
        s = np.sin(2 * np.pi * np.arange(L) / L)
        skl, skt = kernels.stripe_fourier(spins, L, c, s)
        F = np.fft.fft2(spins.reshape(L, L).astype(float))
        P = np.abs(F) ** 2 / L ** 4
        h = L // 2
        # canal x: Q = (pi, 0); k paralelo (1, 0) y perpendicular (0, 1). Canal y: Q = (0, pi).
        skl_fft = P[(h - 1) % L, 0] + P[0, (h - 1) % L]
        skt_fft = P[h, (-1) % L] + P[(-1) % L, h]
        assert skl == pytest.approx(skl_fft, rel=1e-10)
        assert skt == pytest.approx(skt_fft, rel=1e-10)
    stripe = lattice.stripe_configuration(8)
    skl, skt = kernels.stripe_fourier(stripe, 8, np.cos(2 * np.pi * np.arange(8) / 8), np.sin(2 * np.pi * np.arange(8) / 8))
    assert skl == pytest.approx(0.0, abs=1e-12) and skt == pytest.approx(0.0, abs=1e-12)


def test_covariance_identity_exact():
    """d<M>/dT = Cov(M, E)/T^2 a campo fijo, verificado con la enumeración exacta de un toro 4 x 4."""
    E_int, M, SX, SY = lattice.enumerate_states(4, 4, J1, J2)
    N, T, h, dT = 16, 1.3, 0.3, 1e-4
    mom = lambda t: lattice.exact_thermal_moments(E_int, M, SX, SY, N, t, h)
    ex = mom(T)
    fd = N * (mom(T + dT)["mu"] - mom(T - dT)["mu"]) / (2 * dT)
    cov = N ** 2 * (ex["emu"] - ex["e"] * ex["mu"]) / T ** 2
    assert cov == pytest.approx(fd, rel=1e-6)


def test_sampler_matches_exact_enumeration():
    """Los momentos muestreados en L = 4 (PBC, con campo) coinciden con los exactos dentro de 4 errores estándar."""
    L, T, h = 4, 1.5, 0.3
    E_int, M, SX, SY = lattice.enumerate_states(L, L, J1, J2)
    ex = lattice.exact_thermal_moments(E_int, M, SX, SY, L * L, T, h)
    n_rep = 6
    nsamp = 40000 if config.NUMBA_AVAILABLE else 6000
    moms = []
    for r in range(n_rep):
        task = simulation.make_task(L, [T], h=h, nterm=500, nsamp=nsamp, nblocks=10, seed=100 + r)
        moms.append(simulation.run_task(task)["blocks"][0].mean(axis=0))
    moms = np.array(moms)
    mean = moms.mean(axis=0)
    sem = moms.std(axis=0, ddof=1) / np.sqrt(n_rep)
    for key in ("e", "e2", "mu", "mu2", "emu", "m2", "m4"):
        k = kernels.OBS[key]
        assert abs(mean[k] - ex[key]) <= 4 * sem[k] + 1e-4, (key, mean[k], ex[key], sem[k])
    d = derived(mean[None], np.array([T]), L)
    assert np.isfinite(d["Cv"]).all() and np.isfinite(d["dMdT"]).all()
