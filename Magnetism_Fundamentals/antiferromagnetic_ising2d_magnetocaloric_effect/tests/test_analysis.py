"""
Pruebas de los estimadores, del análisis de tamaño finito y del efecto magnetocalórico
con datos sintéticos o con la termodinámica exacta de una red pequeña.
"""

import numpy as np
import pytest

from isingj1j2 import estimators, fss, lattice, mce, simulation
from isingj1j2.kernels import NOBS, OBS

J1, J2 = 1.0, -0.6


def test_tau_int_ar1():
    """Para un proceso AR(1) con coeficiente phi, tau_int = (1 + phi) / (2 (1 - phi))."""
    rng = np.random.default_rng(3)
    phi, n = 0.9, 400_000
    x = np.empty(n)
    x[0] = 0.0
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + noise[t]
    tau, err, _ = estimators.tau_int(x)
    assert tau == pytest.approx((1 + phi) / (2 * (1 - phi)), rel=0.05)
    assert err < 0.05 * tau


def test_jackknife_of_mean_is_sem():
    """Para una función lineal, el error jackknife coincide con el error estándar de la media."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=(12, 5))
    out = estimators.jackknife(s, lambda m: {"x": m})
    assert np.allclose(out["x"][1], s.std(axis=0, ddof=1) / np.sqrt(12))


def test_block_variance_estimator_is_biased_low():
    """Con series correlacionadas, la varianza promediada por bloques cortos queda por debajo de la global."""
    rng = np.random.default_rng(5)
    phi, n, nb = 0.98, 200_000, 20
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    blocks = np.zeros((1, nb, NOBS))
    xb = x.reshape(nb, -1)
    blocks[0, :, OBS["m"]] = xb.mean(axis=1)
    blocks[0, :, OBS["m2"]] = (xb ** 2).mean(axis=1)
    T = np.array([1.0])
    block_est = estimators.block_variance_estimator(blocks, T, 1, "m")[0]
    full = estimators.derived(blocks.mean(axis=1), T, 1)["chi"][0]
    assert block_est < full


def test_parabolic_peak_and_crossing():
    T = np.linspace(0.9, 1.1, 21)
    y = 3.0 - 50.0 * (T - 1.013) ** 2
    Tp, yp, edge = fss.parabolic_peak(T, y)
    assert Tp == pytest.approx(1.013, abs=1e-9) and yp == pytest.approx(3.0) and not edge
    a = 0.5 - 2.0 * (T - 0.98)
    b = 0.5 - 4.0 * (T - 0.98)
    assert fss.find_crossing(T, a, b) == pytest.approx(0.98, abs=1e-9)


def test_weighted_loglog_fit():
    L = np.array([16.0, 24.0, 32.0, 48.0, 64.0])
    y = 3.0 * L ** 2.1
    a, err, b, chi2 = fss.weighted_loglog_fit(L, y, 0.05 * y)
    assert abs(a - 2.1) < 1e-10 and abs(np.exp(b) - 3.0) < 1e-9 and chi2 < 1e-20
    # Con puntos dispersos más allá de sus errores, el error de la pendiente se agranda.
    rng = np.random.default_rng(1)
    y_noisy = y * np.exp(rng.normal(0.0, 0.2, len(L)))
    a1, err1, _, chi2_1 = fss.weighted_loglog_fit(L, y_noisy, 0.02 * y_noisy)
    a2, err2, _, _ = fss.weighted_loglog_fit(L, y_noisy, 0.2 * y_noisy)
    assert chi2_1 > 1 and abs(a1 - a2) < 1e-10 and err1 > 0.5 * err2


def test_fss_recovers_synthetic_exponents():
    """fss_statistics recupera T_c y gamma/nu de curvas sintéticas con la forma de escalado."""
    Tc, inv_nu, gnu = 0.975, 1.5, 1.75
    T = np.round(np.arange(0.90, 1.05, 0.0025), 5)
    data = {}
    for L in (16, 24, 32, 48):
        x = (T - Tc) * L ** inv_nu
        mom = np.zeros((2, len(T), NOBS))
        # m^2 y m^4 con U = 1 - m4/(3 m2^2) monótona en x; skl fijado para que xi/L = f(x)
        u = 0.45 - 0.15 * np.tanh(x)
        m2 = np.full_like(T, 1.0)
        mom[:, :, OBS["m2"]] = m2
        mom[:, :, OBS["m4"]] = 3.0 * (1 - u) * m2 ** 2
        xiL = 0.7 - 0.3 * np.tanh(x)
        k_fac = 2.0 * L * np.sin(np.pi / L)
        sk = m2 / (1.0 + (xiL * k_fac) ** 2)
        mom[:, :, OBS["skl"]] = sk
        mom[:, :, OBS["skt"]] = sk
        chi = 0.5 * L ** gnu * np.exp(-x ** 2)
        mom[:, :, OBS["m"]] = np.sqrt(m2 - chi * T / L ** 2)
        mom[:, :, OBS["e2"]] = np.exp(-x ** 2) * L * T ** 2 / L ** 2
        mom[:, :, OBS["e"]] = 0.0
        data[L] = (T, mom)
    out = fss.fss_statistics(data, L1_min=16, collapse_halfwidth=0.02)
    assert out["T_star"] == pytest.approx(Tc, abs=2e-3)
    assert out["gamma_nu"] == pytest.approx(gnu, abs=0.05)


def _exact_field_dataset(L, H, T):
    """Momentos exactos (enumeración) con la estructura que espera mce.analyze: {h: (1, nT, NOBS)}."""
    E_int, M, SX, SY = lattice.enumerate_states(L, L, J1, J2)
    N = L * L
    groups = {}
    for h in H:
        mom = np.zeros((1, len(T), NOBS))
        for j, t in enumerate(T):
            ex = lattice.exact_thermal_moments(E_int, M, SX, SY, N, t, h)
            for key in ("e", "e2", "mu", "mu2", "emu"):
                mom[0, j, OBS[key]] = ex[key]
        groups[float(h)] = mom
    return groups, E_int, M


def test_mce_routes_match_exact_entropy():
    """
    Con la termodinámica exacta de un toro 4 x 4, los dos caminos para Delta S_M
    (Maxwell e integración en T) reproducen S(T, H) - S(T, 0) calculada con la
    función de partición, y Delta T_ad exacto conserva la entropía.
    """
    L = 4
    N = L * L
    H = np.round(np.arange(0.0, 1.0001, 0.05), 3)
    T = np.round(np.concatenate([np.arange(0.6, 1.5, 0.01), np.arange(1.5, 4.0001, 0.05)]), 4)
    groups, E_int, M = _exact_field_dataset(L, H, T)

    def entropy(t, h):
        E = E_int - h * M
        w = np.exp(-(E - E.min()) / t)
        Z = w.sum()
        return np.log(Z) - E.min() / t + (w * E).sum() / Z / t

    exact = np.array([[entropy(t, h) - entropy(t, 0.0) for t in T] for h in H])
    an = mce.analyze(H, T, groups, L, T_star=1.0)
    assert np.max(np.abs(an["dS"] - exact)) / N < 2e-3
    assert np.max(np.abs(an["dS_maxwell"] - exact)) / N < 2e-3
    # Delta T_ad exacto: S(T + dT, H) = S(T, 0)
    k = int(np.argmin(np.abs(H - 0.5)))
    j = int(np.argmin(np.abs(T - 1.2)))
    Tp = T[j] + an["dTad_exact"][k, j]
    assert entropy(Tp, H[k]) == pytest.approx(entropy(T[j], 0.0), abs=2e-3 * N)


def test_branch_peaks_synthetic():
    T = np.linspace(0.5, 2.0, 151)
    dS = 2.0 * np.exp(-((T - 0.85) / 0.08) ** 2) - 1.0 * np.exp(-((T - 1.2) / 0.2) ** 2)
    pk = mce.branch_peaks(T, dS, T_star=0.976)
    assert pk["T_inv"] == pytest.approx(0.85, abs=0.01)
    assert pk["T_conv"] == pytest.approx(1.2, abs=0.02)
    assert pk["fwhm_conv"] == pytest.approx(2 * 0.2 * np.sqrt(np.log(2)), rel=0.05)


def test_experiment_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ISINGJ1J2_DATA_DIR", str(tmp_path))
    tasks = [simulation.make_task(4, [2.0, 1.5], nterm=5, nsamp=20, nblocks=4, keep_series=True,
                                  keep_snapshots=True, seed=simulation.task_seed("test", r), replica=r)
             for r in range(2)]
    res = simulation.run_experiment("cache_test", tasks, executor="serial", verbose=False)
    again = simulation.run_experiment("cache_test", tasks, executor="serial", verbose=False)
    assert again.select(replica=1) == [1]
    for a, b in zip(res.results, again.results):
        assert np.array_equal(a["blocks"], b["blocks"])
        assert np.array_equal(a["snapshots"], b["snapshots"])
    # La misma semilla reproduce exactamente la misma réplica.
    rerun = simulation.run_task(tasks[0])
    assert np.array_equal(rerun["blocks"], res.results[0]["blocks"])
