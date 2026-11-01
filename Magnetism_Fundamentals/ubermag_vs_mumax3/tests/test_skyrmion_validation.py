"""Réplica en numpy del notebook del skyrmion frente a Ubermag y a las salidas de MuMax3."""
import numpy as np
import pytest

from conftest import ROOT
import validate_skyrmion_numpy as v

NPZ = ROOT / "results" / "ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz"
OUT = ROOT / "mumax3" / "skyrmion_energy_terms_mumax3.out"


@pytest.fixture(scope="module")
def m_rel():
    return v.relax()


def test_initial_disk_has_80_cells():
    assert v.INSIDE.sum() == 80


def test_relaxed_energies_match_ubermag_reference(m_rel):
    e = v.energies(m_rel)
    for k, ref in v.REF.items():
        assert abs(e[k] - ref) / abs(ref) < 2e-7, k
    assert v.max_torque_T(m_rel) < 1e-12


def test_curl_of_uniform_field_vanishes():
    m = np.zeros((v.N, v.N, 3))
    m[..., 2] = 1
    assert np.allclose(v.curl(m), 0) and np.allclose(v.laplace(m), 0)


def test_neumann_boundary_term():
    # En una película de una celda en z, la frontera de Neumann equivale a K' = K - D^2/(2A).
    m = v.initial()
    kp = v.K_U - v.D_DMI ** 2 / (2 * v.A_EX)
    dh = v.heff(m, kp) - v.heff(m, v.K_U)
    assert np.allclose(dh[..., :2], 0)
    assert np.allclose(dh[..., 2], 2 * (kp - v.K_U) / (v.MU0 * v.MS) * m[..., 2])
    en = v.neumann_energies(m)
    e = v.energies(m)
    mperp2 = (m[..., 0] ** 2 + m[..., 1] ** 2).sum()
    assert en["DMI"] - e["DMI"] == pytest.approx(-v.D_DMI ** 2 / (2 * v.A_EX) * mperp2 * v.V_CELL)


def test_euler_stability_estimate():
    lam = v.GAMMA0_UBERMAG * v.ALPHA / (1 + v.ALPHA ** 2) * 2 * v.A_EX / (v.MU0 * v.MS) * 8 / v.D_CELL ** 2
    assert lam * 1e-12 < 2 < 2 * lam * 1e-12      # estable con el factor 1/2, inestable sin él


@pytest.mark.skipif(not NPZ.exists(), reason="sin datos numéricos de Ubermag")
def test_numpy_reproduces_ubermag_fields(m_rel):
    u = np.load(NPZ)
    assert v.max_dm(u["m_relaxed"], m_rel) < 1e-6
    assert v.max_dm(u["M_euler"], v.euler(1e-12, 100)) < 1e-8     # misma gamma0 que el notebook


def test_gamma0_parsing():
    assert v.gamma0_from_log("GammaLL = 0.5 * 2.21276148e5 / mu0   // comentario") == pytest.approx(2.21276148e5)
    assert v.gamma0_from_log("GammaLL = 0.5 * 2.211e5 / mu0") == pytest.approx(2.211e5)
    assert v.gamma0_from_log("// sin esa línea") is None


def test_parse_log_with_both_relaxation_stages(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        'print("MaxTorque (T) tras Run:", MaxTorque.Get())\n//MaxTorque (T) tras Run: 6.6e-05\n'
        'Relax()\nprint("MaxTorque (T) tras Relax:", MaxTorque.Get())\n//MaxTorque (T) tras Relax: 1.5e-07\n'
        '//<m> tras relajar: -2.7e-09 -2.3e-09 0.4288569\n'
        'GammaLL = 0.5 * 2.21276148e5 / mu0\n'
        '//Euler 1 ps: t-t0 (s): 1e-10  <m>: 1e-9 2e-9 0.5151\n'
        '//Euler 0.2 ps: t-t1 (s): 1e-10  <m>: 1e-9 2e-9 0.5156\n', encoding="utf-8")
    d = v.parse_mumax_log(log)
    assert d["maxtorque_run"] == pytest.approx(6.6e-5)
    assert d["maxtorque_relax"] == pytest.approx(1.5e-7)
    assert d["mz_relaxed"] == pytest.approx(0.4288569)
    assert d["mz_euler_1ps"] == pytest.approx(0.5151)
    assert d["mz_euler_02ps"] == pytest.approx(0.5156)
    assert d["gamma0"] == pytest.approx(2.21276148e5)


@pytest.mark.skipif(not (OUT / "log.txt").exists(), reason="sin salidas de MuMax3")
def test_mumax3_outputs_agree_with_numpy(m_rel):
    from ovf_io import read_ovf

    lg = v.parse_mumax_log(OUT / "log.txt")
    g = lg.get("gamma0", v.GAMMA0_UBERMAG)
    assert lg["mz_relaxed"] == pytest.approx(m_rel[..., 2].mean(), abs=1e-4)
    assert v.max_dm(read_ovf(OUT / "m_euler.ovf"), v.euler(1e-12, 100, gamma0=g)) < 1e-5
    d = v.parse_energies(OUT / "energies.txt")
    for key in ("Zeeman", "Anisotropia_conv_Ubermag", "Intercambio", "DMI_OpenBC_true"):
        mx, ref = d[key][:2]
        assert abs(mx - ref) / abs(ref) < 2e-6, key


def test_symmetric_state_is_a_saddle_point(m_rel):
    # Una perturbación pequeña crece: el skyrmion circular no es un mínimo del modelo (sección 8.2).
    import skyrmion_instability_numpy as si

    mode, dm = si.unstable_mode(m_rel, n=8000)
    assert dm > 1e-8                                   # partió de 1e-9 y creció
    assert si.growth_rate(m_rel, mode, n=2000) == pytest.approx(1.175e9, rel=0.05)


@pytest.mark.skipif(not (OUT / "m_relaxed.ovf").exists(), reason="sin salidas de MuMax3")
def test_mumax3_deviation_lies_along_unstable_mode(m_rel):
    import skyrmion_instability_numpy as si
    from ovf_io import read_ovf

    mode, _ = si.unstable_mode(m_rel)
    d = v.unit_layer(read_ovf(OUT / "m_relaxed.ovf")) - m_rel
    assert abs((d * mode).sum()) / np.linalg.norm(d) > 0.99
