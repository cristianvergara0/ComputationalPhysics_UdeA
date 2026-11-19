"""Calculador en Numba: tensor desmagnetizante, términos de energía, minimización e interfaz con Ubermag."""
import numpy as np
import pytest

from conftest import ROOT

nc = pytest.importorskip("numba_calculator", reason="requiere numba y scipy")
import validate_skyrmion_numpy as v  # noqa: E402
import validate_torque_macrospin as vt  # noqa: E402

SKY_NPZ = ROOT / "results" / "ubermag_arrays_micromagnetic_energy_terms_skyrmion.npz"
TQ_NPZ = ROOT / "results" / "ubermag_arrays_torque_magnetometry_uniaxial_box.npz"


def skyrmion_problem():
    return nc.Problem(n=(20, 20, 1), cell=(5e-9,) * 3, Ms=v.MS, pbc=(True, True, False),
                      A=v.A_EX, K=v.K_U, D=v.D_DMI, H=(0, 0, 1e5))


def test_newell_tensor_self_term_and_trace():
    t = nc.newell_tensor((6, 6, 6), (5e-9,) * 3)
    assert np.allclose(t[:3, 0, 0, 0], 1 / 3)                      # celda cúbica
    trace = t[:3].sum(axis=0)
    assert trace[0, 0, 0] == pytest.approx(1)
    assert np.abs(trace.ravel()[1:]).max() < 1e-12                 # traza nula fuera del origen
    t = nc.newell_tensor((3, 3, 3), (2e-9, 5e-9, 9e-9))            # celda no cúbica
    assert t[:3, 0, 0, 0].sum() == pytest.approx(1)
    assert t[0, 0, 0, 0] > t[1, 0, 0, 0] > t[2, 0, 0, 0]


def test_newell_tensor_tends_to_dipole():
    t = nc.newell_tensor((12, 1, 1), (1.0, 1.0, 1.0))
    r = 10.0
    assert t[0, 10, 0, 0] == pytest.approx(-2 / (4 * np.pi * r ** 3), rel=1e-3)
    assert t[1, 10, 0, 0] == pytest.approx(1 / (4 * np.pi * r ** 3), rel=1e-3)


def test_uniform_box_demag_factors_match_aharoni():
    p = nc.Problem(n=(60, 60, 20), cell=(5e-9,) * 3, Ms=9.65e5, demag=True)
    m = np.zeros((60, 60, 20, 3))
    m[..., 2] = 1
    nz = -nc.demag_field(p, m)[..., 2].mean() / p.Ms
    assert nz == pytest.approx(vt.demag_factors(*vt.DIMS)[2], rel=1e-7)


def test_local_fields_match_numpy_replica():
    p = skyrmion_problem()
    rng = np.random.default_rng(3)
    m = rng.standard_normal((20, 20, 3))
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    h = nc.effective_field(p, m[:, :, None, :])[:, :, 0]
    assert np.allclose(h, v.heff(m), rtol=1e-12, atol=1e-6)
    e = v.energies(m)
    names = {"Zeeman": "zeeman", "Anisotropía": "uniaxialanisotropy", "Intercambio": "exchange", "DMI": "dmi"}
    for label, name in names.items():
        assert nc.term_energy(p, m[:, :, None, :], name) == pytest.approx(e[label], rel=1e-10)


@pytest.mark.parametrize("max_angle", [0.05, 0.1, 0.5])
def test_skyrmion_minimization_reaches_symmetric_equilibrium(max_angle):
    p = skyrmion_problem()
    m, info = nc.minimize(p, v.initial()[:, :, None, :], max_angle=max_angle)
    assert info["converged"] and info["max_mxHxm"] < 0.1
    assert v.max_dm(m, v.relax()) < 5e-6
    for label, name in (("Zeeman", "zeeman"), ("Intercambio", "exchange"), ("DMI", "dmi")):
        assert nc.term_energy(p, m, name) == pytest.approx(v.REF[label], rel=2e-6)
    if SKY_NPZ.exists():
        assert v.max_dm(m, np.load(SKY_NPZ)["m_relaxed"]) < 5e-6


@pytest.mark.skipif(not TQ_NPZ.exists(), reason="sin datos numéricos de Ubermag")
def test_torque_matches_ubermag_at_first_angles():
    z = np.load(TQ_NPZ)
    ms, hm = 9.65e5, 2.38e6
    p = nc.Problem(n=(60, 60, 20), cell=(5e-9,) * 3, Ms=ms, A=8.78e-13, K=1.18e6, H=(0, 0, 0), demag=True)
    m = np.zeros((60, 60, 20, 3))
    m[..., 2] = 1
    m, _ = nc.minimize(p, m)
    amp = 0.5 * (z["tx"].max() - z["tx"].min())
    for i in range(6):
        phi = z["angles"][i]
        p.H = (0, hm * np.sin(phi), hm * np.cos(phi))
        m, info = nc.minimize(p, m)
        assert info["converged"]
        tx = nc.MU0 * np.cross(ms * m.reshape(-1, 3).mean(axis=0), p.H)[0]
        assert abs(tx - z["tx"][i]) / amp < 1e-6


def test_ubermag_interface():
    df = pytest.importorskip("discretisedfield")
    mm = pytest.importorskip("micromagneticmodel")
    mesh = df.Mesh(region=df.Region(p1=(-50e-9, -50e-9, 0), p2=(50e-9, 50e-9, 5e-9)), cell=(5e-9,) * 3, bc="xy")
    system = mm.System(name="prueba_numba")
    system.energy = (mm.Exchange(A=v.A_EX) + mm.DMI(D=v.D_DMI, crystalclass="T")
                     + mm.Zeeman(H=(0, 0, 1e5)) + mm.UniaxialAnisotropy(K=v.K_U, u=(0, 0, 1)))
    system.m = df.Field(mesh, nvdim=3, norm=v.MS,
                        value=lambda pt: (0, 0, -1) if pt[0] ** 2 + pt[1] ** 2 < (25e-9) ** 2 else (0, 0, 1))
    p, m = nc.problem_from_system(system)
    assert p.n == (20, 20, 1) and list(p.pbc) == [True, True, False] and p.Ms == pytest.approx(v.MS)
    assert sorted(p.terms) == ["dmi", "exchange", "uniaxialanisotropy", "zeeman"]
    driver = nc.MinDriver()
    driver.drive(system)
    assert driver.info["converged"]
    mu = system.m.orientation
    assert (v.D_DMI * mu.dot(mu.curl)).allclose(nc.compute(system.energy.dmi.density, system))
    assert (2 * v.A_EX / (nc.MU0 * v.MS) * mu.laplace).allclose(nc.compute(system.energy.exchange.effective_field, system))
    assert nc.compute(system.energy.zeeman.energy, system) == pytest.approx(v.REF["Zeeman"], rel=2e-6)
    assert np.allclose(nc.torque(system), 0)                       # <M> paralelo a H
    system.energy += mm.CubicAnisotropy(K=1e3, u1=(1, 0, 0), u2=(0, 1, 0))
    with pytest.raises(NotImplementedError):
        nc.problem_from_system(system)
