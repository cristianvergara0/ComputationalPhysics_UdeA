"""Factores desmagnetizantes, macroespín y curva de torque de MuMax3 frente a Ubermag."""
import numpy as np
import pytest

from conftest import ROOT
import validate_torque_macrospin as t

DATA = ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"
NPZ = ROOT / "results" / "ubermag_arrays_torque_magnetometry_uniaxial_box.npz"


def test_demag_factors_cube_and_sum_rule():
    assert np.allclose(t.demag_factors(1, 1, 1), 1 / 3)
    nx, ny, nz = t.demag_factors(*t.DIMS)
    assert nx + ny + nz == pytest.approx(1, abs=1e-10)
    assert nx == pytest.approx(ny) and nz > ny        # caja aplanada en z
    nx, ny, nz = t.demag_factors(2, 3, 5)               # prisma sin simetría
    assert nx + ny + nz == pytest.approx(1, abs=1e-10)
    assert nx > ny > nz


def test_macrospin_amplitude_is_K_without_shape():
    phis = np.linspace(0, 2 * np.pi, 2001)
    tau = t.macrospin_torque(phis)
    assert np.abs(tau).max() == pytest.approx(t.K, rel=2e-3)


@pytest.mark.skipif(not DATA.exists(), reason="sin salida de MuMax3")
def test_mumax3_curve_matches_macrospin_with_shape_anisotropy():
    phi, tx, ty, tz = t.load_mumax3(DATA)
    assert len(phi) == 71
    amp = 0.5 * (tx.max() - tx.min())
    nx, ny, nz = t.demag_factors(*t.DIMS)
    assert np.abs(t.macrospin_torque(phi, nz, ny) - tx).max() / amp < 0.015
    assert np.abs(t.macrospin_torque(phi) - tx).max() / amp > 0.3      # solo K no la describe
    assert np.abs(tx[:35] - tx[35:70]).max() / amp < 1e-4                # periodo pi


@pytest.mark.skipif(not (DATA.exists() and NPZ.exists()), reason="sin datos de los dos códigos")
def test_mumax3_matches_ubermag_numerically():
    phi, tx, _, _ = t.load_mumax3(DATA)
    z = np.load(NPZ)
    assert np.allclose(z["angles"], phi)
    amp = 0.5 * (tx.max() - tx.min())
    assert np.abs(z["tx"] - tx).max() / amp < 1e-4
