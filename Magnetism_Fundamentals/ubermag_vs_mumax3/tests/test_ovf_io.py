"""Lectura y escritura de OVF 2.0: orden de los índices, formatos y salidas reales de MuMax3."""
import numpy as np
import pytest

from conftest import ROOT
from ovf_io import read_ovf, write_ovf

OUT = ROOT / "mumax3" / "skyrmion_energy_terms_mumax3.out"


@pytest.mark.parametrize("fmt", ["binary 4", "binary 8", "text"])
def test_roundtrip_keeps_index_order(tmp_path, fmt):
    nx, ny, nz = 4, 3, 2
    f = np.zeros((nx, ny, nz, 3))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                f[i, j, k] = (i, 10 * j, 100 * k)        # cada componente identifica un eje
    p = tmp_path / "campo.ovf"
    write_ovf(p, f, fmt=fmt)
    g = read_ovf(p)
    assert g.shape == (nx, ny, nz, 3)
    assert np.allclose(g, f)


def test_bad_control_value_is_rejected(tmp_path):
    p = tmp_path / "malo.ovf"
    write_ovf(p, np.ones((2, 2, 1, 3)), fmt="binary 4")
    raw = bytearray(p.read_bytes())
    i = raw.index(b"# Begin: Data binary 4\n") + len(b"# Begin: Data binary 4\n")
    raw[i:i + 4] = np.array([1.0], "<f4").tobytes()
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        read_ovf(p)


@pytest.mark.skipif(not (OUT / "m_relaxed.ovf").exists(), reason="sin salidas de MuMax3")
def test_mumax3_outputs_are_unit_fields():
    for name in ("m_initial", "m_relaxed", "m_euler", "m_euler_dt02ps"):
        m = read_ovf(OUT / f"{name}.ovf")
        assert m.shape == (20, 20, 1, 3)
        assert np.allclose(np.linalg.norm(m, axis=-1), 1, atol=1e-6)


@pytest.mark.skipif(not (OUT / "m_initial.ovf").exists(), reason="sin salidas de MuMax3")
def test_mumax3_initial_state_is_the_notebook_disk():
    import validate_skyrmion_numpy as v

    m = read_ovf(OUT / "m_initial.ovf")[:, :, 0, :]
    assert np.array_equal(m[..., 2] < 0, v.INSIDE)    # mismo disco de radio L/4 y misma orientación de ejes
