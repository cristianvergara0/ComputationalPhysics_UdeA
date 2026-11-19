"""Análisis experimental de la curva de torque y ejecución de notebooks."""
import sys

import numpy as np
import pytest

from conftest import ROOT
import torque_curve_analysis as tca
import validate_torque_macrospin as vt


def test_shear_correction_recovers_k_of_a_macrospin():
    # Un macroespín sin forma con H = 1.22 H_K: la curva frente a phi está inclinada, la corregida es -K sin 2 theta.
    phi = np.linspace(0, 2 * np.pi, 141)
    tau = vt.macrospin_torque(phi)
    r = tca.analyze(phi, tau)
    assert r["k_ef"] == pytest.approx(vt.K, rel=1e-6)
    assert r["rms"] < 1e-6
    assert abs(r["b2"]) < 0.97 * vt.K and abs(r["b4"]) > 0.1 * vt.K      # sin corregir: amplitud baja y armónico falso


@pytest.mark.skipif(not tca.UBERMAG.exists(), reason="sin datos numéricos de Ubermag")
def test_simulated_curve_gives_back_the_input_k():
    r = tca.analyze(*tca.load_curve("ubermag"))
    assert r["k_ef"] == pytest.approx(9.52e5, rel=2e-3)
    assert r["rms"] < 2e-3
    assert r["k_rec"] == pytest.approx(tca.K_IN, rel=5e-3)


def test_analysis_figure(tmp_path, monkeypatch):
    if not tca.UBERMAG.exists():
        pytest.skip("sin datos numéricos de Ubermag")
    fig = tmp_path / "a.png"
    monkeypatch.setattr(sys, "argv", ["torque_curve_analysis.py", "--fig", str(fig)])
    tca.main()
    assert fig.stat().st_size > 30000


def test_run_notebook_executes_in_project_folder(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    import run_notebook

    out = tmp_path / "salida.txt"
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_code_cell(
        f"from pathlib import Path\nPath(r'{out}').write_text(str(Path.cwd()))\nprint('hecho')"))
    path = tmp_path / "prueba.ipynb"
    nbformat.write(nb, path)
    monkeypatch.setattr(sys, "argv", ["run_notebook.py", str(path)])
    run_notebook.main()
    assert out.read_text() == str(ROOT)
    done = nbformat.read(path, as_version=4)
    assert "hecho" in done.cells[0].outputs[0]["text"]
