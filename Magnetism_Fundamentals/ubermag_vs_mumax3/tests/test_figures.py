"""Las figuras se generan sin GPU ni Ubermag a partir de los datos incluidos (o de datos sintéticos)."""
import csv
import sys

import pytest

from conftest import ROOT

OUT = ROOT / "mumax3" / "skyrmion_energy_terms_mumax3.out"


@pytest.mark.skipif(not (OUT / "m_relaxed.ovf").exists(), reason="sin salidas de MuMax3")
def test_skyrmion_figures(tmp_path, monkeypatch):
    import plot_skyrmion_states as p

    monkeypatch.setattr(sys, "argv", ["plot_skyrmion_states.py", "--save-dir", str(tmp_path)])
    p.main()
    for name in ("fig_skyrmion_ubermag_vs_mumax3.png", "fig_skyrmion_mumax3_default_settings.png"):
        assert (tmp_path / name).stat().st_size > 50000


def test_torque_figure(tmp_path, monkeypatch):
    import validate_torque_macrospin as t

    data = ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"
    if not data.exists():
        pytest.skip("sin salida de MuMax3")
    fig = tmp_path / "torque.png"
    monkeypatch.setattr(sys, "argv", ["validate_torque_macrospin.py", "--fig", str(fig)])
    t.main()
    assert fig.stat().st_size > 30000


def test_benchmark_figure_from_synthetic_csv(tmp_path, monkeypatch):
    import make_benchmark_figure as mbf

    with open(tmp_path / "benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caso", "enfoque", "repeticiones", "mediana_s", "min_s", "max_s", "sesion"])
        for case, (ub, mx) in {"micromagnetic_energy_terms_skyrmion": (88.4, 3.4),
                               "torque_magnetometry_uniaxial_box": (463.7, 1.4)}.items():
            w.writerow([case, mbf.UBERMAG, 3, ub, ub - 0.4, ub + 0.3, "s"])
            w.writerow([case, mbf.MUMAX, 3, mx, mx - 0.02, mx + 0.1, "s"])
    with open(tmp_path / "benchmark_cells.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["caso", "celda", "primera_linea", "mediana_s"])
        for case, spec in mbf.CELLS.items():
            for idx in (k for k in spec if isinstance(k, int)):
                w.writerow([case, idx, "x", 1.0 + spec[idx][2] * 6.2])
    out = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_benchmark_figure.py", "--dir", str(tmp_path), "--out", str(out)])
    mbf.main()
    assert len(list(out.glob("fig_benchmark_*.png"))) == 2


def test_benchmark_figure_without_csv_exits_cleanly(tmp_path, monkeypatch):
    import make_benchmark_figure as mbf

    monkeypatch.setattr(sys, "argv", ["make_benchmark_figure.py", "--dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        mbf.main()


def test_quick_torque_plot(tmp_path, monkeypatch):
    import shutil

    import plot_torque_mumax3 as q

    data = ROOT / "mumax3" / "torque_magnetometry_mumax3.out" / "torque_vs_angle.txt"
    if not data.exists():
        pytest.skip("sin salida de MuMax3")
    shutil.copy(data, tmp_path / "torque_vs_angle.txt")
    monkeypatch.setattr(sys, "argv", ["plot_torque_mumax3.py", str(tmp_path / "torque_vs_angle.txt")])
    q.main()
    assert (tmp_path / "torque_mumax3.png").exists()
