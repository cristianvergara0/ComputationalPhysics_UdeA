"""Funciones del benchmark de tamaño y una ejecución completa con un MuMax3 simulado."""
import os
import stat
import sys
import textwrap

import numpy as np
import pytest

import benchmark_scaling as bs


def test_grid_keeps_notebook_proportions():
    assert bs.grid(60) == (60, 60, 20)
    assert bs.n_cells(60) == 72000
    assert bs.grid(2) == (2, 2, 1)


def test_fit_recovers_line():
    n = np.array([100, 1000, 5000, 20000, 72000])
    t0, c, e0, ec = bs.fit_linear(n, 6.2 + 3e-6 * n)
    assert t0 == pytest.approx(6.2) and c == pytest.approx(3e-6)
    assert e0 < 1e-9 and ec < 1e-12


def test_mx3_script_contents():
    s = bs.mx3_script(24, True)
    assert "SetGridSize(24, 24, 8)" in s and "Minimize()" in s and "MaxTorque" in s
    s = bs.mx3_script(24, False)
    assert "Minimize()" not in s and "E_total.Get()" in s


def test_parse_mz_ignores_command_echo():
    out = 'print("<m> tras Minimize:", m.Comp(0).Average())\n//<m> tras Minimize: 0.0 0.61 0.79\n'
    assert bs.parse_mz(out) == pytest.approx(0.79)


@pytest.fixture
def fake_mumax(tmp_path):
    """Un ejecutable 'mumax3' que imita la salida de MuMax3 (para probar la orquestación)."""
    script = tmp_path / "fake_mumax3.py"
    script.write_text(textwrap.dedent('''
        import sys, time
        from pathlib import Path
        src = Path(sys.argv[-1])
        text = src.read_text()
        out = src.with_suffix(".out"); out.mkdir(exist_ok=True)
        log = [line for line in text.splitlines()]
        if "Minimize()" in text:
            time.sleep(0.05)
            log.append("//<m> tras Minimize: 0.0 0.61 0.79")
        (out / "log.txt").write_text("\\n".join(log))
        print("//mumax 3.12 (simulado)")
    '''), encoding="utf-8")
    if os.name == "nt":
        exe = tmp_path / "mumax3.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        exe = tmp_path / "mumax3"
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_full_run_with_fake_mumax(tmp_path, fake_mumax, monkeypatch):
    res, figs = tmp_path / "results", tmp_path / "figures"
    monkeypatch.setattr(sys, "argv", ["benchmark_scaling.py", "--skip-ubermag", "--sizes", "6", "12",
                                      "--repeats", "2", "--mumax3", fake_mumax,
                                      "--results", str(res), "--figures", str(figs)])
    bs.main()
    rows = bs.read_rows(res / "benchmark_scaling.csv")
    labels = {r["codigo"] for r in rows}
    assert labels == {bs.MX_FULL, bs.MX_SETUP, bs.MX_MIN}
    full = bs.series(rows, bs.MX_FULL)
    assert [r["celdas"] for r in full] == [72, 576]
    assert float(full[0]["mz_final"]) == pytest.approx(0.79)
    assert all(r["mediana_s"] > 0.03 for r in bs.series(rows, bs.MX_MIN))
    assert (figs / "fig_benchmark_scaling.png").stat().st_size > 10000
    assert "Ajuste" in (res / "benchmark_scaling.md").read_text(encoding="utf-8")

    # --plot-only reutiliza el CSV y una nueva medición de un código conserva las filas del otro
    ub = {"codigo": bs.UB, "n": 6, "celdas": 72, "repeticiones": 2, "mediana_s": 6.1, "min_s": 6.0,
          "max_s": 6.3, "mz_final": 0.8, "sesion": "x"}
    bs.write_rows(res / "benchmark_scaling.csv", rows + [ub])
    monkeypatch.setattr(sys, "argv", ["benchmark_scaling.py", "--skip-ubermag", "--sizes", "6", "--repeats", "1",
                                      "--mumax3", fake_mumax, "--results", str(res), "--figures", str(figs)])
    bs.main()
    rows = bs.read_rows(res / "benchmark_scaling.csv")
    assert len(bs.series(rows, bs.UB)) == 1 and len(bs.series(rows, bs.MX_FULL)) == 1
    monkeypatch.setattr(sys, "argv", ["benchmark_scaling.py", "--plot-only", "--results", str(res), "--figures", str(figs)])
    bs.main()
