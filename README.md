# Computational Physics - Universidad de Antioquia

This repository compiles the computational work and simulations carried out as part of the Physics program at **Universidad de Antioquia**.
It includes Jupyter notebooks, data files, and visualizations covering different courses and academic projects.

---

## Repository Organization

* **Advanced Laboratory I/**
  Spring–magnet–coil project: measurement of frequency and damping, and estimation of the magnetic field using Faraday's law.

* **Computational Methods/**
  Numerical methods and simulations, including the double pendulum.

* **Computational Physics I/**
  Monte Carlo model of copper thin-film growth by thermal evaporation, built one mechanism at a time in eight notebooks that share a common simulation package (`thinfilm_mc`): random deposition, surface relaxation, thermally activated hopping, oblique incidence, geometric shadowing with exact ray tracing, gas-phase scattering, and lateral sticking with porous growth. Includes dynamic scaling and universality classes (Edwards–Wilkinson, Kardar–Parisi–Zhang), the shadowing column instability, mound formation in thick films, porosity maps, a 2<sup>5</sup> factorial design of the mechanisms, and automated tests.

* **Experimental Physics IV/**
  Experimental projects with computational modeling (e.g., Boltzmann's constant, absorption spectrum of lager-type beers, and the photoelectric effect).

* **Introduction to Nuclear Physics/**
  Models and calculations in nuclear physics (e.g., radioactive decay, relativistic kinematics, the semi-empirical mass formula, and the Coulomb radius from mirror nuclei).

* **Magnetism Fundamentals/**
  Monte Carlo study of the frustrated J<sub>1</sub>–J<sub>2</sub> Ising model on the square lattice (ferromagnetic J<sub>1</sub>, antiferromagnetic J<sub>2</sub> = &minus;0.6) and its magnetocaloric effect, built on a simulation package (`isingj1j2`) with Numba-compiled Metropolis kernels, process-level parallelism, reproducible seeds and bootstrap error analysis. Includes the fourfold-degenerate stripe order, finite-size scaling for L = 12–64 (correlation length, Binder cumulants, response maxima) locating the transition near the four-state Potts point, the critical temperature as a function of J<sub>2</sub>/J<sub>1</sub> validated against transfer-matrix results, periodic vs. free boundary conditions, the inverse and conventional branches of &Delta;S<sub>M</sub>, the adiabatic temperature change and relative cooling power, and automated tests.

* **Probability and Statistics/**
  Notebooks on statistical inference, probability distributions, and regression.

* **Theoretical Mechanics II/**
  Advanced simulations in theoretical mechanics (e.g., magnetic pendulum, Magnus effect, relativistic harmonic oscillator, and Lagrangian/Hamiltonian analysis of electrical circuits).

---

## Objective

* Document and organize the computational work carried out at the university.
* Serve as an academic portfolio in computational physics and data science.
* Facilitate the reuse and improvement of code and analyses.

---

## Requirements

* Python 3.x
* Jupyter Notebook
* Scientific libraries: `numpy`, `scipy`, `matplotlib`, `numba`, `pandas`, `seaborn`, `scikit-learn`, `sympy`, `Pillow`, `tabulate`, `jinja2`, `pytest`

---

## License

This repository is shared for academic and educational purposes.
The use of the code is permitted for study and reference, provided that the source is properly cited.

---

## Author

**Cristian Camilo Vergara Restrepo**
Physics Student – Universidad de Antioquia
