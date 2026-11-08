# Computational Physics - Universidad de Antioquia

This repository compiles the computational work and simulations carried out as part of the Physics program at **Universidad de Antioquia**.
It includes Jupyter notebooks, data files, and visualizations covering different courses and academic projects.

---

## Repository Organization

* **Advanced Laboratory I/**
  Spring–magnet–coil project: measurement of frequency and damping, and estimation of the magnetic field using Faraday's law.

* **Computational_Methods/**
  Double pendulum: equations of motion derived from the Lagrangian with SymPy, numerical integration, energy conservation, sensitivity to initial conditions and the resulting predictability horizon, and a real-time animation of the motion.

* **Computational Physics I/**
  Monte Carlo model of copper thin-film growth by thermal evaporation, built one mechanism at a time in eight notebooks that share a common simulation package (`thinfilm_mc`): random deposition, surface relaxation, thermally activated hopping, oblique incidence, geometric shadowing with exact ray tracing, gas-phase scattering, and lateral sticking with porous growth. Includes dynamic scaling and universality classes (Edwards–Wilkinson, Kardar–Parisi–Zhang), the shadowing column instability, mound formation in thick films, porosity maps, a 2<sup>5</sup> factorial design of the mechanisms, and automated tests.

* **Experimental_Physics_IV/**
  Analysis of laboratory data: Boltzmann's constant and the ideality factor from the current-voltage characteristic of a diode, the photoelectric effect for three laser colors and two distances, Poisson counting statistics with a scintillation detector (chi-square and dispersion tests, activity of a sample), and the transmission spectra of lager beers, including the effect of detector saturation.

* **Introduction_to_Nuclear_Physics/**
  Radioactive decay (half-life by least squares and maximum likelihood, binomial fluctuations), a relativistic kinematics calculator with the universal ratio between relativistic and classical kinetic energy, the semi-empirical mass formula with its shell-effect residuals and the stable isobar for A = 131, and the nuclear radius from the Coulomb displacement energies of mirror nuclei.

* **Magnetism Fundamentals/**
  Two projects on computational magnetism:
  * `antiferromagnetic_ising2d_magnetocaloric_effect/`: Monte Carlo study of the frustrated J<sub>1</sub>–J<sub>2</sub> Ising model on the square lattice (ferromagnetic J<sub>1</sub>, antiferromagnetic J<sub>2</sub> = &minus;0.6) and its magnetocaloric effect, built on a simulation package (`isingj1j2`) with Numba-compiled Metropolis kernels, process-level parallelism, reproducible seeds and bootstrap error analysis. Includes the fourfold-degenerate stripe order, finite-size scaling for L = 12–64 (correlation length, Binder cumulants, response maxima) locating the transition near the four-state Potts point, the critical temperature as a function of J<sub>2</sub>/J<sub>1</sub> validated against transfer-matrix results, periodic vs. free boundary conditions, the inverse and conventional branches of &Delta;S<sub>M</sub>, the adiabatic temperature change and relative cooling power, and automated tests.
  * `ubermag_vs_mumax3/`: two Ubermag/OOMMF documentation examples (energy terms of a Bloch skyrmion and torque magnetometry of a box with uniaxial anisotropy), explained for the course with their original computations and reimplemented in MuMax3. Includes a cell-by-cell comparison of magnetization states and energies between both codes, the model settings that make them equivalent (DMI boundary condition, torque without precession, gyromagnetic ratio), an independent NumPy replica and a macrospin model with analytical demagnetizing factors for validation, a stability analysis showing that the circular skyrmion of the example is a saddle point, and wall-clock benchmarks that separate the fixed cost of each OOMMF call from the computation.

* **Probability_and_Statistics/**
  Solved exam exercises in statistical inference and simple linear regression: confidence intervals, t tests, sample size and power with the noncentral t distribution, prediction intervals, and the leverage and Cook's distance of an influential observation.

* **Theoretical_Mechanics_II/**
  Nonlinear dynamics and analytical mechanics: basins of attraction and doubly transient chaos in a magnetic pendulum, the Magnus effect on a spinning football with drag and wind, the relativistic harmonic oscillator validated against its exact period, and electrical circuits in the Lagrangian and Hamiltonian formalisms, including Rayleigh dissipation and the Josephson junction.

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
* Micromagnetics: Ubermag (with OOMMF and `mag2exp`) and MuMax3, which requires an NVIDIA GPU

---

## License

This repository is shared for academic and educational purposes.
The use of the code is permitted for study and reference, provided that the source is properly cited.

---

## Author

**Cristian Camilo Vergara Restrepo**
Physics Student – Universidad de Antioquia
