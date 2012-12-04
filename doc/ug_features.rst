Features
========

The main features of Yaff are:

* Error-free implementations of the force-field energy (and forces).
* Full control over the convergence of long-range interactions.
* Readable code.
* Flexible definition of the force field through an extensible parameter file
  format.
* Control of the program through a Python scripting interface.
* Efficient evaluation of the energy, forces and virial in low-level C routines.
* Robust geometry and cell optimization
* Molecular dynamics (NVE, NVT and annealing).
* Elastic constants (0K) and Hessians by taking finite differences of analytical
  first-order derivatives.
* HDF5-based trajectory format.
* Integrated atom-typing language.
* Integrated trajectory analysis.
* Integrated force-field parameter tuning.
* Extensively tested code through a unit-testing framework.


Missing features (work in progress) include:

* Parallel force evaluaton.
* Parallel sampling schemes.
* Constant-pressure sampling schemes: Martyna-Tuckerman-Tobias-Klein.
