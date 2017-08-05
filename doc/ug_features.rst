..
    : YAFF is yet another force-field code.
    : Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
    : Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
    : (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
    : stated.
    :
    : This file is part of YAFF.
    :
    : YAFF is free software; you can redistribute it and/or
    : modify it under the terms of the GNU General Public License
    : as published by the Free Software Foundation; either version 3
    : of the License, or (at your option) any later version.
    :
    : YAFF is distributed in the hope that it will be useful,
    : but WITHOUT ANY WARRANTY; without even the implied warranty of
    : MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    : GNU General Public License for more details.
    :
    : You should have received a copy of the GNU General Public License
    : along with this program; if not, see <http://www.gnu.org/licenses/>
    :
    : --

Features and release notes
==========================

Features
--------

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


Release notes
-------------

* **Version 1.3.0** August 5, 2017

  - Python 3 support

* Version 1.2.0

  - Installable with pip.
  - Installable with conda.
  - Automatic deployment of new releases.
  - Fix parallel unit testing issues.

* Version 1.1.3

  - Correct anharmonic covalent MM3 terms.

* Version 1.1.2

  - Removed flush option from HDF5Writer.

* Version 1.1.1

  - Fixed a few bugs in `System.iter_matches`.
