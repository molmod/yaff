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

Informal TODO list
##################

General
=======

* Never ending: write more documentation.

* Link with ASE.

* More strict unit checking in the parameter files. Change 'e' to 'electron'.

* Literature reference system, so users know what papers to cite when they run a
  simulation.

* ATSELECT

    - Add support for ``@N`` feature to ATSELECT.
    - Add caching to the ATSELECT compiler.

* System IO stuff:

    - Read system from pdb, hdf5 file.
    - Write system to xyz, pdf file.
    - Optional link to openbabel (if installed, not a mandatory dependency).
    - More compact chk file.

* Add topological analysis to System class.

* See TODO comments in code.


``yaff.pes``
============

* FF Generator

    - Add support for scopes to the Generator classes, through sections in the
      parameter file, or by requiring a different file for every scope.
    - Make an FF for methanol, and a methanol-water system to facilitate the testing.
    - ``scope:ffatype`` and ``scope:number`` combinations in the parameter.

* FF models

    - Gaussian charge distributions.
    - ACKS2.
    - Dielectric background for fixed charge models.

* Parallel force evaluation.

* Convert from C to C++.

* Smooth truncation of electrostatic interactions in reciprocal space to obtain
  a perfectly differentiable PES.

* Cell lists.

* NlogN scaling of the electrostatics.

* Correctly treat the periodic boundary conditions in very skewed cells.
  The current implementation of the minimum image convention is, just like in
  most MD codes, rather naive. Check the answers on the scicomp faq.

* Allow one to override the default names of the ForcePart* instances, e.g.
  to allow two different valence parts.



``yaff.sampling``
=================

* Restart functionality.

* Constraints and restraints.

* An optional select argument for the iterative algorithms, e.g. to easily freeze
  the remaining atoms during an MD or optimization. Similarly, one should also
  be able to construct a partial force field model.

* RefTraj derivative of the Iterative class.

* Parallel sampling methods.



``yaff.analysis``
=================

* Port more things from MD-Tracks, including the conversion stuff.


``yaff.tune``
=============

* Stabilize parameter optimizer.
