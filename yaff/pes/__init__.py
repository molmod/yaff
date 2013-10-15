# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
#--
"""Force-field potential energy surfaces (PESs)

   This packages provides the machinery to evaluate to force-field energy and
   its derivatives, like Cartesian gradient (=minus forces) and virial tensor,
   for a given set of atomic coordinates and 0, 1, 2 or 3 cell vectors.

   The priorities of the ``yaff.pes`` module are:

   * Compact, clean and well-tested code.
   * Thin interface. Coordinates and cell vectors are input. Energy and
     optionally some derivatives are output.
   * Easy to add new (types of) force field terms.
   * Handy parameter file format.
   * Computation of energy derivatives is optional.
   * Evaluation of derivatives based on back-propagation algorithm.
   * As much Python as possible without becoming ridiculously slow. Where
     needed, low-level routines are implemented in C.
   * Hide all details of the force-field evaluation inside the
     ForceField.compute method.
   * Allow detailed inspection of different contributions to FF energy.

   The implementation of ``yaff.pes`` is reasonably efficient when taking into
   account the following considerations.

   * Subsequent calls to the ForceField.compute method are done with atom
     coordinates that are displaced a bit (compared to the Verlet skin).

   * Avoid that just one atom moves between two compute calls. The entire energy
     is recomputed regardless of whether all or just a few atoms are displaced.
     Hence, Yaff is not (yet) suitable for naive Monte Carlo simulations.
     However, molecular Dynamics simulations, or Monte Carlo simulations that
     move all atoms at the same time based on gradient data, are suitable
     algorithms for Yaff.

   Working under these assumptions is quite common in simulations codes (e.g.
   ASE, CP2K, ...) and has typically the following benefits:

   * Several complex aspects of the force field evaluation can be implemented
     orthogonally to the sampling module. The sampling algorithms do not have to
     be aware of neighbor lists, scaling of short-range non-bonding
     interactions, atom types, Ewald summation, parallel force evaluation (not
     supported yet in Yaff) ...

   * The interface between the ``yaff.pes`` and ``yaff.sampling`` is very
     small: input = coordinates and cell vectors, output = energy and its
     derivatives.

   The orthogonality between ``yaff.pes`` and ``yaff.sampling`` also has
   some disadvantages that advanced FF codes (OpenKIM, dedicated MC codes,
   ...) can surmount:

   * Global optimizations that can only work for certain FF/sampling
     combinations, are not supported.
"""

from yaff.pes.ext import *
from yaff.pes.dlist import *
from yaff.pes.iclist import *
from yaff.pes.vlist import *
from yaff.pes.generator import *
from yaff.pes.ff import *
from yaff.pes.nlist import *
from yaff.pes.parameters import *
from yaff.pes.scaling import *
