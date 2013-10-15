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
"""Short-range neighbor lists for covalent energy terms

   The short-range neighbor lits are called Delta lists. They are used for the
   covalent energy terms that do not allow for bond breaking.

   The delta list contains all relative vectors that are needed to evaluate
   the covalent energy terms. The minimum image convention (MIC) is used to make
   sure that periodic boundary conditions are taken into account. The current
   implementation of the MIC in Yaff works in principle only for orthorhombic
   cells. In the general case of a triclinic cell, the Yaff implementation is
   known to fail in some corner cases, e.g. in small and very skewed unit cells.
   The derivative of the energy towards the components of the relative vectors
   is computed if the ForceField.compute routine requires energy derivatives.

   The class :class:`yaff.pes.dlist.DeltaList` is intimately related
   to classes :class:`yaff.pes.iclist.InternalCoordinateList` and
   :class:`yaff.pes.vlist.ValenceList`. They work together, just like layers in
   a neural network, and they use the back-propagation algorithm to compute
   partial derivatives. The order of the layers is as follows::

       DeltaList <--> InternalCoordinateList <--> ValenceList

   The class :class:`yaff.pes.ff.ForcePartValence` ties these three lists
   together. The basic idea of the back-propagation algorithm is explained in
   the section :ref:`dg_sec_backprop`.
"""


import numpy as np

from yaff.pes.ext import dlist_forward, dlist_back


__all__ = ['DeltaList']


delta_dtype = [
    ('dx', float), ('dy', float), ('dz', float), # relative vector coordinates.
    ('i', int), ('j', int),                      # involved atoms. vector points from i to j.
    ('gx', float), ('gy', float), ('gz', float), # derivative of energy towards relative vector coordinates.
]


class DeltaList(object):
    """Class to store, manage and evaluate the delta list."""

    def __init__(self, system):
        """
            **Arguments:**

            system
                    A ``System`` instance.

        """
        self.system = system
        self.deltas = np.zeros(10, delta_dtype)
        self.lookup = {}
        self.ndelta = 0

    def add_delta(self, i, j):
        """Register a new relative vector in the delta list

           **Arguments:**

           i, j
                Indexes of the first and second atom. The vector points from
                i to j.

           **Returns:**

           row
                The row index of the newly registered relative vector, for later
                reference.

           sign
                Is -1 when i and j were swapped during the registration. Is +1
                otherwise.
        """
        assert i != j
        assert i >= 0
        assert j >= 0
        assert i < self.system.natom
        assert j < self.system.natom
        row = self.lookup.get((i, j))
        if row is None:
            row = self.lookup.get((j, i))
            if row is None:
                sign = 1
                row = self.ndelta
                if self.ndelta >= len(self.deltas):
                    self.deltas = np.resize(self.deltas, int(len(self.deltas)*1.5))
                self.deltas[row]['i'] = i
                self.deltas[row]['j'] = j
                self.lookup[(i, j)] = row
                self.ndelta += 1
            else:
                sign = -1
        else:
            sign = 1
        return row, sign

    def forward(self):
        """Evaluate the relative vectors for ``self.system.pos``

           The actual computation is carried out by a low-level C routine.
        """
        dlist_forward(self.system.pos, self.system.cell, self.deltas, self.ndelta)

    def back(self, gpos, vtens):
        """Derive gpos and virial from the derivatives towards the relative vectors

           The actual computation is carried out by a low-level C routine.
        """
        dlist_back(gpos, vtens, self.deltas, self.ndelta)
