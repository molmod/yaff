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
"""Internal-coordinate lists for covalent energy terms

   An ``InternalCoordinateList`` object contains a table, where each row
   corresponds to one internal coordinate. This object also contains a
   :class:`yaff.pes.dlist.DeltaList` object that holds all the input relative
   vectors for the internal coordinates.

   Each row in the table contains all the information to compute the internal
   coordinate with the ``forward`` method. Each row can also hold the derivative
   of the energy towards the internal coordinate (computed elsewhere), in order
   to transform this derivative to derivatives of the energy towards the
   relative vectors in the ``DeltaList`` object. (See ``back`` method.)

   Furthermore, a series of ``InternalCoordinate`` classes are defined in this
   module to facilitate the setup of the table in the ``InternalCoordinateList``
   object. An instance of a subclass of ``InternalCoordinate`` can be passed
   to the ``add_ic`` method to register a new internal coordinate in the table.
   The ``add_ic`` method returns the row index of the internal new coordinate.
   If the internal coordinate is already present, no new row is added and the
   index of the existing row is returned. When a new internal coordinate is
   registered, the required relative vectors are registered automatically in the
   ``DeltaList`` object.

   The class :class:`yaff.pes.iclist.InternalCoordinateList` is intimately
   related to classes :class:`yaff.pes.dlist.DeltaList` and
   :class:`yaff.pes.vlist.ValenceList`. They work together, just like layers in
   a neural network, and they use the back-propagation algorithm to compute
   partial derivatives. The order of the layers is as follows::

       DeltaList <--> InternalCoordinateList <--> ValenceList

   The class :class:`yaff.pes.ff.ForcePartValence` ties these three lists
   together. The basic idea of the back-propagation algorithm is explained in
   the section :ref:`dg_sec_backprop`.
"""


import numpy as np

from yaff.log import log
from yaff.pes.ext import iclist_forward, iclist_back


__all__ = [
    'InternalCoordinateList', 'InternalCoordinate', 'Bond', 'BendAngle',
    'BendCos', 'DihedAngle', 'DihedCos', 'UreyBradley', 'OopAngle',
    'OopMeanAngle', 'OopCos', 'OopMeanCos', 'OopDist',
]


iclist_dtype = [
    ('kind', int),                      # Numerical code for the type of internal coordinate, e.g. bond, angle, ...
    ('i0', int), ('sign0', int),        # row index and sign flip of first relative vector in ``DeltaList`` object
    ('i1', int), ('sign1', int),        # row index and sign flip of second ...
    ('i2', int), ('sign2', int),        # ...
    ('i3', int), ('sign3', int),        # ...
    ('value', float), ('grad', float)   # value = value of internal coordinate computed here
                                        # grad = derivative of energy towards internal coordinate, stored here by another part of the code
]


class InternalCoordinateList(object):
    """Contains a table of all internal coordinates used in a covalent force
       field. All computations related to internal coordinates are carried out
       in coordination with a ``DeltaList`` object.
    """
    def __init__(self, dlist):
        """
           **Arugments:**

           dlist
                An instance of the ``DeltaList`` class.
        """
        self.dlist = dlist
        self.ictab = np.zeros(10, iclist_dtype)
        self.lookup = {}
        self.nic = 0

    def add_ic(self, ic):
        '''Register a new or find an existing internal coordinate.

           **Arugments:**

           ic
                An instance of a subclass of the ``InternalCoordinate`` class.

           This method returns the row of the new/existing internal coordinate.
        '''
        # First check whether this ic is already in the table. The
        # get_rows_signs method also registers new relative vectors in the delta
        # list if needed.
        rows_signs = ic.get_rows_signs(self.dlist)
        key = (ic.kind,) + sum(rows_signs, ())
        row = self.lookup.get(key)
        if row is None:
            # No existing ic was found. Add a new ic to table
            if self.nic >= len(self.ictab):
                self.ictab = np.resize(self.ictab, int(len(self.ictab)*1.5))
            row = self.nic
            self.ictab[row]['kind'] = ic.kind
            for i in xrange(len(rows_signs)):
                self.ictab[row]['i%i'%i] = rows_signs[i][0]
                self.ictab[row]['sign%i'%i] = rows_signs[i][1]
            self.lookup[key] = row
            self.nic += 1
        return row

    def forward(self):
        """Compute the internal coordinates based on the relative vectors in
           ``self.dlist``. The result is stored in the table, ``self.ictab``.

           The actual computation is carried out by a low-level C routine.
        """
        iclist_forward(self.dlist.deltas, self.ictab, self.nic)

    def back(self):
        """Transform the derivative of the energy (in ``self.ictab``) to
           derivatives of the energy towards the components of the relative
           vectors in ``self.dlist``.

           The actual computation is carried out by a low-level C routine.
        """
        iclist_back(self.dlist.deltas, self.ictab, self.nic)


class InternalCoordinate(object):
    """Base class for the internal coordinate 'descriptors'.

       The subclasses are merely used to request a new/existing internal
       coordinate in the ``InternalCoordinateList`` class. These classes do
       not carry out any computations.

       The ``kind`` class attribute refers to an integer ID that identifies the
       internal coordinate kind (bond, angle, ...) in the low-level C code.

       Although all of the internal coordinates below are typically associated
       with certain topological patterns, one is free to add internal
       coordinates that have no direct relation with the molecular topology,
       e.g. to define restraints that pull a system over a reaction barrier.
    """

    kind = None
    def __init__(self, index_pairs):
        '''
           **Arguments:**

           index_pairs
                A list of pairs of atom indexes. Each pair corresponds to
                a relative vector used for the computation of the internal
                coordinate.
        '''
        self.index_pairs = index_pairs

    def get_rows_signs(self, dlist):
        '''Request row indexes and sign flips from a delta list

           **Arguments:**

           dlist
                A ``DeltaList`` instance.
        '''
        return [dlist.add_delta(i, j) for i, j in self.index_pairs]

    def get_conversion(self):
        '''Auxiliary routine that allows base classes the specify the unit
           conversion associated with the internal coordinate.
        '''
        raise NotImplementedError

    def get_log(self):
        '''Describe the internal coordinate in a format that is suitable for
           screen logging.
        '''
        return '%s(%s)' % (
            self.__class__.__name__,
            ','.join('%i-%i' % pair for pair in self.index_pairs)
        )


class Bond(InternalCoordinate):
    '''Bond length.'''
    kind = 0
    def __init__(self, i, j):
        '''
           **Arguments:**

           i, j
                The indexes of the atoms involved in the covalent bond.
        '''
        InternalCoordinate.__init__(self, [(i, j)])

    def get_conversion(self):
        return log.length.conversion


class BendCos(InternalCoordinate):
    '''Cosine of a bending (or valence) angle.'''
    kind = 1
    def __init__(self, i, j, k):
        '''
           **Arguments:**

           i, j, k
                The indexes of the atoms involved in the angle. (i-j-k)
        '''
        InternalCoordinate.__init__(self, [(j, i), (j, k)])

    def get_conversion(self):
        return 1.0


class BendAngle(InternalCoordinate):
    '''Bending (or valence) angle.'''
    kind = 2
    def __init__(self, i, j, k):
        '''
           **Arguments:**

           i, j, k
                The indexes of the atoms involved in the angle. (i-j-k)
        '''
        InternalCoordinate.__init__(self, [(j, i), (j, k)])

    def get_conversion(self):
        return log.angle.conversion


class DihedCos(InternalCoordinate):
    '''Cosine of a dihedral (or torsion) angle.'''
    kind = 3
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the dihedral angle. (i-j-k-l)
        '''
        InternalCoordinate.__init__(self, [(j,i), (j,k), (k,l)])

    def get_conversion(self):
        return 1.0


class DihedAngle(InternalCoordinate):
    '''A dihedral (or torsion) angle.'''
    kind = 4
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the dihedral angle. (i-j-k-l)
        '''
        InternalCoordinate.__init__(self, [(j,i), (j,k), (k,l)])

    def get_conversion(self):
        return log.angle.conversion


class UreyBradley(InternalCoordinate):
    '''A Urey-Bradley distance, i.e. the distance over a bending angle'''
    kind = 5
    def __init__(self, i, j, k):
        '''
           **Arguments:**

           i, j, k
                The indexes of the atoms involved in the angle. (i-j-k)
        '''
        InternalCoordinate.__init__(self, [(i, k)])

    def get_conversion(self):
        return log.length.conversion


class OopCos(InternalCoordinate):
    '''Cosine of an out-of-plane angle.'''
    kind = 6
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane angle.
                The central atom is given by the last index (l). This IC gives
                the angle between the plane formed by atoms i, j and l and the
                bond between l and k.
        '''
        InternalCoordinate.__init__(self, [(i,l), (j,l), (k,l)])

    def get_conversion(self):
        return 1.0

class OopMeanCos(InternalCoordinate):
    '''Mean of cosines of all 3 out-of-plane angles in a oop pattern.'''
    kind = 7
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane angle.
                The central atom is given by the last index (l). This IC gives
                the angle between the plane formed by atoms i, j and l and the
                bond between l and k.
        '''
        InternalCoordinate.__init__(self, [(i,l), (j,l), (k,l)])

    def get_conversion(self):
        return 1.0


class OopAngle(InternalCoordinate):
    '''An out-of-plane angle.'''
    kind = 8
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane angle.
                The central atom is given by the last index (l). This IC gives
                the angle between the plane formed by atoms i, j and l and the
                bond between l and k.
        '''
        InternalCoordinate.__init__(self, [(i,l), (j,l), (k,l)])

    def get_conversion(self):
        return log.angle.conversion

class OopMeanAngle(InternalCoordinate):
    '''Mean of all 3 out-of-plane angles in an oop pattern.'''
    kind = 9
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane angle.
                The central atom is given by the last index (l). This IC gives
                the angle between the plane formed by atoms i, j and l and the
                bond between l and k.
        '''
        InternalCoordinate.__init__(self, [(i,l), (j,l), (k,l)])

    def get_conversion(self):
        return log.angle.conversion

class OopDist(InternalCoordinate):
    '''Distance from an atom to the plane formed by three other atoms'''
    kind = 10
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane distance.
                The central atom is given by the last index (l). The plane is
                formed by the other three atoms i,j and k.
        '''
        InternalCoordinate.__init__(self, [(i,j), (j,k), (k,l)])

    def get_conversion(self):
        return log.length.conversion

class SqOopDist(InternalCoordinate):
    '''Squared distance from an atom to the plane formed by three other atoms'''
    kind = 11
    def __init__(self, i, j, k, l):
        '''
           **Arguments:**

           i, j, k, l
                The indexes of the atoms involved in the out-of-plane distance.
                The central atom is given by the last index (l). The plane is
                formed by the other three atoms i,j and k.
        '''
        InternalCoordinate.__init__(self, [(i,j), (j,k), (k,l)])

    def get_conversion(self):
        return log.length.conversion
