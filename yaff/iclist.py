# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
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
# --


import numpy as np

from yaff.ext import iclist_forward, iclist_back


__all__ = [
    'InternalCoordinateList', 'InternalCoordinate', 'Bond', 'BendAngle',
    'BendCos','DihedAngle','DihedCos','UreyBradley',
]

iclist_dtype = [
    ('kind', int),
    ('i0', int), ('sign0', int),
    ('i1', int), ('sign1', int),
    ('i2', int), ('sign2', int),
    ('i3', int), ('sign3', int),
    ('value', float), ('grad', float)
]

class InternalCoordinateList(object):
    def __init__(self, dlist):
        self.dlist = dlist
        self.ictab = np.zeros(10, iclist_dtype)
        self.lookup = {}
        self.nic = 0

    def add_ic(self, ic):
        # First add the deltas
        rows_signs = ic.get_rows_signs(self.dlist)
        key = (ic.kind,) + sum(rows_signs, ())
        row = self.lookup.get(key)
        if row is None:
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
        iclist_forward(self.dlist.deltas, self.ictab, self.nic)

    def back(self):
        iclist_back(self.dlist.deltas, self.ictab, self.nic)


class InternalCoordinate(object):
    kind = None
    def __init__(self, index_pairs):
        self.index_pairs = index_pairs

    def get_rows_signs(self, dlist):
        return [dlist.add_delta(i, j) for i, j in self.index_pairs]


class Bond(InternalCoordinate):
    kind = 0
    def __init__(self, i, j):
        InternalCoordinate.__init__(self, [(i, j)])


class BendCos(InternalCoordinate):
    kind = 1
    def __init__(self, i, j, k):
        InternalCoordinate.__init__(self, [(j, i), (j, k)])


class BendAngle(InternalCoordinate):
    kind = 2
    def __init__(self, i, j, k):
        InternalCoordinate.__init__(self, [(j, i), (j, k)])

class DihedCos(InternalCoordinate):
    kind = 3
    def __init__(self, i, j, k, l):
        InternalCoordinate.__init__(self, [(j,i), (j,k), (k,l)])

class DihedAngle(InternalCoordinate):
    kind = 4
    def __init__(self, i, j, k, l):
        InternalCoordinate.__init__(self, [(j,i), (j,k), (k,l)])

class UreyBradley(InternalCoordinate):
    kind = 5
    def __init__(self, i, j, k):
        InternalCoordinate.__init__(self, [(i, k)])
