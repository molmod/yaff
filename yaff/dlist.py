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

from yaff import dlist_forward


__all__ = ['DeltaList']


delta_dtype = [('dx', float), ('dy', float), ('dz', float), ('i', int), ('j', int)]


class DeltaList(object):
    def __init__(self, system):
        self.system = system
        self.deltas = np.zeros(10, delta_dtype)
        self.lookup = {}
        self.ndelta = 0

    def add_delta(self, i, j):
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
                self.ndelta += 1
            else:
                sign = -1
        else:
            sign = 1
        return row, sign

    def forward(self):
        dlist_forward(self.system.pos, self.system.rvecs, self.system.gvecs, self.deltas)

    def back(self, gradient):
        raise NotImplementedError
