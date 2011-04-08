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

from yaff.topology import Topology


__all__ = ['System']


class System(object):
    def __init__(self, numbers, pos, ffatypes, bonds=None, rvecs=None):
        '''
           *Arguments:*

           numbers
                A numpy array with atomic numbers

           pos
                A numpy array (N,3) with atomic coordinates in bohr.

           ffatypes
                A list of labels of the force field atom types.

           *Optional arguments:*

           bonds
                a numpy array (N, 2) with atom indexes (counting starts from
                zero) to define the chemical bonds.

           rvecs
                An array whose rows are the unit cell vectors. At most three
                rows are allowed, each containg three Cartesian coordinates.
        '''
        self.numbers = numbers
        self.pos = pos
        self.ffatypes = ffatypes
        if bonds is None:
            self.topology = None
        else:
            self.topology = Topology(bonds, self.size)
        self.update_rvecs(rvecs)

    size = property(lambda self: len(self.pos))

    def update_rvecs(self, rvecs):
        if rvecs.size == 0:
            self.rvecs = np.zeros((0,3), float)
            self.gvecs = np.zeros((0,3), float)
            self.rspacings = np.zeros((0,), float)
            self.gspacings = np.zeros((0,), float)
        else:
            self.rvecs = rvecs.reshape((-1,3))
            assert len(self.rvecs) <= 3
            U, S, Vt = np.linalg.svd(rvecs.transpose(), full_matrices=False)
            self.gvecs = np.dot(Vt.transpose(), (U/S).transpose())
            self.rspacings = (self.gvecs**2).sum(axis=1)**(-0.5)
            self.gspacings = (self.rvecs**2).sum(axis=1)**(-0.5)
