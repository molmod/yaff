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


__all__ = ['Scalings']


scaling_dtype = [('a', int), ('b', int), ('scale', float)]


# TODO: include check on periodicity of the topology. In the case of small
# periodic systems, the minimum image convention may not be suitable to decide
# which non-bonding interactions are excluded. To avoid troubles, one can detect
# such cases prior to running the simulation.

class Scalings(object):
    def __init__(self, system, scale1=0.0, scale2=0.0, scale3=1.0):
        self.items = []
        if scale1 < 0 or scale1 > 1:
            raise ValueError('scale1 must be in the range [0,1].')
        if scale2 < 0 or scale2 > 1:
            raise ValueError('scale1 must be in the range [0,1].')
        if scale3 < 0 or scale3 > 1:
            raise ValueError('scale1 must be in the range [0,1].')
        self.scale1 = scale1
        self.scale2 = scale2
        self.scale3 = scale3
        stab = []
        for i0 in xrange(system.natom):
            if scale1 < 1.0:
                for i1 in system.neighs1[i0]:
                    if i0 > i1:
                        stab.append((i0, i1, scale1))
            if scale2 < 1.0:
                for i2 in system.neighs2[i0]:
                    if i0 > i2:
                        stab.append((i0, i2, scale2))
            if scale3 < 1.0:
                for i3 in system.neighs3[i0]:
                    if i0 > i3:
                        stab.append((i0, i3, scale3))
        stab.sort()
        self.stab = np.array(stab, dtype=scaling_dtype)
