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


__all__ = ['Scaling']


class Scaling(object):
    def __init__(self, topology, scale1=0.0, scale2=0.0, scale3=1.0):
        self.items = []
        for i0 in xrange(topology.natom):
            slist = []
            if scale1 < 1.0:
                for i1 in topology.neighs1[i0]:
                    slist.append((i1, scale1))
            if scale2 < 1.0:
                for i2 in  topology.neighs2[i0]:
                    slist.append((i2, scale2))
            if scale3 < 1.0:
                for i3 in topology.neighs3[i0]:
                    slist.append((i3, scale3))
            slist.sort()
            self.items.append(np.array(slist, dtype=[('i', int), ('scale', float)]))

    def __getitem__(self, index):
        return self.items[index]
