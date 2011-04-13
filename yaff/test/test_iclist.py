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
from molmod import bond_length

from yaff import *

from common import get_system_quartz

def test_ic_list_quartz_bonds():
    system = get_system_quartz()
    dlist = DeltaList(system)
    iclist = InternalCoordinateList(dlist)
    for i, j in system.topology.bonds:
        iclist.add_ic(Bond(i, j))
    dlist.forward()
    iclist.forward()
    for row, (i, j) in enumerate(system.topology.bonds):
        delta = system.pos[i] - system.pos[j]
        for c in xrange(len(system.rvecs)):
            delta -= system.rvecs[c]*np.ceil(np.dot(delta, system.gvecs[c]) - 0.5)
        assert abs(iclist.ictab[row]['value'] - bond_length(np.zeros(3, float), delta)[0]) < 1e-5
