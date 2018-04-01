# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
# --


from __future__ import division

import numpy as np

from yaff import COMList, ForcePartValence, Harmonic, Bond

from yaff.test.common import get_system_quartz, get_system_glycine
from yaff.pes.test.common import check_gpos_part, check_vtens_part


def test_glycine():
    system = get_system_glycine()
    groups1 = [
        (np.array([0, 2, 8, 1]),
         np.array([0.3, 0.2, 0.4, 0.3])),
    ]
    groups2 = [
        (np.array([3]),
         np.array([0.5])),
    ]
    groups3 = [
        (np.array([0, 2, 8, 1]),
         np.array([0.3, 0.2, 0.4, 0.3])),
        (np.array([3]),
         np.array([0.5])),
    ]
    for groups in groups1, groups2, groups3:
        comlist = COMList(system, groups)
        comlist.forward()
        for igroup, (iatoms, weights) in enumerate(groups):
            np.testing.assert_almost_equal(
                np.dot(weights, system.pos[iatoms])/weights.sum(),
                comlist.pos[igroup]
            )


def test_glycine_bond():
    system = get_system_glycine()
    groups = [
        (np.array([0, 2, 8, 1]),
         np.array([0.3, 0.2, 0.4, 0.3])),
        (np.array([3, 4, 5]),
         np.array([0.5, 0.2, 0.9])),
    ]
    comlist = COMList(system, groups)
    part = ForcePartValence(system, comlist)
    part.add_term(Harmonic(2.1, 0.5, Bond(0, 1)))
    check_gpos_part(system, part)


def test_quartz():
    system = get_system_quartz()
    groups = [
        (np.array([0, 2, 8, 1]),
         np.array([0.3, 0.2, 0.4, 0.3])),
    ]
    comlist = COMList(system, groups)
    comlist.forward()
    com_a = comlist.pos[0].copy()
    system.pos[2] += system.cell.rvecs[0]
    comlist.forward()
    np.testing.assert_almost_equal(com_a, comlist.pos[0])
