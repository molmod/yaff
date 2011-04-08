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

from common import get_system_water32, get_system_glycine

from yaff import *


def test_scaling_water32():
    system = get_system_water32()
    scaling = Scaling(system.topology)
    for i in xrange(system.natom):
        if system.numbers[i] == 8:
            assert len(scaling[i][0]) == 2
            print scaling[i], i
            assert scaling[i][0]['i'] == i+1
            assert scaling[i][0]['scale'] == 0.0
            assert scaling[i][1]['i'] == i+2
            assert scaling[i][1]['scale'] == 0.0
        elif system.numbers[i] == 8:
            assert len(scaling[i][0]) == 1
            assert scaling[i][0]['i'] == (i/3)*3
            assert scaling[i][0]['scale'] == 0.0


def test_scaling_glycine():
    system = get_system_glycine()
    scaling = Scaling(system.topology, 1.0, 0.5, 0.2) # warning: absurd numbers
    for i in xrange(system.natom):
        assert len(scaling[i]) == len(system.topology.neighs2[i]) + len(system.topology.neighs3[i])
        for j, scale in scaling.items[i]:
            if j in system.topology.neighs2[i]:
                assert scale == 0.5
            if j in system.topology.neighs3[i]:
                assert scale == 0.2
