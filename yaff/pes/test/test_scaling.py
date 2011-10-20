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

from yaff.test.common import get_system_water32, get_system_glycine, \
    get_system_quartz

from yaff import *


def test_scaling_water32():
    system = get_system_water32()
    stab = Scalings(system, 0.5, 0.0, 1.0).stab
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == system.natom
    for i0, i1, scale in stab:
        if system.numbers[i1] == 8:
            assert (i0 == i1+1) or (i0 == i1+2)
            assert scale == 0.5
        elif system.numbers[i1] == 1:
            assert i0 == i1+1
            assert scale == 0.0


def test_scaling_glycine():
    system = get_system_glycine()
    stab = Scalings(system, 1.0, 0.5, 0.2).stab # warning: absurd numbers
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == sum(len(system.neighs2[i]) + len(system.neighs3[i]) for i in xrange(system.natom))/2
    for i0, i1, scale in stab:
        if i0 in system.neighs2[i1]:
            assert scale == 0.5
        elif i0 in system.neighs3[i1]:
            assert scale == 0.2


def test_scaling_quartz():
    system = get_system_quartz()
    stab = Scalings(system).stab
    assert (stab['a'] > stab['b']).all()
    assert len(stab) == sum(len(system.neighs1[i]) + len(system.neighs2[i]) for i in xrange(system.natom))/2
    for i0, i1, scale in stab:
        assert scale == 0.0
        assert i0 in system.neighs1[i1] or i0 in system.neighs2[i1]
