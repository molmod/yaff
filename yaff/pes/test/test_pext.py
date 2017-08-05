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

from yaff import *
from yaff.test.common import get_system_polyethylene4, get_system_graphene8, \
    get_system_water32
from yaff.pes.test.common import check_vtens_part


def test_vtens_pext_1():
    system = get_system_polyethylene4()
    part = ForcePartPressure(system, 1.0)
    check_vtens_part(system, part)

def test_vtens_pext_2():
    system = get_system_graphene8()
    part = ForcePartPressure(system, 1.0)
    check_vtens_part(system, part)

def test_vtens_pext_3():
    system = get_system_water32()
    part = ForcePartPressure(system, 1.0)
    check_vtens_part(system, part)
