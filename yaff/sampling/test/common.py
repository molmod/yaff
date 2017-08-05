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

import pkg_resources
import numpy as np

from yaff import *
from yaff.test.common import get_system_water32, get_system_water, \
    get_system_quartz, get_system_graphene8, get_system_polyethylene4, \
    get_system_nacl_cubic


__all__ = [
    'get_ff_water32', 'get_ff_water', 'get_ff_bks', 'get_ff_graphene',
    'get_ff_polyethylene', 'get_ff_nacl',
]


def get_ff_water32():
    system = get_system_water32()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    return ForceField.generate(system, fn_pars, skin=2)


def get_ff_water():
    system = get_system_water()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_water.txt')
    return ForceField.generate(system, fn_pars)


def get_ff_bks(**kwargs):
    system = get_system_quartz()
    if kwargs.pop('align_ax', False):
        system.align_cell(np.array([[1, 0, 0], [0, 1, 1]]), True)
        rvecs = system.cell.rvecs.copy()
        rvecs[1, 0] = 0.0
        rvecs[2, 0] = 0.0
        rvecs[0, 1] = 0.0
        rvecs[0, 2] = 0.0
        system.cell.update_rvecs(rvecs)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_bks.txt')
    return ForceField.generate(system, fn_pars, **kwargs)


def get_ff_graphene(**kwargs):
    system = get_system_graphene8()
    system = system.supercell(2, 2)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_polyene.txt')
    return ForceField.generate(system, fn_pars, **kwargs)


def get_ff_polyethylene(**kwargs):
    system = get_system_polyethylene4()
    system = system.supercell(2)
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_alkane.txt')
    return ForceField.generate(system, fn_pars, **kwargs)


def get_ff_nacl(**kwargs):
    kwargs.setdefault('rcut', 5.0*angstrom)
    system = get_system_nacl_cubic()
    fn_pars = pkg_resources.resource_filename(__name__, '../../data/test/parameters_nacl.txt')
    return ForceField.generate(system, fn_pars, **kwargs)
