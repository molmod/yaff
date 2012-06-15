# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
#--


from yaff import *
from yaff.test.common import get_system_water32, get_system_water, \
    get_system_quartz


__all__ = ['get_ff_water32', 'get_ff_water', 'get_ff_bks']


def get_ff_water32():
    system = get_system_water32()
    return ForceField.generate(system, 'input/parameters_water.txt')
    return ff


def get_ff_water():
    system = get_system_water()
    return ForceField.generate(system, 'input/parameters_water.txt')
    return ff


def get_ff_bks(**kwargs):
    system = get_system_quartz()
    return ForceField.generate(system, 'input/parameters_bks.txt', **kwargs)
