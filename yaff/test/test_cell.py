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

from molmod import angstrom

from common import get_system_h2o32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine


def test_cell_h2o32():
    system = get_system_h2o32()
    assert (system.rspacings == 9.865*angstrom).all()
    assert (system.gspacings == 1/(9.865*angstrom)).all()


def test_cell_graphene8():
    system = get_system_graphene8()
    assert abs(np.dot(system.gvecs, system.rvecs.transpose()) - np.identity(2)).max() < 1e-5


def test_cell_polyethylene4():
    system = get_system_polyethylene4()
    assert system.rvecs.shape == (1, 3)
    assert system.gvecs.shape == (1, 3)
    assert abs(np.dot(system.gvecs, system.rvecs.transpose()) - 1) < 1e-5


def test_cell_quartz():
    system = get_system_quartz()
    assert system.rvecs.shape == (3, 3)
    assert system.gvecs.shape == (3, 3)
    assert abs(np.dot(system.gvecs, system.rvecs.transpose()) - np.identity(3)).max() < 1e-5


def test_cell_glycine():
    system = get_system_glycine()
    assert system.rvecs.shape == (0, 3)
    assert system.gvecs.shape == (0, 3)
    assert system.rspacings.shape == (0,)
    assert system.gspacings.shape == (0,)
