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

from yaff.test.common import get_system_water32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine


def test_cell_water32():
    cell = get_system_water32().cell
    assert (cell.rspacings == 9.865*angstrom).all()
    assert (cell.gspacings == 1/(9.865*angstrom)).all()
    assert abs(cell.volume - np.linalg.det(cell.rvecs)) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(3)).max() < 1e-5


def test_cell_graphene8():
    cell = get_system_graphene8().cell
    assert abs(cell.volume - np.dot(cell.rvecs[0], cell.rvecs[1])) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(2)).max() < 1e-5


def test_cell_polyethylene4():
    cell = get_system_polyethylene4().cell
    assert cell.rvecs.shape == (1, 3)
    assert cell.gvecs.shape == (1, 3)
    assert abs(cell.volume - np.linalg.norm(cell.rvecs[0])) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - 1) < 1e-5


def test_cell_quartz():
    cell = get_system_quartz().cell
    assert cell.rvecs.shape == (3, 3)
    assert cell.gvecs.shape == (3, 3)
    assert abs(cell.volume - np.linalg.det(cell.rvecs)) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(3)).max() < 1e-5


def test_cell_glycine():
    cell = get_system_glycine().cell
    assert cell.rvecs.shape == (0, 3)
    assert cell.gvecs.shape == (0, 3)
    assert cell.rspacings.shape == (0,)
    assert cell.gspacings.shape == (0,)
