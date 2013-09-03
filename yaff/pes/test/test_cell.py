# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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


import numpy as np

from molmod import angstrom

from yaff.test.common import get_system_water32, get_system_graphene8, \
    get_system_polyethylene4, get_system_quartz, get_system_glycine
from yaff import Cell


def test_cell_water32():
    cell = get_system_water32().cell
    assert (cell.rspacings == 9.865*angstrom).all()
    assert (cell.gspacings == 1/(9.865*angstrom)).all()
    assert abs(cell.volume - abs(np.linalg.det(cell.rvecs))) < 1e-10

    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(3)).max() < 1e-5
    assert abs(np.dot(cell.gvecs.transpose(), cell.rvecs) - np.identity(3)).max() < 1e-5
    vec1 = np.array([10.0, 0.0, 5.0])*angstrom
    cell.mic(vec1)
    assert abs(vec1 - np.array([0.135, 0.0, -4.865])*angstrom).max() < 1e-10
    vec2 = np.array([10.0, 0.0, 5.0])*angstrom
    cell.add_vec(vec2, cell.to_center(vec2))
    assert abs(vec1 - vec2).max() < 1e-10
    cell.add_vec(vec1, np.array([1,2,3]))
    assert abs(vec1 - np.array([10.0, 19.73, 24.73])*angstrom).max() < 1e-10

    cell2 = Cell(-cell.rvecs)
    assert abs(cell2.volume - abs(np.linalg.det(cell.rvecs))) < 1e-10



def test_cell_graphene8():
    cell = get_system_graphene8().cell
    assert abs(cell.volume - np.linalg.norm(np.cross(cell.rvecs[0], cell.rvecs[1]))) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(2)).max() < 1e-5
    vec1 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.mic(vec1)
    assert abs(vec1 - np.array([0.156, 0.0, 105])*angstrom).max() < 1e-3
    vec2 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.add_vec(vec2, cell.to_center(vec2))
    assert abs(vec1 - vec2).max() < 1e-10
    cell.add_vec(vec1, np.array([1,2]))
    assert abs(vec1 - np.array([10.002, 8.524, 105])*angstrom).max() < 1e-3


def test_cell_polyethylene4():
    cell = get_system_polyethylene4().cell
    assert cell.rvecs.shape == (1, 3)
    assert cell.gvecs.shape == (1, 3)
    assert abs(cell.volume - np.linalg.norm(cell.rvecs[0])) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - 1) < 1e-5
    vec1 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.mic(vec1)
    assert abs(vec1 - np.array([-0.15, -0.374, 104.89])*angstrom).max() < 1e-3
    vec2 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.add_vec(vec2, cell.to_center(vec2))
    assert abs(vec1 - vec2).max() < 1e-10
    cell.add_vec(vec1, np.array([1]))
    assert abs(vec1 - np.array([4.925, -0.187, 104.945])*angstrom).max() < 1e-3


def test_cell_quartz():
    cell = get_system_quartz().cell
    assert cell.rvecs.shape == (3, 3)
    assert cell.gvecs.shape == (3, 3)
    assert abs(cell.volume - abs(np.linalg.det(cell.rvecs))) < 1e-10
    assert abs(np.dot(cell.gvecs, cell.rvecs.transpose()) - np.identity(3)).max() < 1e-5


def test_cell_glycine():
    cell = get_system_glycine().cell
    assert cell.rvecs.shape == (0, 3)
    assert cell.gvecs.shape == (0, 3)
    assert cell.rspacings.shape == (0,)
    assert cell.gspacings.shape == (0,)
    vec1 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.mic(vec1)
    assert abs(vec1 - np.array([10.0, 0.0, 105.0])*angstrom).max() < 1e-3
    vec2 = np.array([10.0, 0.0, 105.0])*angstrom
    cell.add_vec(vec2, cell.to_center(vec2))
    assert abs(vec1 - vec2).max() < 1e-10
    cell.add_vec(vec1, np.array([], dtype=int))
    assert abs(vec1 - np.array([10.0, 0.0, 105.0])*angstrom).max() < 1e-3


def test_compute_distances1():
    n = 10
    cell = get_system_water32().cell
    pos = np.random.normal(0, 10, (n, 3))
    output = np.zeros((n*(n-1))/2, float)
    cell.compute_distances(output, pos)
    counter = 0
    for i0 in xrange(n):
        for i1 in xrange(i0):
            delta = pos[i0] - pos[i1]
            cell.mic(delta)
            assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
            counter += 1


def test_compute_distances2():
    n0 = 10
    n1 = 5
    cell = get_system_water32().cell
    pos0 = np.random.normal(0, 10, (n0, 3))
    pos1 = np.random.normal(0, 10, (n1, 3))
    output = np.zeros(n0*n1, float)
    cell.compute_distances(output, pos0, pos1)
    counter = 0
    for i0 in xrange(n0):
        for i1 in xrange(n1):
            delta = pos0[i0] - pos1[i1]
            cell.mic(delta)
            assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
            counter += 1


def test_cell_distances1_exclude_a():
    cell = Cell(np.zeros((0,3),float))
    pos0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])

    # First
    output = np.zeros(1, float)
    exclude = np.zeros((0,2),int)
    cell.compute_distances(output, pos0, pairs=exclude)
    assert output[0] == 1

    # Second
    output = np.zeros(0, float)
    exclude = np.array([[1,0]])
    cell.compute_distances(output, pos0, pairs=exclude)

    # Third
    output = np.zeros(0, float)
    exclude = np.array([[0,1]])
    try:
        cell.compute_distances(output, pos0, pairs=exclude)
        assert False
    except ValueError:
        pass


def test_cell_distances1_exclude_b():
    cell = Cell(np.zeros((0,3),float))
    pos0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    # Zeroth
    output = np.zeros(3, float)
    cell.compute_distances(output, pos0)
    assert output[0] == 1
    assert output[1] == 1
    assert output[2] == 2

    # First
    output = np.zeros(2, float)
    exclude = np.array([[1,0]])
    cell.compute_distances(output, pos0, pairs=exclude)
    assert output[0] == 1
    assert output[1] == 2

    # Second
    output = np.zeros(1, float)
    exclude = np.array([[1,0],[2,1]])
    cell.compute_distances(output, pos0, pairs=exclude)
    assert output[0] == 1

    # Third
    output = np.zeros(1, float)
    exclude = np.array([[2,1],[1,0]])
    try:
        cell.compute_distances(output, pos0, pairs=exclude)
        assert False
    except ValueError:
        pass


def test_cell_distances2_exclude_a():
    cell = Cell(np.zeros((0,3),float))
    pos0 = np.array([
        [0.0, 0.0, 0.0],
    ])
    pos1 = np.array([
        [0.0, 0.0, 1.0],
    ])

    # Zeroth
    output = np.zeros(1, float)
    cell.compute_distances(output, pos0, pos1)
    assert output[0] == 1

    # First
    output = np.zeros(0, float)
    exclude = np.array([[0,0]])
    cell.compute_distances(output, pos0, pos1, pairs=exclude)

    # Second
    output = np.zeros(0, float)
    for exclude in np.array([[-1,0]]), np.array([[0,-1]]), np.array([[0,5]]), np.array([[1,0]]):
        print exclude
        try:
            cell.compute_distances(output, pos0, pos1, pairs=exclude)
            assert False
        except ValueError:
            pass


def test_cell_distances2_exclude_b():
    cell = Cell(np.zeros((0,3),float))
    pos0 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    pos1 = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
    ])

    # Zeroth
    output = np.zeros(4, float)
    cell.compute_distances(output, pos0, pos1)
    assert output[0] == 1
    assert output[1] == 2
    assert output[2] == 0
    assert output[3] == 1

    # First
    output = np.zeros(3, float)
    exclude = np.array([[0,0]])
    cell.compute_distances(output, pos0, pos1, pairs=exclude)
    assert output[0] == 2
    assert output[1] == 0
    assert output[2] == 1

    # Second
    output = np.zeros(2, float)
    exclude = np.array([[0,0],[1,1]])
    cell.compute_distances(output, pos0, pos1, pairs=exclude)
    assert output[0] == 2
    assert output[1] == 0

    # Third
    output = np.zeros(2, float)
    for exclude in np.array([[1,0],[0,1]]), np.array([[1,0],[0,0]]):
        try:
            cell.compute_distances(output, pos0, pos1, pairs=exclude)
            assert False
        except ValueError:
            pass


def test_compute_distances1_nimage():
    natom = 10
    nimage = 1
    cell = get_system_water32().cell
    pos = np.random.normal(0, 10, (natom, 3))
    factor = (1+2*nimage)**3
    output = np.zeros(factor*(natom*(natom-1))/2, float)
    cell.compute_distances(output, pos, nimage=nimage)
    counter = 0
    for r0 in xrange(-1, 2):
        for r1 in xrange(-1, 2):
            for r2 in xrange(-1, 2):
                for i0 in xrange(natom):
                    for i1 in xrange(i0):
                        delta = pos[i0] - pos[i1]
                        cell.mic(delta)
                        cell.add_vec(delta, np.array([r0, r1, r2]))
                        assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
                        counter += 1


def test_compute_distances1_nimage_exclude():
    natom = 10
    nimage = 1
    cell = get_system_water32().cell
    pos = np.random.normal(0, 10, (natom, 3))
    exclude = np.array([[1, 0], [2, 1]])
    factor = (1+2*nimage)**3
    output = np.zeros(factor*(natom*(natom-1))/2 - 2, float)
    cell.compute_distances(output, pos, pairs=exclude, nimage=nimage)
    counter = 0
    ex_counter = 0
    for r0 in xrange(-1, 2):
        for r1 in xrange(-1, 2):
            for r2 in xrange(-1, 2):
                for i0 in xrange(natom):
                    for i1 in xrange(i0):
                        if (r0 == 0) and (r1 == 0) and (r2 == 0):
                            if ex_counter < len(exclude) and i0 == exclude[ex_counter,0] and i1 == exclude[ex_counter, 1]:
                                ex_counter += 1
                                continue
                        delta = pos[i0] - pos[i1]
                        cell.mic(delta)
                        cell.add_vec(delta, np.array([r0, r1, r2]))
                        assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
                        counter += 1
    assert ex_counter == 2


def test_compute_distances2_nimage():
    n0 = 10
    n1 = 5
    nimage = 1
    cell = get_system_water32().cell
    pos0 = np.random.normal(0, 10, (n0, 3))
    pos1 = np.random.normal(0, 10, (n1, 3))
    factor = (1+2*nimage)**3
    output = np.zeros(factor*n0*n1, float)
    cell.compute_distances(output, pos0, pos1, nimage=nimage)
    counter = 0
    for r0 in xrange(-1, 2):
        for r1 in xrange(-1, 2):
            for r2 in xrange(-1, 2):
                for i0 in xrange(n0):
                    for i1 in xrange(n1):
                        delta = pos0[i0] - pos1[i1]
                        cell.mic(delta)
                        cell.add_vec(delta, np.array([r0, r1, r2]))
                        assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
                        counter += 1


def test_compute_distances2_nimage_exclude():
    n0 = 10
    n1 = 5
    nimage = 1
    cell = get_system_water32().cell
    pos0 = np.random.normal(0, 10, (n0, 3))
    pos1 = np.random.normal(0, 10, (n1, 3))
    factor = (1+2*nimage)**3
    exclude = np.array([[1, 0], [2, 1]])
    output = np.zeros(factor*n0*n1 - 2, float)
    cell.compute_distances(output, pos0, pos1, pairs=exclude, nimage=nimage)
    counter = 0
    ex_counter = 0
    for r0 in xrange(-1, 2):
        for r1 in xrange(-1, 2):
            for r2 in xrange(-1, 2):
                for i0 in xrange(n0):
                    for i1 in xrange(n1):
                        if (r0 == 0) and (r1 == 0) and (r2 == 0):
                            if ex_counter < len(exclude) and i0 == exclude[ex_counter,0] and i1 == exclude[ex_counter, 1]:
                                ex_counter += 1
                                continue
                        delta = pos0[i0] - pos1[i1]
                        cell.mic(delta)
                        cell.add_vec(delta, np.array([r0, r1, r2]))
                        assert abs(output[counter] - np.linalg.norm(delta)) < 1e-10
                        counter += 1
    assert ex_counter == 2


def test_compute_distances1_include():
    natom = 10
    cell = get_system_water32().cell
    pos = np.random.normal(0, 10, (natom, 3))
    # all
    output_all = np.zeros((natom*(natom-1))/2, float)
    cell.compute_distances(output_all, pos)
    # exclude
    pairs = np.array([[1, 0], [3, 2], [5, 3]])
    output_ex = np.zeros((natom*(natom-1))/2-3, float)
    cell.compute_distances(output_ex, pos, pairs=pairs, do_include=False)
    # include
    output_in = np.zeros(3, float)
    cell.compute_distances(output_in, pos, pairs=pairs, do_include=True)
    assert set(output_all) == set(output_ex) | set(output_in)


def test_compute_distances2_include():
    n0 = 10
    n1 = 5
    cell = get_system_water32().cell
    pos0 = np.random.normal(0, 10, (n0, 3))
    pos1 = np.random.normal(0, 10, (n1, 3))
    # all
    output_all = np.zeros(n0*n1, float)
    cell.compute_distances(output_all, pos0, pos1)
    # exclude
    pairs = np.array([[1, 0], [3, 2], [5, 3]])
    output_ex = np.zeros(n0*n1-3, float)
    cell.compute_distances(output_ex, pos0, pos1, pairs=pairs, do_include=False)
    # include
    output_in = np.zeros(3, float)
    cell.compute_distances(output_in, pos0, pos1, pairs=pairs, do_include=True)
    assert set(output_all) == set(output_ex) | set(output_in)
