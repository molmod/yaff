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


import tempfile, shutil, numpy as np, h5py as h5

from yaff import System, Cell, angstrom

from common import get_system_water32, get_system_glycine, get_system_quartz, get_system_cyclopropene, get_system_peroxide


def test_chk():
    system0 = get_system_water32()
    dirname = tempfile.mkdtemp('yaff', 'test_chk')
    try:
        system0.to_file('%s/tmp.chk' % dirname)
        system1 = System.from_file('%s/tmp.chk' % dirname)
        assert (system0.numbers == system1.numbers).all()
        assert abs(system0.pos - system1.pos).max() < 1e-10
        assert system0.scopes is None
        assert system1.scopes is None
        assert system0.scope_ids is None
        assert system1.scope_ids is None
        assert (system0.ffatypes == system1.ffatypes).all()
        assert (system0.ffatype_ids == system1.ffatype_ids).all()
        assert (system0.bonds == system1.bonds).all()
        assert abs(system0.cell.rvecs - system1.cell.rvecs).max() < 1e-10
        assert abs(system0.charges - system1.charges).max() < 1e-10
    finally:
        shutil.rmtree(dirname)


def test_xyz():
    system0 = get_system_water32()
    dirname = tempfile.mkdtemp('yaff', 'test_xyz')
    try:
        system0.to_file('%s/tmp.xyz' % dirname)
        system1 = System.from_file('%s/tmp.xyz' % dirname, rvecs=system0.cell.rvecs, ffatypes=system0.ffatypes, ffatype_ids=system0.ffatype_ids)
        assert (system0.numbers == system1.numbers).all()
        assert abs(system0.pos - system1.pos).max() < 1e-10
        assert system0.scopes is None
        assert system1.scopes is None
        assert system0.scope_ids is None
        assert system1.scope_ids is None
        assert (system0.ffatypes == system1.ffatypes).all()
        assert (system0.ffatype_ids == system1.ffatype_ids).all()
        assert abs(system0.cell.rvecs - system1.cell.rvecs).max() < 1e-10
        assert system1.charges is None
    finally:
        shutil.rmtree(dirname)


def test_hdf5():
    system0 = get_system_water32()
    dirname = tempfile.mkdtemp('yaff', 'test_hdf5')
    try:
        #from molmod import Molecule
        fn = '%s/tmp.h5' % dirname
        system0.to_file(fn)
        with h5.File(fn) as f:
            assert 'system' in f
    finally:
        shutil.rmtree(dirname)


def test_ffatypes():
    system = get_system_water32()
    assert (system.ffatypes == ['O', 'H']).all()
    assert (system.ffatype_ids[system.numbers==8] == 0).all()
    assert (system.ffatype_ids[system.numbers==1] == 1).all()


def test_scopes1():
    system = System(
        numbers=np.array([8, 1, 1, 6, 1, 1, 1, 8, 1]),
        pos=np.zeros((9, 3), float),
        scopes=['WAT', 'WAT', 'WAT', 'METH', 'METH', 'METH', 'METH', 'METH', 'METH'],
        ffatypes=['O', 'H', 'H', 'C', 'H_C', 'H_C', 'H_C', 'O', 'H_O'],
    )
    assert (system.scopes == ['WAT', 'METH']).all()
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert (system.ffatypes == ['O', 'H', 'C', 'H_C', 'O', 'H_O']).all()
    assert (system.ffatype_ids == np.array([0, 1, 1, 2, 3, 3, 3, 4, 5])).all()


def test_scopes2():
    system = System(
        numbers=np.array([8, 1, 1, 6, 1, 1, 1, 8, 1]),
        pos=np.zeros((9, 3), float),
        scopes=['WAT', 'METH'],
        scope_ids=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        ffatypes=['O', 'H', 'C', 'H_C', 'O', 'H_O'],
        ffatype_ids=np.array([0, 1, 1, 2, 3, 3, 3, 4, 5])
    )
    assert (system.scopes == ['WAT', 'METH']).all()
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert (system.ffatypes == ['O', 'H', 'C', 'H_C', 'O', 'H_O']).all()
    assert (system.ffatype_ids == np.array([0, 1, 1, 2, 3, 3, 3, 4, 5])).all()


def test_scopes3():
    system = System(
        numbers=np.array([8, 1, 1, 6, 1, 1, 1, 8, 1]),
        pos=np.zeros((9, 3), float),
        scopes=['WAT', 'METH'],
        scope_ids=np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        ffatypes=['O', 'H', 'C', 'H_C', 'H_O'],
        ffatype_ids=np.array([0, 1, 1, 2, 3, 3, 3, 0, 4])
    )
    assert (system.scopes == ['WAT', 'METH']).all()
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert (system.ffatypes == ['O', 'H', 'C', 'H_C', 'H_O', 'O']).all()
    assert (system.ffatype_ids == np.array([0, 1, 1, 2, 3, 3, 3, 5, 4])).all()
    assert system.get_scope(0) == 'WAT'
    assert system.get_scope(3) == 'METH'
    assert system.get_ffatype(0) == 'O'
    assert system.get_ffatype(7) == 'O'
    assert system.get_ffatype(8) == 'H_O'


def test_unravel_triangular():
    from yaff.system import _unravel_triangular
    counter = 0
    for i0 in xrange(100):
        for i1 in xrange(i0):
            assert _unravel_triangular(counter) == (i0, i1)
            counter += 1


def check_detect_bonds(system):
    old_bonds = set([frozenset(pair) for pair in system.bonds])
    system.detect_bonds()
    new_bonds = set([frozenset(pair) for pair in system.bonds])
    assert len(old_bonds) == len(new_bonds)
    assert old_bonds == new_bonds


def test_detect_bonds_glycine():
    system = get_system_glycine()
    check_detect_bonds(system)
    system = System(system.numbers, system.pos)
    system.detect_bonds()
    assert hasattr(system, 'neighs1')
    assert hasattr(system, 'neighs2')
    assert hasattr(system, 'neighs3')


def test_detect_bonds_water32():
    system = get_system_water32()
    assert system.nbond == 64
    check_detect_bonds(system)


def test_detect_bonds_quartz():
    system = get_system_quartz()
    check_detect_bonds(system)


def test_detect_bonds_water_exceptions():
    system = get_system_water32()
    # create system without bonds
    system = System(system.numbers, system.pos)
    # Add bonds between hydrogen atoms (unrealistic but useful for testing)
    system.detect_bonds({(1,1): 2.0*angstrom})
    assert system.nbond >= 96


def test_detect_bonds_cyclopropene_exceptions():
    system = get_system_cyclopropene()
    # create system without bonds
    system = System(system.numbers, system.pos)
    # Add bonds between all hydrogen and carbon atoms (unrealistic but useful for testing)
    system.detect_bonds({(1,6): 8.0*angstrom})
    assert system.nbond == 3+4*3


def test_iter_bonds_empty():
    system = get_system_cyclopropene()
    system.bonds = None
    assert len(list(system.iter_bonds())) == 0
    assert len(list(system.iter_angles())) == 0
    assert len(list(system.iter_dihedrals())) == 0


def check_detect_ffatypes(system, rules):
    old_ffatypes = system.ffatypes
    old_ffatype_ids = system.ffatype_ids
    system.detect_ffatypes(rules)
    assert (system.ffatypes == old_ffatypes).all()
    assert (system.ffatype_ids == old_ffatype_ids).all()


def test_detect_ffatypes():
    system = get_system_quartz()
    rules = [
        ('Si', '14'),
        ('O', '8'),
    ]
    check_detect_ffatypes(system, rules)


def test_align_cell_quartz():
    system = get_system_quartz()
    system.cell = Cell(system.cell.rvecs[::-1].copy())
    lcs = np.array([
        [1, 1, 0],
        [0, 0, 1],
    ])
    system.align_cell(lcs)
    # c should be aligned with z axis
    rvecs = system.cell.rvecs
    assert abs(rvecs[2][0]) < 1e-10
    assert abs(rvecs[2][1]) < 1e-10
    # sum of a and b should be aligned with x axis
    assert abs(rvecs[0][1] + rvecs[1][1]) < 1e-10
    assert abs(rvecs[0][2] + rvecs[1][2]) < 1e-10
    # difference of a and b should be aligned with y axis
    assert abs(rvecs[0][0] - rvecs[1][0]) < 1e-4
    assert abs(rvecs[0][2] - rvecs[1][2]) < 1e-10
    # check if the bonds are the same in the rotated structure
    check_detect_bonds(system)


def test_supercell_quartz_222():
    system111 = get_system_quartz()
    system222 = system111.supercell(2, 2, 2)
    assert abs(system222.cell.volume - system111.cell.volume*8) < 1e-10
    assert abs(system222.cell.rvecs - system111.cell.rvecs*2).max() < 1e-10
    assert system222.natom == system111.natom*8
    assert len(system222.bonds) == len(system111.bonds)*8
    assert abs(system222.pos[9:18] - system111.pos - system111.cell.rvecs[2]).max() < 1e-10
    assert abs(system222.pos[-9:] - system111.pos - system111.cell.rvecs.sum(axis=0)).max() < 1e-10
    assert issubclass(system222.bonds.dtype.type, int)
    rules = [
        ('Si', '14'),
        ('O', '8'),
    ]
    check_detect_ffatypes(system222, rules)
    check_detect_bonds(system222)
    assert issubclass(system222.bonds.dtype.type, int)


def test_supercell_mil53_121():
    system111 = get_system_quartz()
    system121 = system111.supercell(1, 2, 1)
    assert abs(system121.cell.volume - system111.cell.volume*2) < 1e-10
    assert abs(system121.cell.rvecs[0] - system111.cell.rvecs[0]).max() < 1e-10
    assert abs(system121.cell.rvecs[1] - system111.cell.rvecs[1]*2).max() < 1e-10
    assert abs(system121.cell.rvecs[2] - system111.cell.rvecs[2]).max() < 1e-10
    assert system121.natom == system111.natom*2
    assert len(system121.bonds) == len(system111.bonds)*2
    assert abs(system121.pos[:system111.natom] - system111.pos).max() < 1e-10
    assert abs(system121.pos[system111.natom:] - system111.pos - system111.cell.rvecs[1]).max() < 1e-10
    assert (system121.numbers[:system111.natom] == system111.numbers).all()
    assert (system121.numbers[system111.natom:] == system111.numbers).all()
    assert (system121.ffatype_ids[:system111.natom] == system111.ffatype_ids).all()
    assert (system121.ffatype_ids[system111.natom:] == system111.ffatype_ids).all()
    assert (system121.ffatypes == system111.ffatypes).all()
    check_detect_bonds(system121)


def test_supercell_nobonds():
    cellpar = 2.867*angstrom
    sys111 = System(
        numbers=np.array([26, 26]),
        pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])*cellpar,
        ffatypes=['Fe', 'Fe'],
        rvecs=np.identity(3)*cellpar,
    )
    sys333 = sys111.supercell(3,3,3)


def test_supercell_charges():
    cellpar = 2.867*angstrom
    sys111 = System(
        numbers=np.array([26, 26]),
        pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])*cellpar,
        ffatypes=['Fe', 'Fe'],
        charges=np.array([0.1,1.0]),
        radii=np.array([0.0,2.0]),
        rvecs=np.identity(3)*cellpar,
    )
    sys333 = sys111.supercell(3,3,3)
    assert np.all(sys333.charges==np.repeat(np.array([0.1,1.0]),9))==0.0
    assert np.all(sys333.radii==np.repeat(np.array([0.0,2.0]),9))==0.0


def test_supercell_dipoles():
    cellpar = 2.867*angstrom
    sys111 = System(
        numbers=np.array([26, 26]),
        pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])*cellpar,
        ffatypes=['Fe', 'Fe'],
        dipoles=np.array([[0.1,1.0,2.0],[0.5,0.7,0.9]]),
        radii2=np.array([0.0,2.0]),
        rvecs=np.identity(3)*cellpar,
    )
    sys333 = sys111.supercell(3,3,3)
    assert np.all(sys333.charges==np.tile(np.array([[0.1,1.0,2.0],[0.5,0.7,0.9]]),(9,1)))==0.0
    assert np.all(sys333.radii==np.repeat(np.array([0.0,2.0]),9))==0.0


def test_remove_duplicate1():
    system1 = get_system_quartz()
    system2 = system1.remove_duplicate()
    assert system1.natom == system2.natom
    assert system1.nbond == system2.nbond
    assert (system1.numbers == system2.numbers).all()
    assert (system1.pos == system2.pos).all()
    assert (system1.ffatype_ids == system2.ffatype_ids).all()
    assert system1.neighs1 == system2.neighs1


def test_remove_duplicate2():
    system1 = get_system_quartz()
    system2 = system1.supercell(1, 2, 1)
    system2.cell = system1.cell
    system3 = system2.remove_duplicate()
    assert system1.natom == system3.natom
    assert system1.nbond == system3.nbond
    assert system1.numbers.sum() == system3.numbers.sum()
    assert abs(system1.pos.mean(axis=0) - system3.pos.mean(axis=0)).max() < 1e-10
    assert system1.ffatype_ids.sum() == system3.ffatype_ids.sum()


def test_remove_duplicate_dipoles():
    cellpar = 2.867*angstrom
    system1 = System(
        numbers=np.array([26, 27]),
        pos=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])*cellpar,
        ffatypes=['A', 'B'],
        dipoles=np.array([[0.1,1.0,2.0],[0.5,0.7,0.9]]),
        radii2=np.array([0.0,2.0]),
        rvecs=np.identity(3)*cellpar,
    )
    system2 = system1.supercell(1, 2, 1)
    system2.cell = system1.cell
    system3 = system2.remove_duplicate()
    for j, number in enumerate([26, 27]):
        #By removing duplicates, atoms might be reordered
        i = np.where( system3.numbers == number)[0]
        assert system1.radii2[j] == system3.radii2[i]
        assert np.all( system1.dipoles[j] == system3.dipoles[i] )


def test_subsystem():
    system1 = get_system_quartz()
    system1.dipoles = np.random.rand( system1.natom , 3 )
    system2 = system1.subsystem((system1.numbers == 8).nonzero()[0])
    assert system2.natom == 6
    assert (system2.numbers == 8).all()
    assert len(system2.bonds) == 0
    assert system2.scopes is None
    assert system2.get_ffatype(0) == 'O'
    assert (system2.charges == -0.9).all()
    assert (system1.cell.rvecs == system2.cell.rvecs).all()
    assert np.shape(system2.dipoles)[1] == 3


def test_cut_bonds():
    system = get_system_peroxide()
    system.cut_bonds([0,2])
    assert (system.bonds == [[0,2],[1,3]]).all()
