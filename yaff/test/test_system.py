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


import tempfile, shutil, numpy as np

from yaff import System

from common import get_system_water32


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
        assert system0.ffatypes == list(system1.ffatypes)
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
        from molmod import Molecule
        mol = Molecule(system0.numbers, system0.pos)
        mol.write_to_file('%s/tmp.xyz' % dirname)
        system1 = System.from_file('%s/tmp.xyz' % dirname, rvecs=system0.cell.rvecs, ffatypes=system0.ffatypes, ffatype_ids=system0.ffatype_ids)
        assert (system0.numbers == system1.numbers).all()
        assert abs(system0.pos - system1.pos).max() < 1e-10
        assert system0.scopes is None
        assert system1.scopes is None
        assert system0.scope_ids is None
        assert system1.scope_ids is None
        assert system0.ffatypes == system1.ffatypes
        assert (system0.ffatype_ids == system1.ffatype_ids).all()
        assert abs(system0.cell.rvecs - system1.cell.rvecs).max() < 1e-10
        assert system1.charges is None
    finally:
        shutil.rmtree(dirname)


def test_ffatypes():
    system = get_system_water32()
    assert system.ffatypes == ['O', 'H']
    assert (system.ffatype_ids[system.numbers==8] == 0).all()
    assert (system.ffatype_ids[system.numbers==1] == 1).all()

def test_scopes1():
    system = System(
        numbers=np.array([8, 1, 1, 6, 1, 1, 1, 8, 1]),
        pos=np.zeros((9, 3), float),
        scopes=['WAT', 'WAT', 'WAT', 'METH', 'METH', 'METH', 'METH', 'METH', 'METH'],
        ffatypes=['O', 'H', 'H', 'C', 'H_C', 'H_C', 'H_C', 'O', 'H_O'],
    )
    assert system.scopes == ['WAT', 'METH']
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert system.ffatypes == ['O', 'H', 'C', 'H_C', 'O', 'H_O']
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
    assert system.scopes == ['WAT', 'METH']
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert system.ffatypes == ['O', 'H', 'C', 'H_C', 'O', 'H_O']
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
    assert system.scopes == ['WAT', 'METH']
    assert (system.scope_ids == np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])).all()
    assert system.ffatypes == ['O', 'H', 'C', 'H_C', 'H_O', 'O']
    assert (system.ffatype_ids == np.array([0, 1, 1, 2, 3, 3, 3, 5, 4])).all()
    assert system.get_scope(0) == 'WAT'
    assert system.get_scope(3) == 'METH'
    assert system.get_ffatype(0) == 'O'
    assert system.get_ffatype(7) == 'O'
    assert system.get_ffatype(8) == 'H_O'
