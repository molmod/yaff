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


import h5py

from yaff import *
from yaff.sampling.test.common import get_ff_water32

def test_basic_5steps():
    opt = CGOptimizer(get_ff_water32(), CartesianDOF())
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_full_cell_5steps():
    opt = CGOptimizer(get_ff_water32(), CellDOF(FullCell()))
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_aniso_cell_5steps():
    opt = CGOptimizer(get_ff_water32(), CellDOF(AnisoCell()))
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_iso_cell_5steps():
    opt = CGOptimizer(get_ff_water32(), CellDOF(IsoCell()))
    epot0 = opt.epot
    opt.run(5)
    epot1 = opt.epot
    assert opt.counter == 5
    assert epot1 < epot0


def test_basic_until_converged():
    opt = CGOptimizer(get_ff_water32(), CartesianDOF(gpos_max=3e-1, gpos_rms=1e-1, dpos_max=None, dpos_rms=None))
    assert opt.dof.th_gpos_max == 3e-1
    assert opt.dof.th_gpos_rms == 1e-1
    assert opt.dof.th_dpos_max is None
    assert opt.dof.th_dpos_rms is None
    opt.run()
    assert opt.dof.conv_count == 0
    assert opt.dof.conv_val < 1
    assert opt.dof.conv_worst.startswith('gpos_')
    assert opt.dof.gpos_max < 3e-1
    assert opt.dof.gpos_rms < 1e-1


def check_hdf5_common(f):
    assert 'system' in f
    assert 'numbers' in f['system']
    assert 'ffatypes' in f['system']
    assert 'ffatype_ids' in f['system']
    assert 'pos' in f['system']
    assert 'bonds' in f['system']
    assert 'rvecs' in f['system']
    assert 'charges' in f['system']
    assert 'trajectory' in f
    assert 'counter' in f['trajectory']
    assert 'epot' in f['trajectory']
    assert 'pos' in f['trajectory']


def test_hdf5():
    f = h5py.File('tmp.h5', driver='core', backing_store=False)
    try:
        hdf5 = HDF5Writer(f)
        opt = CGOptimizer(get_ff_water32(), CartesianDOF(), hooks=hdf5)
        opt.run(15)
        assert opt.counter == 15
        check_hdf5_common(hdf5.f)
        assert f['trajectory'].attrs['row'] == 16
        assert f['trajectory/counter'][15] == 15
    finally:
        f.close()
