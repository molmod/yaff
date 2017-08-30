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

import shutil
import os
import h5py as h5

from yaff import *
from yaff.analysis.test.common import run_nve_water32
from yaff.sampling.test.common import get_ff_water32


def test_diff_offline():
    with run_nve_water32(__name__, 'test_diff_offline') as (dn_tmp, nve, f):
        select = nve.ff.system.get_indexes('O')
        diff = Diffusion(f, select=select)
        assert 'trajectory/pos_diff' in f
        assert 'trajectory/pos_diff/msds' in f
        assert 'trajectory/pos_diff/time' in f
        assert 'trajectory/pos_diff/msdcounters' in f
        assert 'trajectory/pos_diff/msdsums' in f
        assert 'trajectory/pos_diff/pars' in f
        fn_png = '%s/msds.png' % dn_tmp
        diff.plot(fn_png)
        assert os.path.isfile(fn_png)


def test_diff_online():
    # Setup a test FF
    ff = get_ff_water32()
    # Run a test simulation
    with h5.File('yaff.analysis.test.test_diffusion.test_diff_online.h5', driver='core', backing_store=False) as f:
        hdf5 = HDF5Writer(f)
        select = ff.system.get_indexes('O')
        diff0 = Diffusion(f, select=select)
        nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=[hdf5, diff0])
        nve.run(5)
        assert nve.counter == 5
        # Also run an off-line rdf and compare
        diff1 = Diffusion(f, select=select)
        assert abs(diff0.A - diff1.A) < 1e-10
        assert abs(diff0.B - diff1.B) < 1e-10
        assert abs(diff0.time - diff1.time).max() < 1e-10
        assert abs(diff0.msds - diff1.msds).max() < 1e-10
        assert abs(diff0.msdsums - diff1.msdsums).max() < 1e-10
        assert abs(diff0.msdcounters - diff1.msdcounters).max() < 1e-10


def test_diff_online_blind():
    ff = get_ff_water32()
    select = ff.system.get_indexes('O')
    diff = Diffusion(None, select=select)
    nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=diff)
    nve.run(5)


def test_diff_bsize():
    ff = get_ff_water32()
    select = ff.system.get_indexes('O')
    diff = Diffusion(None, select=select, bsize=3, mult=2)
    nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=diff)
    nve.run(10)
    assert diff.msdcounters[0] == 7
    assert diff.msdcounters[1] == 2
