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
import numpy as np

from yaff import *
from yaff.analysis.test.common import run_nve_water32
from yaff.sampling.test.common import get_ff_water32


def test_spectrum_offline():
    with run_nve_water32(__name__, 'test_spectrum_offline') as (dn_tmp, nve, f):
        for bsize in 2, 4, 5:
            spectrum = Spectrum(f, bsize=bsize)
            assert 'trajectory/vel_spectrum' in f
            assert 'trajectory/vel_spectrum/amps' in f
            assert 'trajectory/vel_spectrum/freqs' in f
            assert 'trajectory/vel_spectrum/ac' in f
            assert 'trajectory/vel_spectrum/time' in f
            fn_png = '%s/spectrum%i.png' % (dn_tmp, bsize)
            spectrum.plot(fn_png)
            assert os.path.isfile(fn_png)
            fn_png = '%s/ac%i.png' % (dn_tmp, bsize)
            spectrum.plot_ac(fn_png)
            assert os.path.isfile(fn_png)
            assert f['trajectory/vel_spectrum'].attrs['nfft'] == 3*3*32*(6//bsize)
            del f['trajectory/vel_spectrum']


def test_spectrum_online():
    for bsize in 2, 4, 5:
        # Setup a test FF
        ff = get_ff_water32()
        # Run a test simulation
        with h5.File('yaff.analysis.test.test_spectrum.test_spectrum_online_%i.h5' % bsize, driver='core', backing_store=False) as f:
            hdf5 = HDF5Writer(f)
            spectrum0 = Spectrum(f, bsize=bsize)
            nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=[hdf5, spectrum0])
            nve.run(5)
            assert nve.counter == 5
            # Also run an off-line spectrum and compare
            spectrum1 = Spectrum(f, bsize=bsize)
            assert abs(spectrum0.timestep - spectrum1.timestep) < 1e-10
            assert abs(spectrum0.amps - spectrum1.amps).max() < 1e-10
            assert abs(spectrum0.freqs - spectrum1.freqs).max() < 1e-10
            assert abs(spectrum0.ac - spectrum1.ac).max() < 1e-10
            assert abs(spectrum0.time - spectrum1.time).max() < 1e-10
            assert f['trajectory/vel_spectrum'].attrs['nfft'] == 3*3*32*(6//bsize)


def test_spectrum_online_blind():
    # Setup a test FF
    ff = get_ff_water32()
    spectrum = Spectrum(bsize=2)
    nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=spectrum)
    nve.run(5)
    assert nve.counter == 5


def test_spectrum_online_weights():
    # Setup a test FF
    ff = get_ff_water32()
    ff.system.set_standard_masses()
    weights = np.array([ff.system.masses]*3).T
    spectrum = Spectrum(bsize=2, weights=weights)
    nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=spectrum)
    nve.run(5)
    assert nve.counter == 5


def test_spectrum_iter_indexes():
    with h5.File('yaff.analysis.test.test_spectrum.test_spectrum_iter_indexes.h5', driver='core', backing_store=False) as f:
        spectrum = Spectrum(f, bsize=10)
    l = list(spectrum._iter_indexes(np.zeros((10, 5, 3), float)))
    assert l == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
                 (2, 2), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2)]
    l = list(spectrum._iter_indexes(np.zeros((10, 5), float)))
    assert l == [(0,), (1,), (2,), (3,), (4,)]
    spectrum = Spectrum(f, bsize=10, select=[1,4])
    l = list(spectrum._iter_indexes(np.zeros((10, 5, 3), float)))
    assert l == [(1, 0), (1, 1), (1, 2), (4, 0), (4, 1), (4, 2)]
    l = list(spectrum._iter_indexes(np.zeros((10, 5), float)))
    assert l == [(1,), (4,)]
