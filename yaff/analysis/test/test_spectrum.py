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

import shutil, os, h5py

from yaff import *
from yaff.analysis.test.common import get_nve_water32
from yaff.sampling.test.common import get_ff_water32


def test_spectrum_offline():
    dn_tmp, nve, f = get_nve_water32()
    try:
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
            del f['trajectory/vel_spectrum']
    finally:
        shutil.rmtree(dn_tmp)


def test_spectrum_online():
    for bsize in 2, 4, 5:
        print 'BSIZE', bsize
        # Setup a test FF
        ff = get_ff_water32()
        # Run a test simulation
        f = h5py.File('tmp%i.h5' % bsize, driver='core', backing_store=False)
        hdf5 = HDF5Writer(f)
        spectrum0 = Spectrum(f, bsize=bsize)
        nve = NVEIntegrator(ff, 1.0*femtosecond, hooks=[hdf5, spectrum0])
        nve.run(5)
        assert nve.counter == 5
        # Also run an off-line spectrum and compare
        spectrum1 = Spectrum(f, bsize=bsize)
        assert abs(spectrum0.timestep - spectrum1.timestep) < 1e-10
        print spectrum0.amps
        print spectrum1.amps
        assert abs(spectrum0.amps - spectrum1.amps).max() < 1e-10
        assert abs(spectrum0.freqs - spectrum1.freqs).max() < 1e-10
        assert abs(spectrum0.ac - spectrum1.ac).max() < 1e-10
        assert abs(spectrum0.time - spectrum1.time).max() < 1e-10