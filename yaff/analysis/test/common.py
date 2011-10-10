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


import tempfile, h5py

from yaff import *
from yaff.pes.test.common import get_system_water32


def get_water_32_simulation():
    # Make a temporary directory
    dn_tmp = tempfile.mkdtemp(suffix='yaff', prefix='water_32')
    # Setup a test FF
    system = get_system_water32()
    system.charges[:] = 0.0
    ff = ForceField.generate(system, 'input/parameters_water.txt')
    # Run a test simulation
    f = h5py.File('%s/output.h5' % dn_tmp)
    hdf5_hook = HDF5Writer(f)
    nve = NVEIntegrator(ff, 1.0*femtosecond, hooks=hdf5_hook)
    nve.run(5)
    assert nve.counter == 5
    return dn_tmp, nve, hdf5_hook.f
