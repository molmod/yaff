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


import numpy as np, tempfile, h5py as h5, shutil
from subprocess import PIPE, Popen, call

from yaff import *
from yaff.sampling.test.common import get_ff_water32

def test_h5_flush():
    # Test if we can read intermediate hdf5 files that are flushed
    # This test relies on the `h5ls` command being  available in the command line.
    # Make a temporary directory
    dn_tmp = tempfile.mkdtemp(suffix='yaff', prefix='nve_water_32')
    # Setup a test FF
    ff = get_ff_water32()
    # Make two different hdf5 hooks
    f = h5.File('%s/output.h5' % dn_tmp)
    hdf5 = HDF5Writer(f)
    f_flushed = h5.File('%s/output_flushed.h5' % dn_tmp)
    hdf5_flushed = HDF5Writer(f_flushed,flush=4)
    # Run a test simulation
    nve = VerletIntegrator(ff, 1.0*femtosecond, hooks=[hdf5,hdf5_flushed])
    nve.run(5)
    try: # Actual test
        # Check that we can read the h5 files from the command line
        # This should not work
        p = Popen(["h5ls", '%s/output.h5' % dn_tmp], stdout=PIPE, stderr=PIPE)
        result = p.communicate()
        assert 'unable' in result[1]
        # This should work
        p = Popen(["h5ls", '%s/output_flushed.h5' % dn_tmp], stdout=PIPE, stderr=PIPE)
        result = p.communicate()
        assert 'trajectory' in result[0]
    finally: # Cleanup
        f.close()
        f_flushed.close()
        shutil.rmtree(dn_tmp)
