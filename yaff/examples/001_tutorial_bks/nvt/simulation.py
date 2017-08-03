#!/usr/bin/env python
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
#!/usr/bin/env python

import numpy as np
import h5py as h5
import sys # Needed for command-line parsing

from yaff import *


# Parse the command line arguments. Don't touch.
args = sys.argv[1:]
assert len(args) == 2
temp = int(args[0])
nstep = int(args[1])
suffix = '%04i_%06i.h5' % (temp, nstep)


# Load system
system = System.from_file('../opt/opt.chk')

# Define force field
ff = ForceField.generate(system, '../bks.pot', rcut=12*angstrom, smooth_ei=True, reci_ei='ewald')

# Output to files and screen
f = h5.File('traj_%s.h5' % suffix, mode='w')
hdf5 = HDF5Writer(f)
xyz = XYZWriter('traj_%s.xyz' % suffix, step=1)
vsl = VerletScreenLog(step=10)

# Pick your thermostat:
#thermo = AndersenThermostat(300, step=10)
thermo = NHCThermostat(300, timecon=100*femtosecond)
#thermo = LangevinThermostat(300, timecon=100*femtosecond)
verlet = VerletIntegrator(ff, femtosecond, temp0=600, hooks=[thermo, xyz, hdf5, vsl])
verlet.run(nstep)

# Plot some results
plot_energies(f, 'ener_%s.png' % suffix)
plot_temperature(f, 'temp_%s.png' % suffix)
plot_pressure(f, 'press_%s.png' % suffix)

# Close the HDF5 file.
f.close()
