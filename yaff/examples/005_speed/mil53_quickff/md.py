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
from __future__ import print_function

import numpy as np
np.random.seed(3)
import pkg_resources
import os, sys
import time


from yaff import System, log
from yaff.pes import ForceField
from yaff.external import swap_noncovalent_lammps
from yaff.sampling import VerletScreenLog, NHCThermostat, VerletIntegrator

from molmod.units import angstrom, femtosecond

from mpi4py import MPI

# Setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# Turn off logging for all processes, it can be turned on for one selected process later on
log.set_level(log.silent)
if rank==0: log.set_level(log.medium)

def load_ff(uselammps=False,supercell=(1,1,1),overwrite_table=False):
    # Load the system
    system = System.from_file('system.chk').supercell(supercell[0],
        supercell[1],supercell[2])
    # Generate the YAFF ForceField
    ff = ForceField.generate(system, 'pars.txt', alpha_scale=3.2,
        gcut_scale=1.5, rcut=15.0*angstrom, smooth_ei=True)
    if uselammps:
        fn_system = 'system_%s.dat'%".".join(["%d"%s for s in supercell])
        fn_table = 'table.dat'
        ff_lammps = swap_noncovalent_lammps(ff, fn_system=fn_system,
             fn_table=fn_table, overwrite_table=overwrite_table, comm=comm)
        return ff_lammps
    else:
        return ff

def load_integrator(ff, T=300.0):
    vsl = VerletScreenLog(step=10)
    thermo = NHCThermostat(T, timecon=100*femtosecond)
    verlet = VerletIntegrator(ff, 1.0*femtosecond, temp0=2*T, hooks=[thermo, vsl])
    return verlet

if __name__=='__main__':
    if sys.argv[1]=='tabulate':
        # Write the table of noncovalent interactions, only needs to be done for
        # one supercell size
        ff = load_ff(uselammps=True,supercell=(1,1,1),overwrite_table=write_table)
    else:
        supercell = [int(s) for s in sys.argv[2].split('.')]
        nsteps = int(sys.argv[3])
        ff = load_ff(uselammps=sys.argv[1]=='liblammps',supercell=supercell)
        verlet = load_integrator(ff)
        start = time.time()
        verlet.run(nsteps)
        end = time.time()
        print("SUMMARY %s %s %d %d %f" % (sys.argv[1],sys.argv[2],ff.system.natom,nprocs,end-start))
