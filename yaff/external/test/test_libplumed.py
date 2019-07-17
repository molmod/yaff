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

import os

import numpy as np

from molmod.test.common import tmpdir
from molmod.units import *
from yaff import *

from yaff.pes.test.common import check_gpos_part, check_vtens_part
from yaff.test.common import get_system_quartz, get_system_peroxide
from yaff.sampling.test.common import get_ff_water32


# The following is included, because otherwise Travis complains about
# incompatible OpenMP libraries, even though tests are not run with OpenMP
# parallelization. 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

try:
    from plumed import Plumed
except:
    from nose.plugins.skip import SkipTest
    raise SkipTest('Could not import PLUMED, skipping all PLUMED related tests')



def check_plumed(system, commands, reference):
    with tmpdir(__name__, 'check_plumed') as dirname:
        # Write PLUMED commands to file
        fn = os.path.join(dirname, 'plumed.dat')
        with open(fn,'w') as f:
            f.write(commands)
        # Setup Plumed
        plumed = ForcePartPlumed(system, fn=fn)
        # Compare with direct calculation
        e = plumed.compute()
        eref = reference(system)
        check_gpos_part(system, plumed)
        check_vtens_part(system, plumed)
        assert np.abs(e-eref)<1e-3*kjmol


def test_plumed_quartz_volume():
    system = get_system_quartz()
    # Harmonic restraint of the volume
    kappa, V0 = 1.6*kjmol/angstrom**6, 110*angstrom**3
    # PLUMED input commands
    commands = "vol: VOLUME\n"
    commands += "RESTRAINT ARG=vol AT=%.20f KAPPA=%.20f LABEL=restraint\n"%\
        (V0/nanometer**3,kappa/kjmol*nanometer**6)
    # Reference calculation
    def reference(system):
        return 0.5*kappa*(system.cell.volume-V0)**2
    check_plumed(system, commands, reference)


def test_plumed_quartz_cell():
    system = get_system_quartz()
    # Harmonic restraints of the cell vector norms
    kappa, a0, b0, c0 = 2.0,90.0,96.0,92.0,
    # PLUMED input commands
    commands = "cell: CELL\n"
    commands += "aaa:    COMBINE ARG=cell.ax,cell.ay,cell.az POWERS=2,2,2 PERIODIC=NO\n"
    commands += "bbb:    COMBINE ARG=cell.bx,cell.by,cell.bz POWERS=2,2,2 PERIODIC=NO\n"
    commands += "ccc:    COMBINE ARG=cell.cx,cell.cy,cell.cz POWERS=2,2,2 PERIODIC=NO\n"
    commands += "RESTRAINT ARG=aaa,bbb,ccc AT=%.20f,%.20f,%.20f "%\
        (a0/nanometer**2, b0/nanometer**2, c0/nanometer**2)
    commands += "KAPPA=%.20f,%.20f,%.20f LABEL=restraint\n"%\
        (kappa/kjmol*nanometer**4, kappa/kjmol*nanometer**4, kappa/kjmol*nanometer**4)
    commands += "PRINT ARG=aaa,bbb,ccc\n"
    # Reference calculation
    def reference(system):
        a = np.dot(system.cell.rvecs[0], system.cell.rvecs[0])
        b = np.dot(system.cell.rvecs[1], system.cell.rvecs[1])
        c = np.dot(system.cell.rvecs[2], system.cell.rvecs[2])
        return 0.5*kappa*( (a-a0)**2+(b-b0)**2+(c-c0)**2 )
    check_plumed(system, commands, reference)


def test_plumed_peroxide_bond():
    system = get_system_peroxide()
    # Linear restraint of the O-O bond
    m, a0 = 2.3*kjmol/angstrom, 0.9*angstrom
    # PLUMED input commands, remember that PLUMED starts counting atoms from 1
    commands = "d: DISTANCE ATOMS=1,2\n"
    commands += "RESTRAINT ARG=d AT=%.20f SLOPE=%.20f LABEL=restraint\n"%\
        (a0/nanometer, m/kjmol*nanometer)
    # Reference calculation:
    def reference(system):
        d = np.linalg.norm(system.pos[1]-system.pos[0])
        return m*(d-a0)
    check_plumed(system, commands, reference)


def test_plumed_md():
    from nose.plugins.skip import SkipTest
    raise SkipTest('The PLUMED interface does not handle multiple bias '
                   'calculation within one time step correctly, see '
                   'discussion on the PLUMED forum')
    with tmpdir(__name__, 'check_plumed') as dirname:
        ff = get_ff_water32()
        kappa, V0 = 1.6*kjmol/angstrom**6, ff.system.cell.volume
        # PLUMED input commands
        commands = "vol: VOLUME\n"
        commands += "RESTRAINT ARG=vol AT=%.20f KAPPA=%.20f LABEL=restraint\n"%\
            (V0/nanometer**3,kappa/kjmol*nanometer**6)
        commands += "PRINT STRIDE=1 ARG=vol FILE=%s\n"%(os.path.join(dirname,'cv.log'))
        commands += "FLUSH STRIDE=1\n"
        # Write PLUMED commands to file
        fn = os.path.join(dirname, 'plumed.dat')
        with open(fn,'w') as f:
            f.write(commands)
        # Setup Plumed
        timestep = 1.0*femtosecond
        plumed = ForcePartPlumed(ff.system, timestep=timestep, fn=fn)
        ff.add_part(plumed)
        # Setup integrator with a barostat, so plumed has to compute forces
        # more than once per timestep
        tbc = TBCombination(MTKBarostat(ff, 300, 1*bar), NHCThermostat(300))
        verlet = VerletIntegrator(ff, timestep, hooks=[tbc])
        # Run a short MD simulation, keeping track of the CV (volume in this case)
        cvref = [ff.system.cell.volume]
        for i in range(4):
            verlet.run(1)
            cvref.append(ff.system.cell.volume)
        # Read the PLUMED output and compare with reference
        cv = np.loadtxt(os.path.join(dirname,'cv.log'))
        assert cv.shape[0]==5
        assert np.allclose(cv[:,0]*picosecond, np.arange(5)*timestep)
        assert np.allclose(cv[:,1]*nanometer**3, cvref)
