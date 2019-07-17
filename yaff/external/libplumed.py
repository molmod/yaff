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
'''libplumed

   This module provides an interface to PLUMED, a library that includes
   enhanced sampling algorithms and free-energy methods.
'''


import numpy as np

from molmod.units import kjmol, nanometer, picosecond

from yaff.log import log, timer
from yaff.pes.ff import ForcePart

__all__ = ['ForcePartPlumed']

class ForcePartPlumed(ForcePart):
    '''Biasing energies computed by PLUMED'''
    def __init__(self, system, timestep=0.0, restart=0,
                       fn='plumed.dat', kernel=None, fn_log='plumed.log'):
        r'''Initialize a PLUMED ForcePart. More information on the interface
            between PLUMED and MD codes can be found on
            http://tcb.ucas.ac.cn/plumed2/developer-doc/html/_how_to_plumed_your_m_d.html

            Unfortunately, PLUMED partially breaks the orthogonality of the pes
            and the sampling modules in Yaff: PLUMED sometimes requires
            information from the integrator. For example, in metadynamics
            PLUMED needs to know when a time integration step has been
            completed (which is not necessarily after each force calculation).
            ForcePartPlumed should therefore also inherit from Hook and
            by attaching this hook to the integrator, it is possible to obtain
            the necessary information from the integrator. Currently, there are
            some problems within PLUMED for this approach to work, particularly
            in the VES module, see the discussion at
            https://groups.google.com/forum/?fromgroups=#!topic/plumed-users/kPZu_tNZtgk

            **Arguments:**

            system
                An instance of the System class

            **Optional Arguments:**

            timestep
                The timestep (in au) of the integrator

            restart
                Set to a value different from 0 to let PLUMED know that this
                is a restarted run

            fn
                A filename from which the PLUMED instructions are read, default
                is plumed.dat

            kernel
                Path to the PLUMED library (something like
                /path/to/libplumedKernel.so). If None is provided, the
                environment variable $PLUMED_KERNEL should point to the library
                file.

            fn_log
                Path to the file where PLUMED logs output, default is
                plumed.log
        '''
        self.system = system
        self.fn = fn
        self.kernel = kernel
        self.fn_log = fn_log
        self.plumedstep = 0
        # TODO In the LAMMPS-PLUMED interface (src/USER-PLUMED/fix_plumed.cpp)
        # it is mentioned that biasing is not possible when tailcorrections are
        # included. Maybe this should need to be checked...
        # Check cell dimensions, only 0D and 3D systems supported
        if not self.system.cell.nvec in [0,3]:
            raise NotImplementedError
        # Setup PLUMED by sending commands to the PLUMED API
        self.setup_plumed(timestep, restart)
        # PLUMED requires masses to be set...
        if self.system.masses is None:
            self.system.set_standard_masses()
        # Initialize the ForcePart
        ForcePart.__init__(self, 'plumed', self.system)
        if log.do_warning:
            log.warn("When using PLUMED as a hook for your integrator "
                     "and PLUMED adds time-dependent forces (for instance "
                     "when performing metadynamics), there is no energy "
                     "conservation. The conserved quantity reported by "
                     "YAFF is irrelevant in this case.")
            log.warn("The PLUMED time step is incremented every time the bias "
                     "is computed, irrespective of whether the integrator "
                     "actually completed an entire time step. For instance "
                     "when performing metadynamics or VES with the MTKBarostat,"
                     " this can be unwanted. In such cases, the easiest "
                     "solution is to adapt the PACE and STRIDE keywords in the "
                     "PLUMED input file, to take into that forces are "
                     "calculated three times during each time step.")

    def setup_plumed(self, timestep, restart):
        r'''Send commands to PLUMED to make it computation-ready.

            **Arguments:**

            timestep
                The timestep (in au) of the integrator

            restart
                Set to a value different from 0 to let PLUMED know that this
                is a restarted run
        '''
        # Try to load the plumed Python wrapper, quit if not possible
        try:
            from plumed import Plumed
        except:
            log("Could not import the PLUMED python wrapper!")
            raise ImportError
        self.plumed = Plumed(kernel=self.kernel)
        # Conversion between PLUMED internal units and YAFF internal units
        # Note that PLUMED output will follow the PLUMED conventions
        # concerning units
        self.plumed.cmd("setMDEnergyUnits", 1.0/kjmol)
        self.plumed.cmd("setMDLengthUnits", 1.0/nanometer)
        self.plumed.cmd("setMDTimeUnits", 1.0/picosecond)
        # Initialize the system in PLUMED
        self.plumed.cmd("setPlumedDat", self.fn)
        self.plumed.cmd("setNatoms", self.system.natom)
        self.plumed.cmd("setMDEngine", "YAFF")
        self.plumed.cmd("setLogFile", self.fn_log)
        self.plumed.cmd("setTimestep", timestep)
        self.plumed.cmd("setRestart", restart)
        self.plumed.cmd("init")


    def _internal_compute(self, gpos, vtens):
        with timer.section('PLUMED'):
            self.plumed.cmd("setStep", self.plumedstep)
            self.plumed.cmd("setPositions", self.system.pos)
            self.plumed.cmd("setMasses", self.system.masses)
            if self.system.charges is not None:
                self.plumed.cmd("setCharges", self.system.charges)
            if self.system.cell.nvec>0:
                rvecs = self.system.cell.rvecs.copy()
                self.plumed.cmd("setBox", rvecs)
            # PLUMED provides a setEnergy command, which should pass the
            # current potential energy. It seems that this is never used, so we
            # don't pass anything for the moment.
            # current_energy = sum([part.energy for part in iterative.ff.parts[:-1] if not isinstance(part, ForcePartPlumed)])
            # self.plumed.cmd("setEnergy", current_energy)
            # PLUMED always needs arrays to write forces and virial to, so
            # provide dummy arrays if Yaff does not provide them
            # Note that gpos and forces differ by a minus sign, which has to be
            # corrected for when interacting with PLUMED
            if gpos is None:
                my_gpos = np.zeros(self.system.pos.shape)
            else:
                gpos[:] *= -1.0
                my_gpos = gpos
            self.plumed.cmd("setForces", my_gpos)
            if vtens is None:
                my_vtens = np.zeros((3,3))
            else: my_vtens = vtens
            self.plumed.cmd("setVirial", my_vtens)
            # Do the actual calculation, including an update; this should
            # actually only be done at the end of a time step, but for now,
            # this is not checked
            self.plumed.cmd("prepareCalc")
            self.plumed.cmd("performCalcNoUpdate")
            self.plumed.cmd("update")
            if gpos is not None:
                gpos[:] *= -1.0
            # Retrieve biasing energy
            energy = np.zeros((1,))
            self.plumed.cmd("getBias",energy)
            self.plumedstep += 1
            return energy[0]
