# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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



import numpy as np

from molmod import boltzmann, femtosecond

from yaff.log import log, timer
from yaff.sampling.iterative import Iterative, StateItem, AttributeStateItem, \
    PosStateItem, DipoleStateItem, DipoleVelStateItem, VolumeStateItem, \
    CellStateItem, EPotContribStateItem, Hook
from yaff.sampling.md import MDScreenLog, ConsErrTracker, get_random_vel, remove_com_vel


__all__ = [
    'NHCNVTIntegrator', 'LNVTIntegrator',
]

# TODO: eliminate common code (also with NVE), possible with more hook types, e.g. to combine different types of thermostats

class NHChain(object):
    # TODO: add citations to nose, hoover and martyna
    def __init__(self, length, timestep, temp, ndof, timecon=100*femtosecond):
        # parameters
        self.length = length
        self.timestep = timestep
        self.temp = temp
        self.ndof = ndof

        # set the masses according to the time constant
        angfreq = 2*np.pi/timecon
        self.masses = np.ones(length)*(boltzmann*temp/angfreq**2)
        self.masses[0] *= ndof
        print 'masses'
        print self.masses

        # allocate degrees of freedom
        self.pos = np.zeros(length)
        self.vel = np.zeros(length) # TODO: sensible random initial velocities?

    def __call__(self, ekin, vel):
        def do_bead(k, ekin):
            # Compute g
            if k == 0:
                # coupling with atoms
                # L = 3N because of equidistant time steps.
                g = 2*ekin - self.ndof*self.temp*boltzmann
            else:
                # coupling between beads
                g = self.masses[k-1]*self.vel[k-1]**2 - self.temp*boltzmann
            g /= self.masses[k]

            # Lioville operators on relevant part of the chain
            if k == self.length-1:
                # iL G_k h/4
                self.vel[k] += g*self.timestep/4
            else:
                # iL vxi_{k-1} h/8
                self.vel[k] *= np.exp(-self.vel[k+1]*self.timestep/8)
                # iL G_k h/4
                self.vel[k] += g*self.timestep/4
                # iL vxi_{k-1} h/8
                #print self.vel[k], self.vel[k+1], self.timestep
                self.vel[k] *= np.exp(-self.vel[k+1]*self.timestep/8)

        # Loop over chain in reverse order
        for k in xrange(self.length-1, -1, -1):
            do_bead(k, ekin)

        # iL xi (all) h/2
        self.pos += self.vel*self.timestep/2

        # iL Cv (all) h/2
        factor = np.exp(-self.vel[0]*self.timestep/2)
        vel *= factor
        ekin *= factor**2

        # Loop over chain in forward order
        for k in xrange(0, self.length):
            do_bead(k, ekin)

        return ekin

    def get_econs_contrib(self):
        kt = boltzmann*self.temp
        return 0.5*(self.vel**2*self.masses).sum() + kt*(self.ndof*self.pos[0] + self.pos[1:].sum())


class NHCNVTIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        PosStateItem(),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        AttributeStateItem('temp'),
        AttributeStateItem('etot'),
        AttributeStateItem('econs'),
        AttributeStateItem('cons_err'),
        AttributeStateItem('ptens'),
        AttributeStateItem('press'),
        DipoleStateItem(),
        DipoleVelStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]

    log_name = 'NHCNVT'

    def __init__(self, ff, timestep, state=None, hooks=None, vel0=None,
                 temp=300, chainlength=3, timecon=100*femtosecond,
                 scalevel0=True, time0=0.0, counter0=0):
        """
           **Arguments:**

           ff
                A ForceField instance

           timestep
                The integration time step (in atomic units)

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           vel0
                An array with initial velocities. If not given, random
                velocities are sampled from the Maxwell-Boltzmann distribution
                corresponding to the optional arguments temp0 and scalevel0

           temp
                The temperature for the random initial velocities and the
                thermostat.

           chainlength
                The number of beads in the Nose-Hoover chain.

           timecon
                The time constant of the Nose-Hoover thermostat.

           scalevel0
                If True (the default), the random velocities are rescaled such
                that the instantaneous temperature coincides with temp0.

           time0
                The time associated with the initial state.

           counter0
                The counter value associated with the initial state.
        """
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses
        if vel0 is None:
            self.vel = get_random_vel(temp, scalevel0, self.masses)
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        remove_com_vel(self.vel, self.masses)
        self.temp_bath = temp
        # TODO: allow for manual override of ndof in case of constraints.
        self.chain = NHChain(chainlength, timestep, temp, self.pos.size, timecon)

        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        self.vtens = np.zeros((3, 3), float)
        # econs_ref should be changed by hooks that change positions or
        # velocities, such that a conserved quantity can be computed.
        self.econs_ref = 0
        # cons_err is an object that keeps track of the error on the conserved
        # quantity.
        self._cons_err_tracker = ConsErrTracker()
        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, MDScreenLog) for hook in self.hooks):
            self.hooks.append(MDScreenLog())

    def initialize(self):
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.ekin = 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()
        self.compute_properties()
        Iterative.initialize(self)

    def propagate(self):
        # call the chain
        self.ekin = self.chain(self.ekin, self.vel)

        # regular verlet step
        self.delta[:] = self.timestep*self.vel + (0.5*self.timestep**2)*self.acc
        self.pos += self.delta
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)
        acc = -self.gpos/self.masses.reshape(-1,1) # new acceleration
        self.vel += 0.5*(acc+self.acc)*self.timestep
        self.ekin = 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()

        # call the chain
        self.ekin = self.chain(self.ekin, self.vel)

        self.acc = acc # override old acc by new
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self)

    def compute_properties(self):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.temp = self.ekin/self.ff.system.natom/3.0*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot + self.econs_ref + self.chain.get_econs_contrib()
        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()
        self.ptens = (np.identity(3)*(boltzmann*self.temp*self.masses.size) + self.vtens)/self.ff.system.cell.volume
        self.press = np.trace(self.ptens)/3

    def finalize(self):
        if log.do_medium:
            log.hline()



class LNVTIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        PosStateItem(),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        AttributeStateItem('temp'),
        AttributeStateItem('etot'),
        AttributeStateItem('econs'),
        AttributeStateItem('cons_err'),
        AttributeStateItem('ptens'),
        AttributeStateItem('press'),
        DipoleStateItem(),
        DipoleVelStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]

    log_name = 'LNVT'
    # TODO: cite Phys. Rev. E 75, 056707 (2007)

    def __init__(self, ff, timestep, state=None, hooks=None, vel0=None,
                 temp=300, timecon=100*femtosecond, scalevel0=True, time0=0.0,
                 counter0=0):
        """
           **Arguments:**

           ff
                A ForceField instance

           timestep
                The integration time step (in atomic units)

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           vel0
                An array with initial velocities. If not given, random
                velocities are sampled from the Maxwell-Boltzmann distribution
                corresponding to the optional arguments temp0 and scalevel0

           temp
                The temperature for the random initial velocities and the
                heat bath.

           timecon
                The timeconstant of the Langevin integrator (1/gamma)

           scalevel0
                If True (the default), the random velocities are rescaled such
                that the instantaneous temperature coincides with temp0.

           time0
                The time associated with the initial state.

           counter0
                The counter value associated with the initial state.
        """
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses
        if vel0 is None:
            self.vel = get_random_vel(temp, scalevel0, self.masses)
            remove_com_vel(self.vel, self.masses)
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        self.temp_bath = temp
        self.timecon = timecon

        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        self.vtens = np.zeros((3, 3), float)
        # econs_ref should be changed by hooks that change positions or
        # velocities, such that a conserved quantity can be computed.
        self.econs_ref = 0
        # cons_err is an object that keeps track of the error on the conserved
        # quantity.
        self._cons_err_tracker = ConsErrTracker()
        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, MDScreenLog) for hook in self.hooks):
            self.hooks.append(MDScreenLog())

    def initialize(self):
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.compute_properties()
        Iterative.initialize(self)

    def thermo(self):
        c1 = np.exp(-self.timestep/self.timecon/2)
        c2 = np.sqrt((1.0-c1**2)*self.temp_bath*boltzmann/self.masses).reshape(-1,1)
        self.econs_ref += 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()
        self.vel = c1*self.vel + c2*np.random.normal(0, 1, self.vel.shape)
        self.econs_ref -= 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()

    def propagate(self):
        self.thermo()

        self.delta[:] = self.timestep*self.vel + (0.5*self.timestep**2)*self.acc
        self.pos += self.delta
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)
        acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*(acc+self.acc)*self.timestep
        self.acc = acc
        self.time += self.timestep

        self.thermo()

        self.compute_properties()
        Iterative.propagate(self)

    def compute_properties(self):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.ekin = 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()
        self.temp = self.ekin/self.ff.system.natom/3.0*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot + self.econs_ref
        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()
        self.ptens = (np.identity(3)*(boltzmann*self.temp*self.masses.size) + self.vtens)/self.ff.system.cell.volume
        self.press = np.trace(self.ptens)/3

    def finalize(self):
        if log.do_medium:
            log.hline()
