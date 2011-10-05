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



import numpy as np

from molmod import boltzmann

from yaff.log import log
from yaff.sampling.iterative import Iterative, StateItem, AttributeStateItem


__all__ = ['NVEIntegrator']


class EKinStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'ekin')

    def get_value(self, sampler):
        return 0.5*(sampler.vel**2*sampler.masses.reshape(-1,1)).sum()


class TempStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'temp')

    def get_value(self, sampler):
        return (sampler.state['ekin'].value/sampler.ff.system.natom/3.0)



class NVEIntegrator(Iterative):
    minimal_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        AttributeStateItem('pos'),
    ]

    default_state = minimal_state + [
        AttributeStateItem('vel'),
        EKinStateItem(),
        TempStateItem(),
    ]

    log_name = 'NVE'

    def __init__(self, ff, timestep, state=None, hooks=None, vel0=None, temp0=300, scalevel0=True, time0=0.0, counter0=0):
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

           temp0
                The (initial) temperature for the random initial velocities

           scalevel0
                If True (the default), the random velocities are rescaled such
                that the instantaneous temperature coincides with temp0.

           time0
                The time associated with the initial state.

           counter0
                The counter value associated with the initial state.
        """
        self.ff = ff
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses
        if vel0 is None:
            self.vel = self.get_initial_vel(temp0, scalevel0)
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        self.gpos = np.zeros(self.pos.shape, float)
        Iterative.__init__(self, state, hooks, counter0)

    def get_initial_vel(self, temp0, scalevel0):
        result = np.random.normal(0, 1, self.pos.shape)*np.sqrt(boltzmann*temp0/self.masses).reshape(-1,1)
        if scalevel0:
            temp = (result**2*self.masses.reshape(-1,1)).mean()/boltzmann
            scale = np.sqrt(temp0/temp)
            result *= scale
        return result

    def initialize(self):
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        Iterative.initialize(self)
        if log.do_medium:
            log.hline()
            log('counter     Epot   d-RMSD   g-RMSD')
            log.hline()

    def propagate(self):
        delta = self.timestep*self.vel + (0.5*self.timestep**2)*self.acc
        self.pos += delta
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.epot = self.ff.compute(self.gpos)
        acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*(acc+self.acc)*self.timestep
        self.acc = acc
        self.time += self.timestep
        Iterative.propagate(self)
        if log.do_medium:
            # TODO: do this via the state items.
            # TODO: add conserved quantity
            # TODO: turn this into more meaningful numbers for error checking
            delta_rmsd = np.sqrt((delta**2).mean())
            gpos_rmsd = np.sqrt((self.gpos**2).mean())
            log('%7i % 8.1e % 8.1e % 8.1e' % (self.counter, self.epot/log.energy, delta_rmsd/log.length, gpos_rmsd/log.force))

    def finalize(self):
        if log.do_medium:
            log.hline()
