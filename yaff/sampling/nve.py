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

from molmod.periodic import periodic
from molmod import boltzmann

from yaff.sampling.state import StateItem, AttributeStateItem


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



nve_minimal_state = [
    AttributeStateItem('epot'),
    AttributeStateItem('time'),
    AttributeStateItem('pos'),
]

nve_normal_state = nve_minimal_state + [
    AttributeStateItem('vel'),
    EKinStateItem(),
    TempStateItem(),
]


class NVEIntegrator(object):
    def __init__(self, ff, timestep, state=None, hooks=None, masses=None, vel0=None, temp0=300, scalevel0=True, time0=0.0):
        """
           **Arguments:**

           ff
                A ForceField instance

           timestep
                The integration time step (in atomic units)

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take are derive a property from the current state of the
                integrator.

           hooks
                A function (or a list of functions) that is called after every
                time step.

           masses
                An array with atomic masses (in atomic units). If not given,
                the standard masses are taken based on the atomic numbers in
                ff.system.numbers

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
                The time associated with the initial structure.
        """
        self.ff = ff
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        if state is None:
            self.state_list = list(nve_normal_state)
        else:
            self.state_list = state
        self.state = dict((item.key, item) for item in self.state_list)
        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, '__length__'):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        if masses is None:
            self.masses = self.get_standard_masses(ff.system)
        else:
            self.masses = masses
        if vel0 is None:
            self.vel = self.get_initial_vel(temp0, scalevel0)
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        self.gpos = np.zeros(self.pos.shape, float)
        self.init_verlet()

    def get_standard_masses(self, system):
        return np.array([periodic[n].mass for n in system.numbers])

    def get_initial_vel(self, temp0, scalevel0):
        result = np.random.normal(0, 1, self.pos.shape)*np.sqrt(boltzmann*temp0/self.masses).reshape(-1,1)
        if scalevel0:
            temp = (result**2*self.masses.reshape(-1,1)).mean()/boltzmann
            scale = np.sqrt(temp0/temp)
            result *= scale
        return result

    def init_verlet(self):
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.call_hooks()

    def call_hooks(self):
        for item in self.state_list:
            item.update(self)
        for hook in self.hooks:
            hook(self.ff, self.state)

    def run(self, nstep):
        for i in xrange(nstep):
            self.prop_verlet()

    def prop_verlet(self):
        self.pos += self.timestep*self.vel + (0.5*self.timestep**2)*self.acc
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*(acc+self.acc)*self.timestep
        self.acc = acc
        self.time += self.timestep
        self.call_hooks()
