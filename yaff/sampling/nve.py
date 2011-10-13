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
from yaff.sampling.iterative import Iterative, StateItem, AttributeStateItem, \
    Hook


__all__ = [
    'NVEScreenLog', 'AndersenThermostat', 'NVEIntegrator',
    'KineticAnnealing'
]


class NVEScreenLog(Hook):
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.ref_econs = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.ref_econs is None:
                self.ref_econs = iterative.econs
                if log.do_medium:
                    log.hline()
                    log('counter  Rel.Econs       Temp     d-RMSD     g-RMSD')
                    log.hline()
            log('%7i %s %s %s %s' % (
                iterative.counter,
                log.energy(iterative.econs - self.ref_econs),
                log.temperature(iterative.temp),
                log.length(iterative.rmsd_delta),
                log.force(iterative.rmsd_gpos),
            ))


class AndersenThermostat(Hook):
    def __init__(self, temp, start=0, step=1, mask=None, annealing=1.0):
        """
           **Arguments:**

           temp
                The average temperature if the NVT ensemble

           **Optional arguments:**

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

           mask
                An array mask to indicate which atoms controlled by the
                thermostat.

           annealing
                After every call to this hook, the temperature is multiplied
                with this annealing factor. This effectively cools down the
                system.
        """
        self.temp = temp
        self.mask = mask
        self.annealing = annealing
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        # Change the velocities
        if self.mask is None:
            iterative.vel[:] = iterative.get_random_vel(self.temp, False)
        else:
            iterative.vel[self.mask] = iterative.get_random_vel(self.temp, False)[self.mask]
        # Update the kinetic energy and the reference for the conserved quantity
        ekin_after = 0.5*(iterative.vel**2*iterative.masses.reshape(-1,1)).sum()
        iterative.econs_ref += iterative.ekin - ekin_after
        iterative.ekin = ekin_after
        # Optional annealing
        self.temp *= self.annealing


class KineticAnnealing(Hook):
    def __init__(self, annealing=0.99999, select=None, start=0, step=1):
        """
           This annealing hook is designed to be used with a plain NVE
           integrator. At every call, the velocities are rescaled with
           the annealing parameter.

           **Arguments:**

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

           annealing
                After every call to this hook, the temperature is multiplied
                with this annealing factor. This effectively cools down the
                system.

           select
                An array mask or a list of indexes to indicate which
                atomic velocities should be annealed.

        """
        self.annealing = annealing
        self.select = select
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        # Change the velocities
        if self.select is None:
            iterative.vel[:] *= self.annealing
        else:
            iterative.vel[self.select] *= self.annealing
        # Update the kinetic energy and the reference for the conserved quantity
        ekin_after = 0.5*(iterative.vel**2*iterative.masses.reshape(-1,1)).sum()
        iterative.econs_ref += iterative.ekin - ekin_after
        iterative.ekin = ekin_after


class NVEIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        AttributeStateItem('pos'),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        AttributeStateItem('temp'),
        AttributeStateItem('etot'),
        AttributeStateItem('econs'),
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
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses
        if vel0 is None:
            self.vel = self.get_random_vel(temp0, scalevel0)
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        # econs_ref should be changed by hooks that change positions or
        # velocities, such that a conserved quantity can be computed.
        self.econs_ref = 0
        Iterative.__init__(self, ff, state, hooks, counter0)
        if not any(isinstance(hook, NVEScreenLog) for hook in self.hooks):
            self.hooks.append(NVEScreenLog())

    def get_random_vel(self, temp0, scalevel0):
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
        self.compute_properties()
        Iterative.initialize(self)

    def propagate(self):
        self.delta[:] = self.timestep*self.vel + (0.5*self.timestep**2)*self.acc
        self.pos += self.delta
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.epot = self.ff.compute(self.gpos)
        acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*(acc+self.acc)*self.timestep
        self.acc = acc
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self)

    def compute_properties(self):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.ekin = 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()
        self.temp = self.ekin/self.ff.system.natom/3.0*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot + self.econs_ref

    def finalize(self):
        if log.do_medium:
            log.hline()
