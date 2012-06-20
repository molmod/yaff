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



import numpy as np, time

from molmod import boltzmann

from yaff.log import log, timer
from yaff.sampling.iterative import Iterative, StateItem, AttributeStateItem, \
    PosStateItem, DipoleStateItem, DipoleVelStateItem, VolumeStateItem, \
    CellStateItem, EPotContribStateItem, Hook


__all__ = [
    'NVEScreenLog', 'AndersenThermostat', 'AndersenThermostatMcDonaldBarostat',
    'KineticAnnealing', 'NVEIntegrator',
]


class NVEScreenLog(Hook):
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log('Cons.Err. =&the root of the ratio of the variance on the conserved quantity and the variance on the kinetic energy.')
                    log('d-rmsd    =&the root-mean-square displacement of the atoms.')
                    log('g-rmsd    =&the root-mean-square gradient of the energy.')
                    log('counter  Cons.Err.       Temp     d-RMSD     g-RMSD   Walltime')
                    log.hline()
            log('%7i %10.5f %s %s %s %10.1f' % (
                iterative.counter,
                iterative.cons_err,
                log.temperature(iterative.temp),
                log.length(iterative.rmsd_delta),
                log.force(iterative.rmsd_gpos),
                time.time() - self.time0,
            ))


class AndersenThermostat(Hook):
    def __init__(self, temp, start=0, step=1, select=None, annealing=1.0):
        """
           **Arguments:**

           temp
                The average temperature of the NVT ensemble

           **Optional arguments:**

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

           select
                An array of atom indexes to indicate which atoms controlled by
                the thermostat.

           annealing
                After every call to this hook, the temperature is multiplied
                with this annealing factor. This effectively cools down the
                system.
        """
        self.temp = temp
        self.select = select
        self.annealing = annealing
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        # Change the velocities
        if self.select is None:
            iterative.vel[:] = iterative.get_random_vel(self.temp, False)
        else:
            iterative.vel[self.select] = iterative.get_random_vel(self.temp, False, self.select)
        # Update the kinetic energy and the reference for the conserved quantity
        ekin_after = 0.5*(iterative.vel**2*iterative.masses.reshape(-1,1)).sum()
        iterative.econs_ref += iterative.ekin - ekin_after
        iterative.ekin = ekin_after
        # Optional annealing
        self.temp *= self.annealing


class AndersenThermostatMcDonaldBarostat(Hook):
    def __init__(self, temp, press, start=0, step=1, amp=1e-3):
        """
           **Arguments:**

           temp
                The average temperature of the NpT ensemble

           press
                The external pressure of the NpT ensemble

           **Optional arguments:**

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

           amp
                The amplitude of the changes in the logarithm of the volume.
        """
        self.temp = temp
        self.press = press
        self.amp = amp
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        def initialize():
            iterative.gpos[:] = 0.0
            iterative.ff.update_pos(iterative.pos)
            iterative.epot = iterative.ff.compute(iterative.gpos)
            iterative.acc = -iterative.gpos/iterative.masses.reshape(-1,1)

        with timer.section('ATMB'):
            # A) Change the logarithm of the volume isotropically.
            scale = np.exp(np.random.uniform(-self.amp, self.amp))
            # A.1) scale the system and recompute the energy
            vol0 = iterative.ff.system.cell.volume
            epot0 = iterative.epot
            rvecs0 = iterative.ff.system.cell.rvecs.copy()
            iterative.ff.update_rvecs(rvecs0*scale)
            pos0 = iterative.pos.copy()
            iterative.pos[:] = pos0*scale
            initialize()
            epot1 = iterative.epot
            vol1 = iterative.ff.system.cell.volume
            # A.2) compute the acceptance ratio
            beta = 1/(boltzmann*self.temp)
            natom = iterative.ff.system.natom
            arg = (epot1 - epot0 + self.press*(vol1 - vol0) - (natom+1)/beta*np.log(vol1/vol0))
            accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
            if accepted:
                # add a correction to the conserved quantity
                iterative.econs_ref += epot0 - epot1
            else:
                # revert the cell and the positions in the original state
                iterative.ff.update_rvecs(rvecs0)
                iterative.pos[:] = pos0
                # reinitialize the iterative algorithm
                initialize()
            # B) Change the velocities
            iterative.vel[:] = iterative.get_random_vel(self.temp, False)
            # C) Update the kinetic energy and the reference for the conserved quantity
            ekin0 = iterative.ekin
            ekin1 = 0.5*(iterative.vel**2*iterative.masses.reshape(-1,1)).sum()
            iterative.econs_ref += ekin0 - ekin1
            iterative.ekin = ekin1
            if log.do_medium:
                with log.section('ATMB'):
                    s = {True: 'accepted', False: 'rejected'}[accepted]
                    log('BARO   volscale %10.7f      arg %s      %s' % (scale, log.energy(arg), s))
                    if accepted:
                        log('BARO   energy change %s      (new vol)**(1/3) %s' % (
                            log.energy(epot1 - epot0), log.length(vol1**(1.0/3.0))
                        ))
                    log('THERMO energy change %s' % log.energy(ekin0 - ekin1))


class KineticAnnealing(Hook):
    def __init__(self, annealing=0.99999, select=None, start=0, step=1):
        """
           This annealing hook is designed to be used with a plain NVE
           integrator. At every call, the velocities are rescaled with
           the annealing parameter.

           **Arguments:**

           annealing
                After every call to this hook, the temperature is multiplied
                with this annealing factor. This effectively cools down the
                system.

           select
                An array mask or a list of indexes to indicate which
                atomic velocities should be annealed.

           start
                The first iteration at which this hook is called

           step
                The number of iterations between two subsequent calls to this
                hook.

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


class ConsErrTracker(object):
    def __init__(self):
        self.counter = 0
        self.ekin_sum = 0.0
        self.ekin_sumsq = 0.0
        self.econs_sum = 0.0
        self.econs_sumsq = 0.0

    def update(self, ekin, econs):
        self.counter += 1
        self.ekin_sum += ekin
        self.ekin_sumsq += ekin**2
        self.econs_sum += econs
        self.econs_sumsq += econs**2

    def get(self):
        if self.counter > 0:
            ekin_var = self.ekin_sumsq/self.counter - (self.ekin_sum/self.counter)**2
            if ekin_var > 0:
                econs_var = self.econs_sumsq/self.counter - (self.econs_sum/self.counter)**2
                return np.sqrt(econs_var/ekin_var)
        return 0.0


class NVEIntegrator(Iterative):
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
        DipoleStateItem(),
        DipoleVelStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
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
            self.remove_com_vel()
        else:
            if vel.shape != self.pos.shape:
                raise TypeError('The vel0 argument does not have the right shape.')
            self.vel = vel0.copy()
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        # econs_ref should be changed by hooks that change positions or
        # velocities, such that a conserved quantity can be computed.
        self.econs_ref = 0
        # cons_err is an object that keeps track of the error on the conserved
        # quantity.
        self._cons_err_tracker = ConsErrTracker()
        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, NVEScreenLog) for hook in self.hooks):
            self.hooks.append(NVEScreenLog())

    def get_random_vel(self, temp0, scalevel0, select=None):
        if select is None:
            masses = self.masses
            shape = self.pos.shape
        else:
            masses = self.masses[select]
            shape = (len(select), 3)
        result = np.random.normal(0, 1, shape)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
        if scalevel0 and temp0 > 0:
            temp = (result**2*masses.reshape(-1,1)).mean()/boltzmann
            scale = np.sqrt(temp0/temp)
            result *= scale
        return result

    def remove_com_vel(self):
        # compute the center of mass velocity
        com_vel = np.dot(self.masses, self.vel)/self.masses.sum()
        # subtract
        self.vel[:] -= com_vel

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
        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()

    def finalize(self):
        if log.do_medium:
            log.hline()
