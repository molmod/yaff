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
'''Generic Verlet integrator'''


import numpy as np, time

from molmod import boltzmann, kjmol

from math import factorial as fact

from yaff.log import log, timer
from yaff.sampling.iterative import Iterative, StateItem, AttributeStateItem, \
    PosStateItem, DipoleStateItem, DipoleVelStateItem, VolumeStateItem, \
    CellStateItem, EPotContribStateItem, Hook
from yaff.sampling.utils import get_random_vel

__all__ = [
    'VerletIntegrator', 'TemperatureStateItem', 'VerletHook', 'VerletScreenLog',
    'ConsErrTracker', 'KineticAnnealing'
]


class TemperatureStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'temp')

    def get_value(self, iterative):
        return getattr(iterative, 'temp', None)

    def iter_attrs(self, iterative):
        yield 'ndof', iterative.ndof


class VerletIntegrator(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('time'),
        AttributeStateItem('epot'),
        PosStateItem(),
        AttributeStateItem('vel'),
        AttributeStateItem('rmsd_delta'),
        AttributeStateItem('rmsd_gpos'),
        AttributeStateItem('ekin'),
        TemperatureStateItem(),
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

    log_name = 'VERLET'

    def __init__(self, ff, timestep, state=None, hooks=None, vel0=None,
                 temp0=300, scalevel0=True, time0=0.0, ndof=None, counter0=0):
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

           ndof
                When given, this option overrides the number of degrees of
                freedom determined from internal heuristics. When ndof is not
                given, its default value depends on the thermostat used. In most
                cases it is 3*natom, except for the NHC thermostat where the
                number if internal degrees of freedom is counted. The ndof
                attribute is used to derive the temperature from the kinetic
                energy.

           counter0
                The counter value associated with the initial state.
        """
        # Assign init arguments
        self.pos = ff.system.pos.copy()
        self.timestep = timestep
        self.time = time0
        self.ndof = ndof
        self.rvecs = ff.system.cell.rvecs.copy()

        # The integrator needs masses. If not available, take default values.
        if ff.system.masses is None:
            ff.system.set_standard_masses()
        self.masses = ff.system.masses

        # Set random initial velocities if needed.
        if vel0 is None:
            self.vel = get_random_vel(temp0, scalevel0, self.masses)
        else:
            self.vel = vel0.copy()

        # Working arrays
        self.gpos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)
        self.vtens = np.zeros((3, 3), float)

        # Tracks quality of the conserved quantity
        self._cons_err_tracker = ConsErrTracker()

        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, VerletScreenLog) for hook in self.hooks):
            self.hooks.append(VerletScreenLog())

    def initialize(self):
        # Standard initialization of Verlet algorithm
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        self.epot = self.ff.compute(self.gpos)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.posoud = self.pos.copy()

        # Allow for specialized initializations by the Verlet hooks.
        self.call_verlet_hooks('init')

        # Configure the number of degrees of freedom if needed
        if self.ndof is None:
            self.ndof = self.pos.size

        # Common post-processing of the initialization
        self.compute_properties()
        Iterative.initialize(self) # Includes calls to conventional hooks

    def propagate(self):
        # Allow specialized hooks to modify the state before the regular verlet
        # step.
        self.call_verlet_hooks('pre')

        # Regular verlet step
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.pos += self.timestep*self.vel
        self.ff.update_pos(self.pos)
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)
        self.acc = -self.gpos/self.masses.reshape(-1,1)
        self.vel += 0.5*self.acc*self.timestep
        self.ekin = self._compute_ekin()

        # Allow specialized verlet hooks to modify the state after the step
        self.call_verlet_hooks('post')

        self.posnieuw = self.pos.copy()
        self.delta[:] = self.posnieuw-self.posoud
        self.posoud[:] = self.posnieuw
        self.gpos[:] = 0.0
        self.vtens[:] = 0.0
        self.epot = self.ff.compute(self.gpos, self.vtens)

        # Common post-processing of a single step
        self.time += self.timestep
        self.compute_properties()
        Iterative.propagate(self) # Includes call to conventional hooks

    def _compute_ekin(self):
        '''Auxiliary routine to compute the kinetic energy

           This is used internally and often also by the Verlet hooks.
        '''
        return 0.5*(self.vel**2*self.masses.reshape(-1,1)).sum()

    def compute_properties(self):
        self.rmsd_gpos = np.sqrt((self.gpos**2).mean())
        self.rmsd_delta = np.sqrt((self.delta**2).mean())
        self.ekin = self._compute_ekin()
        self.temp = self.ekin/self.ndof*2.0/boltzmann
        self.etot = self.ekin + self.epot
        self.econs = self.etot
        for hook in self.hooks:
            if isinstance(hook, VerletHook):
                self.econs += hook.econs_correction
        self._cons_err_tracker.update(self.ekin, self.econs)
        self.cons_err = self._cons_err_tracker.get()
        if self.ff.system.cell.nvec > 0:
            self.ptens = (np.dot(self.vel.T*self.masses, self.vel) - self.vtens)/self.ff.system.cell.volume
            self.press = np.trace(self.ptens)/3

    def finalize(self):
        if log.do_medium:
            log.hline()

    def call_verlet_hooks(self, kind):
        # In this call, the state items are not updated. The pre and post calls
        # of the verlet hooks can rely on the specific implementation of the
        # VerletIntegrator and need not to rely on the generic state item
        # interface.
        with timer.section('%s special hooks' % self.log_name):
            for hook in self.hooks:
                if isinstance(hook, VerletHook) and hook.expects_call(self.counter):
                    if kind == 'init':
                        hook.init(self)
                    elif kind == 'pre':
                        hook.pre(self)
                    elif kind == 'post':
                        hook.post(self)
                    else:
                        raise NotImplementedError


class VerletHook(Hook):
    '''Specialized Verlet hook.

       This is mainly used for the implementation of thermostats and barostats.
    '''
    def __init__(self, start=0, step=1):
        """
           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.econs_correction = 0.0
        Hook.__init__(self, start=0, step=1)

    def __call__(self, iterative):
        pass

    def init(self, iterative):
        raise NotImplementedError

    def pre(self, iterative):
        raise NotImplementedError

    def post(self, iterative):
        raise NotImplementedError



class VerletScreenLog(Hook):
    '''A screen logger for the Verlet algorithm'''
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


class ConsErrTracker(object):
    '''A class that tracks the errors on the conserved quantity'''
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


class KineticAnnealing(VerletHook):
    def __init__(self, annealing=0.99999, select=None, start=0, step=1):
        """
           This annealing hook is designed to be used with a plain Verlet
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
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        pass

    def pre(self, iterative):
        pass

    def post(self, iterative):
        # Compute the kinetic energy before the annealing to correct the
        # conserved quantity/
        ekin_before = iterative._compute_ekin()
        # Change the velocities
        if self.select is None:
            iterative.vel[:] *= self.annealing
        else:
            iterative.vel[self.select] *= self.annealing
        # Update the correction for the conserved quantity
        ekin_after = iterative._compute_ekin()
        self.econs_correction += ekin_before - ekin_after
