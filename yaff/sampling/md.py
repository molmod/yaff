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



from molmod import boltzmann

import numpy as np, time

from yaff.log import log

from yaff.sampling.iterative import Hook


__all__ = [
    'MDScreenLog', 'ConsErrTracker', 'get_random_vel', 'remove_com_vel',
]


class MDScreenLog(Hook):
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


def get_random_vel(temp0, scalevel0, masses, select=None):
    if select is not None:
        masses = masses[select]
    shape = len(masses), 3
    result = np.random.normal(0, 1, shape)*np.sqrt(boltzmann*temp0/masses).reshape(-1,1)
    if scalevel0 and temp0 > 0:
        temp = (result**2*masses.reshape(-1,1)).mean()/boltzmann
        scale = np.sqrt(temp0/temp)
        result *= scale
    return result


def remove_com_vel(vel, masses):
    # compute the center of mass velocity
    com_vel = np.dot(masses, vel)/masses.sum()
    # subtract
    vel[:] -= com_vel
