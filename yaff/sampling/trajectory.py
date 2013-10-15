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
'''Computations on a reference trajectory'''


import numpy as np, time, h5py

from yaff.log import log
from yaff.sampling.iterative import Iterative, AttributeStateItem, \
    PosStateItem, DipoleStateItem, VolumeStateItem, CellStateItem, \
    EPotContribStateItem, Hook

__all__ = [
    'TrajScreenLog', 'RefTrajectory',
]

class TrajScreenLog(Hook):
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log('counter  Walltime')
                    log.hline()
            log('%7i %10.1f' % (
                iterative.counter,
                time.time() - self.time0,
            ))

class RefTrajectory(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        PosStateItem(),
        DipoleStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]

    log_name = 'TRAJEC'

    def __init__(self, ff, fn_traj, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           ff
                A ForceField instance

           fn_traj

                A hdf5 file name containing the trajectory


           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           counter0
                The counter value associated with the initial state.
        """
        self.traj = h5py.File(fn_traj, 'r')
        self.nframes = len(self.traj['trajectory/pos'][:])
        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, TrajScreenLog) for hook in self.hooks):
            self.hooks.append(TrajScreenLog())

    def initialize(self):
        return

    def propagate(self):
        self.ff.update_pos(self.traj['trajectory/pos'][self.counter])
        self.ff.update_rvecs(self.traj['trajectory/cell'][self.counter])
        self.epot = self.ff.compute(None, None)
        self.call_hooks()
        self.counter += 1
        return self.counter==self.nframes

    def finalize(self):
        self.traj.close()
        if log.do_medium:
            log.hline()
