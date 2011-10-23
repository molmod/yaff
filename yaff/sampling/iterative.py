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

from yaff.log import log
from yaff.timer import timer


__all__ = [
    'Iterative', 'StateItem', 'AttributeStateItem', 'PosStateItem',
    'DipoleStateItem', 'DipoleVelStateItem', 'VolumeStateItem', 'CellStateItem',
    'Hook',
]


class Iterative(object):
    default_state = []
    log_name = 'ITER'

    def __init__(self, ff, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           ff
                The ForceField instance used in the iterative algorithm

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
        self.ff = ff
        if state is None:
            self.state_list = list(self.default_state)
        else:
            self.state_list = state
        self.state = dict((item.key, item) for item in self.state_list)
        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, '__len__'):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        self._add_default_hooks()
        self.counter = counter0
        with log.section(self.log_name), timer.section(self.log_name):
            self.initialize()

    def _add_default_hooks(self):
        pass

    def initialize(self):
        self.call_hooks()

    def call_hooks(self):
        with timer.section('%s hooks' % self.log_name):
            state_updated = False
            for hook in self.hooks:
                if self.counter >= hook.start and (self.counter - hook.start) % hook.step == 0:
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    hook(self)

    def run(self, nstep=None):
        with log.section(self.log_name), timer.section(self.log_name):
            if nstep is None:
                while True:
                    if self.propagate():
                        break
            else:
                for i in xrange(nstep):
                    if self.propagate():
                        break
            self.finalize()

    def propagate(self):
        self.counter += 1
        self.call_hooks()

    def finalize():
        raise NotImplementedError


class StateItem(object):
    def __init__(self, key):
        self.key = key
        self.shape = None
        self.dtype = None

    def update(self, iterative):
        self.value = self.get_value(iterative)
        if self.shape is None:
            if isinstance(self.value, np.ndarray):
                self.shape = self.value.shape
                self.dtype = self.value.dtype
            else:
                self.shape = tuple([])
                self.dtype = type(self.value)

    def get_value(self, iterative):
        raise NotImplementedError


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key)


class PosStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'pos')

    def get_value(self, iterative):
        return iterative.ff.system.pos


class DipoleStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'dipole')

    def get_value(self, iterative):
        sys = iterative.ff.system
        return np.dot(sys.charges, sys.pos)


class DipoleVelStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'dipole_vel')

    def get_value(self, iterative):
        return np.dot(iterative.ff.system.charges, iterative.vel)


class VolumeStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'volume')

    def get_value(self, iterative):
        return iterative.ff.system.cell.volume


class CellStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'cell')

    def get_value(self, iterative):
        return iterative.ff.system.cell.rvecs


class Hook(object):
    def __init__(self, start=0, step=1):
        """
           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.start = start
        self.step = step

    def __call__(self, iterative):
        raise NotImplementedError
