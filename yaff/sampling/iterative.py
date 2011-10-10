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


__all__ = ['Iterative', 'StateItem', 'AttributeStateItem', 'Hook']


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
        self.counter = counter0
        log.enter(self.log_name)
        self.initialize()
        log.leave()

    def initialize(self):
        self.call_hooks()

    def call_hooks(self):
        state_updated = False
        for hook in self.hooks:
            if self.counter >= hook.start and (self.counter - hook.start) % hook.step == 0:
                if not state_updated:
                    for item in self.state_list:
                        item.update(self)
                    state_updated = True
                hook(self)

    def run(self, nstep):
        log.enter(self.log_name)
        for i in xrange(nstep):
            self.propagate()
        self.finalize()
        log.leave()

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

    def update(self, sampler):
        self.value = self.get_value(sampler)
        if self.shape is None:
            if isinstance(self.value, np.ndarray):
                self.shape = self.value.shape
                self.dtype = self.value.dtype
            else:
                self.shape = tuple([])
                self.dtype = type(self.value)

    def get_value(self, sampler):
        raise NotImplementedError


class AttributeStateItem(StateItem):
    def get_value(self, sampler):
        return getattr(sampler, self.key)


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
