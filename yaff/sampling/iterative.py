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
'''Base class for iterative algorithms'''


import numpy as np

from molmod.units import *

from yaff import *

from yaff.log import log, timer
from yaff.pes.ff import ForcePartValence, ForcePartPair
from yaff.pes.ext import PairPotEI


__all__ = [
    'Iterative', 'StateItem', 'AttributeStateItem', 'PosStateItem',
    'DipoleStateItem', 'DipoleVelStateItem', 'VolumeStateItem', 'CellStateItem',
    'EPotContribStateItem', 'EpotBondsStateItem', 'EpotBendsStateItem',
    'EpotDihedsStateItem', 'Hook',
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
            self.state_list = [state_item.copy() for state_item in self.default_state]
        else:
            #self.state_list = state
            self.state_list = [state_item.copy() for state_item in self.default_state]
            self.state_list += state
        self.state = dict((item.key, item) for item in self.state_list)
        if hooks is None:
            self.hooks = []
        elif hasattr(hooks, '__len__'):
            self.hooks = hooks
        else:
            self.hooks = [hooks]
        self._add_default_hooks()
        self.counter0 = counter0
        self.counter = counter0
        with log.section(self.log_name), timer.section(self.log_name):
            self.initialize()

        # Initialize restart hook if present
        from yaff.sampling.io import RestartWriter
        for hook in self.hooks:
            if isinstance(hook, RestartWriter):
                hook.init_state(self)

    def _add_default_hooks(self):
        pass

    def initialize(self):
        self.call_hooks()

    def call_hooks(self):
        with timer.section('%s hooks' % self.log_name):
            state_updated = False
            from yaff.sampling.io import RestartWriter
            for hook in self.hooks:
                if hook.expects_call(self.counter) and not (isinstance(hook, RestartWriter) and self.counter==self.counter0):
                    if not state_updated:
                        for item in self.state_list:
                            item.update(self)
                        state_updated = True
                    if isinstance(hook, RestartWriter):
                        for item in hook.state_list:
                            item.update(self)
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

    def iter_attrs(self, iterative):
        return []

    def copy(self):
        return self.__class__()


class AttributeStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative, self.key, None)

    def copy(self):
        return self.__class__(self.key)


class ConsErrStateItem(StateItem):
    def get_value(self, iterative):
        return getattr(iterative._cons_err_tracker, self.key, None)

    def copy(self):
        return self.__class__(self.key)


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
        if sys.charges is not None:
            return np.dot(sys.charges, sys.pos)


class DipoleVelStateItem(StateItem):
    def __init__(self):
        StateItem.__init__(self, 'dipole_vel')

    def get_value(self, iterative):
        charges = iterative.ff.system.charges
        if charges is not None:
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


class EPotContribStateItem(StateItem):
    """Keeps track of all the contributions to the potential energy."""
    def __init__(self):
        StateItem.__init__(self, 'epot_contribs')

    def get_value(self, iterative):
        return np.array([part.energy for part in iterative.ff.parts])

    def iter_attrs(self, iterative):
        yield 'epot_contrib_names', tuple(part.name for part in iterative.ff.parts)


class EpotBondsStateItem(StateItem):
    """Keeps track of all the Valence Bond contributions to the potential energy."""
    def __init__(self, do_ei=False):
        """
           **Optional arguments:**

           do_ei
                If True, the electrostatic contributions of the bonded atom pair
                will also be tracked.
        """
        self.do_ei = do_ei
        StateItem.__init__(self, 'epot_bonds')

    def get_value(self, iterative):
        val = 0.0
        ei = 0.0
        for part in iterative.ff.parts:
            if isinstance(part, ForcePartValence):
                vtab = part.vlist.vtab
                ictab = part.vlist.iclist.ictab
                dtab = part.vlist.iclist.dlist.deltas
        for term in vtab:
            if term['kind']!=3 and ictab[term['ic0']]['kind']==0:
                val += term['energy']
                if self.do_ei:
                    ic = ictab[term['ic0']]
                    qi = iterative.ff.system.charges[dtab[ic['i0']]['i']]
                    qj = iterative.ff.system.charges[dtab[ic['i0']]['j']]
                    d  = ic['value']
                    ei += qi*qj/d
        if self.do_ei:
            return np.array([val, val+ei])
        else:
            return val

    def iter_atts(self, iterative):
        if self.do_ei:
            yield 'epot_bonds_names', tuple('val', 'val+ei')
        else:
            yield []


class EpotBendsStateItem(StateItem):
    """Keeps track of all the Valence Bend contributions to the potential energy"""
    def __init__(self, do_ei=False):
        """
           **Optional arguments:**

           do_ei
                If True, the electrostatic contributions of the 1-3 non-bonded
                atom pair will also be tracked.
        """
        self.do_ei = do_ei
        StateItem.__init__(self, 'epot_bends')

    def get_value(self, iterative):
        val = 0.0
        ei = 0.0
        for part in iterative.ff.parts:
            if isinstance(part, ForcePartValence):
                vtab = part.vlist.vtab
                ictab = part.vlist.iclist.ictab
                dtab = part.vlist.iclist.dlist.deltas
        for term in vtab:
            if term['kind']!=3 and ictab[term['ic0']]['kind'] in [1,2]:
                val += term['energy']
                if self.do_ei:
                    ic = ictab[term['ic0']]
                    d0 = dtab[ic['i0']]
                    d1 = dtab[ic['i1']]
                    rji = np.array([d0['dx'], d0['dy'], d0['dz']])*ic['sign0']
                    rjk = np.array([d1['dx'], d1['dy'], d1['dz']])*ic['sign1']
                    ji = np.array([d0['i'], d0['j']])[::ic['sign0']]
                    jk = np.array([d1['i'], d1['j']])[::ic['sign1']]
                    qi = iterative.ff.system.charges[ji[1]]
                    qk = iterative.ff.system.charges[jk[1]]
                    d  = np.linalg.norm(rji-rjk)
                    ei += qi*qk/d
        if self.do_ei:
            return np.array([val, val+ei])
        else:
            return val

    def iter_atts(self, iterative):
        if self.do_ei:
            yield 'epot_bends_names', tuple('val', 'val+ei')
        else:
            yield []


class EpotDihedsStateItem(StateItem):
    """Keeps track of all the Valence Dihedral contributions to the potential energy"""
    def __init__(self, do_ei=False):
        """
           **Optional arguments:**

           do_ei
                If True, the electrostatic contributions of the 1-4 non-bonded
                atom pair will also be tracked.
        """
        self.do_ei = do_ei
        StateItem.__init__(self, 'epot_diheds')

    def get_value(self, iterative):
        val = 0.0
        ei = 0.0
        for part in iterative.ff.parts:
            if isinstance(part, ForcePartValence):
                vtab = part.vlist.vtab
                ictab = part.vlist.iclist.ictab
                dtab = part.vlist.iclist.dlist.deltas
        for term in vtab:
            if term['kind']!=3 and ictab[term['ic0']]['kind'] in [3,4]:
                val += term['energy']
                if self.do_ei:
                    ic = ictab[term['ic0']]
                    d0 = dtab[ic['i0']]
                    d1 = dtab[ic['i1']]
                    d2 = dtab[ic['i2']]
                    rji = np.array([d0['dx'], d0['dy'], d0['dz']])*ic['sign0']
                    rjk = np.array([d1['dx'], d1['dy'], d1['dz']])*ic['sign1']
                    rkl = np.array([d2['dx'], d2['dy'], d2['dz']])*ic['sign2']
                    ji = np.array([d0['i'], d0['j']])[::ic['sign0']]
                    jk = np.array([d1['i'], d1['j']])[::ic['sign1']]
                    kl = np.array([d2['i'], d2['j']])[::ic['sign2']]
                    qi = iterative.ff.system.charges[ji[1]]
                    ql = iterative.ff.system.charges[kl[1]]
                    d  = np.linalg.norm(rji-rjk-rkl)
                    ei += qi*ql/d
        if self.do_ei:
            return np.array([val, val+ei])
        else:
            return val

    def iter_atts(self, iterative):
        if self.do_ei:
            yield 'epot_diheds_names', tuple('val', 'val+ei')
        else:
            yield []


class Hook(object):
    name = None
    kind = None
    method = None
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

    def expects_call(self, counter):
        return counter >= self.start and (counter - self.start) % self.step == 0

    def __call__(self, iterative):
        raise NotImplementedError
