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


import numpy as np, time

from molmod.minimizer import ConjugateGradient, NewtonLineSearch, Minimizer

from yaff.log import log
from yaff.sampling.iterative import Iterative, AttributeStateItem, \
    VolumeStateItem, CellStateItem, Hook


__all__ = ['OptScreenLog', 'CGOptimizer']


class OptScreenLog(Hook):
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, iterative):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log('Conv.val. =&the highest ratio of a convergence criterion over its threshold.')
                    log('N         =&the number of convergence criteria that is not met.')
                    log('Worst     =&the name of the convergence criterion that is worst.')
                    log('counter  Conv.val.  N        Worst   Walltime')
                    log.hline()
            log('%7i % 10.3e %2i %12s %10.1f' % (
                iterative.counter,
                iterative.conv_val,
                iterative.conv_count,
                iterative.conv_worst,
                time.time() - self.time0,
            ))


class CGOptimizer(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        AttributeStateItem('pos'),
        VolumeStateItem(),
        CellStateItem(),
    ]
    log_name = 'CGOPT'

    def __init__(self, ff, state=None, hooks=None, grad_max=3e-4, grad_rms=1e-4, disp_max=3e-2, disp_rms=1e-2):
        """
           **Arguments:**

           ff
                A ForceField instance

           **Optional arguments:**

           state
                A list with state items. State items are simple objects
                that take or derive a property from the current state of the
                iterative algorithm.

           hooks
                A function (or a list of functions) that is called after every
                iterative.

           grad_max, grad_rms, step_max, step_rms
                Thresholds that define the convergence. If all of the actual
                values drop below these thresholds, the minimizer stops.

           **Convergence conditions:**

           grad_max
                The maximum of the norm of the gradient on an atom.

           grad_rms
                The root-mean-square of the norm of the gradients on the atoms.

           disp_max
                The maximum of the norm of the displacement of an atom.

           disp_rms
                The root-mean-square of the norm of the displacements of the
                atoms.
        """
        self.pos = ff.system.pos.copy()
        self.gpos = np.zeros(self.pos.shape, float)
        self.last_pos = np.zeros(self.pos.shape, float)
        self.delta = np.zeros(self.pos.shape, float)

        self.th_grad_max = grad_max
        self.th_grad_rms = grad_rms
        self.th_disp_max = disp_max
        self.th_disp_rms = disp_rms
        self.minimizer = Minimizer(
            self.pos.ravel(), self.fun, ConjugateGradient(), NewtonLineSearch(),
            None, None, anagrad=True, verbose=False,
        )

        Iterative.__init__(self, ff, state, hooks)

    def _add_default_hooks(self):
        if not any(isinstance(hook, OptScreenLog) for hook in self.hooks):
            self.hooks.append(OptScreenLog())

    def fun(self, x, do_gradient=False):
        self.pos[:] = x.reshape(-1,3)
        self.gpos[:] = 0.0
        self.ff.update_pos(self.pos)
        if do_gradient:
            self.epot = self.ff.compute(self.gpos)
            return self.epot, self.gpos.ravel()
        else:
            self.epot = self.ff.compute()
            return self.epot

    def initialize(self):
        self.last_pos[:] = self.pos
        self.minimizer.initialize()
        self.compute_properties()
        Iterative.initialize(self)

    def propagate(self):
        self.last_pos[:] = self.pos
        self.minimizer.propagate()
        self.compute_properties()
        Iterative.propagate(self)
        return self.converged

    def compute_properties(self):
        # Compute the values that have to be compared to the thresholds
        gradsq = (self.gpos**2).sum(axis=1)
        self.grad_max = np.sqrt(gradsq.max())
        self.grad_rms = np.sqrt(gradsq.mean())
        self.delta[:] = self.pos
        self.delta -= self.last_pos
        dispsq = (self.delta**2).sum(axis=1)
        self.disp_max = np.sqrt(dispsq.max())
        self.disp_rms = np.sqrt(dispsq.mean())
        # Compute a general value that has to go below one to have convergence
        conv_vals = []
        if self.th_grad_max is not None:
            conv_vals.append((self.grad_max/self.th_grad_max, 'grad_max'))
        if self.th_grad_rms is not None:
            conv_vals.append((self.grad_rms/self.th_grad_rms, 'grad_rms'))
        if self.th_disp_max is not None:
            conv_vals.append((self.disp_max/self.th_disp_max, 'disp_max'))
        if self.th_disp_rms is not None:
            conv_vals.append((self.disp_rms/self.th_disp_rms, 'disp_rms'))
        if len(conv_vals) == 0:
            raise RuntimeError('At least one convergence criterion must be present.')
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for  v, n in conv_vals)
        self.converged = (self.conv_count == 0) and self.counter > 0

    def finalize(self):
        if log.do_medium:
            log.hline()
