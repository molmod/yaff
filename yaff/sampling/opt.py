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

from molmod.minimizer import ConjugateGradient, QuasiNewton, NewtonLineSearch, \
    Minimizer, check_delta

from yaff.log import log
from yaff.sampling.iterative import Iterative, AttributeStateItem, \
    PosStateItem, VolumeStateItem, CellStateItem, Hook


__all__ = [
    'OptScreenLog', 'CartesianDOF', 'FullCell', 'AnisoCell', 'IsoCell',
    'CellDOF', 'BaseOptimizer', 'CGOptimizer',
]


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
                iterative.dof.conv_val,
                iterative.dof.conv_count,
                iterative.dof.conv_worst,
                time.time() - self.time0,
            ))


class DOF(object):
    def __init__(self):
        self._gx = None

    def _init_gx(self, length):
        self._gx = np.zeros(length, float)


class CartesianDOF(DOF):
    """Cartesian degrees of freedom for the optimizers"""
    def __init__(self, gpos_max=3e-5, gpos_rms=1e-5, dpos_max=3e-3, dpos_rms=1e-3):
        """
           **Optional arguments:**

           gpos_max, gpos_rms, dpos_max, dpos_rms
                Thresholds that define the convergence. If all of the actual
                values drop below these thresholds, the minimizer stops.

           **Convergence conditions:**

           gpos_max
                The maximum of the norm of the gradient of an atom.

           gpos_rms
                The root-mean-square of the norm of the gradients of the atoms.

           dpos_max
                The maximum of the norm of the displacement of an atom.

           dpos_rms
                The root-mean-square of the norm of the displacements of the
                atoms.
        """
        self.th_gpos_max = gpos_max
        self.th_gpos_rms = gpos_rms
        self.th_dpos_max = dpos_max
        self.th_dpos_rms = dpos_rms
        DOF.__init__(self)
        self._pos = None
        self._last_pos = None

    def get_initial(self, system):
        """Return the initial value of the unknowns"""
        x = system.pos.ravel()
        self._init_gx(len(x))
        self._pos = system.pos.copy()
        self._dpos = np.zeros(system.pos.shape, float)
        self._gpos = np.zeros(system.pos.shape, float)
        return x

    def fun(self, x, ff, do_gradient=False):
        """computes the energy i.f.o x, and optionally the gradient.

           **Arguments:**

           x
                The degrees of freedom

           system
                The system object in which these
        """
        self._pos[:] = x.reshape(-1,3)
        ff.update_pos(self._pos[:])
        if do_gradient:
            self._gpos[:] = 0.0
            v = ff.compute(self._gpos)
            self._gx[:] = self._gpos.ravel()
            return v, self._gx
        else:
            return ff.compute()

    def check_convergence(self):
        # When called for the first time, initialize _last_pos
        if self._last_pos is None:
            self._last_pos = self._pos.copy()
            self.converged = False
            self.conv_val = 2
            self.conv_worst = 'first_step'
            self.conv_count = -1
            return
        # Compute the values that have to be compared to the thresholds
        gpossq = (self._gpos**2).sum(axis=1)
        self.gpos_max = np.sqrt(gpossq.max())
        self.gpos_rms = np.sqrt(gpossq.mean())
        self._dpos[:] = self._pos
        self._dpos -= self._last_pos
        #
        dpossq = (self._dpos**2).sum(axis=1)
        self.dpos_max = np.sqrt(dpossq.max())
        self.dpos_rms = np.sqrt(dpossq.mean())
        # Compute a general value that has to go below 1.0 to have convergence.
        conv_vals = []
        if self.th_gpos_max is not None:
            conv_vals.append((self.gpos_max/self.th_gpos_max, 'gpos_max'))
        if self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, 'gpos_rms'))
        if self.th_dpos_max is not None:
            conv_vals.append((self.dpos_max/self.th_dpos_max, 'dpos_max'))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, 'dpos_rms'))
        if len(conv_vals) == 0:
            raise RuntimeError('At least one convergence criterion must be present.')
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for v, n in conv_vals)
        self.converged = (self.conv_count == 0)
        self._last_pos[:] = self._pos[:]



class FullCell(object):
    def get_initial(self, system):
        self.nvec = system.cell.nvec
        if self.nvec == 0:
            raise ValueError('A cell optimization requires a system that is periodic.')
        self._cell_scale = system.cell.volume**(1.0/self.nvec)*10
        if log.do_debug:
            log('Cell scale set to %s.' % log.length(self._cell_scale))
        return system.cell.rvecs.ravel()/self._cell_scale

    def x_to_rvecs(self, x):
        index = self.nvec*3
        return x[:index].reshape(-1,3)*self._cell_scale, index

    def grvecs_to_gx(self, grvecs):
        return grvecs.ravel()*self._cell_scale


class AnisoCell(object):
    def get_initial(self, system):
        self.nvec = system.cell.nvec
        if self.nvec == 0:
            raise ValueError('A cell optimization requires a system that is periodic.')
        self.rvecs0 = system.cell.rvecs.copy()
        return np.ones(self.nvec, float)

    def x_to_rvecs(self, x):
        index = self.nvec
        return self.rvecs0*x[:index].reshape(-1,1), index

    def grvecs_to_gx(self, grvecs):
        return (grvecs*self.rvecs0).sum(axis=1)


class IsoCell(object):
    def get_initial(self, system):
        self.nvec = system.cell.nvec
        if self.nvec == 0:
            raise ValueError('A cell optimization requires a system that is periodic.')
        self.rvecs0 = system.cell.rvecs.copy()
        return np.ones(1, float)

    def x_to_rvecs(self, x):
        return self.rvecs0*x[0], 1

    def grvecs_to_gx(self, grvecs):
        return (grvecs*self.rvecs0).sum()


class CellDOF(DOF):
    """Fractional coordinates and cell parameters"""
    def __init__(self, cell_spec, gpos_max=3e-5, gpos_rms=1e-5, dpos_max=3e-3, dpos_rms=1e-3, gcell_max=3e-5, gcell_rms=1e-5, dcell_max=3e-3, dcell_rms=1e-3):
        """
           **Arguments:**

           cell_spec
                A cell specification. This object defines which parts of the
                cell parameters are fixed/free. Some examples: FullCell(),
                AnisoCell(), IsoCell().

           **Optional arguments:**

           gpos_max, gpos_rms, dpos_max, dpos_rms, gcell_max, gcell_rms, dcell_max, dcell_rms
                Thresholds that define the convergence. If all of the actual
                values drop below these thresholds, the minimizer stops.

           **Convergence conditions:**

           gpos_max
                The maximum of the norm of the gradient of an atom.

           gpos_rms
                The root-mean-square of the norm of the gradients of the atoms.

           dpos_max
                The maximum of the norm of the displacement of an atom.

           dpos_rms
                The root-mean-square of the norm of the displacements of the
                atoms.

           gcell_max
                The maximum of the norm of the gradient of acell vector.

           gcell_rms
                The root-mean-square of the norm of the gradients of the cell
                vectors.

           dcell_max
                The maximum of the norm of the displacement of a cell vector.

           dcell_rms
                The root-mean-square of the norm of the displacements of the
                cell vectors.
        """
        self.cell_spec = cell_spec
        self.th_gpos_max = gpos_max
        self.th_gpos_rms = gpos_rms
        self.th_dpos_max = dpos_max
        self.th_dpos_rms = dpos_rms
        self.th_gcell_max = gcell_max
        self.th_gcell_rms = gcell_rms
        self.th_dcell_max = dcell_max
        self.th_dcell_rms = dcell_rms
        DOF.__init__(self)
        self._pos = None
        self._last_pos = None
        self._cell = None
        self._last_cell = None
        self._cell_scale = 1.0

    def get_initial(self, system):
        """Return the initial value of the unknowns"""
        cell_vars = self.cell_spec.get_initial(system)
        frac = np.dot(system.pos, system.cell.gvecs.T)
        x = np.concatenate([cell_vars, frac.ravel()])
        self._init_gx(len(x))
        self._pos = system.pos.copy()
        self._dpos = np.zeros(system.pos.shape, float)
        self._gpos = np.zeros(system.pos.shape, float)
        self._cell = system.cell.rvecs.copy()
        self._dcell = np.zeros(self._cell.shape, float)
        self._vtens = np.zeros(self._cell.shape, float)
        self._gcell = np.zeros(self._cell.shape, float)
        return x

    def fun(self, x, ff, do_gradient=False):
        """computes the energy i.f.o x, and optionally the gradient.

           **Arguments:**

           x
                The degrees of freedom

           system
                The system object in which these
        """
        self._cell, index = self.cell_spec.x_to_rvecs(x)
        frac = x[index:].reshape(-1,3)
        self._pos[:] = np.dot(frac, self._cell)
        ff.update_pos(self._pos[:])
        ff.update_rvecs(self._cell[:])
        if do_gradient:
            self._gpos[:] = 0.0
            self._vtens[:] = 0.0
            v = ff.compute(self._gpos, self._vtens)
            self._gcell[:] = np.dot(ff.system.cell.gvecs, self._vtens)
            self._gx[:index] = self.cell_spec.grvecs_to_gx(self._gcell)
            self._gx[index:] = np.dot(self._gpos, self._cell.T).ravel()
            return v, self._gx
        else:
            return ff.compute()

    def check_convergence(self):
        # When called for the first time, initialize _last_pos
        if self._last_pos is None:
            self._last_pos = self._pos.copy()
            self._last_cell = self._cell.copy()
            self.converged = False
            self.conv_val = 2
            self.conv_worst = 'first_step'
            self.conv_count = -1
            return
        # Compute the values that have to be compared to the thresholds
        gpossq = (self._gpos**2).sum(axis=1)
        self.gpos_max = np.sqrt(gpossq.max())
        self.gpos_rms = np.sqrt(gpossq.mean())
        self._dpos[:] = self._pos
        self._dpos -= self._last_pos
        #
        dpossq = (self._dpos**2).sum(axis=1)
        self.dpos_max = np.sqrt(dpossq.max())
        self.dpos_rms = np.sqrt(dpossq.mean())
        #
        gcellsq = (self._gcell**2).sum(axis=1)
        self.gcell_max = np.sqrt(gcellsq.max())
        self.gcell_rms = np.sqrt(gcellsq.mean())
        self._dcell[:] = self._cell
        self._dcell -= self._last_cell
        #
        dcellsq = (self._dcell**2).sum(axis=1)
        self.dcell_max = np.sqrt(dcellsq.max())
        self.dcell_rms = np.sqrt(dcellsq.mean())
        # Compute a general value that has to go below 1.0 to have convergence.
        conv_vals = []
        if self.th_gpos_max is not None:
            conv_vals.append((self.gpos_max/self.th_gpos_max, 'gpos_max'))
        if self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, 'gpos_rms'))
        if self.th_dpos_max is not None:
            conv_vals.append((self.dpos_max/self.th_dpos_max, 'dpos_max'))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, 'dpos_rms'))
        if self.th_gcell_max is not None:
            conv_vals.append((self.gcell_max/self.th_gcell_max, 'gcell_max'))
        if self.th_gcell_rms is not None:
            conv_vals.append((self.gcell_rms/self.th_gcell_rms, 'gcell_rms'))
        if self.th_dcell_max is not None:
            conv_vals.append((self.dcell_max/self.th_dcell_max, 'dcell_max'))
        if self.th_dcell_rms is not None:
            conv_vals.append((self.dcell_rms/self.th_dcell_rms, 'dcell_rms'))
        if len(conv_vals) == 0:
            raise RuntimeError('At least one convergence criterion must be present.')
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for v, n in conv_vals)
        self.converged = (self.conv_count == 0)
        self._last_pos[:] = self._pos[:]
        self._last_cell[:] = self._cell[:]


class BaseOptimizer(Iterative):
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        PosStateItem(),
        VolumeStateItem(),
        CellStateItem(),
    ]
    log_name = 'XXOPT'

    def __init__(self, ff, dof, search_direction, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           ff
                A ForceField instance

           dof
                A specification of the degrees of freedom. The convergence
                criteria are also part of this argument.

           search_direction
                A search direction object

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
        self.dof = dof
        self.minimizer = Minimizer(
            self.dof.get_initial(ff.system), self.fun, search_direction,
            NewtonLineSearch(), None, None, anagrad=True, verbose=False,
        )
        Iterative.__init__(self, ff, state, hooks, counter0)

    def _add_default_hooks(self):
        if not any(isinstance(hook, OptScreenLog) for hook in self.hooks):
            self.hooks.append(OptScreenLog())

    def fun(self, x, do_gradient=False):
        if do_gradient:
            self.epot, gx = self.dof.fun(x, self.ff, True)
            return self.epot, gx
        else:
            self.epot = self.dof.fun(x, self.ff, False)
            return self.epot


    def initialize(self):
        self.minimizer.initialize()
        # The first call to check_convergence will never flag convergence, but
        # it is need to keep track of some convergence criteria.
        self.dof.check_convergence()
        Iterative.initialize(self)

    def propagate(self):
        success = self.minimizer.propagate()
        if success == False:
            if log.do_warning:
                log.warn('Line search failed in optimizer. Aborting optimization. This is probably due to a dicontinuity in the energy or the forces. Check the truncation of the non-bonding interactions and the Ewald summation parameters.')
            return True
        self.dof.check_convergence()
        Iterative.propagate(self)
        return self.dof.converged

    def finalize(self):
        if log.do_medium:
            log.hline()

    def check_delta(self, eps=1e-4):
        """Test the analytical derivatives"""
        x = self.minimizer.x
        dxs = np.random.uniform(-eps, eps, (100, len(x)))
        check_delta(self.fun, x, dxs)


class CGOptimizer(BaseOptimizer):
    log_name = 'CGOPT'

    def __init__(self, ff, dof, state=None, hooks=None, counter0=0):
        BaseOptimizer.__init__(self, ff, dof, ConjugateGradient(), state, hooks, counter0)


# TODO: figure out why this one does not work well:
#class QNOptimizer(BaseOptimizer):
#    log_name = 'QNOPT'
#
#    def __init__(self, ff, dof, state=None, hooks=None, counter0=0):
#        BaseOptimizer.__init__(self, ff, dof, QuasiNewton(), state, hooks, counter0)
