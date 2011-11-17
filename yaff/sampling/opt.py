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
    PosStateItem, DipoleStateItem, VolumeStateItem, CellStateItem, \
    EPotContribStateItem, Hook


__all__ = [
    'OptScreenLog', 'CartesianDOF', 'FullCell', 'AnisoCell', 'IsoCell',
    'CellDOF', 'BaseOptimizer', 'CGOptimizer', 'BFGSHessianModel',
    'BFGSOptimizer',
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
    def __init__(self, gpos_rms=1e-5, dpos_rms=1e-3):
        """
           **Optional arguments:**

           gpos_rms, dpos_rms
                Thresholds that define the convergence. If all of the actual
                values drop below these thresholds, the minimizer stops.

                For each rms threshold, a corresponding max threshold is
                included automatically. The maximum of the absolute value of a
                component should be smaller than 3/sqrt(N) times the rms
                threshold, where N is the number of degrees of freedom.

           **Convergence conditions:**

           gpos_rms
                The root-mean-square of the norm of the gradients of the atoms.

           dpos_rms
                The root-mean-square of the norm of the displacements of the
                atoms.
        """
        self.th_gpos_rms = gpos_rms
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
        return x.copy()

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
            return v, self._gx.copy()
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
        if self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, 'gpos_rms'))
            conv_vals.append((self.gpos_max/(self.th_gpos_rms*3/np.sqrt(gpossq.size)), 'gpos_max'))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, 'dpos_rms'))
            conv_vals.append((self.dpos_max/(self.th_dpos_rms*3/np.sqrt(dpossq.size)), 'dpos_max'))
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
    def __init__(self, cell_spec, gpos_rms=1e-5, dpos_rms=1e-3, gcell_rms=1e-5, dcell_rms=1e-3):
        """
           **Arguments:**

           cell_spec
                A cell specification. This object defines which parts of the
                cell parameters are fixed/free. Some examples: FullCell(),
                AnisoCell(), IsoCell().

           **Optional arguments:**

           gpos_rms, dpos_rms, gcell_rms, dcell_rms
                Thresholds that define the convergence. If all of the actual
                values drop below these thresholds, the minimizer stops.

                For each rms threshold, a corresponding max threshold is
                included automatically. The maximum of the absolute value of a
                component should be smaller than 3/sqrt(N) times the rms
                threshold, where N is the number of degrees of freedom.

           **Convergence conditions:**

           gpos_rms
                The root-mean-square of the norm of the gradients of the atoms.

           dpos_rms
                The root-mean-square of the norm of the displacements of the
                atoms.

           gcell_rms
                The root-mean-square of the norm of the gradients of the cell
                vectors.

           dcell_rms
                The root-mean-square of the norm of the displacements of the
                cell vectors.
        """
        self.cell_spec = cell_spec
        self.th_gpos_rms = gpos_rms
        self.th_dpos_rms = dpos_rms
        self.th_gcell_rms = gcell_rms
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
        if self.th_gpos_rms is not None:
            conv_vals.append((self.gpos_rms/self.th_gpos_rms, 'gpos_rms'))
            conv_vals.append((self.gpos_max/(self.th_gpos_rms*3/np.sqrt(gpossq.size)), 'gpos_max'))
        if self.th_dpos_rms is not None:
            conv_vals.append((self.dpos_rms/self.th_dpos_rms, 'dpos_rms'))
            conv_vals.append((self.dpos_max/(self.th_dpos_rms*3/np.sqrt(dpossq.size)), 'dpos_max'))
        if self.th_gcell_rms is not None:
            conv_vals.append((self.gcell_rms/self.th_gcell_rms, 'gcell_rms'))
            conv_vals.append((self.gcell_max/(self.th_gcell_rms*3/np.sqrt(gcellsq.size)), 'gcell_max'))
        if self.th_dcell_rms is not None:
            conv_vals.append((self.dcell_rms/self.th_dcell_rms, 'dcell_rms'))
            conv_vals.append((self.dcell_max/(self.th_dcell_rms*3/np.sqrt(dcellsq.size)), 'dcell_max'))
        if len(conv_vals) == 0:
            raise RuntimeError('At least one convergence criterion must be present.')
        self.conv_val, self.conv_worst = max(conv_vals)
        self.conv_count = sum(int(v>=1) for v, n in conv_vals)
        self.converged = (self.conv_count == 0)
        self._last_pos[:] = self._pos[:]
        self._last_cell[:] = self._cell[:]


class BaseOptimizer(Iterative):
    # TODO: This should be copied upon initialization. As it is now, two
    # consecutive simulations with a different number of atoms will raise an
    # exception.
    default_state = [
        AttributeStateItem('counter'),
        AttributeStateItem('epot'),
        PosStateItem(),
        DipoleStateItem(),
        VolumeStateItem(),
        CellStateItem(),
        EPotContribStateItem(),
    ]
    log_name = 'XXOPT'

    def __init__(self, ff, dof, state=None, hooks=None, counter0=0):
        """
           **Arguments:**

           ff
                A ForceField instance

           dof
                A specification of the degrees of freedom. The convergence
                criteria are also part of this argument.

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
        # The first call to check_convergence will never flag convergence, but
        # it is need to keep track of some convergence criteria.
        self.dof.check_convergence()
        Iterative.initialize(self)

    def propagate(self):
        self.dof.check_convergence()
        Iterative.propagate(self)
        return self.dof.converged

    def finalize(self):
        if log.do_medium:
            log.hline()

    def check_delta(self, x, eps=1e-4):
        """Test the analytical derivatives"""
        dxs = np.random.uniform(-eps, eps, (100, len(x)))
        check_delta(self.fun, x, dxs)


class CGOptimizer(BaseOptimizer):
    log_name = 'CGOPT'

    def __init__(self, ff, dof, state=None, hooks=None, counter0=0):
        self.minimizer = Minimizer(
            dof.get_initial(ff.system), self.fun, ConjugateGradient(),
            NewtonLineSearch(), None, None, anagrad=True, verbose=False,
        )
        BaseOptimizer.__init__(self, ff, dof, state, hooks, counter0)

    def initialize(self):
        self.minimizer.initialize()
        BaseOptimizer.initialize(self)

    def propagate(self):
        success = self.minimizer.propagate()
        if success == False:
            if log.do_warning:
                log.warn('Line search failed in optimizer. Aborting optimization. This is probably due to a dicontinuity in the energy or the forces. Check the truncation of the non-bonding interactions and the Ewald summation parameters.')
            return True
        return BaseOptimizer.propagate(self)

    def check_delta(self, x=None, eps=1e-4):
        if x is None:
            x = self.minimizer.x
        BaseOptimizer.check_delta(self, x, eps=1e-4)


class BFGSHessianModel(object):
    def __init__(self, size):
        self.hessian = np.identity(size, float)

    def update(self, dx, dg):
        tmp = np.dot(self.hessian, dx)
        hmax = abs(self.hessian).max()
        # Only compute updates if the denominators do not blow up
        denom1 = np.dot(dx, tmp)
        if hmax*denom1 <= 1e-5*abs(tmp).max():
            return
        denom2 = np.dot(dg, dx)
        if hmax*denom2 <= 1e-5*abs(dg).max():
            return
        if log.do_debug:
            log('Updating BFGS Hessian.    denom1=%10.3e   denom2=%10.3e' % (denom1, denom2))
        self.hessian -= np.outer(tmp, tmp)/denom1
        self.hessian += np.outer(dg, dg)/denom2

    def get_spectrum(self):
        return np.linalg.eigh(self.hessian)


class BFGSOptimizer(BaseOptimizer):
    """BFGS optimizer

       This is just a basic implementation of the algorithm, but it has the
       potential to become more advanced and efficient. The following
       improvements will be made when time permits:

       1) Support for non-linear constraints. This should be relatively easy. We
          need a routine that can bring the unknowns back to the constraints,
          and a routine to solve a constrained second order problem with linear
          equality/inequality constraints. These should be methods of an object
          that is an attribute of the dof object, which is need to give the
          constraint code access to the Cartesian coordinates. In the code
          below, some comments are added to mark where the constraint methods
          should be called.

       2) The Hessian updates and the diagonalization are currently very slow
          for big systems. This can be fixed with a rank-1 update algorithm for
          the spectral decomposition.

       3) The optimizer would become much more efficient if redundant
          coordinates were used. This can be implemented efficiently by using
          the same machinery as the constraint code, but using the dlist and
          iclist concepts for the sake of efficiency.

       4) It is in practice not needed to keep track of the full Hessian. The
          L-BFGS algorithm is a nice method to obtain a linear memory usage and
          computational cost. However, L-BFGS is not compatible with the trust
          radius used in this class, while we want to keep the trust radius for
          the sake of efficiency, robustness and support for constraints. Using
          the rank-1 updates mentioned above, it should be relatively easy to
          keep track of the decomposition of a subspace of the Hessian.
          This subspace can be defined as the basis of the last N rank-1
          updates. Simple assumptions about the remainder of the spectrum should
          be sufficient to keep the algorithm efficient.
    """
    log_name = 'BFGSOPT'

    def __init__(self, ff, dof, state=None, hooks=None, counter0=0):
        self.x_old = dof.get_initial(ff.system)
        self.hessian = BFGSHessianModel(len(self.x_old))
        self.trust_radius = 1.0
        BaseOptimizer.__init__(self, ff, dof, state, hooks, counter0)

    def initialize(self):
        self.f_old, self.g_old = self.fun(self.x_old, True)
        self.x, self.f, self.g = self.make_step()
        BaseOptimizer.initialize(self)

    def propagate(self):
        # Update the Hessian
        self.hessian.update(self.x - self.x_old, self.g - self.g_old)
        # Move new to old
        self.x_old = self.x
        self.f_old = self.f
        self.g_old = self.g
        # Compute a step
        self.x, self.f, self.g = self.make_step()
        return BaseOptimizer.propagate(self)

    def make_step(self):
        evals, evecs = self.hessian.get_spectrum()
        tmp1 = -np.dot(evecs.T, self.g_old)

        # Initial ridge parameter
        if evals[0] <= 0:
            ridge = abs(evals[0]) + 1e-3*abs(evals).max()
        else:
            ridge = 0.0

        # Trust radius loop
        if log.do_high:
            log.hline()
            log('       Ridge      Radius       Trust')
            log.hline()
        while True:
            # Increase ridge until step is smaller than trust radius
            while True:
                # MARKER FOR CONSTRAINT CODE: instead of the following line, a
                # constrained harmonic solver should be added to find the step
                # that minimes the local quadratic problem under a set of linear
                # equality/inequality constraints. This can be implemented using
                # the active set algorithm.
                tmp2 = tmp1*evals/(evals**2 + ridge**2)
                radius = np.linalg.norm(tmp2)
                if log.do_high:
                    log('%12.5e %12.5e %12.5e' % (ridge, radius, self.trust_radius))
                if radius < self.trust_radius:
                    break
                if ridge == 0.0:
                    ridge = abs(evals[evals!=0.0]).min()
                else:
                    ridge *= 1.2
            # Check if the step is trust worthy
            delta_x = np.dot(evecs, tmp2)
            # MARKER FOR CONSTRAINT CODE: the following line should be replaced
            # by something of the sort: x = self.shaker.fix(self.x_old + delta_x)
            x = self.x_old + delta_x
            f = self.fun(x, False)
            delta_f = f - self.f_old
            if delta_f > 0:
                # The function must decrease, if not the trust radius is too big
                if log.do_high:
                    log('Function incraeses.')
                self.trust_radius *= 0.5
                continue
            # If we get here, we are done.
            if log.do_high:
                log.hline()
            self.trust_radius *= 1.2
            f, g = self.fun(x, True)
            return x, f, g

    def check_delta(self, x=None, eps=1e-4):
        if x is None:
            x = self.x
        BaseOptimizer.check_delta(self, x, eps=1e-4)
