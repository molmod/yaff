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
from yaff.log import timer
from yaff.sampling.dof import CartesianDOF


__all__ = ['estimate_hessian', 'estimate_cart_hessian']


def estimate_hessian(dof, eps=1e-4):
    """Estimate the Hessian using the symmetric finite difference approximation.

       **Arguments:**

       dof
            A DOF object

       **Optional arguments:**

       eps
            The magnitude of the displacements
    """
    with log.section('HESS'), timer.section('Hessian'):
        # Loop over all displacements
        if log.do_medium:
            log('The following displacements are computed:')
            log('DOF     Dir Energy')
            log.hline()
        x1 = dof.x0.copy()
        rows = np.zeros((len(x1), len(x1)), float)
        for i in xrange(len(x1)):
            x1[i] = dof.x0[i] + eps
            epot, gradient_p = dof.fun(x1, do_gradient=True)
            if log.do_medium:
                log('% 7i neg %s' % (i, log.energy(epot)))
            x1[i] = dof.x0[i] - eps
            epot, gradient_m = dof.fun(x1, do_gradient=True)
            if log.do_medium:
                log('% 7i neg %s' % (i, log.energy(epot)))
            rows[i] = (gradient_p-gradient_m)/(2*eps)
            x1[i] = dof.x0[i]
        dof.reset()
        if log.do_medium:
            log.hline()

        # Enforce symmetry and return
        return 0.5*(rows + rows.T)


def estimate_cart_hessian(ff, eps=1e-4, select=None):
    """Estimate the Cartesian Hessian with symmetric finite differences.

       **Arguments:**

       ff
            A force field object

       **Optional arguments:**

       eps
            The magnitude of the Cartesian displacements

       select
            A selection of atoms for which the hessian must be computed. If not
            given, the entire hessian is computed.
    """
    dof = CartesianDOF(ff, select=select)
    return estimate_hessian(dof, eps)
