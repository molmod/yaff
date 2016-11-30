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
'''Harmonic models'''


import numpy as np

from yaff.log import log
from yaff.log import timer
from yaff.sampling.dof import CartesianDOF, StrainCellDOF


__all__ = ['estimate_hessian', 'estimate_cart_hessian', 'estimate_elastic']


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
                log('% 7i pos %s' % (i, log.energy(epot)))
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


def estimate_elastic(ff, eps=1e-4, do_frozen=False, ridge=1e-4):
    """Estimate the elastic constants using the symmetric finite difference
       approximation.

       **Arguments:**

       ff
            A force field object

       **Optional arguments:**

       eps
            The magnitude of the Cartesian displacements

       do_frozen
            By default this is False, which means that the changes in fractional
            atomic coordinates due to cell deformations are properly taken into
            account. When this is set to True, such displacements (other than
            uniform scaling) are ignored. The latter is much faster, but only
            correct for the simplest materials.

       ridge
            Threshold for the eigenvalues of the Cartesian Hessian. This only
            matters if ``do_frozen==False``.

       The elastic constants are second order derivatives of the strain energy
       density with respect to uniform deformations. At the molecular scale,
       uniform deformations can be describe by a linear transformation of the
       atoms and the cell parameters::

            x' = e . x0

       When x0 corresponds to the reference point for the second order
       expansion, small deviations from that reference point correspond to small
       deviations of e from the unit matrix. The strain energy density is
       nothing but the energy of the system divided by its volume, minus a the
       energy density of the fully relaxed system. The adjective `strain` refers
       to deviations from the relaxed reference. When computing the second order
       derivatives, this reference point can be ignored. Assuming the reference
       point is the relaxed system, the strain energy density can be
       approximated to second order as::

           u = 1/2 \sum_ijkl e_ij c_ijkl e_kl

       In this equation the matrix c_ijkl contains all the elastic constants.
       In principle, it contains 81 values, but due to symmetry considerations,
       only 21 of these values are independent. [The symmetry considerations are
       as follows: (i) it is sufficient to consider symmetric deformations e
       when excluding cell rotations, (ii) the index pairs (ij) and (kl) are
       interchangeable.] These 21 numbers are typically represented using the
       (inclusive) lower diagonal of a 6x6 matrix, using the following
       convention to map pair indexes (ij) and (kl) onto single indexes n and m:

       ============ ======
       (ij) or (kl) n or m
       ============ ======
       11           1
       22           2
       22           3
       23           4
       31           5
       12           6
       ============ ======

       (This table is also known as the compressed Voight notation.)

       This routine returns the elastic constants in a symmetric 6x6 matrix,
       using the index conventions by Voight.
    """
    cell = ff.system.cell
    if cell.nvec == 0:
        raise ValueError('The elastic constants can only be computed if the system is periodic.')
    dof = StrainCellDOF(ff, do_frozen=do_frozen)
    vol0 = cell.volume
    if do_frozen:
        return estimate_hessian(dof, eps)/vol0
    else:
        hessian = estimate_hessian(dof, eps)/vol0
        # Do a VSA-like trick...
        i = (cell.nvec*(cell.nvec+1))/2
        h11 = hessian[:i,:i]
        h12 = hessian[:i,i:]
        h22 = hessian[i:,i:]
        # Do some special effort to perform a well-conditioned inverse of h22.
        # In case of extremely floppy materials, it may be necessary to tune eps
        # and ridge.
        evals, evecs = np.linalg.eigh(h22)
        h12evecs = np.dot(h12, evecs)
        evalsinv = evals/(evals**2+ridge**2)
        return h11 - np.dot(h12evecs*evalsinv, h12evecs.T)
