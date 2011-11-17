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

from yaff.log import log, timer


__all__ = ['estimate_elastic']


def estimate_elastic(ff, eps=1e-4):
    """Estimate the elastic constants using the symmetric finite difference
       approximation.

       **Arguments:**

       ff
            A force field object

       **Optional arguments:**

       eps
            The magnitude of the Cartesian displacements

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
    # This auxiliary routine is used to compute the Cauchy stress tensor for a
    # given deformation. In this context, the Cauchy tensor is nothing but the
    # Virial tensor divided by the cell volume. (This is the right answer at
    # zero Klevin.)
    def get_cauchy(vol0, pos0, rvecs0, deform):
        pos = np.dot(pos0, deform)
        rvecs = np.dot(rvecs0, deform)
        ff.update_pos(pos)
        ff.update_rvecs(rvecs)
        vtens = np.zeros((3, 3), float)
        epot = ff.compute(vtens=vtens)
        return epot, vtens/vol0

    with log.section('ELASTIC'), timer.section('Elastic const.'):
        # Keep some reference values
        vol0 = ff.system.cell.volume
        if vol0 == 0.0:
            # In case this function is used on an isolated system, it makes no
            # sense to try to compute derivatives of an energy density. It is
            # in general a weird idea to use this function for anything else
            # than 3D periodic systems.
            vol0 = 1
        pos0 = ff.system.pos.copy()
        rvecs0 = ff.system.cell.rvecs.copy()

        # A) Compute all derivatives of the Cauchy tensor with respect to all
        # possible deformation, including rotations etc. We do not bother to
        # consider only symmetric deformations.
        if log.do_medium:
            log('The following deformations are computed:')
            log('i j Dir Energy')
            log.hline()
        elastic = np.zeros((3, 3, 3, 3), float)
        for i in xrange(3):
            for j in xrange(3):
                # Use symmetric finite differences for each derivative...
                # positive displacement
                deform = np.identity(3, float)
                deform[i,j] += eps
                epot, cauchy_p = get_cauchy(vol0, pos0, rvecs0, deform)
                if log.do_medium:
                    log('%i %i neg %s' % (i, j, log.energy(epot)))
                # negative displacement
                deform = np.identity(3, float)
                deform[i,j] -= eps
                energy, cauchy_m = get_cauchy(vol0, pos0, rvecs0, deform)
                if log.do_medium:
                    log('%i %i pos %s' % (i, j, log.energy(epot)))
                # assign
                elastic[i,j] = (cauchy_p - cauchy_m)/(2*eps)
        if log.do_medium:
            log.hline()

        # B) Symmetrize the elastic constants. The following index swaps should
        # all be equivalent: ijkl jikl ijlk jilk klij lkij klji lkji
        elastic = (
            elastic +
            elastic.transpose(1, 0, 2, 3) +
            elastic.transpose(0, 1, 3, 2) +
            elastic.transpose(1, 0, 3, 2) +
            elastic.transpose(2, 3, 0, 1) +
            elastic.transpose(3, 2, 0, 1) +
            elastic.transpose(2, 3, 1, 0) +
            elastic.transpose(3, 2, 1, 0)
        )/8

        # C) Make a compact 6x6 matrix, using Voight indexes.
        voight = [(0,0,0), (1,1,1), (2,2,2), (3,1,2), (4,2,0), (5,0,1)]
        result = np.zeros((6, 6), float)
        for n, i, j in voight:
            for m, k, l in voight:
                result[n,m] = elastic[i,j,k,l]
        return result
