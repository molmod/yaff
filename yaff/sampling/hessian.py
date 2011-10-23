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


__all__ = ['estimate_hessian']


def estimate_hessian(ff, eps=1e-4, select=None):
    """Estimate the Hessian using the symmetric finite difference approximation.

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
    with log.section('HESS'), timer.section('Hessian'):
        rows = []
        if select is None:
            iterator = xrange(ff.system.natom)
        else:
            iterator = select

        # Loop over all displacements
        if log.do_medium:
            log('The following displacements are computed:')
            log('Atom    Axis Dir Energy')
            log.hline()
        gpos = np.zeros((ff.system.natom, 3), float)
        pos0 = ff.system.pos.copy()
        pos1 = pos0.copy()
        for i in iterator:
            for j in xrange(3):
                gpos[:] = 0.0
                pos1[:] = pos0
                pos1[i,j] -= eps
                ff.update_pos(pos1)
                epot = ff.compute(gpos)
                if log.do_medium:
                    log('% 7i %4i neg %s' % (i, j, epot))
                gpos *= -1
                pos1[:] = pos0
                pos1[i,j] += eps
                ff.update_pos(pos1)
                epot = ff.compute(gpos)
                if log.do_medium:
                    log('% 7i %4i pos %s' % (i, j, epot))
                rows.append(gpos/(2*eps))
        if log.do_medium:
            log.hline()

        # Select subsystem if needed
        rows = np.array(rows)
        if select is not None:
            rows = rows[:,select,:]
            print rows.shape
        ndof = rows.shape[0]
        rows = rows.reshape((ndof, ndof))

        # Enforce symmetry and return
        hessian = 0.5*(rows + rows.T)

        # Restore the original positions in the system object
        ff.update_pos(pos0)

        # Done
        return hessian
