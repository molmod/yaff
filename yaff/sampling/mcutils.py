# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
# --
'''Utilities supporting Monte-Carlo simulations'''


from __future__ import division

import numpy as np
import time

from yaff.sampling.iterative import Hook
from yaff.system import System
from yaff.log import log, timer


__all__ = ['get_random_rotation_matrix', 'random_insertion',
           'GCMCScreenLog',
          ]


def get_random_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def random_insertion(guest):
    """
    Place a guest molecule at a random position with a random orientation

    Arguments:**

        guest
            A System instance that is 3D periodic
    """
    # Only 3D periodic systems supported for now
    assert guest.cell.nvec==3
    # Center at origin
    pos = guest.pos.copy()
    pos -= np.average(pos, axis=0)
    # Perform rotation
    M = get_random_rotation_matrix()
    pos = np.einsum('ib,ab->ia', pos, M)
    # Perform translation
    translation = np.dot(guest.cell.rvecs.T, np.random.rand(3)-0.5)
    return pos+translation


class GCMCScreenLog(Hook):
    '''A screen logger for GCMC simulations'''
    def __init__(self, start=0, step=1):
        Hook.__init__(self, start, step)
        self.time0 = None

    def __call__(self, mc):
        if log.do_medium:
            if self.time0 is None:
                self.time0 = time.time()
                if log.do_medium:
                    log.hline()
                    log('     counter          N          <N>          E        <E>   Walltime')
                    log.hline()
            log('%12i %10d %12.6f %s %s %10.1f' % (
                mc.counter,
                mc.N,
                mc.Nmean,
                log.energy(mc.energy),
                log.energy(mc.emean),
                time.time() - self.time0,
            ))
