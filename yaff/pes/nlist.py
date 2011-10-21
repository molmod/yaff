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
# MERCHAnlistILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


import numpy as np

from yaff.log import log
from yaff.timer import timer
from yaff.pes.ext import nlist_status_init, nlist_status_finish, nlist_build, \
    nlist_recompute


__all__ = ['NeighborList']


neigh_dtype = [
    ('a', int), ('b', int), ('d', float),
    ('dx', float), ('dy', float), ('dz', float),
    ('r0', int), ('r1', int), ('r2', int)
]


class NeighborList(object):
    def __init__(self, system, skin=0):
        """Algorithms to keep track of all distances below a given rcut

           **Arguments:**

           system
                A System instance.

           **Optional arguments:**

           skin
                A margin added to the rcut parameter. Only when atoms are
                displaced by half this distance, the neighborlist is rebuilt
                from scratch. In the other case, the distances of the known
                pairs are just recomputed. If set to zero, the default, the
                neighborlist is rebuilt at each update.

                A reasonable skin setting can drastically improve the
                performance of the neighborlist updates. For example, when
                ``rcut`` is ``10*angstrom``, a ``skin`` of ``2*angstrom`` is
                reasonable. If the skin is set too large, the updates will
                become very inefficient. Some tuning of ``rcut`` and ``skin``
                may be beneficial.
        """
        if skin < 0:
            raise ValueError('The skin parameter must be positive.')
        self.system = system
        self.skin = skin
        self.rcut = 0.0
        # the neighborlist:
        self.neighs = np.empty(10, dtype=neigh_dtype)
        self.nneigh = 0
        # rmax determines the number of periodic images that are considered.
        # Along the a direction, images are taken from -rmax[0] to rmax[0]
        # (inclusive), etc.
        self.rmax = None
        # for skin algorithm:
        self._pos_old = None
        self.rebuild_next = False

    def request_rcut(self, rcut):
        """Make sure the internal rcut parameter is at least is high as rcut."""
        self.rcut = max(self.rcut, rcut)
        self.update_rmax()

    def update_rmax(self):
        """Update the rmax attribute.

           This may be necessary for two reasons: (i) the cutoff has changed,
           and (ii) the cell vectors have changed.
        """
        # determine the number of periodic images
        self.rmax = np.ceil((self.rcut+self.skin)/self.system.cell.rspacings-0.5).astype(int)
        if log.do_high:
            if len(self.rmax) == 1:
                log('rmax a       = %i' % tuple(self.rmax))
            elif len(self.rmax) == 2:
                log('rmax a,b     = %i,%i' % tuple(self.rmax))
            elif len(self.rmax) == 3:
                log('rmax a,b,c   = %i,%i,%i' % tuple(self.rmax))
        # Request a rebuild of the neighborlist because there is no simple way
        # to figure out whether an update is sufficient. TODO: look at other
        # codes to see how this is done.
        self.rebuild_next = True

    def update(self):
        with log.section('NLIST'), timer.section('Nlists'):
            assert self.rcut > 0

            if self._need_rebuild():
                # *rebuild* the entire neighborlist
                # 1) make an initial status object for the neighbor list algorithm
                status = nlist_status_init(self.rmax)
                # 2) a loop of consecutive update/allocate calls
                last_start = 0
                while True:
                    done = nlist_build(
                        self.system.pos, self.rcut + self.skin, self.rmax,
                        self.system.cell, status, self.neighs[last_start:]
                    )
                    if done:
                        break
                    last_start = len(self.neighs)
                    new_neighs = np.empty((len(self.neighs)*3)/2, dtype=neigh_dtype)
                    new_neighs[:last_start] = self.neighs
                    self.neighs = new_neighs
                    del new_neighs
                # 3) get the number of neighbors in the list.
                self.nneigh = nlist_status_finish(status)
                if log.do_debug:
                    log('Rebuilt, size = %i' % self.nneigh)
                # 4) store the current state to check in future calls if we
                #    need to do a rebuild or a recompute.
                self._checkpoint()
                self.rebuild_next = False
            else:
                # just *recompute* the deltas and the distance in the
                # neighborlist
                nlist_recompute(self.system.pos, self._pos_old, self.system.cell, self.neighs[:self.nneigh])
                if log.do_debug:
                    log('Recomputed')

    def _checkpoint(self):
        if self.skin > 0:
            # Only use the skin algorithm if this parameter is larger than zero.
            if self._pos_old is None:
                self._pos_old = self.system.pos.copy()
            else:
                self._pos_old[:] = self.system.pos

    def _need_rebuild(self):
        if self.skin <= 0 or self._pos_old is None or self.rebuild_next:
            return True
        else:
            # Compute an upper bound for the maximum relative displacement.
            disp = np.sqrt(((self.system.pos - self._pos_old)**2).sum(axis=1).max())
            disp *= 2*(self.rmax.max()+1)
            if log.do_debug:
                log('Maximum relative displacement %s      Skin %s' % (log.length(disp), log.length(self.skin)))
            # Compare with skin parameter
            return disp >= self.skin
