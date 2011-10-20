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
from yaff.pes.ext import nlist_status_init, nlist_status_finish, nlist_update


__all__ = ['NeighborList']


neigh_dtype = [
    ('a', int), ('b', int), ('d', float),
    ('dx', float), ('dy', float), ('dz', float),
    ('r0', int), ('r1', int), ('r2', int)
]


class NeighborList(object):
    def __init__(self, system):
        self.system = system
        self.rcut = 0.0
        self.neighs = np.empty(10, dtype=neigh_dtype)
        self.nneigh = 0
        self.rmax = None

    def request_rcut(self, rcut):
        self.rcut = max(self.rcut, rcut)
        self.update_rmax()

    def update_rmax(self):
        # determine the number of periodic images
        self.rmax = np.ceil(self.rcut/self.system.cell.rspacings-0.5).astype(int)
        if log.do_high:
            if len(self.rmax) == 1:
                log('rmax a       = %i' % tuple(self.rmax))
            elif len(self.rmax) == 2:
                log('rmax a,b     = %i,%i' % tuple(self.rmax))
            elif len(self.rmax) == 3:
                log('rmax a,b,c   = %i,%i,%i' % tuple(self.rmax))

    def update(self):
        with log.section('NLIST'), timer.section('Nlists'):
            assert self.rcut > 0
            # build all neighbor lists
            last_start = 0
            # make an initial status object for the neighbor list algorithm
            status = nlist_status_init(self.rmax)
            while True:
                done = nlist_update(
                    self.system.pos, self.rcut, self.rmax,
                    self.system.cell, status, self.neighs[last_start:]
                )
                if done:
                    break
                last_start = len(self.neighs)
                new_neighs = np.empty((len(self.neighs)*3)/2, dtype=neigh_dtype)
                new_neighs[:last_start] = self.neighs
                self.neighs = new_neighs
                del new_neighs
            self.nneigh = nlist_status_finish(status)
            if log.do_debug:
                log('nlist size = %i' % self.nneigh)
