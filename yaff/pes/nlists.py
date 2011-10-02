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

from yaff.pes.ext import nlist_status_init, nlist_status_finish, nlist_update


__all__ = ['NeighborLists']


nlist_dtype = [
    ('i', int), ('d', float), ('dx', float), ('dy', float), ('dz', float),
    ('r0', int), ('r1', int), ('r2', int)
]


class NeighborLists(object):
    def __init__(self, system):
        self.system = system
        self.rcut = 0.0
        self.nlists = None
        self.nlist_sizes = None
        self.rmax = None

    natom = property(lambda self: self.system.natom)

    def request_rcut(self, rcut):
        self.rcut = max(self.rcut, rcut)

    def __len__(self):
        return len(self.nlists)

    def __getitem__(self, index):
        return self.nlists[index][:self.nlist_sizes[index]]

    def update(self):
        assert self.rcut > 0
        # if there are no items yet, lets make them first:
        if self.nlists is None:
            self.nlists = [np.empty(10, dtype=nlist_dtype) for i in xrange(self.system.natom)]
            self.nlist_sizes = np.zeros(self.system.natom, dtype=int)
        # determine the number of periodic images
        self.rmax = np.ceil(self.rcut/self.system.cell.rspacings-0.5).astype(int)
        # build all neighbor lists
        for i in xrange(self.system.natom):
            # make an initial nlist array
            nlist = self.nlists[i]
            last_start = 0
            # make an initial status object for the nlist algorithm
            nlist_status = nlist_status_init(i, self.rmax)
            while True:
                done = nlist_update(
                    self.system.pos, i, self.rcut, self.rmax,
                    self.system.cell, nlist_status, nlist[last_start:]
                )
                if done:
                    break
                last_start = len(nlist)
                new_nlist = np.empty((len(nlist)*3)/2, dtype=nlist_dtype)
                new_nlist[:last_start] = nlist
                nlist = new_nlist
                del new_nlist
            self.nlists[i] = nlist
            self.nlist_sizes[i] = nlist_status_finish(nlist_status)
