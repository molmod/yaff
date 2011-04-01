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


def nlist_status_init(rmax):
    # five integer status fields:
    # * other_index
    # * number of rows consumed
    # * r0
    # * r1
    # * r2
    return np.array([-rmax[0], -rmax[1], -rmax[2], 0, 0], float)

def nlist_update(pos, center_index, cutoff, rmax, rvecs, gvecs, nlist_status, nlist):
    r0 = nlist_status[0]
    r1 = nlist_status[1]
    r2 = nlist_status[2]
    other_index = nlist_status[3]

    update_delta0 = True
    row = 0
    nvec = len(rmax)

    while row < len(nlist):
        if other_index >= len(pos):
            nlist_status[4] += row
            return True
        if update_delta0:
            delta0 = pos[center_index] - pos[other_index]
            delta0 -= np.dot(rvecs, np.ceil(np.dot(gvecs.transpose(), delta0)-0.5))
            update_delta0 = False
        delta = delta0.copy()
        if nvec > 0:
            delta += r0*rvecs[:,0]
        if nvec > 1:
            delta += r1*rvecs[:,1]
        if nvec > 2:
            delta += r2*rvecs[:,2]
        d = np.linalg.norm(delta)
        if d < cutoff:
            nlist[row] = (other_index, d, delta, (r0, r1, r2))
            row += 1
        if nvec > 0:
            r0 += 1
            if r0 > rmax[0]:
                r0 = -rmax[0]
                if nvec > 1:
                    r1 += 1
                    if r1 > rmax[1]:
                        r1 = -rmax[1]
                        if nvec > 2:
                            r2 += 1
                            if r2 > rmax[2]:
                                r2 = -rmax[2]
                                other_index += 1
                                update_delta0 = True
                        else:
                            other_index += 1
                            update_delta0 = True
                else:
                    other_index += 1
                    update_delta0 = True
        else:
            other_index += 1
            update_delta0 = True

    nlist_status[0] = r0
    nlist_status[1] = r1
    nlist_status[2] = r2
    nlist_status[3] = other_index
    nlist_status[4] += row
    return False

def nlist_status_finish(nlist_status):
    return nlist_status[4]
