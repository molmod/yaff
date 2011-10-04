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


import h5py


__all__ = ['HDF5TrajectoryHook']


class HDF5TrajectoryHook(object):
    def __init__(self, *args, **kwargs):
        self.f = h5py.File(*args, **kwargs)

    def __del__(self):
        self.f.close()

    def __call__(self, ff, state):
        if 'system' not in self.f:
            self.dump_system(ff.system)
        if 'trajectory' not in self.f:
            self.init_trajectory(state)
        tgrp = self.f['trajectory']
        row = tgrp.attrs['row']
        for key, item in state.iteritems():
            ds = tgrp[key]
            if ds.shape[0] < row:
                ds.resize(int(row*1.2)+5, axis=0)
            ds[row] = item.value
        tgrp.attrs['row'] += 1

    def dump_system(self, system):
        sgrp = self.f.create_group('system')
        sgrp.create_dataset('numbers', data=system.numbers)
        sgrp.create_dataset('pos', data=system.pos)
        sgrp.create_dataset('ffatypes', data=system.ffatypes, dtype='a10')
        if system.topology is not None:
            sgrp.create_dataset('bonds', data=system.topology.bonds)
        if system.cell.nvec > 0:
            sgrp.create_dataset('rvecs', data=system.cell.rvecs)
        if system.charges is not None:
            sgrp.create_dataset('charges', data=system.charges)

    def init_trajectory(self, state):
        tgrp = self.f.create_group('trajectory')
        for key, item in state.iteritems():
            maxshape = (None,) + item.shape
            shape = (10,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
        tgrp.attrs['row'] = 0
