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

from yaff.sampling.iterative import Hook


__all__ = ['HDF5TrajectoryHook', 'XYZWriterHook']


class HDF5TrajectoryHook(Hook):
    def __init__(self, *args, **kwargs):
        # Extract the arguments for the Hook base class.
        hook_kwargs = {}
        for key in 'start', 'step':
            value = kwargs.get(key)
            if value is not None:
                hook_kwargs[key] = value
                del kwargs[key]
        # By default, the trajectory file is overwritten.
        if not 'mode' in kwargs:
            kwargs['mode'] = 'w'
        # Create file and wrap up
        self.f = h5py.File(*args, **kwargs)
        Hook.__init__(self, **hook_kwargs)

    def __del__(self):
        self.f.close()

    def __call__(self, iterative):
        if 'system' not in self.f:
            self.dump_system(iterative.ff.system)
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative.state)
        tgrp = self.f['trajectory']
        # Get the number of rows written so far. It may seem redundant to have
        # the number of rows stored as an attribute, while each dataset also
        # has a shape from which the number of rows can be determined. However,
        # this helps to keep the dataset sane in case this write call got
        # interrupted in the loop below. Only when the last line below is
        # executed, the data from this iteration is officially written.
        row = tgrp.attrs['row']
        for key, item in iterative.state.iteritems():
            ds = tgrp[key]
            if ds.shape[0] <= row:
                # do not over-allocate. hdf5 works with chunks internally.
                ds.resize(row+1, axis=0)
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
        if system.masses is not None:
            sgrp.create_dataset('masses', data=system.masses)

    def init_trajectory(self, state):
        tgrp = self.f.create_group('trajectory')
        for key, item in state.iteritems():
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
        tgrp.attrs['row'] = 0


class XYZWriterHook(Hook):
    def __init__(self, fn_xyz, start=0, step=1):
        self.fn_xyz = fn_xyz
        self.xyz_writer = None
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        from molmod import angstrom
        if self.xyz_writer is None:
            from molmod.periodic import periodic
            from molmod.io import XYZWriter
            symbols = [periodic[n].symbol for n in iterative.ff.system.numbers]
            self.xyz_writer = XYZWriter(self.fn_xyz, symbols)
        title = '%7i E_pot = %.10f' % (iterative.counter, iterative.epot)
        self.xyz_writer.dump(title, iterative.pos)
