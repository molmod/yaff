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
'''Trajectory writers'''


from yaff.sampling.iterative import Hook


__all__ = ['HDF5Writer', 'XYZWriter']


class HDF5Writer(Hook):
    def __init__(self, f, start=0, step=1, flush=None):
        """
           **Argument:**

           f
                A h5.File object to write the trajectory to.

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.

           flush
                Flush the h5.File object every `flush` iterations so it can be
                read up to the latest flush. This is useful to avoid data loss
                for long calculations or to monitor running calculations.
                Note that flushing too often might impact performance of hdf5.
        """
        self.f = f
        self.flush = flush
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        if 'system' not in self.f:
            self.dump_system(iterative.ff.system)
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f['trajectory']
        # determine the row to write the current iteration to. If a previous
        # iterations was not completely written, then the last row is reused.
        row = min(tgrp[key].shape[0] for key in iterative.state if key in tgrp.keys())
        for key, item in iterative.state.iteritems():
            if item.value is None:
                continue
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.dtype is type(None):
                continue
            ds = tgrp[key]
            if ds.shape[0] <= row:
                # do not over-allocate. hdf5 works with chunks internally.
                ds.resize(row+1, axis=0)
            ds[row] = item.value
        if self.flush is not None:
            if row%self.flush == 0: self.f.flush()

    def dump_system(self, system):
        system.to_hdf5(self.f)

    def init_trajectory(self, iterative):
        tgrp = self.f.create_group('trajectory')
        for key, item in iterative.state.iteritems():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.dtype is type(None):
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
               tgrp.attrs[name] = value


class XYZWriter(Hook):
    def __init__(self, fn_xyz, select=None, start=0, step=1):
        """
           **Argument:**

           fn_xyz
                A filename to write the XYZ trajectory too.

           **Optional arguments:**

           select
                A list of atom indexes that should be written to the trajectory
                output. If not given, all atoms are included.

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.fn_xyz = fn_xyz
        self.select = select
        self.xyz_writer = None
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        from molmod import angstrom
        if self.xyz_writer is None:
            from molmod.periodic import periodic
            from molmod.io import XYZWriter
            numbers = iterative.ff.system.numbers
            if self.select is None:
                symbols = [periodic[n].symbol for n in numbers]
            else:
                symbols = [periodic[numbers[i]].symbol for i in self.select]
            self.xyz_writer = XYZWriter(self.fn_xyz, symbols)
        rvecs = iterative.ff.system.cell.rvecs.copy()
        rvecs_string = " ".join([str(x[0]) for x in rvecs.reshape((-1,1))])
        title = '%7i E_pot = %.10f    %s' % (iterative.counter, iterative.epot, rvecs_string)
        if self.select is None:
            pos = iterative.ff.system.pos
        else:
            pos = iterative.ff.system.pos[self.select]
        self.xyz_writer.dump(title, pos)
