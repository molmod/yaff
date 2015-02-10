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


from yaff.sampling.iterative import Hook, AttributeStateItem, PosStateItem, CellStateItem
from yaff.sampling.nvt import NHCThermostat
from yaff.sampling.npt import MTKBarostat, TBCombination

__all__ = ['HDF5Writer', 'XYZWriter', 'RestartWriter']


class HDF5Writer(Hook):
    def __init__(self, f, start=0, step=1):
        """
           **Argument:**

           f
                A h5.File object to write the trajectory to.

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.f = f
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


class RestartWriter(Hook):
    def __init__(self, fn_restart, step):
        """
            **Argument:**

            fn_restart
                A filename to write the restart information too.

            step
                The hook will be called every 'step' iterations

        """

        self.fn_restart = fn_restart
        Hook.__init__(self, 0, step)

    def show_vec(self, f, name, data):
        c1 = data.shape[0]
        f.write(name + '\t\t\t' + str(c1) + '\n')
        for i in xrange(c1):
            f.write(str(data[i]) + '\t')
        f.write('\n\n')

    def show_mat(self, f, name, data):
        c1 = data.shape[0]
        c2 = data.shape[1]
        f.write(name + '\t\t\t' + str(c1) + ',' + str(c2) + '\n')
        for i in xrange(c1):
            for j in xrange(c2):
                f.write(str(data[i,j]) + '\t')
            f.write('\n')
        f.write('\n')

    def __call__(self, iterative):
        f = open(self.fn_restart, 'a')
        f.write('counter\t\t\t1\n' + str(iterative.counter) + '\n\n')
        f.write('time\t\t\t1\n' + str(iterative.time) + '\n\n')
        self.show_mat(f, 'positions', iterative.pos)
        self.show_mat(f, 'velocities', iterative.vel)
        self.show_mat(f, 'cell tensor', iterative.ff.system.cell.rvecs)

        thermo = None
        baro = None
        for hook in iterative.hooks:
            if isinstance(hook, NHCThermostat):
                thermo = hook.chain
            if isinstance(hook, MTKBarostat):
                baro = hook
            if isinstance(hook, TBCombination):
                if isinstance(hook.thermostat, NHCThermostat):
                    thermo = hook.thermostat.chain
                if isinstance(hook.barostat, MTKBarostat):
                    baro = hook.barostat
        if thermo is not None:
            self.show_vec(f, 'thermostat positions', thermo.pos)
            self.show_vec(f, 'thermostat velocities', thermo.vel)

        if baro is not None:
            self.show_mat(f, 'barostat velocity', baro.vel_press)

        if baro.baro_thermo is not None:
                self.show_vec(f, 'barostat thermostat positions', baro.baro_thermo.chain.pos)
                self.show_vec(f, 'barostat thermostat velocities', baro.baro_thermo.chain.vel)
        f.close()
