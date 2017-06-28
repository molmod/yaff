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


from yaff.sampling.iterative import Hook, AttributeStateItem, PosStateItem, CellStateItem, ConsErrStateItem
from yaff.sampling.nvt import NHCThermostat, NHCAttributeStateItem
from yaff.sampling.npt import MTKBarostat, MTKAttributeStateItem, TBCombination


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
        rvecs_string = " ".join([str(x[0]/angstrom) for x in rvecs.reshape((-1,1))])
        title = '%7i E_pot = %.10f    %s' % (iterative.counter, iterative.epot, rvecs_string)
        if self.select is None:
            pos = iterative.ff.system.pos
        else:
            pos = iterative.ff.system.pos[self.select]
        self.xyz_writer.dump(title, pos)


class RestartWriter(Hook):
    def __init__(self, f, start=0, step=1000):
        """
            **Argument:**

            f
                A h5.File object to write the restart information to.

            **Optional arguments:**

            start
                The first iteration at which this hook should be called.

            step
                The hook will be called every `step` iterations.
        """
        self.f = f
        self.state = None
        self.default_state = None
        Hook.__init__(self, start, step)

    def init_state(self, iterative):
        # Basic properties needed for the restart
        self.default_state = [
            AttributeStateItem('counter'),
            AttributeStateItem('time'),
            PosStateItem(),
            AttributeStateItem('vel'),
            CellStateItem(),
            AttributeStateItem('econs'),
            ConsErrStateItem('econs_counter'),
            ConsErrStateItem('ekin_sum'),
            ConsErrStateItem('ekin_sumsq'),
            ConsErrStateItem('econs_sum'),
            ConsErrStateItem('econs_sumsq')
        ]

        # Dump the timestep
        rgrp = self.f.create_group('restart')
        rgrp.create_dataset('timestep', data = iterative.timestep)

        # Verify whether there are any deterministic thermostats / barostats, and add them if present
        thermo = None
        baro = None

        for hook in iterative.hooks:
            if hook.kind == 'deterministic':
                if hook.method == 'thermostat': thermo = hook
                elif hook.method == 'barostat': baro = hook
            elif hook.name == 'TBCombination':
                if hook.thermostat.kind == 'deterministic': thermo = hook.thermostat
                if hook.barostat.kind == 'deterministic': baro = hook.barostat

        if thermo is not None:
            self.dump_restart(thermo)
            if thermo.name == 'NHC':
                self.default_state.append(NHCAttributeStateItem('pos'))
                self.default_state.append(NHCAttributeStateItem('vel'))
        if baro is not None:
            self.dump_restart(baro)
            if baro.name == 'MTTK':
                self.default_state.append(MTKAttributeStateItem('vel_press'))
                if baro.baro_thermo is not None:
                    self.default_state.append(MTKAttributeStateItem('chain_pos'))
                    self.default_state.append(MTKAttributeStateItem('chain_vel'))

        # Finalize the restart state items
        self.state_list = [state_item.copy() for state_item in self.default_state]
        self.state = dict((item.key, item) for item in self.state_list)


    def __call__(self, iterative):
        if 'system' not in self.f:
            self.dump_system(iterative.ff.system)
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f['trajectory']
        # determine the row to write the current iteration to. If a previous
        # iterations was not completely written, then the last row is reused.
        row = min(tgrp[key].shape[0] for key in self.state if key in tgrp.keys())
        for key, item in self.state.iteritems():
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
        for key, item in self.state.iteritems():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.dtype is type(None):
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
               tgrp.attrs[name] = value

    def dump_restart(self, hook):
        rgrp = self.f['/restart']
        if hook.method == 'thermostat':
            # Dump the thermostat properties
            rgrp.create_dataset('thermo_name', data = hook.name)
            rgrp.create_dataset('thermo_temp', data = hook.temp)
            rgrp.create_dataset('thermo_timecon', data = hook.chain.timecon)
        elif hook.method == 'barostat':
            # Dump the barostat properties
            rgrp.create_dataset('baro_name', data = hook.name)
            rgrp.create_dataset('baro_timecon', data = hook.timecon_press)
            rgrp.create_dataset('baro_temp', data = hook.temp)
            rgrp.create_dataset('baro_press', data = hook.press)
            rgrp.create_dataset('baro_anisotropic', data = hook.anisotropic)
            rgrp.create_dataset('vol_constraint', data = hook.vol_constraint)
            if hook.name == 'Berendsen':
                rgrp.create_dataset('beta', data = hook.beta)
            if hook.name == 'MTTK' and hook.baro_thermo is not None:
                # Dump the barostat thermostat properties
                rgrp.create_dataset('baro_chain_temp', data = hook.baro_thermo.temp)
                rgrp.create_dataset('baro_chain_timecon', data = hook.baro_thermo.chain.timecon)
