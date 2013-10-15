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
'''Gaussian09 BOMD log Files'''


import numpy as np

from molmod import amu, second, femtosecond
from yaff.conversion.common import get_trajectory_group, \
    get_trajectory_datasets, write_to_dataset, get_last_trajectory_row, \
    check_trajectory_rows
from yaff.log import log


__all__ = ['g09log_to_hdf5']


def _scan_to_line(f, marker):
    while True:
        line = f.next()
        if line.startswith(marker):
            return line


def _scan_g09_forces(f):
    '''Search for the next forces block and return numbers and forces'''
    _scan_to_line(f, " Center     Atomic                   Forces (Hartrees/Bohr)")
    f.next() # skip line
    f.next() # skip line
    # read the numbers and forces
    numbers = []
    frc = []
    while True:
        line = f.next()
        if line.startswith(" ---------------"):
            break
        words = line.split()
        numbers.append(int(words[1]))
        frc.append([float(words[2]), float(words[3]), float(words[4])])
    return np.array(numbers), np.array(frc)


def _scan_g09_time(f):
    line = _scan_to_line(f, " Summary information for step")
    step = int(line[30:])
    line = f.next()
    time = float(line[12:])*femtosecond
    line = f.next()
    parts = line.split(';')
    ekin = float(parts[0].split()[2])
    epot = float(parts[1].split()[2])
    etot = float(parts[2].split()[2])
    return time, step, ekin, epot, etot


def _scan_g09_pos_vel(f):
    _scan_to_line(f, " Cartesian coordinates: (bohr)")
    convert = lambda s: float(s.replace('D', 'E'))

    pos = []
    while True:
        line = f.next()
        if not line.startswith(" I="):
            break
        words = line.split()
        pos.append([convert(words[3]), convert(words[5]), convert(words[7])])

    vel = []
    while True:
        line = f.next()
        if not line.startswith(" I="):
            break
        words = line.split()
        vel.append([convert(words[3]), convert(words[5]), convert(words[7])])

    return np.array(pos), np.array(vel)*(np.sqrt(amu)/second)


def _iter_frames_g09(fn_g09):
    '''Step through a G09 BOMD log file and yield relevant properties at each step.

       The following values (in a.u.) are present in the yield statement:

       numbers
            The element numbers (N,)

       pos
            The atomic positions (N,3)

       vel
            The atomic velocities (N,3)

       frc
            The atomic velocities (N,3)

       time
            The time

       step
            The MD step

       epot
            The potential energy

       ekin
            The kinetic energy

       etot
            The total energy
    '''
    with open(fn_g09) as f:
        # Skip the first and second block of Forces
        for i in xrange(2):
            _scan_to_line(f, " Center     Atomic                   Forces (Hartrees/Bohr)")

        # Keep reading MD steps until the file ends
        while True:
            try:
                numbers, frc = _scan_g09_forces(f)
                time, step, ekin, epot, etot = _scan_g09_time(f)
                pos, vel = _scan_g09_pos_vel(f)
                yield numbers, pos, vel, frc, time, step, epot, ekin, etot
            except StopIteration:
                return


def g09log_to_hdf5(f, fn_log):
    """Convert Gaussian09 BOMD log file to Yaff HDF5 format.

       **Arguments:**

       f
            An open and writable HDF5 file.

       fn_log
            The name of the Gaussian log file.
    """
    with log.section('G09H5'):
        if log.do_medium:
            log('Loading Gaussian 09 file \'%s\' into \'trajectory\' of HDF5 file \'%s\'' % (
                fn_log, f.filename
            ))

        # First make sure the HDF5 file has a system description that is consistent
        # with the XYZ file.
        if 'system' not in f:
            raise ValueError('The HDF5 file must contain a system group.')
        if 'numbers' not in f['system']:
            raise ValueError('The HDF5 file must have a system group with atomic numbers.')
        natom = f['system/numbers'].shape[0]

        # Take care of the trajectory group
        tgrp = get_trajectory_group(f)

        # Take care of the pos and vel datasets
        dss = get_trajectory_datasets(tgrp,
            ('pos', (natom, 3)),
            ('vel', (natom, 3)),
            ('frc', (natom, 3)),
            ('time', (1,)),
            ('step', (1,)),
            ('epot', (1,)),
            ('ekin', (1,)),
            ('etot', (1,)),
        )
        ds_pos, ds_vel, ds_frc, ds_time, ds_step, ds_epot, ds_ekin, ds_etot = dss

        # Load frame by frame
        row = get_last_trajectory_row(dss)
        for numbers, pos, vel, frc, time, step, epot, ekin, etot in _iter_frames_g09(fn_log):
            if (numbers != f['system/numbers']).any():
                log.warn('The element numbers of the HDF5 and LOG file do not match.')
            write_to_dataset(ds_pos, pos, row)
            write_to_dataset(ds_vel, vel, row)
            write_to_dataset(ds_frc, frc, row)
            write_to_dataset(ds_time, time, row)
            write_to_dataset(ds_step, step, row)
            write_to_dataset(ds_epot, epot, row)
            write_to_dataset(ds_ekin, ekin, row)
            write_to_dataset(ds_etot, etot, row)
            row += 1

        # Check number of rows
        check_trajectory_rows(tgrp, dss, row)
