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
'''XYZ Files'''



from molmod import angstrom
from molmod.io import XYZReader, slice_match
from yaff.conversion.common import get_trajectory_group, \
    get_trajectory_datasets, write_to_dataset, get_last_trajectory_row, \
    check_trajectory_rows
from yaff.log import log


__all__ = ['xyz_to_hdf5']


def xyz_to_hdf5(f, fn_xyz, sub=slice(None), file_unit=angstrom, name='pos'):
    """Convert XYZ trajectory file to Yaff HDF5 format.

       **Arguments:**

       f
            An open and writable HDF5 file.

       fn_xyz
            The filename of the XYZ trajectory file.

       **Optional arguments:**

       sub
            The sub argument for the XYZReader. This must be a slice object that
            defines the subsampling of the XYZ file reader. By default all
            frames are read.

       file_unit
            The unit of the data in the XYZ file. [default=angstrom]

       name
            The name of the HDF5 dataset where the trajectory is stored. This
            array is stored in the 'trajectory' group.

       This routine will also test the consistency of the row attribute of the
       trajectory group. If some trajectory data is already present, it will be
       replaced by the new data. It is highly recommended to first initialize
       the HDF5 file with the ``to_hdf5`` method of the System class.
    """
    with log.section('XYZH5'):
        if log.do_medium:
            log('Loading XYZ file \'%s\' into \'trajectory/%s\' of HDF5 file \'%s\'' % (
                fn_xyz, name, f.filename
            ))

        # First make sure the HDF5 file has a system description that is consistent
        # with the XYZ file.
        if 'system' not in f:
            raise ValueError('The HDF5 file must contain a system group.')
        if 'numbers' not in f['system']:
            raise ValueError('The HDF5 file must have a system group with atomic numbers.')

        xyz_reader = XYZReader(fn_xyz, sub=sub, file_unit=file_unit)
        if len(xyz_reader.numbers) != len(f['system/numbers']):
            raise ValueError('The number of atoms in the HDF5 and the XYZ files does not match.')
        if (xyz_reader.numbers != f['system/numbers']).any():
            log.warn('The atomic numbers of the HDF5 and XYZ file do not match.')

        # Take care of the trajectory group
        tgrp = get_trajectory_group(f)

        # Take care of the dataset
        ds, = get_trajectory_datasets(tgrp, (name, (len(xyz_reader.numbers), 3)))

        # Fill the dataset with data.
        row = get_last_trajectory_row([ds])
        for title, coordinates in xyz_reader:
            write_to_dataset(ds, coordinates, row)
            row += 1

        # Check number of rows
        check_trajectory_rows(tgrp, [ds], row)
