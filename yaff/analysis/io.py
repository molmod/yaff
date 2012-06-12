# YAFF is yet another force-field code
# Copyright (C) 2008 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
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


from molmod import angstrom, femtosecond
from molmod.io import XYZReader, slice_match

from yaff.log import log


__all__ = ['xyz_to_hdf5', 'cp2k_ener_to_hdf5']



def _get_trajectory_group(f):
    if 'trajectory' not in f:
        if log.do_high:
            log('Creating new trajectory datagroup in %s' % f.filename)
        tgrp = f.create_group('trajectory')
        existing_row = None
    else:
        tgrp = f['trajectory']
        existing_row = tgrp.attrs['row']
        if log.do_high:
            log('Using existing trajectory datagroup in %s with %i rows of data' % (f.filename, existing_row))
    return tgrp, existing_row


def _get_trajectory_datasets(tgrp, *fields):
    result = []
    for name, row_shape in fields:
        if name in tgrp:
            if log.do_medium:
                log('Overwriting existing dataset %s in group %s.' % (name, tgrp.name))
            del tgrp[name]

        if log.do_high:
            log('Creating new dataset %s with row shape %s' % (name, row_shape))
        # Create a new dataset
        shape = (0,) + row_shape
        maxshape = (None,) + row_shape
        ds = tgrp.create_dataset(name, shape, maxshape=maxshape, dtype=float)
        result.append(ds)
    return result


def _append_to_dataset(ds, value, row):
    if ds.shape[0] <= row:
        ds.resize(row+1, axis=0)
    ds[row] = value


def _check_trajectory_rows(tgrp, existing_row, row):
    if existing_row is None:
        tgrp.attrs['row'] = row
    else:
        if existing_row != row:
            raise ValueError('The amount of data loaded into the HDF5 file is not consistent with number of rows already present in the trajectory.')


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

        xyz_reader = XYZReader(fn_xyz, sub=sub)
        if len(xyz_reader.numbers) != len(f['system/numbers']):
            raise ValueError('The number of atoms in the HDF5 and the XYZ files does not match.')
        if (xyz_reader.numbers != f['system/numbers']).any():
            log.warn('The atomic numbers of the HDF5 and XYZ file do not match.')

        # Take care of the trajectory group
        tgrp, existing_row = _get_trajectory_group(f)

        # Take care of the dataset
        ds, = _get_trajectory_datasets(tgrp, ('pos', (len(xyz_reader.numbers), 3)))

        # Fill the dataset with data.
        row = 0
        for title, coordinates in xyz_reader:
            _append_to_dataset(ds, coordinates, row)
            row += 1

        # Check number of rows
        _check_trajectory_rows(tgrp, existing_row, row)


def cp2k_ener_to_hdf5(f, fn_ener, sub=slice(None)):
    """Convert a CP2K energy trajectory file to Yaff HDF5 format.

       **Arguments:**

       f
            An open and writable HDF5 file.

       fn_ener
            The filename of the CP2K energy trajectory file.

       **Optional arguments:**

       sub
            This must be a slice object that defines the sub-sampling of the
            CP2K energy file. By default all time steps are read.

       This routine will also test the consistency of the row attribute of the
       trajectory group. If some trajectory data is already present, it will be
       replaced by the new data. Furthermore, this routine also checks the
       header of the CP2K energy file to make sure the values are interpreted
       correctly.

       It is highly recommended to first initialize the HDF5 file with the
       ``to_hdf5`` method of the System class.
    """
    with log.section('CP2KEH5'):
        if log.do_medium:
            log('Loading CP2K energy file \'%s\' into \'trajectory\' of HDF5 file \'%s\'' % (
                fn_ener, f.filename
            ))

        # Take care of the data group
        tgrp, existing_row = _get_trajectory_group(f)

        # Take care of the datasets
        ds_step, ds_time, ds_ke, ds_temp, ds_pe, ds_cq = _get_trajectory_datasets(
            tgrp,
            ('step', (1,)),
            ('time', (1,)),
            ('ekin', (1,)),
            ('temp', (1,)),
            ('epot', (1,)),
            ('econs', (1,)),
        )

        # Fill the datasets with data.
        row = 0
        counter = 0
        fin = file(fn_ener)

        # check header line
        line = fin.next()
        words = line.split()
        if words[0] != '#':
            raise ValueError('The first line in the energies file should be a header line starting with #.')
        if words[3] != 'Time[fs]' or words[4] != 'Kin.[a.u.]' or \
           words[5] != 'Temp[K]' or words[6] != 'Pot.[a.u.]' or \
           words[7] + ' ' + words[8] != 'Cons Qty[a.u.]':
            raise ValueError('The fields in the header line indicate that this file contains unsupported data.')

        # Load lines
        for line in fin:
            if slice_match(sub, counter):
                words = line.split()
                _append_to_dataset(ds_step, float(words[0]), row)
                _append_to_dataset(ds_time, float(words[1])*femtosecond, row)
                _append_to_dataset(ds_ke, float(words[2]), row)
                _append_to_dataset(ds_temp, float(words[3]), row)
                _append_to_dataset(ds_pe, float(words[4]), row)
                _append_to_dataset(ds_cq, float(words[5]), row)
                row += 1
            counter += 1
        fin.close()

        # Check number of rows
        _check_trajectory_rows(tgrp, existing_row, row)
