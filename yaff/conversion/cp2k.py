# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
# --
'''CP2K Files'''


from __future__ import division

from molmod import femtosecond
from molmod.io import slice_match

from yaff.conversion.common import get_trajectory_group, \
    get_trajectory_datasets, get_last_trajectory_row, write_to_dataset, \
    check_trajectory_rows
from yaff.log import log

__all__ = ['cp2k_ener_to_hdf5']


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
        tgrp = get_trajectory_group(f)

        # Take care of the datasets
        dss = get_trajectory_datasets(
            tgrp,
            ('step', (1,)),
            ('time', (1,)),
            ('ekin', (1,)),
            ('temp', (1,)),
            ('epot', (1,)),
            ('econs', (1,)),
        )
        ds_step, ds_time, ds_ke, ds_temp, ds_pe, ds_cq = dss

        # Fill the datasets with data.
        row = get_last_trajectory_row(dss)
        counter = 0
        with open(fn_ener) as fin:
            # check header line
            line = next(fin)
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
                    write_to_dataset(ds_step, float(words[0]), row)
                    write_to_dataset(ds_time, float(words[1])*femtosecond, row)
                    write_to_dataset(ds_ke, float(words[2]), row)
                    write_to_dataset(ds_temp, float(words[3]), row)
                    write_to_dataset(ds_pe, float(words[4]), row)
                    write_to_dataset(ds_cq, float(words[5]), row)
                    row += 1
                counter += 1

        # Check number of rows
        check_trajectory_rows(tgrp, dss, row)
