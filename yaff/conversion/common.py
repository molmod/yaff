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
'''Tools for writing trajectory data'''


import h5py as h5

from yaff.log import log


__all__ = [
    'get_trajectory_group', 'get_trajectory_datasets',
    'get_last_trajectory_row', 'write_to_dataset', 'check_trajectory_rows'
]


def get_trajectory_group(f):
    '''Create or return an existing trajectory group

       **Arguments:**

       f
            An open HDF5 File or Group object.
    '''
    if 'trajectory' not in f:
        if log.do_high:
            log('Creating new trajectory datagroup in %s.' % f.filename)
        tgrp = f.create_group('trajectory')
    else:
        tgrp = f['trajectory']
        if log.do_high:
            log('Using existing trajectory datagroup in %s.' % f.filename)
    return tgrp


def get_trajectory_datasets(tgrp, *fields):
    '''Return a list of new/existing datasets corresponding to the given fields

       **Arguments:**

       tgrp
            The trajectory group

       fields
            A list of fields, i.e. pairs of name and row_shape.
    '''
    result = []
    for name, row_shape in fields:
        if name in tgrp:
            ds = tgrp[name]
            if ds.shape[1:] != row_shape:
                raise TypeError('The shape of the existing dataset is not compatible with the new data.')
            if log.do_medium:
                log('Found an existing dataset %s in group %s with %i rows.' % (name, tgrp.name, ds.shape[0]))
        else:
            if log.do_high:
                log('Creating new dataset %s with row shape %s' % (name, row_shape))
            # Create a new dataset
            shape = (0,) + row_shape
            maxshape = (None,) + row_shape
            ds = tgrp.create_dataset(name, shape, maxshape=maxshape, dtype=float)
        result.append(ds)
    return result


def get_last_trajectory_row(dss):
    '''Find the first row to write new trajectory data.

       **Arguments:**

       dss
            A list of datasets or the trajectory group.
    '''
    if isinstance(dss, h5.Group):
        dss = dss.itervalues()
    row = min(ds.shape[0] for ds in dss)
    return row


def write_to_dataset(ds, value, row):
    '''Write a result at a given row in a trajectory. If needed, the dataset is extended.

       **Arguments:**

       ds
            The dataset

       value
            The data to be written

       row
            The row index.
    '''
    if ds.shape[0] <= row:
        ds.resize(row+1, axis=0)
    ds[row] = value


def check_trajectory_rows(tgrp, dss, row):
    '''Check if the datasets with the new trajectory data have consistent sizes.

       **Arguments:**

       tgrp
            The trajectory group.

       dss
            The list of datasets that was filled with data.

       row
            The last row.
    '''
    # check the sizes of the modified datasets
    for ds in dss:
        assert ds.shape[0] >= row
