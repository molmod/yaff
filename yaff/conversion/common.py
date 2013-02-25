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


from yaff.log import log


__all__ = [
    'get_trajectory_group', 'get_trajectory_datasets', 'append_to_dataset',
    'check_trajectory_rows'
]


def get_trajectory_group(f):
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


def get_trajectory_datasets(tgrp, *fields):
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


def append_to_dataset(ds, value, row):
    if ds.shape[0] <= row:
        ds.resize(row+1, axis=0)
    ds[row] = value


def check_trajectory_rows(tgrp, existing_row, row):
    if existing_row is None:
        tgrp.attrs['row'] = row
    else:
        if existing_row != row:
            raise ValueError('The amount of data loaded into the HDF5 file is not consistent with number of rows already present in the trajectory.')
