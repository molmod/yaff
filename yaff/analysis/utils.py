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
'''Auxiliary analysis routines'''


import h5py as h5


__all__ = ['get_slice']


def get_slice(f, start=0, end=-1, max_sample=None, step=None):
    """
       **Argument:**

       f
            A HDF5.File instance, may be None if it is not available. If it
            contains a trajectory group, this group will be used to determine
            the number of time steps in the trajectory.

       **Optional arguments:**

       start
            The first sample to be considered for analysis. This may be negative
            to indicate that the analysis should start from the -start last
            samples.

       end
            The last sample to be considered for analysis. This may be negative
            to indicate that the last -end sample should not be considered.

       max_sample
            When given, step is set such that the number of samples does not
            exceed max_sample.

       step
            The spacing between the samples used for the analysis

       The optional arguments can be given to all of the analysis routines. Just
       make sure you never specify step and max_sample at the same time. The
       max_sample argument assures that the step is set such that the number of
       samples does (just) not exceed max_sample. The max_sample option only
       works when f is not None, or when end is positive.

       if f is present or start and end are positive, and max_sample and step or
       not given, max_sample defaults to 1000.

       Returns start, end and step. When f is given, start and end are always
       positive.
    """
    if f is None or 'trajectory' not in f:
        nrow = None
    else:
        nrow = min(ds.shape[0] for ds in f['trajectory'].itervalues() if isinstance(ds, h5.Dataset))
        if end < 0:
            end = nrow + end + 1
        else:
            end = min(end, nrow)
        if start < 0:
            start = nrow + start + 1
    if start > 0 and end > 0 and step is None and max_sample is None:
        max_sample = 1000
    if step is None:
        if max_sample is None:
            return start, end, 1
        else:
            if end < 0:
                raise ValueError('When max_sample is given and end is negative, a file must be present.')
            step = max(1, (end - start)/max_sample + 1)
    elif max_sample is not None:
        raise ValueError('Both step and max_sample are given at the same time.')
    return start, end, step
