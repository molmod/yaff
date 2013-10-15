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
'''The block-average method'''


import numpy as np


__all__ = ['blav']


def blav(signal, minblock=100, fn_png=None, unit=None):
    """Analyze the signal with the block average method.

       The variance on the block average error as function of block size is
       fitted using the ``a+b/bsize`` model, where ``a`` is a measure for the
       error on the average, i.e. when the block size becomes infintely large.
       If the fit fails, a coarse estimate of the error on the average is
       returned, i.e. the largest block average error.

       **Arguments:**

       signal
            An array containing time-depedendent data.

       **Optional arguments:**

       minblock
            The minimum number of blocks to be considered.

       fn_png
            When given, the data used for the fit and the fitted model are
            plotted.

       unit
            This is only relevant when a plot is made. It is used as the unit
            for the y-axis, e.g. ``log.length``.

       **Returns:**

       error
            The fitted error.

       sinef
            The fitted statistical inefficiency.
    """
    x = [] # block sizes
    e = [] # errors on the mean

    for bsize in xrange(1, len(signal)/minblock):
        nblock = len(signal)/bsize
        total_size = nblock * bsize
        averages = signal[:total_size].reshape((nblock, bsize)).mean(axis=1)
        x.append(bsize)
        e.append(averages.std()/np.sqrt(nblock))

    x = np.array(x, dtype=float)
    e = np.array(e)

    # perform the fit on the last two thirds of the data points
    l = len(e)*2/3
    if l == 0:
        raise ValueError("Too few blocks to do a proper estimate of the error.")
    # estimate the limit of the error towards large block sizes
    dm = np.array([np.ones(l), 1/x[-l:]]).transpose()
    ev = e[-l:]
    error, b = np.linalg.lstsq(dm, ev)[0]
    # improve robustness, in case the fitting went wrong
    if error < 0 or b > 0:
        error = e.max()
        b = 0.0
    # compute the ratio with the naive error
    sinef = error/e[0]

    if fn_png is not None:
        import matplotlib.pyplot as pt
        if unit is None:
            conversion = 1
            notation = '1'
        else:
            converions = unit.conversion
            notation = unit.notation
        pt.clf()
        pt.plot(x[:-l], e[:-l]/conversion, 'k+', alpha=0.5)
        pt.plot(x[-l:], e[-l:]/conversion, 'k+')
        pt.plot(x[-l:], (error+b/x[-l:])/conversion, 'r-')
        pt.axhline(error/conversion, color='r')
        pt.ylabel('Error on the block average [%s]' % notation)
        pt.xlabel('Block size')
        pt.savefig(fn_png)

    return error, sinef
