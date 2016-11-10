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
from molmod.units import *
import matplotlib.pyplot as pt


__all__ = ['blav', 'inefficiency']


def blav(signal, minblock=100, fn_png=None, unit=None):
    """Analyze the signal with the block average method.

       The variance on the block average error as function of block size is
       fitted using the ``a+b/bsize`` model, where ``a`` is a measure for the
       error on the average, i.e. when the block size becomes infintely large.
       If the fit fails, a coarse estimate of the error on the average is
       returned, i.e. the largest block average error.

       **Arguments:**

       signal
            An array containing time-dependent data.

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
            conversion = unit.conversion
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


def inefficiency(signal, time = None, fn_png = 'stat_ineff.png', taus = None, eq_limits = None):
    """
        Analyze the signal to determine the statistical inefficiency. The statistical
        inefficiency of a signal is defined as the limiting ratio of the observed
        variance of its long-term averages to their expected variance. It can hence be
        regarded as the factor (>1) with which the sample size should be multiplied in
        order to compensate for correlation. This is derived in:

            Friedberg, R; Cameron, J.E. J. Chem. Phys. 1970, 52, 6049-6058.


        **Arguments:**

        signal
            An array containing time-dependent data.

        **Optional arguments:**

        time
            An array containing the time information of the provided signal.

        fn_png
            Name of the plot to which the data is written.

        taus
            An array, where for each element tau the block averages with this
            block size are determined.

        eq_limits
            An array containing the fractions of the signal to be considered
            as equilibration.

       **Returns:**
    """

    total_length = len(signal)
    if taus is None:
        # By default, take the block lengths between 0.02 % and 1%
        # of the total simulation time, and make sure that all entries are integers
        taus = np.arange(total_length/5000, total_length/100, total_length/5000)
        taus = taus.astype(int)
    if eq_limits is None:
        # By default, take the equilibration time between 0 % and 80 %
        # of the total simulation time with a stepsize of 10 %,
        # and make sure that all entries are integers
        eq_limits = np.arange(0, 0.8*total_length, 0.1*total_length)
        eq_limits = eq_limits.astype(int)

    # Initialize the statistical inefficiency phi
    phi = np.zeros((len(eq_limits), len(taus)))
    varX = np.var(signal)

    # Calculate the statisticial inefficiencies
    for i in xrange(len(eq_limits)):
        eq_limit = eq_limits[i]
        for j in xrange(len(taus)):
            tau = taus[j]
            nblock = (len(signal)-eq_limit)/tau
            total_size = nblock*tau
            averages = signal[eq_limit:eq_limit+total_size].reshape((nblock, tau)).mean(axis=1)
            phi[i,j] = 1.*tau*np.var(averages)/varX

    # Plot the statistical ineffiency
    if time is not None:
        xlabel = 'Length of segment [ps]'
        unit = 1./((time[1]-time[0])/picosecond)
        unit_ab = 'ps'
    else:
        xlabel = 'Length of segment []'
        unit = 1
        unit_ab = 'steps'

    pt.clf()
    comap = pt.cm.get_cmap(name='jet')
    pt.xlabel(xlabel)
    pt.ylabel('Statistical inefficiency')

    for i in xrange(len(eq_limits)):
        clr = 1.*i/len(eq_limits)
        pt.plot(taus/unit, phi[i,:], color=comap(clr), label='Equilibrated for %i %s' %(eq_limits[i]/unit, unit_ab))
    pt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    pt.savefig(fn_png, bbox_inches='tight')
