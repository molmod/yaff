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
'''Spectral analysis and autocorrelation functions'''


from __future__ import division

import numpy as np

from molmod.constants import lightspeed
from molmod.units import centimeter
from yaff.log import log
from yaff.analysis.utils import get_slice
from yaff.analysis.hook import AnalysisInput, AnalysisHook


__all__ = ['Spectrum']


class Spectrum(AnalysisHook):
    def __init__(self, f=None, start=0, end=-1, step=1, bsize=4096, select=None,
                 path='trajectory/vel', key='vel', outpath=None, weights=None):
        """
           **Optional arguments:**

           f
                An h5.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           start
                The first sample to be considered for analysis. This may be
                negative to indicate that the analysis should start from the
                -start last samples.

           end
                The last sample to be considered for analysis. This may be
                negative to indicate that the last -end sample should not be
                considered.

           step
                The spacing between the samples used for the analysis

           bsize
                The size of the blocks used for individual FFT calls.

           select
                A list of atom indexes that are considered for the computation
                of the spectrum. If not given, all atoms are used.

           path
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis. The spectra are summed over the other axes.

           key
                In case of an on-line analysis, this is the key of the state
                item that contains the data from which the spectrum is derived.

           outpath
                The output path for the frequency computation in the HDF5 file.
                If not given, it defaults to '%s_spectrum' % path. If this path
                already exists, it will be removed first.

           weights
                If not given, the spectrum is just a simple sum of contributions
                from different time-dependent functions. If given, a linear
                combination is made based on these weights.

           The max_sample argument from get_slice is not used because the choice
           step value is an important parameter: it is best to choose step*bsize
           such that it coincides with a part of the trajectory in which the
           velocities (or other data) are continuous.

           The block size should be set such that it corresponds to a decent
           resolution on the frequency axis, i.e. 33356 fs of MD data
           corresponds to a resolution of about 1 cm^-1. The step size should be
           set such that the highest frequency is above the highest relevant
           frequency in the spectrum, e.g. a step of 10 fs corresponds to a
           frequency maximum of 3336 cm^-1. The total number of FFT's, i.e.
           length of the simulation divided by the block size multiplied by the
           number of time-dependent functions in the data, determines the noise
           reduction on the (the amplitude of) spectrum. If there is sufficient
           data to perform 10K FFT's, one should get a reasonably smooth
           spectrum.

           Depending on the FFT implementation in numpy, it may be interesting
           to tune the bsize argument. A power of 2 is typically a good choice.

           When f is None, or when the path does not exist in the HDF5 file, the
           class can be used as an on-line analysis hook for the iterative
           algorithms in yaff.sampling package. This means that the spectrum
           is built up as the iterative algorithm progresses. The end option is
           ignored for an on-line analysis.
        """
        self.bsize = bsize
        self.select = select
        self.weights = weights
        self.ssize = self.bsize//2+1 # the length of the spectrum array
        self.amps = np.zeros(self.ssize, float)
        self.nfft = 0 # the number of fft calls, for statistics
        if outpath is None:
            outpath = path + '_spectrum'
        analysis_inputs = {'signal': AnalysisInput(path, key)}
        AnalysisHook.__init__(self, f, start, end, None, step, analysis_inputs, outpath, True)

    def init_online(self):
        AnalysisHook.init_online(self)
        self.ncollect = 0

    def _iter_indexes(self, array):
        if self.select is None:
            for indexes in np.ndindex(array.shape[1:]):
                yield indexes
        else:
            for i0 in self.select:
                for irest in np.ndindex(array.shape[2:]):
                    yield (i0,) + irest

    def _get_weight(self, indexes):
        if self.weights is None:
            return 1.0
        else:
            return self.weights[indexes]

    def init_timestep(self):
        self.freqs = np.arange(self.ssize)/(self.timestep*self.bsize)
        self.time = np.arange(self.ssize)*self.timestep
        if self.outg is not None:
            self.outg['freqs'][:] = self.freqs
            self.outg['time'][:] = self.time

    def offline_loop(self, ds_signal):
        # Compute the amplitudes of the spectrum
        current = self.start
        stride = self.step*self.bsize
        work = np.zeros(self.bsize, float)
        while current <= self.end - stride:
            for indexes in self._iter_indexes(ds_signal):
                ds_signal.read_direct(work, (slice(current, current+stride, self.step),) + indexes)
                self.amps += self._get_weight(indexes)*abs(np.fft.rfft(work))**2
                self.nfft += 1
            current += stride
        # Compute related arrays
        self.compute_derived()

    def configure_online(self, iterative, st_signal):
        if self.select is None:
            shape = (self.bsize,) + st_signal.shape
        else:
            shape = (self.bsize, self.select) + st_signal.shape[1:]
        self.work = np.zeros(shape, float)

    def configure_offline(self, ds_signal):
        self.work = np.zeros(self.bsize)

    def init_first(self):
        AnalysisHook.init_first(self)
        if self.outg is not None:
            self.outg.create_dataset('amps', (self.ssize,), float)
            self.outg.create_dataset('freqs', (self.ssize,), float)
            self.outg.create_dataset('ac', (self.ssize,), float)
            self.outg.create_dataset('time', (self.ssize,), float)

    def read_online(self, st_signal):
        if self.select is None:
            self.work[self.ncollect] = st_signal.value
        else:
            self.work[self.ncollect] = st_signal.value[self.select]
        self.ncollect += 1

    def compute_iteration(self):
        if self.ncollect == self.bsize:
            # collected sufficient data to fill one block, computing FFT
            for indexes in self._iter_indexes(self.work):
                work = self.work[(slice(0, self.bsize),) + indexes]
                self.amps += self._get_weight(indexes)*abs(np.fft.rfft(work))**2
                self.nfft += 1
            # compute some derived stuff
            self.compute_derived()
            # reset some things
            self.work[:] = 0.0
            self.ncollect = 0

    def compute_derived(self):
        self.ac = np.fft.irfft(self.amps)[:self.ssize]
        if self.outg is not None:
            self.outg['amps'][:] = self.amps
            self.outg['ac'][:] = self.ac
            self.outg.attrs['nfft'] = self.nfft

    def plot(self, fn_png='spectrum.png', do_wavenum=True, xlim=None, verticals=None, thermostat=None, ndof=None):
        """
            verticals: array containing as first entry the timeconstant of the original
            system, and as following entries the wavenumbers of the original system.
        """

        import matplotlib.pyplot as pt
        if do_wavenum:
            xunit = lightspeed/centimeter
            xlabel = 'Wavenumber [1/cm]'
        else:
            xunit = 1/log.time.conversion
            xlabel = 'Frequency [1/%s]' % log.time.notation
        pt.clf()
        pt.plot(self.freqs/xunit, self.amps)
        if verticals is not None:
            thermo_freq = 1.0/verticals[0]/lightspeed*centimeter
            #plot frequencies original system, and coupling to thermostat
            for i in np.arange(1, len(verticals)):
                pt.axvline(verticals[i], color='r', ls='--')
                pt.axvline(verticals[i] + thermo_freq, color='g', ls='--')
                pt.axvline(verticals[i] - thermo_freq, color='g', ls='--')
        if thermostat is not None and ndof is not None:
            thermo_freq = 1.0/thermostat/lightspeed*centimeter
            pt.axvline(thermo_freq, color='k', ls='--')
            pt.axvline(thermo_freq/np.sqrt(ndof), color='k', ls='--')
            pt.axvline(thermo_freq+thermo_freq/np.sqrt(ndof), color='r', ls='--')
            pt.axvline(thermo_freq+2.0*thermo_freq/np.sqrt(ndof), color='r', ls='--')

        if xlim is not None:
            pt.xlim(xlim[0]/xunit, xlim[1]/xunit)
        else:
            pt.xlim(0, self.freqs[-1]/xunit)



        pt.xlabel(xlabel)
        pt.ylabel('Amplitude')
        pt.savefig(fn_png)

    def plot_ac(self, fn_png='ac.png'):
        import matplotlib.pyplot as pt
        pt.clf()
        pt.plot(self.time/log.time.conversion, self.ac/self.ac[0])
        pt.xlabel('Time [%s]' % log.time.notation)
        pt.ylabel('Autocorrelation')
        pt.savefig(fn_png)
