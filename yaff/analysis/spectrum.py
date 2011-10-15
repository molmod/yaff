# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
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


import numpy as np

from molmod.constants import lightspeed
from molmod.units import centimeter
from yaff.log import log
from yaff.analysis.utils import get_slice
from yaff.sampling.iterative import Hook


__all__ = ['Spectrum']


class Spectrum(Hook):
    def __init__(self, f, start=0, end=-1, step=1, bsize=4096, select=None,
                 path='trajectory/vel', key='vel', outpath=None):
        """
           **Argument:**

           f
                An h5py.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           **Optional arguments:**

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
           is built up as the itertive algorithm progresses. The end option is
           ignored for an on-line analysis.
        """
        self.f = f
        self.start, self.end, self.step = get_slice(self.f, start, end, step=step)
        self.bsize = bsize
        self.select = select
        self.ssize = self.bsize/2+1 # the length of the spectrum array
        self.amps = np.zeros(self.ssize, float)
        self.timestep = None
        self.time = None
        self.freqs = None

        self.path = path
        self.key = key
        if outpath is None:
            self.outpath = '%s_spectrum' % path
        else:
            self.outpath = outpath
        self.online = self.f is None or path not in self.f
        if not self.online:
            self.compute_offline()
        else:
            # some attributes that are only relevant for on-line analysis
            self.work = None
            self.ncollect = 0
            self.lasttime = None

    def _iter_indexes(self, array):
        if self.select is None:
            for indexes in np.ndindex(array.shape[1:]):
                yield indexes
        else:
            for i0 in self.select:
                for irest in np.ndindex(array.shape[2:]):
                    yield (i0,) + irest

    def __call__(self, iterative):
        if self.work is None:
            self.work = np.zeros((self.bsize,) + iterative.state[self.key].shape, float)
        if self.timestep is None:
            if self.lasttime is None:
                self.lasttime = iterative.state['time'].value
            else:
                self.timestep = iterative.state['time'].value - self.lasttime
                del self.lasttime
        self.work[self.ncollect] = iterative.state[self.key].value
        self.ncollect += 1
        if self.ncollect == self.bsize:
            # collected sufficient data to fill one block, computing FFT
            for indexes in self._iter_indexes(self.work):
                work = self.work[(slice(0, self.bsize),) + indexes]
                self.amps += abs(np.fft.rfft(work))**2
            # compute some derived stuff
            self.compute_derived()
            # reset some things
            self.work[:] = 0.0
            self.ncollect = 0

    def compute_offline(self):
        # Compute the amplitudes of the spectrum
        current = self.start
        stride = self.step*self.bsize
        work = np.zeros(self.bsize, float)
        ds = self.f[self.path]
        while current <= self.end - stride:
            for indexes in self._iter_indexes(ds):
                ds.read_direct(work, (slice(current, current+stride, self.step),) + indexes)
                self.amps += abs(np.fft.rfft(work))**2
            current += stride
        # Compute related arrays
        self.timestep = self.f['trajectory/time'][self.start+self.step] - self.f['trajectory/time'][self.start]
        self.compute_derived()

    def compute_derived(self):
        first = self.freqs is None
        if first:
            self.freqs = np.arange(self.ssize)/(self.timestep*self.ssize)
            self.time = np.arange(self.ssize)*self.timestep
        self.ac = np.fft.irfft(self.amps)[:self.ssize]
        # Write the results to the HDF5 file
        if self.f is not None:
            if self.outpath in self.f:
                del self.f[self.outpath]
            g = self.f.create_group(self.outpath)
            g['amps'] = self.amps
            g['ac'] = self.ac
            if first:
                g['freqs'] = self.freqs
                g['time'] = self.time

    def plot(self, fn_png='spectrum.png', do_wavenum=True):
        import matplotlib.pyplot as pt
        if do_wavenum:
            xunit = lightspeed/centimeter
            xlabel = 'Wavenumber [1/cm]'
        else:
            xunit = 1/log.time.conversion
            xlabel = 'Frequency [1/%s]' % log.time.notation
        pt.clf()
        pt.plot(self.freqs/xunit, self.amps)
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
