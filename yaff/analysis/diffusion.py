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

from yaff.log import log
from yaff.analysis.hook import AnalysisHook


__all__ = ['Diffusion']


class Diffusion(AnalysisHook):
    label = 'diff'

    def __init__(self, f=None, start=0, end=-1, step=1, mult=20, select=None,
                 path='trajectory/pos', key='pos', outpath=None):
        """Computes mean-squared displacements and diffusion constants

           **Optional arguments:**

           f
                An h5py.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           start, end, step
                Optional arguments for the ``get_slice`` function. max_sample is
                not supported because the choice of the step argument is
                critical for a useful result.

           mult
                In the first place, the mean square displacement (MSD) between
                subsequent step is computed. The MSD is also computed between
                every, two, three, ..., until ``mult`` steps.

           select
                A list of atom indexes that are considered for the computation
                of the MSD's. If not given, all atoms are used.

           path
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis.

           key
                In case of an on-line analysis, this is the key of the state
                item that contains the data from which the MSD's are derived.

           outpath
                The output path for the MSD results. If not given, it defaults
                to '%s_diff' % path. If this path already exists, it will be
                removed first.

        """
        # TODO: Add bsize optional argument, to avoids intervals that contain
        #       boundaries between two blocks.
        self.mult = mult
        self.select = select
        self.msdsums = np.zeros(self.mult, float)
        self.msdcounters = np.zeros(self.mult, int)
        self.counter = 0
        AnalysisHook.__init__(self, f, start, end, None, step, path, key, outpath, True)

    def init_timestep(self):
        pass

    def configure_online(self, iterative):
        self.shape = iterative.state[self.key].shape

    def configure_offline(self, ds):
        self.shape = ds.shape[1:]

    def init_first(self):
        # update the shape if select is present
        if self.select is not None:
            self.shape = (len(self.select),) + self.shape[1:]
        # compute the number of dimensions, i.e. 3 for atoms
        self.ndim = 1
        for s in self.shape[1:]:
            self.ndim *= s
        # allocate working arrays
        self.last_poss = [np.zeros(self.shape, float) for i in xrange(self.mult)]
        self.pos = np.zeros(self.shape, float)
        # prepare the hdf5 output file, if present.
        AnalysisHook.init_first(self)
        if self.outg is not None:
            for m in xrange(self.mult):
                self.outg.create_dataset('msd%03i' % (m+1), shape=(0,), maxshape=(None,), dtype=float)
            self.outg.create_dataset('msdsums', data=self.msdsums)
            self.outg.create_dataset('msdcounters', data=self.msdcounters)
            self.outg.create_dataset('pars', shape=(2,), dtype=float)

    def read_online(self, iterative):
        if self.select is None:
            self.pos[:] = iterative.state[self.key].value
        else:
            self.pos[:] = iterative.state[self.key].value[self.select]

    def read_offline(self, ds, i):
        if self.select is None:
            ds.read_direct(self.pos, (i,))
        else:
            ds.read_direct(self.pos, (i, self.select))

    def compute_iteration(self):
        for m in xrange(self.mult):
            if self.counter % (m+1) == 0:
                if self.counter > 0:
                    msd = ((self.pos - self.last_poss[m])**2).mean()*self.ndim
                    self.update_msd(msd, m)
                self.last_poss[m][:] = self.pos
        self.counter += 1

    def update_msd(self, msd, m):
        self.msdsums[m] += msd
        self.msdcounters[m] += 1
        if self.outg is not None:
            ds = self.outg['msd%03i' % (m+1)]
            row = ds.shape[0]
            ds.resize(row+1, axis=0)
            ds[row] = msd

    def compute_derived(self):
        mask = self.msdcounters > 0
        if mask.sum() > 1:
            self.msds = self.msdsums[mask]/self.msdcounters[mask]
            self.time = np.arange(1, self.mult+1)[mask]*self.timestep
            dm = np.array([self.time, np.ones(len(self.time))]).T
            # TODO: add error estimates of outg is not None
            self.A, self.B = np.linalg.lstsq(dm, self.msds)[0]
            if self.outg is not None:
                if 'time' in self.outg:
                    del self.outg['time']
                    del self.outg['msds']
                self.outg['time'] = self.time
                self.outg['msds'] = self.msds
                self.outg['msdsums'][:] = self.msdsums
                self.outg['msdcounters'][:] = self.msdcounters
                self.outg['pars'][:] = np.array([self.A, self.B])

    def plot(self, fn_png='msds.png'):
        import matplotlib.pyplot as pt
        pt.clf()
        pt.plot(self.time/log.time.conversion, self.msds/log.area.conversion, 'k+')
        t2 = np.array([0, self.time[-1]+self.timestep], float)
        pt.plot(t2/log.time.conversion, (self.A*t2+self.B)/log.area.conversion, 'r-')
        pt.xlabel('Time [%s]' % log.time.notation)
        pt.ylabel('MSD [%s]' % log.area.notation)
        pt.title('Diffusion constant: %s [%s]' % (log.diffconst(self.A), log.diffconst.notation))
        pt.savefig(fn_png)
