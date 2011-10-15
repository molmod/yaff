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
from yaff.analysis.utils import get_slice
from yaff.sampling.iterative import Hook


__all__ = ['Diffusion']


class Diffusion(Hook):
    def __init__(self, f, start=0, end=-1, step=1, mult=20, select=None,
                 path='trajectory/pos', key='pos', outpath=None):
        """Computes mean-squared displacements and diffusion constants

           **Arguments:**

           f
                An h5py.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           **Optional arguments:**

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
        self.f = f
        self.start, self.end, self.step = get_slice(self.f, start, end, step=step)
        self.mult = mult
        self.select = select
        self.path = path
        self.key = key
        # prepare the output path
        if outpath is None:
            self.outpath = '%s_diff' % path
        else:
            self.outpath = outpath
        # create the output group
        if self.f is not None:
            if self.outpath in self.f:
                del self.f[outpath]
            self.outg = self.f.create_group(self.outpath)
        else:
            self.outg = None

        # prepare some attributes
        self.msdsums = np.zeros(self.mult, float)
        self.msdcounters = np.zeros(self.mult, int)
        if self.outg is not None:
            for m in xrange(self.mult):
                self.outg.create_dataset('msd%03i' % (m+1), shape=(0,), maxshape=(None,), dtype=float)

        self.online = self.f is None or path not in self.f
        if not self.online:
            self.compute_offline()
        else:
            raise NotImplementedError

    def compute_offline(self):
        # prepare data
        self.timestep = self.f['trajectory/time'][self.start+self.step] - self.f['trajectory/time'][self.start]
        ds = self.f[self.path]
        if self.select is None:
            shape = ds.shape[1:]
        else:
            shape = (len(self.select),) + ds.shape[2:]
        last_poss = [np.zeros(shape, float) for i in xrange(self.mult)]
        pos = np.zeros(shape, float)
        # Iterate over the dataset
        counter = 0
        for i in xrange(self.start, self.end, self.step):
            # load data
            if self.select is None:
                ds.read_direct(pos, (i,))
            else:
                ds.read_direct(pos, (i,self.select))
            # compute MSD's and update last_poss
            for m in xrange(self.mult):
                if counter % (m+1) == 0:
                    if counter > 0:
                        msd = ((pos - last_poss[m])**2).mean()*3
                        self.update_msd(msd, m)
                    last_poss[m][:] = pos
            # update last_poss
            counter += 1
        # compute derive properties
        self.compute_derived()

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
        self.msds = self.msdsums[mask]/self.msdcounters[mask]
        self.time = np.arange(1, self.mult+1)[mask]*self.timestep
        dm = np.array([self.time, np.ones(len(self.time))]).T
        # TODO: add error estimates of outg is not None
        self.A, self.B = np.linalg.lstsq(dm, self.msds)[0]
        if self.outg is not None:
            self.outg['time'] = self.time
            self.outg['msds'] = self.msds
            self.outg['msdsums'] = self.msdsums
            self.outg['msdcounters'] = self.msdcounters
            self.outg['pars'] = np.array([self.A, self.B])

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
