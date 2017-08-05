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
'''Diffusion constants'''


from __future__ import division

import numpy as np

from yaff.log import log
from yaff.analysis.hook import AnalysisInput, AnalysisHook
from yaff.analysis.blav import blav


__all__ = ['Diffusion']


class Diffusion(AnalysisHook):
    def __init__(self, f=None, start=0, end=-1, step=1, mult=20, select=None,
                 bsize=None, pospath='trajectory/pos', poskey='pos', outpath=None):
        """Computes mean-squared displacements and diffusion constants

           **Optional arguments:**

           f
                An h5.File instance containing the trajectory data. If ``f``
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

           bsize
                If given, time intervals that coincide with the boundaries of
                the block size, will not be considered form the analysis. This
                is useful when there is a significant monte carlo move between
                subsequent blocks. If step > 1, the intervals will be left out
                if the overlap with boundaries of blocks with size bsize*step.

           pospath
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis.

           poskey
                In case of an on-line analysis, this is the key of the state
                item that contains the data from which the MSD's are derived.

           outpath
                The output path for the MSD results. If not given, it defaults
                to '%s_diff' % path. If this path already exists, it will be
                removed first.

        """
        if bsize is not None and bsize < mult:
            raise ValueError('The bsize parameter must be larger than mult.')
        self.mult = mult
        self.select = select
        self.bsize = bsize
        self.msdsums = np.zeros(self.mult, float)
        self.msdcounters = np.zeros(self.mult, int)
        self.counter = 0
        if outpath is None:
            outpath = pospath + '_diff'
        analysis_inputs = {'pos': AnalysisInput(pospath, poskey)}
        AnalysisHook.__init__(self, f, start, end, None, step, analysis_inputs, outpath, True)

    def init_first(self):
        # update the shape if select is present
        if self.select is not None:
            self.shape = (len(self.select),) + self.shape[1:]
        # compute the number of dimensions, i.e. 3 for atoms
        self.ndim = 1
        for s in self.shape[1:]:
            self.ndim *= s
        # allocate working arrays
        self.last_poss = [np.zeros(self.shape, float) for i in range(self.mult)]
        self.pos = np.zeros(self.shape, float)
        # prepare the hdf5 output file, if present.
        AnalysisHook.init_first(self)
        if self.outg is not None:
            for m in range(self.mult):
                self.outg.create_dataset('msd%03i' % (m+1), shape=(0,), maxshape=(None,), dtype=float)
            self.outg.create_dataset('msdsums', data=self.msdsums)
            self.outg.create_dataset('msdcounters', data=self.msdcounters)
            self.outg.create_dataset('pars', shape=(2,), dtype=float)
            self.outg.create_dataset('pars_error', shape=(2,), dtype=float)

    def configure_online(self, iterative, st_pos):
        self.shape = st_pos.shape

    def configure_offline(self, ds_pos):
        self.shape = ds_pos.shape[1:]

    def init_timestep(self):
        pass

    def read_online(self, st_pos):
        if self.select is None:
            self.pos[:] = st_pos.value
        else:
            self.pos[:] = st_pos.value[self.select]

    def read_offline(self, i, ds_pos):
        if self.select is None:
            ds_pos.read_direct(self.pos, (i,))
        else:
            ds_pos.read_direct(self.pos, (i, self.select))

    def compute_iteration(self):
        for m in range(self.mult):
            if self.counter % (m+1) == 0:
                if self.counter > 0 and not self.overlap_bsize(m):
                    msd = ((self.pos - self.last_poss[m])**2).mean()*self.ndim
                    self.update_msd(msd, m)
                self.last_poss[m][:] = self.pos
        self.counter += 1

    def compute_derived(self):
        positive = (self.msdcounters > 0).nonzero()[0]
        if len(positive) > 0:
            self.msds = self.msdsums[positive]/self.msdcounters[positive]
            self.time = np.arange(1, self.mult+1)[positive]*self.timestep
            # make error estimates for A and B, first compute error
            # estimates for the msds array.
            if self.outg is None:
                self.msds_error = None
            else:
                self.msds_error = []
                for m in positive:
                    try:
                        error = blav(self.outg['msd%03i' % (m+1)][:], minblock=10)[0]
                        self.msds_error.append(error)
                    except ValueError:
                        self.msds_error = None
                        break
                if self.msds_error is not None:
                    self.msds_error = np.array(self.msds_error)
            dm = np.array([self.time, np.ones(len(self.time))])
            if self.msds_error is None:
                self.A, self.B = np.linalg.lstsq(dm.T, self.msds)[0]
                self.A_error = None
                self.B_error = None
            else:
                dm = (dm/self.msds_error).T
                ev = self.msds/self.msds_error
                U, S, Vt = np.linalg.svd(dm, full_matrices=False)
                self.A, self.B = np.dot(Vt.T, np.dot(U.T, ev)/S)
                self.A_error, self.B_error = np.dot(Vt.T, U.sum(axis=0)/S)
            if self.outg is not None:
                # write out all results
                if 'time' in self.outg:
                    del self.outg['time']
                    del self.outg['msds']
                self.outg['time'] = self.time
                self.outg['msds'] = self.msds
                self.outg['msdsums'][:] = self.msdsums
                self.outg['msdcounters'][:] = self.msdcounters
                self.outg['pars'][0] = self.A
                self.outg['pars'][1] = self.B
                if self.msds_error is not None:
                    if 'msds_error' in self.outg:
                        del self.outg['msds_error']
                    self.outg['msds_error'] = self.msds_error
                    self.outg['pars_error'][0] = self.A_error
                    self.outg['pars_error'][1] = self.B_error

    def overlap_bsize(self, m):
        if self.bsize is None:
            return False
        else:
            return self.counter - (self.counter//self.bsize)*self.bsize - m - 1 < 0

    def update_msd(self, msd, m):
        self.msdsums[m] += msd
        self.msdcounters[m] += 1
        if self.outg is not None:
            ds = self.outg['msd%03i' % (m+1)]
            row = ds.shape[0]
            ds.resize(row+1, axis=0)
            ds[row] = msd

    def plot(self, fn_png='msds.png'):
        import matplotlib.pyplot as pt
        pt.clf()
        if self.msds_error is None:
            pt.plot(self.time/log.time.conversion, self.msds/log.area.conversion, 'k+')
        else:
            pt.errorbar(
                self.time/log.time.conversion, self.msds/log.area.conversion,
                self.msds_error/log.area.conversion, fmt='k+'
            )
        t2 = np.array([0, self.time[-1]+self.timestep], float)
        pt.plot(t2/log.time.conversion, (self.A*t2+self.B)/log.area.conversion, 'r-')
        pt.xlabel('Time [%s]' % log.time.notation)
        pt.ylabel('MSD [%s]' % log.area.notation)
        A_repr = log.diffconst(self.A).strip()
        if self.msds_error is not None:
            A_repr += ' +- ' + log.diffconst(self.A_error).strip()
        pt.title('Diffusion constant: %s [%s]' % (A_repr, log.diffconst.notation))
        pt.savefig(fn_png)
