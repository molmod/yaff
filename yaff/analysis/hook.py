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


from yaff.log import log
from yaff.sampling.iterative import Hook
from yaff.analysis.utils import get_slice


__all__ = ['AnalysisHook']


class AnalysisHook(Hook):
    label = None

    def __init__(self, f=None, start=0, end=-1, max_sample=None, step=None,
                 path=None, key=None, outpath=None, do_timestep=False):
        """Base class for the ansysis hooks.

           Analysis hooks in Yaff support both off-line and on-line analysis.

           **Optional arguments:**

           f
                An h5py.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           start, end, max_sample, step
                Optional arguments for the ``get_slice`` function.

           path
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis.

           key
                In case of an on-line analysis, this is the key of the state
                item that contains the main data that will be used for the
                analysis.

           outpath
                The output path for the analysis. If not given, it defaults
                to '%s_%' % (path, self.label). If this path already exists, it
                will be removed first.

           do_timestep
                When True, a self.timestep attribute will be initialized as
                early as possible.
        """
        self.f = f
        self.start, self.end, self.step = get_slice(self.f, start, end, max_sample, step)
        self.path = path
        self.key = key
        # prepare the output path
        if outpath is None:
            self.outpath = '%s_%s' % (path, self.label)
        else:
            self.outpath = outpath
        self.do_timestep = do_timestep

        self.online = self.f is None or path not in self.f
        if not self.online:
            self.compute_offline()
        else:
            self.init_online()

    def __call__(self, iterative):
        # prepare some data structures
        if self._first_iteration:
            self.configure_online(iterative)
            self.init_first()
            self._first_iteration = False
        # get the time step
        if self.do_timestep and self.timestep is None:
            if self.lasttime is None:
                self.lasttime = iterative.state['time'].value
            else:
                self.timestep = iterative.state['time'].value - self.lasttime
                del self.lasttime
        self.read_online(iterative)
        self.compute_iteration()
        self.compute_derived()

    def compute_offline(self):
        # prepare data
        if self.do_timestep:
            self.timestep = self.f['trajectory/time'][self.start+self.step] - self.f['trajectory/time'][self.start]
        ds = self.f[self.path]
        self.configure_offline(ds)
        self.init_first()
        self.offline_loop(ds)
        # compute derived properties
        self.compute_derived()

    def configure_online(self, iterative):
        raise NotImplementedError

    def configure_offline(self, ds):
        raise NotImplementedError

    def init_first(self):
        # create the output group
        if self.f is not None:
            if self.outpath in self.f:
                del self.f[self.outpath]
            self.outg = self.f.create_group(self.outpath)
        else:
            self.outg = None

    def offline_loop(self, ds):
        # Iterate over the dataset
        for i in xrange(self.start, self.end, self.step):
            self.read_offline(ds, i)
            self.compute_iteration()

    def init_online(self):
        self._first_iteration = True
        if self.do_timestep:
            self.timestep = None
            self.lasttime = None

    def read_online(self, iterative):
        raise NotImplementedError

    def read_offline(self, ds, i):
        raise NotImplementedError

    def configure_online(self, iterative):
        self.shape = iterative.state[self.key].shape

    def configure_offline(self, ds):
        self.shape = ds.shape[1:]

    def compute_iteration(self):
        raise NotImplementedError

    def compute_derived(self):
        raise NotImplementedError
