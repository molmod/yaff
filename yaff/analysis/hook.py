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
'''Abstract hook implementation for trajectory analysis'''


from __future__ import division

from yaff.log import log
from yaff.sampling.iterative import Hook
from yaff.analysis.utils import get_slice


__all__ = ['AnalysisInput', 'AnalysisHook']



class AnalysisInput(object):
    '''Describes the location of the time-dependent input for an analysis.'''
    def __init__(self, path, key, required=True):
        '''
           **Arguments:**

           path
                The path in the HDF5 file were the input can be found. This
                is only relevant for an off-line analysis. This must be a
                dataset in the trajectory folder.

           key
                The key of the state item that contains the input. This is
                only relevant for an on-line analysis.

           **Optional arguments:**

           required
                When true, this input is mandatory. When false, it is optional.
        '''
        if path is not None and not path.startswith('trajectory/'):
            raise ValueError('When the path is given, it must start with "trajectory/".')
        self.path = path
        self.key = key
        self.required = required


class AnalysisHook(Hook):
    def __init__(self, f=None, start=0, end=-1, max_sample=None, step=None,
                 analysis_inputs={}, outpath='trajectory/noname', do_timestep=False):
        """Base class for the analysis hooks.

           Analysis hooks in Yaff support both off-line and on-line analysis.

           **Optional arguments:**

           f
                An h5.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           start, end, max_sample, step
                Optional arguments for the ``get_slice`` function.

           analysis_inputs
                A list with AnalysisInput instances

           outpath
                The output path for the analysis.

           do_timestep
                When True, a self.timestep attribute will be initialized as
                early as possible.
        """
        self.f = f
        self.start, self.end, self.step = get_slice(self.f, start, end, max_sample, step)
        self.analysis_inputs = analysis_inputs
        self.outpath = outpath
        self.do_timestep = do_timestep

        self.online = self.f is None
        if not self.online:
            for ai in self.analysis_inputs.values():
                self.online |= (ai.path is None and ai.required)
                self.online |= not (ai.path is None or ai.path in self.f)
        if not self.online:
            self.compute_offline()
        else:
            self.init_online()

    def __call__(self, iterative):
        # get the requested state items
        state_items = {}
        for key, ai in self.analysis_inputs.items():
            if ai.key is not None:
                state_items['st_'+key] = iterative.state[ai.key]
        # prepare some data structures
        if self._first_iteration:
            self.configure_online(iterative, **state_items)
            self.init_first()
            self._first_iteration = False
        # get the time step
        if self.do_timestep and self.timestep is None:
            if self.lasttime is None:
                self.lasttime = iterative.state['time'].value
            else:
                self.timestep = iterative.state['time'].value - self.lasttime
                del self.lasttime
                self.init_timestep()
        # do the analysis on one iteration
        self.read_online(**state_items)
        self.compute_iteration()
        self.compute_derived()

    def compute_offline(self):
        datasets = {}
        for key, ai in self.analysis_inputs.items():
            if ai.path is not None:
                datasets['ds_' + key] = self.f[ai.path]
        self.configure_offline(**datasets)
        self.init_first()
        if self.do_timestep:
            self.timestep = self.f['trajectory/time'][self.start+self.step] - self.f['trajectory/time'][self.start]
            self.init_timestep()
        self.offline_loop(**datasets)
        self.compute_derived()

    def init_first(self):
        # create the output group
        if self.f is not None:
            if self.outpath in self.f:
                del self.f[self.outpath]
            self.outg = self.f.create_group(self.outpath)
        else:
            self.outg = None

    def init_online(self):
        self._first_iteration = True
        if self.do_timestep:
            self.timestep = None
            self.lasttime = None

    def offline_loop(self, **datasets):
        # Iterate over the dataset
        for i in range(self.start, self.end, self.step):
            self.read_offline(i, **datasets)
            self.compute_iteration()

    def configure_online(self, **state_items):
        pass

    def configure_offline(self, **kwargs):
        pass

    def init_timestep(self):
        raise NotImplementedError

    def read_online(self, **state_items):
        raise NotImplementedError

    def read_offline(self, i, **kwargs):
        raise NotImplementedError

    def compute_iteration(self):
        raise NotImplementedError

    def compute_derived(self):
        raise NotImplementedError
