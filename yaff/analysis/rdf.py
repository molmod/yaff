# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2012 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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


import numpy as np

from yaff.log import log
from yaff.analysis.utils import get_slice
from yaff.analysis.hook import AnalysisHook
from yaff.pes.ext import Cell


__all__ = ['RDF']


class RDF(AnalysisHook):
    label = 'rdf'

    def __init__(self, rcut, rspacing, f=None, start=0, end=-1, max_sample=None,
                 step=None, select0=None, select1=None, exclude=None,
                 path='trajectory/pos', key='pos', outpath=None):
        """Computes a radial distribution function (RDF)

           **Argument:**

           rcut
                The cutoff for the RDF analysis. This should be lower than the
                spacing between the primitive cell planes.

           rspacing
                The width of the bins to build up the RDF.

           **Optional arguments:**

           f
                An h5py.File instance containing the trajectory data. If ``f``
                is not given, or it does not contain the dataset referred to
                with the ``path`` argument, an on-line analysis is carried out.

           start, end, max_sample, step
                arguments to setup the selection of time slices. See
                ``get_slice`` for more information.

           select0
                A list of atom indexes that are considered for the computation
                of the rdf. If not given, all atoms are used.

           select1
                A list of atom indexes that are needed to compute an RDF between
                two disjoint sets of atoms. (If there is some overlap between
                select0 and select1, an error will be raised.) If this is None,
                an 'internal' RDF will be computed for the atoms specified in
                select0.

           exclude
                An array with pairs of atoms (shape K x 2) with pairs of atom
                indexes to be excluded from the RDF.

           path
                The path of the dataset that contains the time dependent data in
                the HDF5 file. The first axis of the array must be the time
                axis.

           key
                In case of an on-line analysis, this is the key of the state
                item that contains the data from which the RDF is derived.

           outpath
                The output path for the frequency computation in the HDF5 file.
                If not given, it defaults to '%s_rdf' % path. If this path
                already exists, it will be removed first.

           When f is None, or when the path does not exist in the HDF5 file, the
           class can be used as an on-line analysis hook for the iterative
           algorithms in yaff.sampling package. This means that the RDF
           is built up as the itertive algorithm progresses. The end option is
           ignored and max_sample is not applicable to an on-line analysis.
        """
        if select0 is not None and len(select0) != len(set(select0)):
            raise ValueError('No duplicates are allowed in select0')
        if select1 is not None and len(select1) != len(set(select1)):
            raise ValueError('No duplicates are allowed in select1')
        if select0 is not None and select1 is not None and len(select0) + len(select1) != len(set(select0) | set(select1)):
            raise ValueError('No overlap is allowed between select0 and select1')
        if select0 is None and select1 is not None:
            raise ValueError('select1 can not be given without select0.')
        self.rcut = rcut
        self.rspacing = rspacing
        self.select0 = select0
        self.select1 = select1
        self.exclude_atoms = exclude
        self.nbin = int(self.rcut/self.rspacing)
        self.bins = np.arange(self.nbin+1)*self.rspacing
        self.d = self.bins[:-1] + 0.5*self.rspacing
        self.counts = np.zeros(self.nbin, int)
        self.nsample = 0
        self._init_exclude()
        AnalysisHook.__init__(self, f, start, end, max_sample, step, path, key, outpath, False)

    def _init_exclude(self):
        if self.exclude_atoms is None:
            self.exclude = None
            return
        elif self.select1 is None:
            index0 = dict((atom0, i0) for i0, atom0 in enumerate(self.select0))
            index1 = index0
        else:
            index0 = dict((atom0, i0) for i0, atom0 in enumerate(self.select0))
            index1 = dict((atom1, i1) for i1, atom1 in enumerate(self.select1))
        exclude = []
        for atom0, atom1 in self.exclude_atoms:
            i0 = index0.get(atom0)
            i1 = index1.get(atom1)
            if i0 is None or i1 is None:
                i0 = index0.get(atom1)
                i1 = index1.get(atom0)
            if i0 is None or i1 is None:
                continue
            if self.select1 is None and i0 < i1:
                i0, i1 = i1, i0
            exclude.append((i0, i1))
        exclude.sort()
        if len(exclude) > 0:
            self.exclude = np.array(exclude)
        else:
            self.exclude = None

    def configure_online(self, iterative):
        self.cell = iterative.ff.system.cell
        self.natom = iterative.ff.system.natom

    def configure_offline(self, ds):
        if 'rvecs' in self.f['system']:
            rvecs = self.f['system/rvecs'][:]
            self.cell = Cell(rvecs)
            if (2*self.rcut > self.cell.rspacings).any():
                raise ValueError('The 2*rcut argument should not exceed any of the cell spacings.')
        else:
            self.cell = Cell(None)
        if self.cell.nvec != 3:
            raise ValueError('RDF can only be computed for 3D periodic systems.')
        # get the total number of atoms
        self.natom = self.f['system/numbers'].shape[0]

    def init_first(self):
        self.volume = self.cell.volume
        # Setup some work arrays
        if self.select0 is None:
            self.natom0 = self.natom
        else:
            self.natom0 = len(self.select0)
        self.pos0 = np.zeros((self.natom0, 3), float)
        if self.select1 is None:
            self.npair = (self.natom0*(self.natom0-1))/2
            self.pos1 = None
        else:
            self.natom1 = len(self.select1)
            self.pos1 = np.zeros((self.natom1, 3), float)
            self.npair = self.natom0*self.natom1
        if self.exclude is not None:
            self.npair -= len(self.exclude)
        self.work = np.zeros(self.npair, float)
        # Prepare the output
        AnalysisHook.init_first(self)
        if self.outg is not None:
            self.outg.create_dataset('rdf', (self.nbin,), float)
            self.outg.create_dataset('crdf', (self.nbin,), float)
            self.outg.create_dataset('counts', (self.nbin,), int)
            self.outg['d'] = self.d

    def read_online(self, iterative):
        pos = iterative.state[self.key].value
        if self.select0 is None:
            self.pos0[:] = pos
        else:
            self.pos0[:] = pos[self.select0]
        if self.select1 is not None:
            self.pos1[:] = pos[self.select1]

    def read_offline(self, ds, i):
        if self.select0 is None:
            ds.read_direct(self.pos0, (i,))
        else:
            ds.read_direct(self.pos0, (i,self.select0))
        if self.select1 is not None:
            ds.read_direct(self.pos1, (i,self.select1))

    def compute_iteration(self):
        self.cell.compute_distances(self.work, self.pos0, self.pos1, self.exclude)
        self.counts += np.histogram(self.work, bins=self.bins)[0]
        self.nsample += 1

    def compute_derived(self):
        # derive the RDF
        ref_count = self.npair/self.volume*4*np.pi*self.d**2*self.rspacing
        self.rdf = self.counts/ref_count/self.nsample
        # derived the cumulative RDF
        self.crdf = self.counts.cumsum()/float(self.nsample*self.natom0)
        # store everything in the h5py file
        if self.outg is not None:
            self.outg['rdf'][:] = self.rdf
            self.outg['crdf'][:] = self.crdf
            self.outg['counts'][:] = self.counts

    def plot(self, fn_png='rdf.png'):
        import matplotlib.pyplot as pt
        pt.clf()
        xunit = log.length.conversion
        pt.plot(self.d/xunit, self.rdf, 'k-', drawstyle='steps-mid')
        pt.xlabel('Distance [%s]' % log.length.notation)
        pt.ylabel('RDF')
        pt.xlim(self.bins[0]/xunit, self.bins[-1]/xunit)
        pt.savefig(fn_png)

    def plot_crdf(self, fn_png='crdf.png'):
        import matplotlib.pyplot as pt
        pt.clf()
        xunit = log.length.conversion
        pt.plot(self.d/xunit, self.crdf, 'k-', drawstyle='steps-mid')
        pt.xlabel('Distance [%s]' % log.length.notation)
        pt.ylabel('CRDF')
        pt.xlim(self.bins[0]/xunit, self.bins[-1]/xunit)
        pt.savefig(fn_png)
