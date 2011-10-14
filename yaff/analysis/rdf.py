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


__all__ = ['RDF']


class RDF(Hook):
    def __init__(self, f, rcut, rspacing, start=0, end=-1, max_sample=None, step=None,
                 select1=None, select2=None, path='trajectory/pos', key='pos',
                 outpath=None):
        """Computes a radial distribution function (RDF)
        
           **Argument:**

           f
                An h5py.File instance containing the trajectory data.
           
           rcut
                The cutoff for the RDF analysis. This should be lower than the
                spacing between the primitive cell planes.
           
           rspacing
                The width of the bins to build up the RDF.

           **Optional arguments:**

           start, end, max_sample, step
                arguments to setup the selection of time slices. See
                ``get_slice`` for more information.

           select1
                A list of atom indexes that are considered for the computation
                of the ref data. If not given, all atoms are used.

           select2
                A list of atom indexes that are needed to compute an RDF between
                two disjoint sets of atoms. (If there is some overlap between
                select1 and select2, an error will be raised.) If this is None,
                an 'internal' RDF will be computed for the atoms specified in
                select1.

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
        self.f = f
        self.rcut = rcut
        self.rspacing = rspacing
        self.start, self.end, self.step = get_slice(self.f, start, end, max_sample, step)
        self.select1 = select1
        self.select2 = selectt
        self.path = path
        self.key = key
        if outpath is None:
            self.outpath = '%s_rdf' % path
        else:
            self.outpath = outpath
        
        if self.select1 is not None and len(self.select1) != len(set(self.select1)):
            raise ValueError('No duplicates are allowed in select1')
        if self.select2 is not None and len(self.select2) != len(set(self.select2)):
            raise ValueError('No duplicates are allowed in select2')
        if self.select1 is not None and self.select2 is not None and len(self.select1) + len(select2) != len(set(select1) + set(self.select2)):
            raise ValueError('No overlap is allowed between select1 and select2')
        
        self.nbin = int(self.rcut/self.rspacing)
        self.counts = np.zeros(self.nbins, int)
        
        self.online = self.f is None or path not in self.f
        if not self.online:
            self.compute_offline()
        else:
            raise NotImplementedError

    def compute_offline(self):
        # TODO: check if rcut is smaller than cell spacings
        # Compute the counts for the RDF
        if self.select1 is None:
            natom1 = self.f['system/numbers'].shape[0]
        else:
            natom1 = len(self.select1)
        if self.select2 is None:
            natom2 = natom1
        else
            natom2 = len(self.select2)
        # setup work arrays for storing positions and distances
        
        raise NotImplementedError
        # Compute related arrays
        self.compute_derived()

    def compute_derived(self):
        raise NotImplementedError

    def plot(self, fn_png='rdf.png', do_wavenum=True):
        import matplotlib.pyplot as pt
        raise NotImplementedError

