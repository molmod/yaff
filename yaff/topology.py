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


__all__ = ['Topology']


class Topology(object):
    def __init__(self, bonds, natom):
        '''Bundle of derived properties from bonds array.

           *Arguments:*

           bonds
                The array with bonds (numpy array with integers, shape=N,2)

           natom
                The total number of atoms in the system
        '''
        self.bonds = bonds
        # 1-bond neighbors
        self.neighs1 = dict((i,[]) for i in xrange(natom))
        for i0, i1 in self.bonds:
            self.neighs1[i0].append(i1)
            self.neighs1[i1].append(i0)
        # 2-bond neighbors
        self.neighs2 = dict((i,[]) for i in xrange(natom))
        for i0, n0 in self.neighs1.iteritems():
            for i1 in n0:
                for i2 in self.neighs1[i1]:
                    # Require that there are no shorter paths than two bonds between
                    # i0 and i2. Also avoid duplicates.
                    if i2 > i0 and i2 not in self.neighs1[i0]:
                        self.neighs2[i0].append(i2)
                        self.neighs2[i2].append(i0)
        # 3-bond neighbors
        self.neighs3 = dict((i,[]) for i in xrange(natom))
        for i0, n0 in self.neighs1.iteritems():
            for i1 in n0:
                for i3 in self.neighs2[i1]:
                    # Require that there are no shorter paths than three bonds
                    # between i0 and i3. Also avoid duplicates.
                    if i3 != i0 and i3 not in self.neighs1[i0] and i3 not in self.neighs2[i0]:
                        self.neighs3[i0].append(i3)
                        self.neighs3[i3].append(i0)
        # Derive array formatted version of the neighs* dictionaries
        #self.narrs1 = [np.array(self.neighs1.get(i, [])) for i in xrange(system.size)]
        #self.narrs2 = [np.array(self.neighs2.get(i, [])) for i in xrange(system.size)]
        #self.narrs3 = [np.array(self.neighs3.get(i, [])) for i in xrange(system.size)]
