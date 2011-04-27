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

from yaff.ext import vlist_forward, vlist_back


__all__ = ['ValenceList', 'ValenceTerm', 'Harmonic', 'PolyFour']


vlist_dtype = [
    ('kind', int),
    ('par0', float), ('par1', float), ('par2', float), ('par3', float),
    ('ic0', int), ('ic1', int),
]


class ValenceList(object):
    def __init__(self, iclist):
        self.iclist = iclist
        self.vtab = np.zeros(10, vlist_dtype)
        self.nv = 0

    def add_term(self, term):
        if self.nv >= len(self.vtab):
            self.vtab = np.resize(self.vtab, int(len(self.vtab)*1.5))
        row = self.nv
        self.vtab[row]['kind'] = term.kind
        for i in xrange(len(term.pars)):
            self.vtab[row]['par%i'%i] = term.pars[i]
        ic_indexes = term.get_ic_indexes(self.iclist)
        for i in xrange(len(ic_indexes)):
            self.vtab[row]['ic%i'%i] = ic_indexes[i]
        self.nv += 1

    def forward(self):
        return vlist_forward(self.iclist.ictab, self.vtab, self.nv)

    def back(self):
        vlist_back(self.iclist.ictab, self.vtab, self.nv)


class ValenceTerm(object):
    kind = None
    def __init__(self, pars, ics):
        self.pars = pars
        self.ics = ics

    def get_ic_indexes(self, iclist):
        return [iclist.add_ic(ic) for ic in self.ics]


class Harmonic(ValenceTerm):
    kind = 0
    def __init__(self, fc, rv, ic):
        ValenceTerm.__init__(self, [fc, rv], [ic])


class PolyFour(ValenceTerm):
    kind = 1
    def __init__(self, pars, ic):
        while len(pars)<4:
            pars.append(0.0)
        ValenceTerm.__init__(self, pars, [ic])
