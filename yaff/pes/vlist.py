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
from yaff.pes.ext import vlist_forward, vlist_back


__all__ = [
    'ValenceList', 'ValenceTerm', 'Harmonic', 'PolyFour', 'Fues', 'Cross',
    'Cosine', 'Chebychev1', 'Chebychev2'
]


vlist_dtype = [
    ('kind', int),
    ('par0', float), ('par1', float), ('par2', float), ('par3', float),
    ('ic0', int), ('ic1', int),
    ('energy', float),
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

    def get_log(self):
        raise NotImplementedError


class Harmonic(ValenceTerm):
    kind = 0
    def __init__(self, fc, rv, ic):
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )


class PolyFour(ValenceTerm):
    kind = 1
    def __init__(self, pars, ic):
        if len(pars)>4:
            raise ValueError("PolyFour term can have maximum 4 parameters, received %i" %len(pars))
        while len(pars)<4:
            pars.append(0.0)
        ValenceTerm.__init__(self, pars, [ic])

    def get_log(self):
        u = self.ics[0].get_conversion()
        return '%s(C1=%.5e,C2=%.5e,C3=%.5e,C4=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/u),
            self.pars[1]/(log.energy.conversion/u**2),
            self.pars[2]/(log.energy.conversion/u**3),
            self.pars[3]/(log.energy.conversion/u**4),
        )

class Fues(ValenceTerm):
    kind = 2
    def __init__(self, fc, rv, ic):
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )


class Cross(ValenceTerm):
    kind = 3
    def __init__(self, fc, rv0, rv1, ic0, ic1):
        ValenceTerm.__init__(self,[fc,rv0,rv1],[ic0,ic1])

    def get_log(self):
        c0 = self.ics[0].get_conversion()
        c1 = self.ics[1].get_conversion()
        return '%s(FC=%.5e,RV0=%.5e,RV1=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c0/c1),
            self.pars[1]/c0,
            self.pars[2]/c1,
        )


class Cosine(ValenceTerm):
    kind = 4
    def __init__(self, m, a, phi0, ic):
        ValenceTerm.__init__(self, [m, a, phi0], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(M=%i,A=%.5e,PHI0=%.5e)' % (
            self.__class__.__name__,
            int(self.pars[0]),
            self.pars[1]/log.energy.conversion,
            self.pars[2]/c
        )


class Chebychev1(ValenceTerm):
    kind = 5
    def __init__(self, A, ic):
        ValenceTerm.__init__(self, [A], [ic])

    def get_log(self):
        return '%s(A=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/log.energy.conversion,
        )


class Chebychev2(ValenceTerm):
    kind = 6
    def __init__(self, A, ic):
        ValenceTerm.__init__(self, [A], [ic])

    def get_log(self):
        return '%s(A=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/log.energy.conversion,
        )
