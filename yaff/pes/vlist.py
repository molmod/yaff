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
'''Module for the complete list of covalent energy terms.

   A ``ValenceList`` object contains a table with all the energy terms that
   contribute (additively) to the total energy. The values of the internal
   coordinates, needed to compute the energy, are taken from an
   :class:`yaff.pes.iclist.InternalCoordinateList` class.

   Each row in the table contains all the information to evaluate one energy
   term, which is done by the ``forward`` method. The ``back`` method **adds**
   the derivative of the energy towards the internal coordinate to the right
   entry in the ``InternalCoordinateList`` object.

   A series of ``ValenceTerm`` classes is defined. These are used to register
   new energy terms in a ``ValenceList`` object. Each subclass of ``ValenceList``
   represents a kind of energy term, e.g. harmonic, Fues, class-2 cross term,
   etc. Instances of these classes are passed to the ``add_term`` method,
   which will append a new row to the table and register the required
   internal coordinates in the ``InternalCoordinateList`` object. (That will
   in turn register the requires relative vectors in a ``DeltaList`` object.)

   The class :class:`yaff.pes.vlist.ValenceList` is intimately related to
   classes :class:`yaff.pes.dlist.DeltaList` and
   :class:`yaff.pes.iclist.InternalCoordinateList`. They work together, just
   like layers in a neural network, and they use the back-propagation algorithm
   to compute partial derivatives. The order of the layers is as follows::

       DeltaList <--> InternalCoordinateList <--> ValenceList

   The class :class:`yaff.pes.ff.ForcePartValence` ties these three lists
   together. The basic idea of the back-propagation algorithm is explained in
   the section :ref:`dg_sec_backprop`.
'''


from __future__ import division

import numpy as np

from yaff.log import log
from yaff.pes.ext import vlist_forward, vlist_back


__all__ = [
    'ValenceList', 'ValenceTerm', 'Harmonic', 'PolyFour', 'Fues', 'Cross',
    'Cosine', 'Chebychev1', 'Chebychev2', 'Chebychev3', 'Chebychev4',
    'Chebychev6', 'PolySix', 'MM3Quartic', 'MM3Bend', 'BondDoubleWell',
    'Morse',
]


vlist_dtype = [
    ('kind', int),                      # The kind of energy term, e.g. harmonic, fues, ...
    ('par0', float), ('par1', float),   # The parameters for the energy term. Meaning of par0, par1, ... depends on kind.
    ('par2', float), ('par3', float),
    ('par4', float), ('par5', float),
    ('ic0', int), ('ic1', int),         # Indexes of rows in the table of internal coordinates. (See InternalCoordinatList class.)
#    ('ic2', int),
    ('energy', float),                  # The computed value of the energy, output of forward method.
]


class ValenceList(object):
    '''Contains a complete list of all valence energy terms. Computations are
       carried out in coordination with an ``InternalCoordinateList`` object.
    '''
    def __init__(self, iclist):
        '''
           **Arguments:**

           iclist
                An instance of the ``InternalCoordinateList`` object.
        '''
        self.iclist = iclist
        self.vtab = np.zeros(10, vlist_dtype)
        self.nv = 0

    def add_term(self, term):
        '''Register a new covalent energy term

           **Arguments:**

           term
                An instance of a subclass of the ``ValenceTerm`` class.
        '''
        # extend the table if needed.
        if self.nv >= len(self.vtab):
            self.vtab = np.resize(self.vtab, int(len(self.vtab)*1.5))
        # fill in the new term
        row = self.nv
        self.vtab[row]['kind'] = term.kind
        for i in range(len(term.pars)):
            self.vtab[row]['par%i'%i] = term.pars[i]
        ic_indexes = term.get_ic_indexes(self.iclist) # registers ics in InternalCoordinateList.
        for i in range(len(ic_indexes)):
            self.vtab[row]['ic%i'%i] = ic_indexes[i]
        self.nv += 1

    def forward(self):
        """Compute the values of the energy terms, based on the values of the
           internal coordinates list, and store the result in the ``self.vtab``
           table.

           The actual computation is carried out by a low-level C routine.
        """
        return vlist_forward(self.iclist.ictab, self.vtab, self.nv)

    def back(self):
        """Compute the derivatives of the energy terms towards the internal
           coordinates and store the results in the ``self.iclist.ictab`` table.

           The actual computation is carried out by a low-level C routine.
        """
        vlist_back(self.iclist.ictab, self.vtab, self.nv)


class ValenceTerm(object):
    '''Base class for valence energy terms 'descriptors'.

       The subclasses are merely used to request a new covalent energy terms in
       the ``ValenceList`` class. These classes do not carry out any
       computations.

       The ``kind`` class attribute refers to an integer ID that identifies the
       valence term kind (harmonic, fues, ...) in the low-level C code.
    '''
    kind = None
    def __init__(self, pars, ics):
        '''
           **Arguments:**

           pars
                A list of parameters to be stored for this energy term. This list
                may at most contain four elements.

           ics
                A list of row indexes in the table of internal coordinates. This
                list may contain either one or two elements.
        '''
        self.pars = pars
        self.ics = ics

    def get_ic_indexes(self, iclist):
        '''Request row indexes for the internal coordinates from the given
           ``InternalCoordinateList`` object.
        '''
        return [iclist.add_ic(ic) for ic in self.ics]

    def get_log(self):
        '''Describe the covalent energy term in a format that is suitable for
           screen logging.
        '''
        raise NotImplementedError


class Harmonic(ValenceTerm):
    '''The harmonic energy term: 0.5*K*(q-q0)^2'''
    kind = 0
    def __init__(self, fc, rv, ic):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           ic
                An ``InternalCoordinate`` object.
        '''
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )


class PolyFour(ValenceTerm):
    '''Fourth-order polynomical term: par0*q + par1*q^2 + par2*q^3 + par3*q^4'''
    kind = 1
    def __init__(self, pars, ic):
        '''
           **Arguments:**

           pars
                The constant linear coefficients of the polynomial, in atomic
                units, starting from first order. This list may at most contain
                four coefficients.

           ic
                An ``InternalCoordinate`` object.
        '''
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
    '''The Fues energy term: 0.5*K*q0^2*(1-q/q0)^2'''
    kind = 2
    def __init__(self, fc, rv, ic):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           ic
                An ``InternalCoordinate`` object.
        '''
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )


class Cross(ValenceTerm):
    '''A traditional class-2 cross term: K*(x-x0)*(y-y0)'''
    kind = 3
    def __init__(self, fc, rv0, rv1, ic0, ic1):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv0, rv1
                The rest values (in atomic units).

           ic0, ic1
                The ``InternalCoordinate`` objects. ic0 corresponds to rv0, and
                ic1 corresponds to rv1.
        '''
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
    '''A cosine energy term: 0.5*a*(1-cos(m*(phi-phi0)))'''
    kind = 4
    def __init__(self, m, a, phi0, ic):
        '''
           **Arguments:**

           m
                The multiplicity of the cosine function, which may be useful
                for torsional barriers.

           a
                The amplitude of the cosine function (in atomic units).

           phi0
                The rest angle of cosine term (in radians).

           ic
                An ``InternalCoordinate`` object. This must be an internal
                coordinate that computes some angle in radians.
        '''
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
    '''A first degree polynomial: 0.5*A*(1 -+ T1)
       where T1=x is the first Chebychev polynomial of the first kind.

       This is used for a computationally efficient implementation of torsional
       energy terms, because the only computation of the cosine of the dihedral
       angle is needed, not the angle itself.

       This term corresponds to multiplicity 1. The minus sign corresponds to a
       rest value of 0 degrees. With a the plus sign, the rest value becomes
       180 degrees.
    '''
    kind = 5
    def __init__(self, A, ic, sign=-1):
        '''
           **Arguments:**

           A
                The energy scale of the function (in atomic units).

           ic
                An ``InternalCoordinate`` object.

           sign
                Choose positive or negative sign in the polynomial.
        '''
        ValenceTerm.__init__(self, [A,sign], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(A=%.5e,sign=%+2d)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c),
            self.pars[1],
        )


class Chebychev2(ValenceTerm):
    '''A second degree polynomial: 0.5*A*(1 -+ T2)
       where T2=2*x**2-1 is the second Chebychev polynomial of the first kind.

       This is used for a computationally efficient implementation of torsional
       energy terms, because the only computation of the cosine of the dihedral
       angle is needed, not the angle itself.

       This term corresponds to multiplicity 2. The minus sign corresponds to a
       rest value of 0 degrees. With a the plus sign, the rest value becomes
       90 degrees.
    '''
    kind = 6
    def __init__(self, A, ic, sign=-1):
        '''
           **Arguments:**

           A
                The energy scale of the function (in atomic units).

           ic
                An ``InternalCoordinate`` object.

           sign
                Choose positive or negative sign in the polynomial.
        '''
        ValenceTerm.__init__(self, [A,sign], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(A=%.5e,sign=%+2d)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1],
        )


class Chebychev3(ValenceTerm):
    '''A third degree polynomial: 0.5*A*(1 -+ T3)
       where T3=4*x**3-3*x is the third Chebychev polynomial of the first kind.

       This is used for a computationally efficient implementation of torsional
       energy terms, because the only computation of the cosine of the dihedral
       angle is needed, not the angle itself.

       This term corresponds to multiplicity 3. The minus sign corresponds to a
       rest value of 0 degrees. With a the plus sign, the rest value becomes
       60 degrees.
    '''
    kind = 7
    def __init__(self, A, ic, sign=-1):
        '''
           **Arguments:**

           A
                The energy scale of the function (in atomic units).

           ic
                An ``InternalCoordinate`` object.

           sign
                Choose positive or negative sign in the polynomial.
        '''
        ValenceTerm.__init__(self, [A,sign], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(A=%.5e,sign=%+2d)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1],
        )


class Chebychev4(ValenceTerm):
    '''A fourth degree polynomial: 0.5*A*(1 -+ T4)
       where T4=8*x**4-8*x**2+1 is the fourth Chebychev polynomial of the
       first kind.

       This is used for a computationally efficient implementation of torsional
       energy terms, because the only computation of the cosine of the dihedral
       angle is needed, not the angle itself.

       This term corresponds to multiplicity 4. The minus sign corresponds to a
       rest value of 0 degrees. With a the plus sign, the rest value becomes
       45 degrees.
    '''
    kind = 8
    def __init__(self, A, ic, sign=-1):
        '''
           **Arguments:**

           A
                The energy scale of the function (in atomic units).

           ic
                An ``InternalCoordinate`` object.

           sign
                Choose positive or negative sign in the polynomial.
        '''
        ValenceTerm.__init__(self, [A,sign], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(A=%.5e,sign=%+2d)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1],
        )

class Chebychev6(ValenceTerm):
    '''A sixth degree polynomial: 0.5*A*(1 -+ T6)
       where T6=32*x**6-48*x**4+18*x**2-1 is the sixth Chebychev polynomial of
       the first kind.

       This is used for a computationally efficient implementation of torsional
       energy terms, because the only computation of the cosine of the dihedral
       angle is needed, not the angle itself.

       This term corresponds to multiplicity 6. The minus sign corresponds to a
       rest value of 0 degrees. With a the plus sign, the rest value becomes
       30 degrees.
    '''
    kind = 9
    def __init__(self, A, ic, sign=-1):
        '''
           **Arguments:**

           A
                The energy scale of the function (in atomic units).

           ic
                An ``InternalCoordinate`` object.

           sign
                Choose positive or negative sign in the polynomial.
        '''
        ValenceTerm.__init__(self, [A,sign], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(A=%.5e,sign=%+2d)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1],
        )


class PolySix(ValenceTerm):
    '''Sixth-order polynomical term: par0*q + par1*q^2 + par2*q^3 + par3*q^4 + par4*q^5 + par5*q^6'''
    kind = 10
    def __init__(self, pars, ic):
        '''
           **Arguments:**

           pars
                The constant linear coefficients of the polynomial, in atomic
                units, starting from first order. This list may at most contain
                six coefficients.

           ic
                An ``InternalCoordinate`` object.
        '''
        if len(pars)>6:
            raise ValueError("PolySix term can have maximum 6 parameters, received %i" %len(pars))
        while len(pars)<6:
            pars.append(0.0)
        ValenceTerm.__init__(self, pars, [ic])

    def get_log(self):
        u = self.ics[0].get_conversion()
        return '%s(C1=%.5e,C2=%.5e,C3=%.5e,C4=%.5e,C5=%.5e,C6=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/u),
            self.pars[1]/(log.energy.conversion/u**2),
            self.pars[2]/(log.energy.conversion/u**3),
            self.pars[3]/(log.energy.conversion/u**4),
            self.pars[4]/(log.energy.conversion/u**5),
            self.pars[5]/(log.energy.conversion/u**6),
        )


class MM3Quartic(ValenceTerm):
    '''The quartic energy term used for the bond stretch in MM3: 0.5*K*(q-q0)^2*(1-2.55*(q-q0)^2+7/12*(2.55*(q-q0))^2)'''
    kind = 11
    def __init__(self, fc, rv, ic):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           ic
                An ``InternalCoordinate`` object.
        '''
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )


class MM3Bend(ValenceTerm):
    '''The sixth-order energy term used for the bends in MM3: 0.5*K*(q-q0)^2*(1-0.14*(q-q0)+5.6*10^(-5)*(q-q0)^2-7*10^(-7)*(q-q0)^3+2.2*10^(-7)*(q-q0)^4)'''
    kind = 12
    def __init__(self, fc, rv, ic):
        '''
           **Arguments:**

           fc
                The force constant (in atomic units).

           rv
                The rest value (in atomic units).

           ic
                An ``InternalCoordinate`` object.
        '''
        ValenceTerm.__init__(self, [fc, rv], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(FC=%.5e,RV=%.5e)' % (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c
        )

class BondDoubleWell(ValenceTerm):
    '''Sixth-order polynomial term: K/(2*(r1-r2)^4)*(r-r1)^2*(r-r2)^4'''
    kind = 13
    def __init__(self, K, r1, r2, ic):
        '''
            **Arguments:**

            K
                Force constant corresponding with V-O double bond
            r1, r2
                Two rest values for the V-O chain (r1, 'double bond', r2, 'weak bond')
            ic
                An "InternalCoordinate" object (here bond distance)
        '''
        ValenceTerm.__init__(self, [K,r1,r2], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(K=%.5e,R1=%.5e,R2=%.5e)' % (
            self.__class__.name__,
            self.pars[0]/(log.energy.conversion/c**2),
            self.pars[1]/c,
            self.pars[2]/c
        )

class Morse(ValenceTerm):
    ''' The morse potential: E0*( exp(-2*k*(r-r1)) - 2*exp(-k*(r-r1)) )'''
    kind = 14
    def __init__(self, E0, k, r1, ic):
        '''
            **Arguments:**

            E0
                The well depth
            k
                The well width
            r1
                The rest value
            ic
                An "InternalCoordinate" object, typically a distance
        '''
        ValenceTerm.__init__(self, [E0,k,r1], [ic])

    def get_log(self):
        c = self.ics[0].get_conversion()
        return '%s(E0=%.5e,k=%.5e,r1=%.5e)' %  (
            self.__class__.__name__,
            self.pars[0]/(log.energy.conversion),
            self.pars[1]*c,
            self.pars[2]/c,
        )
