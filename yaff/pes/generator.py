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
"""Automatically generate force field models

   This module contains all the machinery needed to support the
   :meth:`yaff.pes.ff.ForceField.generate` method.
"""


from __future__ import division

import numpy as np

from molmod.units import parse_unit

from itertools import permutations

from yaff.log import log
from yaff.pes.ext import PairPotEI, PairPotLJ, PairPotMM3, PairPotExpRep, \
    PairPotQMDFFRep, PairPotDampDisp, PairPotDisp68BJDamp, Switch3
from yaff.pes.ff import ForcePartPair, ForcePartValence, \
    ForcePartEwaldReciprocal, ForcePartEwaldCorrection, \
    ForcePartEwaldNeutralizing
from yaff.pes.iclist import Bond, BendAngle, BendCos, \
    UreyBradley, DihedAngle, DihedCos, OopAngle, OopMeanAngle, OopCos, \
    OopMeanCos, OopDist, SqOopDist
from yaff.pes.nlist import NeighborList
from yaff.pes.scaling import Scalings
from yaff.pes.vlist import Harmonic, PolyFour, Fues, Cross, Cosine, \
    Chebychev1, Chebychev2, Chebychev3, Chebychev4, Chebychev6, PolySix, \
    MM3Quartic, MM3Bend, BondDoubleWell, Morse


__all__ = [
    'FFArgs', 'Generator',

    'ValenceGenerator', 'BondGenerator', 'BondHarmGenerator', 'BondDoubleWellGenerator',
    'BondDoubleWell2Generator', 'BondFuesGenerator', 'MM3QuarticGenerator',
    'BendGenerator', 'BendAngleHarmGenerator', 'BendCosHarmGenerator', 'BendCosGenerator', 'MM3BendGenerator',
    'TorsionGenerator', 'TorsionCosHarmGenerator', 'TorsionCos2HarmGenerator',
    'UreyBradleyHarmGenerator', 'OopAngleGenerator', 'OopMeanAngleGenerator',
    'OopCosGenerator', 'OopMeanCosGenerator', 'OopDistGenerator', 'BondMorseGenerator',

    'ValenceCrossGenerator', 'CrossGenerator',

    'NonbondedGenerator', 'LJGenerator', 'MM3Generator', 'ExpRepGenerator',
    'DampDispGenerator', 'FixedChargeGenerator', 'D3BJGenerator',

    'apply_generators',
]


class FFArgs(object):
    '''Data structure that holds all arguments for the ForceField constructor

       The attributes of this object are gradually filled up by the various
       generators based on the data in the ParsedPars object.
    '''
    def __init__(self, rcut=18.89726133921252, tr=Switch3(7.558904535685008),
                 alpha_scale=3.5, gcut_scale=1.1, skin=0, smooth_ei=False,
                 reci_ei='ewald'):
        """
           **Optional arguments:**

           Some optional arguments only make sense if related parameters in the
           parameter file are present.

           rcut
                The real space cutoff used by all pair potentials.

           tr
                Default truncation model for everything except the electrostatic
                interactions. The electrostatic interactions are not truncated
                by default.

           alpha_scale
                Determines the alpha parameter in the Ewald summation based on
                the real-space cutoff: alpha = alpha_scale / rcut. Higher
                values for this parameter imply a faster convergence of the
                reciprocal terms, but a slower convergence in real-space.

           gcut_scale
                Determines the reciprocale space cutoff based on the alpha
                parameter: gcut = gcut_scale * alpha. Higher values for this
                parameter imply a better convergence in the reciprocal space.

           skin
                The skin parameter for the neighborlist.

           smooth_ei
                Flag for smooth truncations for the electrostatic interactions.

           reci_ei
                The method to be used for the reciprocal contribution to the
                electrostatic interactions in the case of periodic systems. This
                must be one of 'ignore' or 'ewald'. The 'ewald' option is only
                supported for 3D periodic systems.

           The actual value of gcut, which depends on both gcut_scale and
           alpha_scale, determines the computational cost of the reciprocal term
           in the Ewald summation. The default values are just examples. An
           optimal trade-off between accuracy and computational cost requires
           some tuning. Dimensionless scaling parameters are used to make sure
           that the numerical errors do not depend too much on the real space
           cutoff and the system size.
        """
        if reci_ei not in ['ignore', 'ewald']:
            raise ValueError('The reci_ei option must be one of \'ignore\' or \'ewald\'.')
        self.rcut = rcut
        self.tr = tr
        self.alpha_scale = alpha_scale
        self.gcut_scale = gcut_scale
        self.skin = skin
        self.smooth_ei = smooth_ei
        self.reci_ei = reci_ei
        # arguments for the ForceField constructor
        self.parts = []
        self.nlist = None

    def get_nlist(self, system):
        if self.nlist is None:
            self.nlist = NeighborList(system, self.skin)
        return self.nlist

    def get_part(self, ForcePartClass):
        for part in self.parts:
            if isinstance(part, ForcePartClass):
                return part

    def get_part_pair(self, PairPotClass):
        for part in self.parts:
            if isinstance(part, ForcePartPair) and isinstance(part.pair_pot, PairPotClass):
                return part

    def get_part_valence(self, system):
        part_valence = self.get_part(ForcePartValence)
        if part_valence is None:
            part_valence = ForcePartValence(system)
            self.parts.append(part_valence)
        return part_valence

    def add_electrostatic_parts(self, system, scalings, dielectric):
        if self.get_part_pair(PairPotEI) is not None:
            return
        nlist = self.get_nlist(system)
        if system.cell.nvec == 0:
            alpha = 0.0
        elif system.cell.nvec == 3:
            #TODO: the choice of alpha should depend on the radii of the
            #charge distributions. Following expression is OK for point charges.
            alpha = self.alpha_scale/self.rcut
        else:
            raise NotImplementedError('Only zero- and three-dimensional electrostatics are supported.')
        # Real-space electrostatics
        if self.smooth_ei:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, self.tr, dielectric, system.radii)
        else:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, None, dielectric, system.radii)
        part_pair_ei = ForcePartPair(system, nlist, scalings, pair_pot_ei)
        self.parts.append(part_pair_ei)
        if self.reci_ei == 'ignore':
            # Nothing to do
            pass
        elif self.reci_ei == 'ewald':
            if system.cell.nvec == 3:
                # Reciprocal-space electrostatics
                part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, self.gcut_scale*alpha, dielectric)
                self.parts.append(part_ewald_reci)
                # Ewald corrections
                part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings, dielectric)
                self.parts.append(part_ewald_corr)
                # Neutralizing background
                part_ewald_neut = ForcePartEwaldNeutralizing(system, alpha, dielectric)
                self.parts.append(part_ewald_neut)
            elif system.cell.nvec != 0:
                raise NotImplementedError('The ewald summation is only available for 3D periodic systems.')
        else:
            raise NotImplementedError


class Generator(object):
    """Creates (part of a) ForceField object automatically.

       A generator is a class that describes how a part of a parameter file
       must be turned into a part of ForceField object. As the generator
       proceeds, it will modify and extend the current arguments of the FF. They
       should be implemented such that the order of the generators is not
       important.

       **Important class attributes:**

       prefix
            The prefix string that must match the prefix in the parameter file.
            If this is None, it is assumed that the Generator class is abstract.
            In that case it will be ignored by the apply_generators function
            at the bottom of this module.

       par_info
            A description of the parameters on a single line (PARS suffix)

       suffixes
            The supported suffixes

       allow_superpositions
            Whether multiple PARS lines with the same atom types are allowed.
            This is rarely the case, except for the TORSIONS and a few other
            weirdos.
    """
    prefix = None
    par_info = None
    suffixes = None
    allow_superposition = False

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from this generator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        raise NotImplementedError

    def check_suffixes(self, parsec):
        for suffix in parsec.definitions:
            if suffix not in self.suffixes:
                parsec.complain(None, 'contains a suffix (%s) that is not recognized by generator %s.' % (suffix, self.prefix))

    def process_units(self, pardef):
        '''Load parameter conversion information

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary with (name, converion) pairs.
        '''
        result = {}
        expected_names = [name for name, dtype in self.par_info if dtype is float]
        for counter, line in pardef:
            words = line.split()
            if len(words) != 2:
                pardef.complain(counter, 'must have two arguments in UNIT suffix.')
            name = words[0].upper()
            if name not in expected_names:
                pardef.complain(counter, 'specifies a unit for an unknown parameter. (Must be one of %s, but got %s.)' % (expected_names, name))
            try:
                result[name] = parse_unit(words[1])
            except (NameError, ValueError):
                pardef.complain(counter, 'has a UNIT suffix with an unknown unit.')
        if len(result) != len(expected_names):
            raise IOError('Not all units are specified for generator %s in file %s. Got %s, should have %s.' % (
                self.prefix, pardef.complain.filename, list(result.keys()), expected_names
            ))
        return result

    def process_pars(self, pardef, conversions, nffatype, par_info=None):
        '''Load parameter and apply conversion factors

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           conversions
                A dictionary with (name, conversion) items.

           nffatype
                The number of ffatypes per line of parameters.

           **Optional arguments:**

           par_info
                A custom description of the parameters. If not present,
                self.par_info is used. This is convenient when this method
                is used to parse other definitions than PARS.
        '''
        if par_info is None:
            par_info = self.par_info
        par_table = {}
        for counter, line in pardef:
            words = line.split()
            num_args = nffatype + len(par_info)
            if len(words) != num_args:
                pardef.complain(counter, 'should have %s arguments.' % num_args)
            key = tuple(words[:nffatype])
            try:
                pars = []
                for i, (name, dtype) in enumerate(par_info):
                    word = words[i+nffatype]
                    if dtype is float:
                        pars.append(float(word)*conversions[name])
                    else:
                        pars.append(dtype(word))
                pars = tuple(pars)
            except ValueError:
                pardef.complain(counter, 'has parameters that can not be converted to numbers.')
            par_list = par_table.get(key, [])
            if len(par_list) > 0 and not self.allow_superposition:
                pardef.complain(counter, 'conts duplicate parameters, which is not allowed for generator %s.' % self.prefix)
            par_list.append(pars)
            for key in self.iter_alt_keys(key):
                par_table[key] = par_list
        return par_table

    def iter_alt_keys(self, key):
        '''Iterates of all equivalent reorderings of a tuple of ffatypes'''
        if len(key) == 1:
            yield key
        else:
            raise NotImplementedError


class ValenceGenerator(Generator):
    '''All generators for diagonal valence terms derive from this class.

       **More important attributes:**

       nffatype
            The number of atoms involved in the internal coordinates. Hence
            this is also the number ffatypes in a single row in the force field
            parameter file.

       ICClass
            The ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       VClass
            The ``ValenceTerm`` class. See ``yaff.pes.vlist``.
    '''

    suffixes = ['UNIT', 'PARS']
    nffatype = None
    ICClass = None
    VClass = None

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from a ValenceGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args)

    def apply(self, par_table, system, ff_args):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence terms.')
        part_valence = ff_args.get_part_valence(system)
        for indexes in self.iter_indexes(system):
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            if len(par_list) == 0 and log.do_warning:
                log.warn('No valence %s parameters found for atoms %s with key %s' % (self.prefix, indexes, key))
                continue
            for pars in par_list:
                vterm = self.get_vterm(pars, indexes)
                part_valence.add_term(vterm)

    def get_vterm(self, pars, indexes):
        '''Return an instance of the ValenceTerm class with the proper InternalCoordinate instance

           **Arguments:**

           pars
                The parameters for the ValenceTerm class.

           indexes
                The atom indices used to define the internal coordinate
        '''
        args = pars + (self.ICClass(*indexes),)
        return self.VClass(*args)

    def iter_indexes(self, system):
        '''Iterate over all tuples of indices for the internal coordinate'''
        raise NotImplementedError


class BondGenerator(ValenceGenerator):
    par_info = [('K', float), ('R0', float)]
    nffatype = 2
    ICClass = Bond
    VClass = None

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_bonds()


class BondHarmGenerator(BondGenerator):
    prefix = 'BONDHARM'
    VClass = Harmonic


class BondFuesGenerator(BondGenerator):
    prefix = 'BONDFUES'
    VClass = Fues


class MM3QuarticGenerator(BondGenerator):
    prefix = 'MM3QUART'
    VClass = MM3Quartic


class BondDoubleWellGenerator(ValenceGenerator):
    par_info = [('K', float), ('R1', float), ('R2', float)]
    nffatype = 2
    ICClass = Bond
    prefix = 'DOUBWELL'
    VClass = BondDoubleWell

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_bonds()


class BondMorseGenerator(ValenceGenerator):
    prefix = 'BONDMORSE'
    par_info = [('E0', float), ('K', float), ('R0', float)]
    nffatype = 2
    ICClass = Bond
    VClass = Morse

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_bonds()


class BondDoubleWell2Generator(ValenceGenerator):
    nffatype = 2
    prefix = 'DOUBWELL2'
    ICClass = Bond
    VClass = PolySix
    par_info = [('K', float), ('R1', float), ('R2', float)]

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_bonds()

    def process_pars(self, pardef, conversions, nffatype, par_info=None):
        '''
            Transform the 3 parameters given in the parameter file to the 6
            parameters required by PolySix. The parameters of PolySix are
            given as a single argument (a list) containing all 6 parameters, not
            6 arguments with each a parameter.
        '''
        tmp = Generator.process_pars(self, pardef, conversions, nffatype, par_info=par_info)
        par_table = {}

        for key, oldpars in tmp.items():
            K = oldpars[0][0]
            r0 = oldpars[0][1]
            r1 = oldpars[0][2]
            a = K/(2.0*(r0-r1)**4)
            c1 = -2.0*r0*r1**4 - 4.0*r0**2*r1**3
            c2 = r1**4 + 8.0*r0*r1**3 + 6.0*r0**2*r1**2
            c3 = -4.0*r1**3 - 12.0*r0*r1**2 - 4.0*r1*r0**2
            c4 = 6.0*r1**2 + 8.0*r0*r1 + r0**2
            c5 = -4.0*r1 - 2.0*r0
            c6 = 1.0
            pars = [a*c1, a*c2, a*c3, a*c4, a*c5, a*c6]
            par_table[key] = [(pars,)]
        return par_table


class PolySixGenerator(ValenceGenerator):
    nffatype = 2
    prefix = 'POLYSIX'
    ICClass = Bond
    VClass = PolySix
    par_info = [('C0', float), ('C1', float), ('C2', float), ('C3', float), ('C4', float), ('C5', float), ('C6', float)]

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_bonds()


class BendGenerator(ValenceGenerator):
    nffatype = 3
    ICClass = None
    VClass = Harmonic

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_angles()


class BendAngleHarmGenerator(BendGenerator):
    par_info = [('K', float), ('THETA0', float)]
    prefix = 'BENDAHARM'
    ICClass = BendAngle


class BendCosHarmGenerator(BendGenerator):
    par_info = [('K', float), ('COS0', float)]
    prefix = 'BENDCHARM'
    ICClass = BendCos


class MM3BendGenerator(BendGenerator):
    nffatype = 3
    par_info = [('K', float), ('THETA0', float)]
    ICClass = BendAngle
    VClass = MM3Bend
    prefix = 'MM3BENDA'


class UreyBradleyHarmGenerator(BendGenerator):
    par_info = [('K', float), ('R0', float)]
    prefix = 'UBHARM'
    ICClass = UreyBradley


class BendCosGenerator(BendGenerator):
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'BENDCOS'
    ICClass = BendAngle
    VClass = Cosine

class BendCLinGenerator(BendGenerator):
    par_info = [('A', float)]
    prefix = 'BENDCLIN'
    ICClass = BendCos
    VClass = Chebychev1

    def get_vterm(self, pars, indexes):
        args = pars + (self.ICClass(*indexes),)
        return self.VClass(*args, sign=1.0)


class TorsionAngleHarmGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('A', float), ('PHI0', float)]
    prefix = 'TORSAHARM'
    ICClass = DihedAngle
    VClass = Harmonic

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_dihedrals()


class TorsionCosHarmGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('A', float), ('COS0', float)]
    prefix = 'TORSCHARM'
    ICClass = DihedCos
    VClass = Harmonic

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_dihedrals()


class TorsionGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'TORSION'
    ICClass = DihedAngle
    VClass = Cosine
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_dihedrals()

    def get_vterm(self, pars, indexes):
        # A torsion term with multiplicity m and rest value either 0 or pi/m
        # degrees, can be treated as a polynomial in cos(phi). The code below
        # selects the right polynomial.
        if pars[2] == 0.0 and pars[0] == 1:
            ic = DihedCos(*indexes)
            return Chebychev1(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/1)<1e-6 and pars[0] == 1:
            ic = DihedCos(*indexes)
            return Chebychev1(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 2:
            ic = DihedCos(*indexes)
            return Chebychev2(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/2)<1e-6 and pars[0] == 2:
            ic = DihedCos(*indexes)
            return Chebychev2(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 3:
            ic = DihedCos(*indexes)
            return Chebychev3(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/3)<1e-6 and pars[0] == 3:
            ic = DihedCos(*indexes)
            return Chebychev3(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 4:
            ic = DihedCos(*indexes)
            return Chebychev4(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/4)<1e-6 and pars[0] == 4:
            ic = DihedCos(*indexes)
            return Chebychev4(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 6:
            ic = DihedCos(*indexes)
            return Chebychev6(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/6)<1e-6 and pars[0] == 6:
            ic = DihedCos(*indexes)
            return Chebychev6(pars[1], ic, sign=1)
        else:
            return ValenceGenerator.get_vterm(self, pars, indexes)


class TorsionCos2HarmGenerator(ValenceGenerator):
    'A term harmonic in the cos(2*psi)'
    nffatype = 4
    par_info = [('A', float), ('COS0', float)]
    prefix = 'TORSC2HARM'
    ICClass = DihedCos
    VClass = PolyFour

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_dihedrals()

    def process_pars(self, pardef, conversions, nffatype, par_info=None):
        '''
            Transform the 2 parameters given in the parameter file to the 4
            parameters required by PolyFour. The parameters of PolyFour are
            given as a single argument (a list) containing all 4 parameters, not
            4 arguments with each a parameter.
        '''
        tmp = Generator.process_pars(self, pardef, conversions, nffatype, par_info=par_info)
        par_table = {}
        for key, oldpars in tmp.items():
            pars = [0.0, -4*oldpars[0][0]*oldpars[0][1]**2, 0.0, 2.0*oldpars[0][0]]
            par_table[key] = [(pars,)]
        return par_table


class OopAngleGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('PSI0', float)]
    prefix = 'OOPAngle'
    ICClass = OopAngle
    VClass = Harmonic
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield (key[1],key[0],key[2],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopAngle term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                #Yield a term for all three out-of-plane angles
                #with atom as center atom
                yield neighbours[0],neighbours[1],neighbours[2],atom
                yield neighbours[1],neighbours[2],neighbours[0],atom
                yield neighbours[2],neighbours[0],neighbours[1],atom

class OopMeanAngleGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('PSI0', float)]
    prefix = 'OOPMANGLE'
    ICClass = OopMeanAngle
    VClass = Harmonic
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield (key[1],key[2],key[0],key[3])
        yield (key[2],key[0],key[1],key[3])
        yield (key[1],key[0],key[2],key[3])
        yield (key[0],key[2],key[1],key[3])
        yield (key[2],key[1],key[0],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopAngle term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom


class OopCosGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('A', float)]
    prefix = 'OOPCOS'
    ICClass = OopCos
    VClass = Chebychev1
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield (key[1],key[0],key[2],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopCos term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                #Yield a term for all three out-of-plane angles
                #with atom as center atom
                yield neighbours[0],neighbours[1],neighbours[2],atom
                yield neighbours[1],neighbours[2],neighbours[0],atom
                yield neighbours[2],neighbours[0],neighbours[1],atom

    def get_vterm(self, pars, indexes):
        ic = OopCos(*indexes)
        return Chebychev1(pars[0], ic)


class OopMeanCosGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('A', float)]
    prefix = 'OOPMCOS'
    ICClass = OopMeanCos
    VClass = Chebychev1
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield (key[1],key[2],key[0],key[3])
        yield (key[2],key[0],key[1],key[3])
        yield (key[1],key[0],key[2],key[3])
        yield (key[0],key[2],key[1],key[3])
        yield (key[2],key[1],key[0],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopCos term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

    def get_vterm(self, pars, indexes):
        ic = OopMeanCos(*indexes)
        return Chebychev1(pars[0], ic)


class OopDistGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'OOPDIST'
    ICClass = OopDist
    VClass = Harmonic
    allow_superposition = False

    def iter_alt_keys(self, key):
        yield key
        yield (key[2],key[0],key[1],key[3])
        yield (key[1],key[2],key[0],key[3])
        yield (key[2],key[1],key[0],key[3])
        yield (key[1],key[0],key[2],key[3])
        yield (key[0],key[2],key[1],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom


class SquareOopDistGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'SQOOPDIST'
    ICClass = SqOopDist
    VClass = Harmonic
    allow_superposition = False

    def iter_alt_keys(self, key):
        yield key
        yield (key[2],key[0],key[1],key[3])
        yield (key[1],key[2],key[0],key[3])
        yield (key[2],key[1],key[0],key[3])
        yield (key[1],key[0],key[2],key[3])
        yield (key[0],key[2],key[1],key[3])

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom


class ImproperGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'IMPROPER'
    ICClass = DihedAngle
    VClass = Cosine
    allow_superposition = True

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an Improper term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                for i1, i2, i3 in permutations([1,2,3]):
                    yield atom, i1, i2, i3
                for i1, i2, i3 in permutations([1,2,3]):
                    yield i1, atom, i2, i3


class ValenceCrossGenerator(Generator):
    '''All generators for cross valence terms derive from this class.

       **More important attributes:**

       nffatype
            The number of atoms involved in the internal coordinates. Hence
            this is also the number ffatypes in a single row in the force field
            parameter file.

       ICClass0
            The first ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       ICClass1
            The second ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       ICClass2
            The third ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       VClass01
            The ``ValenceTerm`` class for the cross term between IC0 and IC1.
            See ``yaff.pes.vlist``.

       VClass02
            The ``ValenceTerm`` class for the cross term between IC0 and IC2.
            See ``yaff.pes.vlist``.

       VClass12
            The ``ValenceTerm`` class for the cross term between IC1 and IC2.
            See ``yaff.pes.vlist``.
    '''
    suffixes = ['UNIT', 'PARS']
    nffatype = None
    ICClass0 = None
    ICClass1 = None
    ICClass2 = None
    VClass01 = None
    VClass02 = None
    VClass12 = None

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from a ValenceCrossGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args)

    def apply(self, par_table, system, ff_args):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence cross terms.')
        part_valence = ff_args.get_part_valence(system)
        for indexes in self.iter_indexes(system):
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            if len(par_list) == 0 is None and log.do_warning:
                log.warn('No valence %s parameters found for atoms %s with key %s' % (self.prefix, indexes, key))
                continue
            for pars in par_list:
                indexes0 = self.get_indexes0(indexes)
                indexes1 = self.get_indexes1(indexes)
                indexes2 = self.get_indexes2(indexes)
                args_01 = (pars[0], pars[3], pars[4]) + (self.ICClass0(*indexes0), self.ICClass1(*indexes1))
                args_02 = (pars[1], pars[3], pars[5]) + (self.ICClass0(*indexes0), self.ICClass2(*indexes2))
                args_12 = (pars[2], pars[4], pars[5]) + (self.ICClass1(*indexes1), self.ICClass2(*indexes2))
                part_valence.add_term(self.VClass01(*args_01))
                part_valence.add_term(self.VClass02(*args_02))
                part_valence.add_term(self.VClass12(*args_12))

    def iter_indexes(self, system):
        '''Iterate over all tuples of indexes for the pair of internal coordinates'''
        raise NotImplementedError

    def get_indexes0(self, indexes):
        '''Get the indexes for the first internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes1(self, indexes):
        '''Get the indexes for the second internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes2(self, indexes):
        '''Get the indexes for the third internal coordinate from the whole'''
        raise NotImplementedError


class CrossGenerator(ValenceCrossGenerator):
    prefix = 'CROSS'
    par_info = [('KSS', float), ('KBS0', float), ('KBS1', float), ('R0', float), ('R1', float), ('THETA0', float)]
    nffatype = 3
    ICClass0 = Bond
    ICClass1 = Bond
    ICClass2 = BendAngle
    VClass01 = Cross
    VClass02 = Cross
    VClass12 = Cross

    def iter_alt_keys(self, key):
        yield key

    def iter_indexes(self, system):
        return system.iter_angles()

    def get_indexes0(self, indexes):
        return indexes[:2]

    def get_indexes1(self, indexes):
        return indexes[1:]

    def get_indexes2(self, indexes):
        return indexes


class NonbondedGenerator(Generator):
    '''All generators for the non-bonding interactions derive from this class

       **One more important class attribute:**

       mixing_rules
            A dictionary with (par_name, rule_name): (narg, rule_id) items
    '''
    mixing_rules = None

    def process_scales(self, pardef):
        '''Process the SCALE definitions

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary with (numbonds, scale) items.
        '''
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 2:
                pardef.complain(counter, 'must have 2 arguments.')
            try:
                num_bonds = int(words[0])
                scale = float(words[1])
            except ValueError:
                pardef.complain(counter, 'has parameters that can not be converted. The first argument must be an integer. The second argument must be a float.')
            if num_bonds in result and result[num_bonds] != scale:
                pardef.complain(counter, 'contains a duplicate incompatible scale suffix.')
            if scale < 0 or scale > 1:
                pardef.complain(counter, 'has a scale that is not in the range [0,1].')
            result[num_bonds] = scale
        if len(result) < 3 or len(result) > 4:
            pardef.complain(None, 'must contain three or four SCALE suffixes for each non-bonding term.')
        if 1 not in result or 2 not in result or 3 not in result:
            pardef.complain(None, 'must contain a scale parameter for atoms separated by 1, 2 and 3 bonds, for each non-bonding term.')
        if 4 not in result:
            result[4] = 1.0
        return result

    def process_mix(self, pardef):
        '''Process mixing rules

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary of (par_name, (rule_id, rule_args)) items.
        '''
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) < 2:
                pardef.complain(counter, 'contains a mixing rule with to few arguments. At least 2 are required.')
            par_name = words[0].upper()
            rule_name = words[1].upper()
            key = par_name, rule_name
            if key not in self.mixing_rules:
                pardef.complain(counter, 'contains an unknown mixing rule.')
            narg, rule_id = self.mixing_rules[key]
            if len(words) != narg+2:
                pardef.complain(counter, 'does not have the correct number of arguments. %i arguments are required.' % (narg+2))
            try:
                args = tuple([float(word) for word in words[2:]])
            except ValueError:
                pardef.complain(counter, 'contains parameters that could not be converted to floating point numbers.')
            result[par_name] = rule_id, args
        expected_num_rules = len(set([par_name for par_name, rule_id in self.mixing_rules]))
        if len(result) != expected_num_rules:
            pardef.complain(None, 'does not contain enough mixing rules for the generator %s.' % self.prefix)
        return result


class LJGenerator(NonbondedGenerator):
    prefix = 'LJ'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No LJ parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                sigmas[i], epsilons[i] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotLJ)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJ part should not be present yet.')

        pair_pot = PairPotLJ(sigmas, epsilons, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class LJCrossGenerator(NonbondedGenerator):
    prefix = 'LJCROSS'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        #TODO:
        # Prepare the atomic parameters
        sigmas = np.zeros([system.natom,system.natom])
        epsilons = np.zeros([system.natom,system.natom])
        for i in range(system.natom):
            for j in range(system.natom):
                key = (system.get_ffatype(i),system.get_ffatype(j))
                par_list = par_table.get(key, [])
                if len(par_list) == 0:
                    if log.do_warning:
                        log.warn('No LJ cross parameters found for atom tupple %i,%i with fftypes %s,%s.' % (i, system.get_ffatype(i)))
                else:
                    sigmas[i,j], epsilons[i,j] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotLJCross)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJ part should not be present yet.')

        pair_pot = PairPotLJ(sigmas, epsilons, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class MM3Generator(NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No MM3 parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotMM3)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the MM3 part should not be present yet.')

        pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class ExpRepGenerator(NonbondedGenerator):
    prefix = 'EXPREP'
    suffixes = ['UNIT', 'SCALE', 'MIX', 'PARS', 'CPARS']
    par_info = [('A', float), ('B', float)]
    mixing_rules = {
        ('A', 'GEOMETRIC'): (0, 0),
        ('A', 'GEOMETRIC_COR'): (1, 1),
        ('B', 'ARITHMETIC'): (0, 0),
        ('B', 'ARITHMETIC_COR'): (1, 1),
    }

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        cpar_table = self.process_pars(parsec['CPARS'], conversions, 2)
        scale_table = self.process_scales(parsec['SCALE'])
        mixing_rules = self.process_mix(parsec['MIX'])
        self.apply(par_table, cpar_table, scale_table, mixing_rules, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, par_table, cpar_table, scale_table, mixing_rules, system, ff_args):
        # Prepare the atomic parameters
        amps = np.zeros(system.nffatype, float)
        bs = np.zeros(system.nffatype, float)
        for i in range(system.nffatype):
            key = (system.ffatypes[i],)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No EXPREP parameters found for ffatype %s.' % system.ffatypes[i])
            else:
                amps[i], bs[i] = par_list[0]

        # Prepare the cross parameters
        amp_cross = np.zeros((system.nffatype, system.nffatype), float)
        b_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in range(system.nffatype):
            for i1 in range(i0+1):
                cpar_list = cpar_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(cpar_list) == 0:
                    if log.do_high:
                        log('No EXPREP cross parameters found for ffatypes %s,%s. Mixing rule will be used' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    amp_cross[i0,i1], b_cross[i0,i1] = cpar_list[0]
                    if i0 != i1:
                        amp_cross[i1,i0], b_cross[i1,i0] = cpar_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])
        amp_mix, amp_mix_coeff = mixing_rules['A']
        if amp_mix == 0:
            amp_mix_coeff = 0.0
        elif amp_mix == 1:
            amp_mix_coeff = amp_mix_coeff[0]
        b_mix, b_mix_coeff = mixing_rules['B']
        if b_mix == 0:
            b_mix_coeff = 0.0
        elif b_mix == 1:
            b_mix_coeff = b_mix_coeff[0]

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotExpRep)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the EXPREP part should not be present yet.')

        pair_pot = PairPotExpRep(
            system.ffatype_ids, amp_cross, b_cross, ff_args.rcut, ff_args.tr,
            amps, amp_mix, amp_mix_coeff, bs, b_mix, b_mix_coeff,
        )
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class QMDFFRepGenerator(NonbondedGenerator):
    prefix = 'QMDFFREP'
    suffixes = ['UNIT', 'SCALE', 'CPARS']
    par_info = [('A', float), ('B', float)]
    pairpar_info = [('A', float), ('B', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        cpar_table = self.process_pars(parsec['CPARS'], conversions, 2)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(cpar_table, scale_table, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, cpar_table, scale_table, system, ff_args):
        # Prepare the cross parameters
        amp_cross = np.zeros((system.nffatype, system.nffatype), float)
        b_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in range(system.nffatype):
            for i1 in range(i0+1):
                cpar_list = cpar_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(cpar_list) == 0:
                    if log.do_high:
                        log('No EXPREP cross parameters found for ffatypes %s,%s.' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    amp_cross[i0,i1], b_cross[i0,i1] = cpar_list[0]
                    if i0 != i1:
                        amp_cross[i1,i0], b_cross[i1,i0] = cpar_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotQMDFFRep)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the QMDFFREP part should not be present yet.')

        pair_pot = PairPotQMDFFRep(
            system.ffatype_ids, amp_cross, b_cross, ff_args.rcut, ff_args.tr
        )
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class DampDispGenerator(NonbondedGenerator):
    prefix = 'DAMPDISP'
    suffixes = ['UNIT', 'SCALE', 'PARS', 'CPARS']
    par_info = [('C6', float), ('B', float), ('VOL', float)]
    cpar_info = [('C6', float), ('B', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        cpar_table = self.process_pars(parsec['CPARS'], conversions, 2, self.cpar_info)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, cpar_table, scale_table, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, par_table, cpar_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        c6s = np.zeros(system.nffatype, float)
        bs = np.zeros(system.nffatype, float)
        vols = np.zeros(system.nffatype, float)
        for i in range(system.nffatype):
            key = (system.ffatypes[i],)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No DAMPDISP parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                c6s[i], bs[i], vols[i] = par_list[0]

        # Prepare the cross parameters
        c6_cross = np.zeros((system.nffatype, system.nffatype), float)
        b_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in range(system.nffatype):
            for i1 in range(i0+1):
                cpar_list = cpar_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(cpar_list) == 0:
                    if log.do_high:
                        log('No DAMPDISP cross parameters found for ffatypes %s,%s. Mixing rule will be used' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    c6_cross[i0,i1], b_cross[i0,i1] = cpar_list[0]
                    if i0 != i1:
                        c6_cross[i1,i0], b_cross[i1,i0] = cpar_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotDampDisp)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the DAMPDISP part should not be present yet.')

        pair_pot = PairPotDampDisp(system.ffatype_ids, c6_cross, b_cross, ff_args.rcut, ff_args.tr, c6s, bs, vols)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class D3BJGenerator(NonbondedGenerator):
    prefix = 'D3BJ'
    suffixes = ['UNIT', 'SCALE', 'CPARS', 'GLOBALPARS']
    par_info = [('C6', float), ('C8', float),('S6', float), ('S8', float),('A1', float), ('A2', float)]
    pairpar_info = [('C6', float), ('C8', float)]
    globalpar_info = [('S6', float), ('S8', float),('A1', float), ('A2', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        #Parameters for every couple of ffatypes
        par_table = self.process_pars(parsec['CPARS'], conversions, 2, self.pairpar_info)
        #Global parameters, specifically for D3BJ these are s6,s8,a1,a2
        globalpar_table = self.process_pars(parsec['GLOBALPARS'], conversions, 0, self.globalpar_info)
        scale_table = self.process_scales(parsec['SCALE'])
        self.apply(par_table, globalpar_table, scale_table, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, par_table, globalpar_table, scale_table, system, ff_args):
        # Prepare the cross parameters
        c6_cross = np.zeros((system.nffatype, system.nffatype), float)
        c8_cross = np.zeros((system.nffatype, system.nffatype), float)
        R_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in range(system.nffatype):
            for i1 in range(i0+1):
                par_list = par_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(par_list) == 0:
                    if log.do_high:
                        log('No D3BJ cross parameters found for ffatypes %s,%s. Parameters reset to zero.' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    c6_cross[i0,i1], c8_cross[i0,i1] = par_list[0]
                    if i0 != i1:
                        c6_cross[i1,i0], c8_cross[i1,i0] = par_list[0]
        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])
        gps = globalpar_table.get(())[0]
        s6,s8,a1,a2 = gps[0],gps[1],gps[2],gps[3]
        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotDampDisp)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the DAMPDISP part should not be present yet.')

        pair_pot = PairPotDisp68BJDamp(system.ffatype_ids, c6_cross, c8_cross, R_cross, c6_scale=s6, c8_scale=s8, bj_a=a1, bj_b=a2, rcut=ff_args.rcut, tr=ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class FixedChargeGenerator(NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        self.apply(atom_table, bond_table, scale_table, dielectric, system, ff_args)

    def process_atoms(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments.')
            ffatype = words[0]
            if ffatype in result:
                pardef.complain(counter, 'has an atom type that was already encountered earlier.')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to a floating point number.')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments.')
            key = tuple(words[:2])
            if key in result:
                pardef.complain(counter, 'has a combination of atom types that were already encountered earlier.')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to floating point numbers.')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, pardef):
        result = None
        for counter, line in pardef:
            if result is not None:
                pardef.complain(counter, 'is redundant. The DIELECTRIC suffix may only occur once.')
            words = line.split()
            if len(words) != 1:
                pardef.complain(counter, 'must have one argument.')
            try:
                result = float(words[0])
            except ValueError:
                pardef.complain(counter, 'must have a floating point argument.')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system, ff_args):
        if system.charges is None:
            system.charges = np.zeros(system.natom)
        elif log.do_warning and abs(system.charges).max() != 0:
            log.warn('Overwriting charges in system.')
        system.charges[:] = 0.0
        system.radii = np.zeros(system.natom)

        # compute the charges
        for i in range(system.natom):
            pars = atom_table.get(system.get_ffatype(i))
            if pars is not None:
                charge, radius = pars
                system.charges[i] += charge
                system.radii[i] = radius
            elif log.do_warning:
                log.warn('No charge defined for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
        for i0, i1 in system.iter_bonds():
            ffatype0 = system.get_ffatype(i0)
            ffatype1 = system.get_ffatype(i1)
            if ffatype0 == ffatype1:
                continue
            charge_transfer = bond_table.get((ffatype0, ffatype1))
            if charge_transfer is None:
                if log.do_warning:
                    log.warn('No charge transfer parameter for atom pair (%i,%i) with fftype (%s,%s).' % (i0, i1, system.get_ffatype(i0), system.get_ffatype(i1)))
            else:
                system.charges[i0] += charge_transfer
                system.charges[i1] -= charge_transfer

        # prepare other parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Setup the electrostatic pars
        ff_args.add_electrostatic_parts(system, scalings, dielectric)


def apply_generators(system, parameters, ff_args):
    '''Populate the attributes of ff_args, prepares arguments for ForceField

       **Arguments:**

       system
            A System instance for which the force field object is being made

       ff_args
            An instance of the FFArgs class.

       parameters
            An instance of the Parameters, typically made by
            ``Parmaeters.from_file('parameters.txt')``.
    '''

    # Collect all the generators that have a prefix.
    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if log.do_warning:
                log.warn('There is no generator named %s.' % prefix)
        else:
            generator(system, section, ff_args)
