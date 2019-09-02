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
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from molmod.units import *
from yaff.log import log

from itertools import permutations

__all__ = ['apply_lammps_generators']

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

    def __call__(self, system, parsec):
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
                pardef.complain(counter, 'must have two arguments in UNIT suffix')
            name = words[0].upper()
            if name not in expected_names:
                pardef.complain(counter, 'specifies a unit for an unknown parameter. (Must be one of %s, but got %s.)' % (expected_names, name))
            try:
                result[name] = parse_unit(words[1])
            except (NameError, ValueError):
                pardef.complain(counter, 'has a UNIT suffix with an unknown unit')
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
            # Parsing PARS
            par_info = self.par_info
            allow_superposition = self.allow_superposition
        else:
            # Parsing other fields than PARS, so supperposition should never be allowed.
            allow_superposition = False

        par_table = {}
        par_values = []
        for counter, line in pardef:
            words = line.split()
            num_args = nffatype + len(par_info)
            if len(words) != num_args:
                pardef.complain(counter, 'should have %s arguments' % num_args)
            # Extract the key
            key = tuple(words[:nffatype])
            # Extract the parameters
            pars = []
            for i, (name, dtype) in enumerate(par_info):
                word = words[i + nffatype]
                try:
                    if issubclass(dtype, float):
                        pars.append(float(word)*conversions[name])
                    else:
                        pars.append(dtype(word))
                except ValueError:
                    pardef.complain(counter, 'contains a parameter that can not be converted to a number: {}'.format(word))
            pars = tuple(pars)

            # Process the new key + pars pair, taking into account equivalent permutations
            # of the atom types and corresponding permutations of parameters.
            current_par_table = {}
            for alt_key, alt_pars in self.iter_equiv_keys_and_pars(key, pars):
                # When permuted keys are identical to the original, no new items are
                # added.
                if alt_key in current_par_table:
                    if current_par_table[alt_key] != alt_pars:
                        pardef.complain(counter, 'contains parameters that are not consistent with the permutational symmetry of the atom types')
                else:
                    current_par_table[alt_key] = alt_pars

            # Add the parameters and their permutations to the parameter table, checking
            # for superposition.
            for alt_key, alt_pars in current_par_table.items():
                if not alt_pars in par_values: par_values.append(alt_pars)
                index = par_values.index(alt_pars)
                par_list = par_table.setdefault(alt_key, [])
                if len(par_list) > 0 and not allow_superposition:
                    pardef.complain(counter, 'contains a duplicate energy term, possibly with different parameters, which is not allowed for generator %s' % self.prefix)
                par_list.append(index)
        return par_table, par_values

    def iter_equiv_keys_and_pars(self, key, pars):
        '''Iterates of all equivalent re-orderings of a tuple of ffatypes (keys) and corresponding parameters.'''
        if len(key) == 1:
            yield key, pars
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

    def __call__(self, system, parsec):
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
        par_table, par_values = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            output0 = self.apply(par_table, system)
        return output0, par_values

    def apply(self, par_table, system):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence terms.')
        out0 = []
        for indexes in self.iter_indexes(system):
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                out0.append((pars,[indexes[iindex] for iindex in self.index_order]))
        return out0

    def iter_indexes(self, system):
        '''Iterate over all tuples of indices for the internal coordinate'''
        raise NotImplementedError

class BondGenerator(ValenceGenerator):
    par_info = [('K', float), ('R0', float)]
    nffatype = 2
    index_order = np.arange(nffatype, dtype=int)

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_bonds()

class BondHarmGenerator(BondGenerator):
    prefix = 'BONDHARM'

class MM3QuarticGenerator(BondGenerator):
    prefix = 'MM3QUART'

class BondMorseGenerator(ValenceGenerator):
    prefix = 'BONDMORSE'
    par_info = [('E0', float), ('K', float), ('R0', float)]
    nffatype = 2
    index_order = np.arange(nffatype, dtype=int)

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_bonds()

class BendGenerator(ValenceGenerator):
    nffatype = 3
    index_order = np.arange(nffatype)
    par_info = [('K', float), ('THETA0', float)]

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_angles()


class BendAngleHarmGenerator(BendGenerator):
    prefix = 'BENDAHARM'

class MM3BendGenerator(BendGenerator):
    prefix = 'MM3BENDA'

class BendCosHarmGenerator(BendGenerator):
    par_info = [('K', float), ('COS0', float)]
    prefix = 'BENDCHARM'

class BendCosGenerator(ValenceGenerator):
    nffatype = 3
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    index_order = np.arange(nffatype)
    prefix = 'BENDCOS'

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_angles()


class TorsionGenerator(ValenceGenerator):
    nffatype = 4
    index_order = np.arange(nffatype)
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'TORSION'
    allow_superposition = True

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_dihedrals()


class OopDistGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    index_order = np.array([3,0,1,2],dtype=int)
    prefix = 'OOPDIST'
    allow_superposition = False

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[2], key[0], key[1], key[3]), pars
        yield (key[1], key[2], key[0], key[3]), pars
        yield (key[2], key[1], key[0], key[3]), pars
        yield (key[1], key[0], key[2], key[3]), pars
        yield (key[0], key[2], key[1], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

    def write_lammps(self, par_values):
        counter = 1
        out1 = "%s\n\n" % (self.lammps1)
        for pars in par_values:
#            assert pars[1]==0.0
            out1 += "%5d %15.8f %15.8f\n" % (counter,0.5*pars[0]/self.par_units[0],pars[1]/self.par_units)
            counter += 1
        return out1

class SquareOopDistGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'SQOOPDIST'
    index_order = np.array([3,0,1,2],dtype=int)
    allow_superposition = False

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[2], key[0], key[1], key[3]), pars
        yield (key[1], key[2], key[0], key[3]), pars
        yield (key[2], key[1], key[0], key[3]), pars
        yield (key[1], key[0], key[2], key[3]), pars
        yield (key[0], key[2], key[1], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

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

    def __call__(self, system, parsec):
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
        par_table, par_values = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            indexes = self.apply(par_table, system)
        return indexes, par_values

    def apply(self, par_table, system):
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
        allindexes = []
        for indexes in self.iter_indexes(system):
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                allindexes.append((pars,indexes))
        return allindexes

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

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], (pars[0], pars[2], pars[1], pars[4], pars[3], pars[5])

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
                pardef.complain(counter, 'must have 2 arguments')
            try:
                num_bonds = int(words[0])
                scale = float(words[1])
            except ValueError:
                pardef.complain(counter, 'has parameters that can not be converted. The first argument must be an integer. The second argument must be a float')
            if num_bonds in result and result[num_bonds] != scale:
                pardef.complain(counter, 'contains a duplicate incompatible scale suffix')
            if scale < 0 or scale > 1:
                pardef.complain(counter, 'has a scale that is not in the range [0,1]')
            result[num_bonds] = scale
        if len(result) < 3 or len(result) > 4:
            pardef.complain(None, 'must contain three or four SCALE suffixes for each non-bonding term')
        if 1 not in result or 2 not in result or 3 not in result:
            pardef.complain(None, 'must contain a scale parameter for atoms separated by 1, 2 and 3 bonds, for each non-bonding term')
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
                pardef.complain(counter, 'contains a mixing rule with to few arguments. At least 2 are required')
            par_name = words[0].upper()
            rule_name = words[1].upper()
            key = par_name, rule_name
            if key not in self.mixing_rules:
                pardef.complain(counter, 'contains an unknown mixing rule')
            narg, rule_id = self.mixing_rules[key]
            if len(words) != narg+2:
                pardef.complain(counter, 'does not have the correct number of arguments. %i arguments are required' % (narg+2))
            try:
                args = tuple([float(word) for word in words[2:]])
            except ValueError:
                pardef.complain(counter, 'contains parameters that could not be converted to floating point numbers')
            result[par_name] = rule_id, args
        expected_num_rules = len(set([par_name for par_name, rule_id in self.mixing_rules]))
        if len(result) != expected_num_rules:
            pardef.complain(None, 'does not contain enough mixing rules for the generator %s' % self.prefix)
        return result

class LJGenerator(NonbondedGenerator):
    prefix = 'LJ'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsec):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        return par_table, scale_table

class MM3Generator(NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        return par_table, scale_table

class FixedChargeGenerator(NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        self.apply(atom_table, bond_table, scale_table, dielectric, system)
        return None, scale_table

    def process_atoms(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            ffatype = words[0]
            if ffatype in result:
                pardef.complain(counter, 'has an atom type that was already encountered earlier')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to a floating point number')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            key = tuple(words[:2])
            if key in result:
                pardef.complain(counter, 'has a combination of atom types that were already encountered earlier')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to floating point numbers')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, pardef):
        result = None
        for counter, line in pardef:
            if result is not None:
                pardef.complain(counter, 'is redundant. The DIELECTRIC suffix may only occur once')
            words = line.split()
            if len(words) != 1:
                pardef.complain(counter, 'must have one argument')
            try:
                result = float(words[0])
            except ValueError:
                pardef.complain(counter, 'must have a floating point argument')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system):
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


def apply_lammps_generators(system, parameters):
    '''Populate the attributes of ff_args, prepares arguments for ForceField

       **Arguments:**

       system
            A System instance for which the force field object is being made

       parameters
            An instance of the Parameters, typically made by
            ``Parmaeters.from_file('parameters.txt')``.
    '''

    # Collect all the generators that have a prefix.
    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    output = {}
    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if log.do_warning:
                log.warn('There is no generator named %s. It will be ignored.' % prefix)
        else:
            output[prefix] = generator(system, section)
    return output
