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

from molmod.units import parse_unit

from yaff.log import log
from yaff.pes.ext import PairPotEI, PairPotLJ, PairPotMM3, PairPotExpRep, \
    PairPotDampDisp, Switch3
from yaff.pes.ff import ForcePartPair, ForcePartValence, \
    ForcePartEwaldReciprocal, ForcePartEwaldCorrection, \
    ForcePartEwaldNeutralizing
from yaff.pes.iclist import Bond, BendAngle, BendCos, \
    UreyBradley, DihedAngle, DihedCos
from yaff.pes.nlist import NeighborList
from yaff.pes.scaling import Scalings
from yaff.pes.vlist import Harmonic, Fues, Cross, Cosine, \
    Chebychev1, Chebychev2


__all__ = [
    'ParsedPars', 'FFArgs', 'Generator',

    'ValenceGenerator', 'BondGenerator', 'BondHarmGenerator',
    'BondFuesGenerator', 'BendGenerator', 'BendAngleHarmGenerator',
    'BendCosHarmGenerator', 'UreyBradleyHarmGenerator',

    'ValenceCrossGenerator', 'BondCrossGeneratorGenerator',

    'NonbondedGenerator', 'LJGenerator', 'MM3Generator', 'ExpRepGenerator',
    'DampDispGenerator', 'FixedChargeGenerator',

    'generators',
]


class ParsedPars(object):
    def __init__(self, fn, info=None):
        self.fn = fn
        if info is None:
            f = open(fn)
            try:
                self.load(f)
            finally:
                f.close()
        else:
            self.info = info

    def complain(self, counter, message=None):
        if counter is None:
            raise IOError('The parameter file %s %s.' % (self.fn, message))
        else:
            raise IOError('Line %i in the parameter file %s %s.' % (counter, self.fn, message))

    def load(self, f):
        self.info = {}
        counter = 1
        for line in f:
            line = line[:line.find('#')].strip()
            if len(line) > 0:
                pos = line.find(':')
                if pos == -1:
                    self.complain(counter, 'does not contain a colon')
                prefix = line[:pos].upper()
                rest = line[pos+1:].strip()
                if len(rest) == 0:
                    self.complain(counter, 'does not have text after the colon')
                if len(prefix.split()) > 1:
                    self.complain(counter, 'has a prefix that contains whitespace')
                pos = rest.find(' ')
                if pos == 0:
                    self.complain(counter, 'does not have a command after the prefix')
                elif pos == -1:
                    self.complain(counter, 'does not have data after the command')
                command = rest[:pos].upper()
                data = rest[pos+1:].strip()
                l1 = self.info.setdefault(prefix, {})
                l2 = l1.setdefault(command, [])
                l2.append((counter, data))
            counter += 1

    def get_section(self, key):
        if key in self.info:
            return ParsedPars(self.fn, self.info[key])
        else:
            return ParsedPars(self.fn, {})


class FFArgs(object):
    def __init__(self, rcut=18.89726133921252, tr=Switch3(7.558904535685008),
                 alpha_scale=3.0, gcut_scale=1.1, skin=0, smooth_ei=False,
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
                the real-space cutoff: alpha = alpha_scale / rcut. The higher
                this parameter, the better the real-space convergence.

           gcut_scale
                Determines the reciprocale space cutoff based on the alpha
                parameter: gcut = gcut_scale * alpha. The higher this parameter,
                the more accurate the computation.

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
           optimal tradeoff between accuracy and computational cost requires
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

    def add_electrostatic_parts(self, system, scalings):
        if self.get_part_pair(PairPotEI) is not None:
            return
        nlist = self.get_nlist(system)
        if system.cell.nvec == 0:
            alpha = 0.0
        elif system.cell.nvec == 3:
            alpha = self.alpha_scale/self.rcut
        else:
            raise NotImplementedError('Only zero- and three-dimensional electrostatics are supported.')
        # Real-space electrostatics
        if self.smooth_ei:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, self.tr)
        else:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut)
        part_pair_ei = ForcePartPair(system, nlist, scalings, pair_pot_ei)
        self.parts.append(part_pair_ei)
        if self.reci_ei == 'ignore':
            # Nothing to do
            pass
        elif self.reci_ei == 'ewald':
            if system.cell.nvec == 3:
                # Reciprocal-space electrostatics
                # TODO: also make these smooth if smooth_ei is True
                part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, gcut=self.gcut_scale*alpha)
                self.parts.append(part_ewald_reci)
                # Ewald corrections
                part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings)
                self.parts.append(part_ewald_corr)
                # Neutralizing background
                part_ewald_neut = ForcePartEwaldNeutralizing(system, alpha)
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
    """
    prefix = None
    par_info = None
    commands = None
    allow_superposition = False

    def __call__(self, system, parsed_pars, ff_args):
        raise NotImplementedError

    def check_commands(self, parsed_pars):
        for command in parsed_pars.info:
            if command not in self.commands:
                parsed_pars.complain(None, 'contains a command (%s) that is not recognized by generator %s.' % (command, self.prefix))

    def process_units(self, parsed_pars):
        result = {}
        expected_names = [name for name, dtype in self.par_info if dtype is float]
        for counter, line in parsed_pars.info:
            words = line.split()
            if len(words) != 2:
                parsed_pars.complain(counter, 'must have two arguments in UNIT command.')
            name = words[0].upper()
            if name not in expected_names:
                parsed_pars.complain(counter, 'specifies a unit for an unknown parameter. (Must be one of %s, but got %s.)' % (expected_names, name))
            try:
                result[name] = parse_unit(words[1])
            except (NameError, ValueError):
                parsed_pars.complain(counter, 'has a UNIT command with an unknown unit.')
        if len(result) != len(expected_names):
            raise IOError('Not all units are specified for generator %s in file %s. Got %s, should have %s.' % (
                self.prefix, parsed_pars.fn, result.keys(), expected_names
            ))
        return result

    def process_pars(self, parsed_pars, conversions, nffatype, par_info=None):
        if par_info is None:
            par_info = self.par_info
        par_table = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            num_args = nffatype + len(par_info)
            if len(words) != num_args:
                parsed_pars.complain(counter, 'should have %s arguments.' % num_args)
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
                parsed_pars.complain(counter, 'has parameters that can not be converted to numbers.')
            par_list = par_table.get(key, [])
            if len(par_list) > 0 and not self.allow_superposition:
                parsed_pars.complain(counter, 'conts duplicate parameters, which is not allowed for generator %s.' % self.prefix)
            par_list.append(pars)
            for key in self.iter_alt_keys(key):
                par_table[key] = par_list
        return par_table

    def iter_alt_keys(self, key):
        if len(key) == 1:
            yield key
        else:
            raise NotImplementedError


class ValenceGenerator(Generator):
    commands = ['UNIT', 'PARS']
    nffatype = None
    ICClass = None
    VClass = None

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, self.nffatype)
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args)

    def apply(self, par_table, system, ff_args):
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
        args = pars + (self.ICClass(*indexes),)
        return self.VClass(*args)



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


class UreyBradleyHarmGenerator(BendGenerator):
    par_info = [('K', float), ('R0', float)]
    prefix = 'UBHARM'
    ICClass = UreyBradley


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
        if pars[2] == 0.0 and pars[0] == 1:
            ic = DihedCos(*indexes)
            return Chebychev1(pars[1], ic)
        elif pars[2] == 0.0 and pars[0] == 2:
            ic = DihedCos(*indexes)
            return Chebychev2(pars[1], ic)
        else:
            return ValenceGenerator.get_vterm(self, pars, indexes)



class ValenceCrossGenerator(Generator):
    commands = ['UNIT', 'PARS']
    nffatype = None
    ICClass0 = None
    ICClass1 = None
    VClass = None

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, self.nffatype)
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args)

    def apply(self, par_table, system, ff_args):
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
                args = pars + (self.ICClass0(*indexes0), self.ICClass1(*indexes1))
                part_valence.add_term(self.VClass(*args))

    def get_indexes0(self, indexes):
        raise NotImplementedError

    def get_indexes1(self, indexes):
        raise NotImplementedError


class BondCrossGeneratorGenerator(ValenceCrossGenerator):
    prefix = 'BONDCROSS'
    par_info = [('K', float), ('R0', float), ('R1', float)]
    nffatype = 3
    ICClass0 = Bond
    ICClass1 = Bond
    VClass = Cross

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def iter_indexes(self, system):
        return system.iter_angles()

    def get_indexes0(self, indexes):
        return indexes[:2]

    def get_indexes1(self, indexes):
        return indexes[1:]


class NonbondedGenerator(Generator):
    mixing_rules = None

    def process_scales(self, parsed_pars):
        result = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            if len(words) != 2:
                parsed_pars.complain(counter, 'must have 2 arguments.')
            try:
                num_bonds = int(words[0])
                scale = float(words[1])
            except ValueError:
                parsed_pars.complain(counter, 'has parameters that can not be converted. The first argument must be an integer. The second argument must be a float.')
            if num_bonds in result and result[num_bonds] != scale:
                parsed_pars.complain(counter, 'contains a duplicate incompatible scale command.')
            if scale < 0 or scale > 1:
                parsed_pars.complain(counter, 'has a scale that is not in the range [0,1].')
            result[num_bonds] = scale
        if len(result) != 3:
            parsed_pars.complain(None, 'must contain three SCALE commands for each non-bonding term.')
        if 1 not in result or 2 not in result or 3 not in result:
            parsed_pars.complain(None, 'must contain a scale parameter for atoms separated by 1, 2 and 3 bonds, for each non-bonding term.')
        return result

    def process_mix(self, parsed_pars):
        result = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            if len(words) < 2:
                parsed_pars.complain(counter, 'contains a mixing rule with to few arguments. At least 2 are required.')
            par_name = words[0].upper()
            rule_name = words[1].upper()
            key = par_name, rule_name
            if key not in self.mixing_rules:
                parsed_pars.complain(counter, 'contains an unknown mixing rule.')
            narg, rule_id = self.mixing_rules[key]
            if len(words) != narg+2:
                parsed_pars.complain(counter, 'does not have the correct number of arguments. %i arguments are required.' % (narg+2))
            try:
                args = tuple([float(word) for word in words[2:]])
            except ValueError:
                parsed_pars.complain(counter, 'contains parameters that could not be converted to floating point numbers.')
            result[par_name] = rule_id, args
        expected_num_rules = len(set([par_name for par_name, rule_id in self.mixing_rules]))
        if len(result) != expected_num_rules:
            parsed_pars.complain(None, 'does not contain enough mixing rules for the generator %s.' % self.prefix)
        return result


class LJGenerator(NonbondedGenerator):
    prefix = 'LJ'
    commands = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float)]

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, 1)
        scale_table = self.process_scales(parsed_pars.get_section('SCALE'))
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        for i in xrange(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No LJ parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                sigmas[i], epsilons[i] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotLJ)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the LJ part should not be present yet.')

        pair_pot = PairPotLJ(sigmas, epsilons, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class MM3Generator(NonbondedGenerator):
    prefix = 'MM3'
    commands = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, 1)
        scale_table = self.process_scales(parsed_pars.get_section('SCALE'))
        self.apply(par_table, scale_table, system, ff_args)

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in xrange(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No MM3 parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3])

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
    commands = ['UNIT', 'SCALE', 'MIX', 'PARS', 'CPARS']
    par_info = [('A', float), ('B', float)]
    mixing_rules = {
        ('A', 'GEOMETRIC'): (0, 0),
        ('A', 'GEOMETRIC_COR'): (1, 1),
        ('B', 'ARITHMETIC'): (0, 0),
        ('B', 'ARITHMETIC_COR'): (1, 1),
    }

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, 1)
        cpar_table = self.process_pars(parsed_pars.get_section('CPARS'), conversions, 2)
        scale_table = self.process_scales(parsed_pars.get_section('SCALE'))
        mixing_rules = self.process_mix(parsed_pars.get_section('MIX'))
        self.apply(par_table, cpar_table, scale_table, mixing_rules, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, par_table, cpar_table, scale_table, mixing_rules, system, ff_args):
        # Prepare the atomic parameters
        amps = np.zeros(system.nffatype, float)
        bs = np.zeros(system.nffatype, float)
        for i in xrange(system.nffatype):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No EXPREP parameters found for ffatype %s.' % system.ffatypes[i])
            else:
                amps[i], bs[i] = par_list[0]

        # Prepare the cross parameters
        amp_cross = np.zeros((system.nffatype, system.nffatype), float)
        b_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in xrange(system.nffatype):
            for i1 in xrange(i0+1):
                cpar_list = cpar_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(cpar_list) == 0:
                    if log.do_high:
                        log('No EXPREP cross parameters found for ffatypes %s,%s. Mixing rule will be used' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    amp_cross[i0,i1], b_cross[i0,i1] = cpar_list[0]
                    if i0 != i1:
                        amp_cross[i1,i0], b_cross[i1,i0] = cpar_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3])
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


class DampDispGenerator(NonbondedGenerator):
    # TODO: it seems that the dispersion interactions are just too strong. check
    # if this is a bug.
    prefix = 'DAMPDISP'
    commands = ['UNIT', 'SCALE', 'PARS', 'CPARS']
    par_info = [('C6', float), ('B', float), ('VOL', float)]
    cpar_info = [('C6', float), ('B', float)]

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        par_table = self.process_pars(parsed_pars.get_section('PARS'), conversions, 1)
        cpar_table = self.process_pars(parsed_pars.get_section('CPARS'), conversions, 2, self.cpar_info)
        scale_table = self.process_scales(parsed_pars.get_section('SCALE'))
        self.apply(par_table, cpar_table, scale_table, system, ff_args)

    def iter_alt_keys(self, key):
        yield key
        yield key[::-1]

    def apply(self, par_table, cpar_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        c6s = np.zeros(system.nffatype, float)
        bs = np.zeros(system.nffatype, float)
        vols = np.zeros(system.nffatype, float)
        for i in xrange(system.nffatype):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) == 0:
                if log.do_warning:
                    log.warn('No DAMPDISP parameters found for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                c6s[i], bs[i], vols[i] = par_list[0]

        # Prepare the cross parameters
        c6_cross = np.zeros((system.nffatype, system.nffatype), float)
        b_cross = np.zeros((system.nffatype, system.nffatype), float)
        for i0 in xrange(system.nffatype):
            for i1 in xrange(i0+1):
                cpar_list = cpar_table.get((system.ffatypes[i0], system.ffatypes[i1]), [])
                if len(cpar_list) == 0:
                    if log.do_high:
                        log('No DAMPDISP cross parameters found for ffatypes %s,%s. Mixing rule will be used' % (system.ffatypes[i0], system.ffatypes[i1]))
                else:
                    c6_cross[i0,i1], b_cross[i0,i1] = cpar_list[0]
                    if i0 != i1:
                        c6_cross[i1,i0], b_cross[i1,i0] = cpar_list[0]

        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotDampDisp)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the DAMPDISP part should not be present yet.')

        pair_pot = PairPotDampDisp(system.ffatype_ids, c6_cross, b_cross, ff_args.rcut, ff_args.tr, c6s, bs, vols)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)


class FixedChargeGenerator(NonbondedGenerator):
    prefix = 'FIXQ'
    commands = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsed_pars, ff_args):
        self.check_commands(parsed_pars)
        conversions = self.process_units(parsed_pars.get_section('UNIT'))
        atom_table = self.process_atoms(parsed_pars.get_section('ATOM'), conversions)
        bond_table = self.process_bonds(parsed_pars.get_section('BOND'), conversions)
        scale_table = self.process_scales(parsed_pars.get_section('SCALE'))
        dielectric = self.process_dielectric(parsed_pars.get_section('DIELECTRIC'))
        self.apply(atom_table, bond_table, scale_table, dielectric, system, ff_args)

    def process_atoms(self, parsed_pars, conversions):
        result = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            if len(words) != 3:
                parsed_pars.complain(counter, 'should have 3 arguments.')
            ffatype = words[0]
            if ffatype in result:
                parsed_pars.complain(counter, 'has an atom type that was already encountered earlier.')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                parsed_pars.complain(counter, 'contains a parameter that can not be converted to a floating point number.')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, parsed_pars, conversions):
        result = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            if len(words) != 3:
                parsed_pars.complain(counter, 'should have 3 arguments.')
            key = tuple(words[:2])
            if key in result:
                parsed_pars.complain(counter, 'has a combination of atom types that were already encountered earlier.')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                parsed_pars.complain(counter, 'contains a parameter that can not be converted to floating point numbers.')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, parsed_pars):
        result = None
        for counter, line in parsed_pars.info:
            if result is not None:
                parsed_pars.complain(counter, 'is redundant. The DIELECTRIC command may only occur once.')
            words = line.split()
            if len(words) != 1:
                parsed_pars.complain(counter, 'must have one argument.')
            try:
                result = float(words[0])
            except ValueError:
                parsed_pars.complain(counter, 'must have a floating point argument.')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system, ff_args):
        if system.charges is None:
            system.charges = np.zeros(system.natom)
        elif log.do_warning and abs(system.charges).max() != 0:
            log.warn('Overwriting charges in system.')
            system.charges[:] = 0.0
        radii = np.zeros(system.natom)

        # compute the charges
        for i in xrange(system.natom):
            pars = atom_table.get(system.get_ffatype(i))
            if pars is None and log.do_warning:
                log.warn('No charge defined for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
            else:
                charge, radius = pars
                if radius > 0:
                    raise NotImplementedError('TODO: support smeared charges.')
                system.charges[i] += charge
                radii[i] = radius
        for i0, i1 in system.bonds:
            ffatype0 = system.get_ffatype(i0)
            ffatype1 = system.get_ffatype(i1)
            if ffatype0 == ffatype1:
                continue
            charge_transfer = bond_table.get((ffatype0, ffatype1))
            if charge_transfer is None and log.do_warning:
                log.warn('No charge transfer parameter for atom pair (%i,%i) with fftype (%s,%s).' % (i0, i1, system.get_ffatype(i0), system.get_ffatype(i1)))
            else:
                system.charges[i0] += charge_transfer
                system.charges[i1] -= charge_transfer

        # prepare other parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3])

        if dielectric != 1.0:
            raise NotImplementedError('Only a relative dielectric constant of 1 is supported.')

        # Setup the electrostatic pars
        ff_args.add_electrostatic_parts(system, scalings)


# Collect all the generators that have a prefix.
generators = {}
for x in globals().values():
    if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
        generators[x.prefix] = x()
