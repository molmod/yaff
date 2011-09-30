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


from molmod.units import parse_unit

from yaff.ff import ForcePartPair, ForcePartValence
from yaff.nlists import NeighborLists


__all__ = ['ParsedPars', 'FFArgs', 'Generator', 'BondHarmGenerator', 'generators']


class ParsedPars(object):
    def __init__(self, fn, info=None):
        self.fn = fn
        if info is None:
            self.load(fn)
        else:
            self.info = info

    def complain(self, counter, message=None):
        if counter is None:
            raise IOError('The parameter file %s %s.' % (self.fn, message))
        else:
            raise IOError('Line %i in the parameter file %s %s.' % (counter, self.fn, message))

    def load(self, fn):
        self.info = {}
        f = file(fn)
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
        f.close()

    def get_section(self, key):
        if key in self.info:
            return ParsedPars(self.fn_paramters, self.info[key])
        else:
            return ParsedPars(self.fn_paramters, {})


class FFArgs(object):
    # TODO: fix default alpha
    def __init__(self, rcut=18.89726133921252, alpha=0.0, gcut=0.35):
        """
           **Optional arguments:**

           Some optional arguments only make sense if related parameters in the
           parameter file are present.

           rcut
                The real space cutoff used by all pair potentials.

           alpha
                The alpha parameter in the Ewald summation.

           gcut
                The reciprocal space cutoff for the Ewald summation.
        """
        self.parts = []
        self.nlists = None
        self.rcut = rcut
        self.alpha = alpha
        self.gcut = gcut

    def get_nlists(self, system):
        if self.nlists is None:
            self.nlists = NeighborLists(system)
        return self.nlists

    def get_part(self, ForcePartClass):
        for part in self.parts:
            if isinstance(part, ForcePartClass):
                return part

    def get_part_pair(self, PairPotClass):
        for part in self.parts:
            if isinstance(part, ForcePartPair) and isisntance(part.pair_pot, PairPotClass):
                return part

    def get_part_valence(self, system):
        part_valence = ff_args.get_part(ForcePartValence)
        if part_valence is None:
            part_valence = ForcePartValence(system)
            ff_args.append(part_valence)
        return part_valence


class Generator(object):
    """Creates (part of a) ForceField object automatically.

       A generator is a class that describes how a part of a parameter file
       must be turned into a part of ForceField object. As the generator
       proceeds, it will modify and extend the current arguments of the FF. They
       should be implemented such that the order of the generators is not
       important.
    """
    prefix = None
    num_ffatypes = None
    par_names = None

    def __call__(self, system, parsed_pars, ff_args):
        conversions = self.process_units(parsed_pars.get_section('UNITS'))
        par_table = self.process_pars(parsed_pars.get_section('UNITS'), conversions)
        if len(par_table) > 0:
            aux_info = self.process_aux(parsed_pars)
            self.apply(par_table, aux_info, system, ff_args)

    def process_units(self, parsed_pars):
        result = {}
        for counter, line in parsed_pars.info:
            line = words.split()
            if len(words) != 2:
                parsed_pars.complain(counter, 'should have two arguments in UNIT command.')
            name = words[0].upper()
            if name not in self.par_names:
                parsed_pars.complain(counter, 'specifies a unit for an unknown parameter. (Should be one of %s, but got %s.)' % (self.par_names, name))
            try:
                result[name] = parse_unit(words[1])
            except (NameError, ValueError):
                parsed_pars.complain(counter, 'has a UNIT command with an unknown unit.')
        if len(results) != len(self.par_names):
            raise IOError('Not all units are specified for generator %s in file %s. Got %s, should have %s.' % (
                self.prefix, parsed_pars.fn, result.keys(), self.par_names
            ))
        return result

    def process_pars(self, parsed_pars, conversions):
        par_table = {}
        for counter, line in parsed_pars.info:
            words = line.split()
            num_args = self.num_ffatypes + len(self.par_names)
            if len(words) != num_args:
                parsed_pars.complain(counter, 'should have %s arguments.' % num_args)
            key = tuple(words[:self.num_ffatypes])
            try:
                pars = tuple(
                    float(words[i+self.num_ffatypes])/conversions[par_name]
                    for i, par_name in enumerate(self.par_names)
                )
            except ValueError:
                parsed_pars.complain(counter, 'has parameters that can not be converted to floating point numbers.')
            if key in par_table and log.do_warning:
                log('WARNING!! Duplicate parameters found on line %i in %s. Later ones override earlier ones.' % (counter, parsed_pars.fn))
            for key in self.iter_alt_keys(key):
                par_table[key] = pars
        return par_table

    def process_aux(self, parsed_pars):
        for command, info in parsed_pars.info.iteritems():
            if command == 'UNITS':
                continue
            elif command == 'PARS':
                continue
            else:
                raise parsed_pars.complain('contains an unknown command: %s' % command)

    def apply(self, par_table, system, ff_args):
        raise NotImplementedError


class BondHarmGenerator(Generator):
    prefix = 'BONDHARM'
    num_ffatypes = 2
    par_names = ['K', 'R0']

    def iter_alt_keys(key):
        yield key
        yield key[::-1]

    def apply(self, par_table, system, ff_args):
        if system.topology is None:
            raise ValueError('The system must have a topology (i.e. bonds) in order to define valence terms.')
        part_valence = ff_args.get_part_valence(system)
        from yaff.iclist import Bond
        from yaff.vlist import Harmonic
        for i0, i1 in system.topology.bonds:
            pars = par_table.get((system.ffatypes[i0], system.ffatypes[1]))
            part_valence.add_term(Harmonic(pars, Bond(i0, i1)))


# Collect all the generators that have a prefix.
generators = {}
for x in globals().values():
    if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
        generators[x.prefix] = x
