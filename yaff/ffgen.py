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

from yaff.ff import ForcePartPair


__all__ = ['ParsedPars', 'FFArgs', 'Generator', 'BondHarmGenerator', 'generators']


class ParsedPars(object):
    def __init__(self, fn_parameters, info=None):
        self.fn_parameter = fn_parameters
        if info is None:
            self.load(fn_parameters)
        else:
            self.info = info

    def complain(self, counter, message=None):
        raise IOError('Line %i in the parameter file %s %s.' % (counter, self.fn_parameters, message))

    def load(self, fn_parameters):
        self.info = {}
        f = file(fn_parameters)
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

    def get_part(self, ForcePartClass):
        for part in self.parts:
            if isinstance(part, ForcePartClass):
                return part

    def get_part_pair(self, PairPotClass):
        for part in self.parts:
            if isinstance(part, ForcePartPair) and isisntance(part.pair_pot, PairPotClass):
                return part


class Generator(object):
    """Creates (part of a) ForceField object automatically.

       A generator is a class that describes how a part of a parameter file
       must be turned into a part of ForceField object. As the generator
       proceeds, it will modify and extend the current arguments of the FF. They
       should be implemented such that the order of the generators is not
       important.
    """
    prefix = None

    def __call__(self, parsed_pars, ff_args):
        my_parsed_pars = parsed_pars.get_section(self.prefix)
        if my_pardata is not None:
            conversions = self.process_units(my_parsed_pars.get_section('UNITS'))
            self.process(my_parsed_pars, conversions, ff_args)

    def process_units(self, parsed_pars):
        result = {}
        if parsed_pars is not None:
            for counter, line in parse_pars.info:
                line = words.split()
                if len(words) != 2:
                    raise parse_pars.complain(counter, 'Can not process the following unit command: %s' % line)
                result[words[0]] = parse_unit(words[1])
        return result


    def process(self, pardata, conversions, ff_args):
        raise NotImplementedError


class BondHarmGenerator(Generator):
    prefix = 'BONDHARM'

    def process(self, pardata, conversions, ff_args):
        raise NotImplementedError


# TODO: make generators a dictionary. only call generators that have a prefix in
# the parameters file.
generators = [BondHarmGenerator]
