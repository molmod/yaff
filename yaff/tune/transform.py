# -*- coding: utf-8 -*-
# YAFF is yet another force-field code
# Copyright (C) 2011 - 2013 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
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
'''Specification of variables in a parameters file'''


import re
import numpy as np


__all__ = [
    'ParameterTransform', 'ParameterModifier', 'ModifierRule', 'ScaleRule',
    'IncrementRule'
]


class ParameterTransform(object):
    def __init__(self, parameters0, mods):
        self.parameters0 = parameters0
        self.mods = mods

    def get_init(self):
        result = []
        for mod in self.mods:
            result.append(mod.get_init())
        return np.array(result)

    def __call__(self, x):
        assert len(x) == len(self.mods)
        result = self.parameters0.copy()
        for i in xrange(len(x)):
            self.mods[i](x[i], result)
        return result


class ParameterModifier(object):
    def __init__(self, rules):
        self.rules = rules

    def get_init(self):
        init_vals = np.array([rule.get_init() for rule in self.rules])
        assert init_vals.max() == init_vals.min()
        return init_vals[0]

    def __call__(self, x, parameters):
        for rule in self.rules:
            rule(x, parameters)


class ModifierRule(object):
    def __init__(self, prefix, suffix, regex, index):
        self.prefix = prefix
        self.suffix = suffix
        self.regex = regex
        self.pattern = re.compile(regex)
        self.index = index

    def __call__(self, x, parameters):
        definition = parameters[self.prefix][self.suffix]
        lines = []
        for counter, data in definition:
            if self.pattern.search(data) is not None:
                words = data.split()
                value = float(words[self.index])
                value = self.modify_value(x, value)
                words[self.index] = '% 17.10e' % value
                data = ' '.join(words)
            lines.append((counter, data))
        definition.lines = lines

    def modify_value(self, x, value):
        raise NotImplementedError

    def get_init(self):
        raise NotImplementedError


class ScaleRule(ModifierRule):
    def modify_value(self, x, value):
        return x*value

    def get_init(self):
        return 1.0


class IncrementRule(ModifierRule):
    def modify_value(self, x, value):
        return x+value

    def get_init(self):
        return 0.0
