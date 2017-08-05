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
'''Object-oriented representation of parameter files'''


from __future__ import division
from __future__ import print_function


__all__ = ['Complain', 'Parameters', 'ParameterSection', 'ParameterDefinition']


class Complain(object):
    '''Class for complain method of ParameterFile and ParameterSection'''
    def __init__(self, filename='__nofile__'):
        self.filename = filename

    def __call__(self, counter, message):
        if counter is None:
            raise IOError('The parameter file %s %s.' % (self.filename, message))
        else:
            raise IOError('Line %i in the parameter file %s %s.' % (counter, self.filename, message))


class Parameters(object):
    '''Object that represents a force field parameter file

       The parameter file is first parsed by this object into a convenient
       data structure with dictionaries. The actual force field is then
       generated based on these dictionaries.

       The parameter file has a purely line-based syntax. The order of the lines
       has no meaning. Comments begin with a hash sign (#) and continue till the
       end of a line. If the line is empty after stripping the comments, it is
       ignored. Every non-empty line should have the following format:

       PREFIX:SUFFIX DATA

       The prefix is used for sections, the suffix for definitions and the
       remainder of the line contains arguments for the definition. Definitions
       may be repeated with different or the same arguments.
    '''
    def __init__(self, sections=None):
        if sections is None:
            self.sections = {}
        else:
            self.sections = sections

    @classmethod
    def from_file(cls, filenames):
        '''Create a Parameters instance from one or more text files.

           **Arguments:**

           filenames
                A single filename or a list of filenames
        '''

        def parse_line(line, complain):
            '''parse a single line'''
            pos = line.find(':')
            if pos == -1:
                complain(counter, 'does not contain a colon')
            prefix = line[:pos].upper()
            rest = line[pos+1:].strip()
            if len(rest) == 0:
                complain(counter, 'does not have text after the colon')
            if len(prefix.split()) > 1:
                complain(counter, 'has a prefix that contains whitespace')
            pos = rest.find(' ')
            if pos == 0:
                complain(counter, 'does not have a definition after the prefix')
            elif pos == -1:
                complain(counter, 'does not have data after the definition')
            suffix = rest[:pos].upper()
            data = rest[pos+1:]
            return prefix, suffix, data

        if isinstance(filenames, str):
            filenames = [filenames]

        result = cls({})
        for filename in filenames:
            with open(filename) as f:
                counter = 1
                complain = Complain(filename)
                for line in f:
                    line = line[:line.find('#')].strip()
                    if len(line) > 0:
                        # parse single line
                        prefix, suffix, data = parse_line(line, complain)
                        # get/make section
                        section = result.sections.get(prefix)
                        if section is None:
                            section = ParameterSection(prefix, {}, complain)
                            result.sections[prefix] = section
                        # get/make definition
                        definition = section.definitions.get(suffix)
                        if definition is None:
                            definition = ParameterDefinition(suffix, [], complain)
                            section.definitions[suffix] = definition
                        definition.lines.append((counter, data))
                    counter += 1

        return result

    def copy(self):
        '''Return an independent copy'''
        sections = {}
        for prefix, section in self.sections.items():
            sections[prefix] = section.copy()
        return Parameters(sections)

    def write_to_file(self, filename):
        '''Write the parameters back to a file

           The outut file will not contain any comments.
        '''
        with open(filename, 'w') as f:
            for prefix, section in self.sections.items():
                for suffix, definition in section.definitions.items():
                    for counter, data in definition.lines:
                        print('%s:%s %s' % (prefix, suffix, data), file=f)
                    print(file=f)
                print(file=f)
                print(file=f)

    def __getitem__(self, prefix):
        result = self.sections.get(prefix.upper())
        if result is None:
            result = ParameterSection(prefix, {})
        return result


class ParameterSection(object):
    '''Object that represents one section in a force field parameter file'''
    def __init__(self, prefix, definitions=None, complain=None):
        self.prefix = prefix
        if definitions is None:
            self.definitions = {}
        else:
            self.definitions = definitions
        if complain is None:
            self.complain = Complain()
        else:
            self.complain = complain

    def __getitem__(self, suffix):
        result = self.definitions.get(suffix.upper())
        if result is None:
            result = ParameterDefinition(suffix, [], self.complain)
        return result

    def copy(self):
        '''Return an independent copy'''
        definitions = {}
        for suffix, definition in self.definitions.items():
            definitions[suffix] = definition.copy()
        return ParameterSection(self.prefix, definitions, self.complain)


class ParameterDefinition(object):
    '''Object that represents a set of data lines from a parameter file'''
    def __init__(self, suffix, lines=None, complain=None):
        self.suffix = suffix
        if lines is None:
            self.lines = []
        else:
            self.lines = lines
        if complain is None:
            self.complain = Complain()
        else:
            self.complain = complain

    def __getitem__(self, index):
        return self.lines[index]

    def __iter__(self):
        return iter(self.lines)

    def copy(self):
        lines = []
        for counter, data in self.lines:
            lines.append((counter, data))
        return ParameterDefinition(self.suffix, lines, self.complain)
