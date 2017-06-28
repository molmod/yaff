#!/usr/bin/env python
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
#!/usr/bin/env python

import re, sys



rules = [
    ('setup.py', '^    version=\'(...+)\',$'),
    ('yaff/__init__.py', '^__version__ = \'(...+)\'$'),
    ('doc/conf.py', '^version = \'(...+)\'$'),
    ('doc/conf.py', '^release = \'(...+)\'$'),
    ('doc/ug_install.rst', '^    https://github.com/molmod/yaff/releases/download/1.1.1/yaff-(...+).tar.gz$'),
    ('doc/ug_install.rst', '^    wget https://github.com/molmod/yaff/releases/download/1.1.1/yaff-(...+).tar.gz$'),
    ('doc/ug_install.rst', '^    tar -xvzf yaff-(...+).tar.gz$'),
    ('doc/ug_install.rst', '^    cd yaff-(...+)$'),
    ('yaff/log.py', '^ *Welcome to Yaff (...+) - Yet another force field'),
]


if __name__ == '__main__':
    newversion = sys.argv[1]

    for fn, regex in rules:
        r = re.compile(regex)
        with open(fn) as f:
            lines = f.readlines()
        for i in xrange(len(lines)):
            line = lines[i]
            m = r.match(line)
            if m is not None:
                lines[i] = line[:m.start(1)] + newversion + line[m.end(1):]
        with open(fn, 'w') as f:
            f.writelines(lines)
