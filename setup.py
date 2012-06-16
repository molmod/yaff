#!/usr/bin/env python
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


import glob
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name='YAFF',
    version='0.0',
    description='YAFF is yet another force-field code.',
    author='Toon Verstraelen',
    author_email='Toon.Verstraelen@UGent.be',
    url='http://molmod.ugent.be/code/',
    package_dir = {'yaff': 'yaff'},
    packages=['yaff', 'yaff/test', 'yaff/pes', 'yaff/pes/test', 'yaff/sampling',
              'yaff/sampling/test', 'yaff/analysis', 'yaff/analysis/test'],
    cmdclass = {'build_ext': build_ext},
    ext_modules=[
        Extension("yaff.pes.ext",
            sources=['yaff/pes/ext.pyx', 'yaff/pes/nlist.c',
                     'yaff/pes/pair_pot.c', 'yaff/pes/ewald.c',
                     'yaff/pes/dlist.c', 'yaff/pes/iclist.c',
                     'yaff/pes/vlist.c', 'yaff/pes/cell.c',
                     'yaff/pes/truncation.c'],
            depends=['yaff/pes/nlist.h', 'yaff/pes/nlist.pxd',
                     'yaff/pes/pair_pot.h', 'yaff/pes/pair_pot.pxd',
                     'yaff/pes/ewald.h', 'yaff/pes/ewald.pxd',
                     'yaff/pes/dlist.h', 'yaff/pes/dlist.pxd',
                     'yaff/pes/iclist.h', 'yaff/pes/iclist.pxd',
                     'yaff/pes/vlist.h', 'yaff/pes/vlist.pxd',
                     'yaff/pes/cell.h', 'yaff/pes/cell.pxd',
                     'yaff/pes/truncation.h', 'yaff/pes/truncation.pxd',
                     'yaff/pes/constants.h'],
        ),
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Topic :: Science/Engineering :: Molecular Science'
    ],
)
