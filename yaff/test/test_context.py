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


import os, subprocess

from yaff import context


def test_context():
    fn = context.get_fn('test/parameters_bks.txt')
    assert os.path.isfile(fn)
    fns = context.glob('test/parameters_*')
    assert fn in fns


def test_data_files():
    # Find files in data that were not checked in.
    # This test only makes sense if ran inside the source tree. The purpose is
    # to detect mistakes in the development process.
    if context.data_dir == os.path.abspath('data/') and os.path.isdir('.git'):
        lines = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', 'data']).split('\n')
        for line in lines:
            line = line.strip()
            if len(line) != 0:
                raise ValueError('The following file is not checked in: %s' % line)
