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


import os, glob, subprocess, sys


def run_example(dirname, fn_py, *args):
    # fix python path
    env = dict(os.environ)
    python_path = env.get('PYTHONPATH')
    if python_path is None:
        python_path = os.getcwd()
    else:
        python_path += ':' + os.getcwd()
    env['PYTHONPATH'] = python_path

    # prepare Popen arguments
    root = os.path.join("examples", dirname)
    assert os.path.isdir(root)
    assert os.path.isfile(os.path.join(root, fn_py))

    # run example and pass through the output
    p = subprocess.Popen(['./%s' % fn_py] + list(args), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    p.wait()
    sys.stdout.write(p.stdout.read())
    sys.stderr.write(p.stderr.read())

    # final check
    assert p.returncode == 0

def test_example_000_overview():
    run_example('000_overview', 'simulation.py')

def test_example_001_tutorial_bks():
    run_example('001_tutorial_bks/init', 'mksystem.py')
    run_example('001_tutorial_bks/opt', 'simulation.py')
    run_example('001_tutorial_bks/opt', 'analysis.py')
    run_example('001_tutorial_bks/nvt', 'simulation.py', '300', '310')
    run_example('001_tutorial_bks/nvt', 'analysis.py', '300', '310', '10')

def test_example_002_external_trajectory():
    run_example('002_external_trajectory', 'rdf.py')

def test_example_999_back_propagation():
    run_example('999_back_propagation', 'bp.py')
