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

import os, shlex, subprocess

from yaff import context


def run_example(workdir, command):
    env = dict(os.environ)
    if os.path.isfile('setup.py') and os.path.isdir('data') and os.path.isdir('yaff'):
        # Needed in case the tests are executed on an in-place build:
        rootdir = os.getcwd()
        env['PYTHONPATH'] = rootdir + ':' + env.get('PYTHONPATH', '')
        env['YAFFDATA'] = os.path.join(rootdir, 'data')
    workdir = context.get_fn(workdir)
    proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir, env=env)
    outdata, errdata = proc.communicate()
    if proc.returncode != 0:
        print 'Standard output'
        print '+'*80
        print outdata
        print '+'*80
        print 'Standard error'
        print '+'*80
        print errdata
        print '+'*80
        assert False
    fn_clean = os.path.join(workdir, 'clean.sh')
    if os.path.isfile(fn_clean):
        assert os.system('cd %s; ./clean.sh &> /dev/null' % workdir) == 0


def test_example_000_overview():
    run_example('examples/000_overview', './simulation.py')


def test_example_001_tutorial_bks():
    run_example('examples/001_tutorial_bks', './all.sh')


def test_example_002_external_trajectory():
    run_example('examples/002_external_trajectory', './rdf.py')


def test_example_003_water_thermostat():
    run_example('examples/003_water_thermostat', './md.py')


def test_example_999_back_propagation():
    run_example('examples/999_back_propagation', './bp.py')
