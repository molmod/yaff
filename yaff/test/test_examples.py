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


from __future__ import division

import os
import shlex
import subprocess
import pkg_resources
import stat

from molmod.test.common import tmpdir


def check_example(dirname, fn_script, fns_data):
    """Run an example in a temporary directory and check its exit code.

    Parameters
    ----------
    dirname : str
        The directory with the example, relative to the __file__ of where you call this
        function.
    fn_script : str
        The name of the script to be executed, assumed to be present in the given
        directory.
    fns_data : list of str:
        A list of data files needed by the example, which will be copied over to the
        temporary directory.
    """
    with tmpdir(__name__, dirname + fn_script) as dntmp:
        for fn in [fn_script] + fns_data:
            with pkg_resources.resource_stream("yaff", "examples/{}/{}".format(dirname, fn)) as fin:
                # Create the directory if needed.
                if '/' in fn:
                    subdntmp = os.path.join(dntmp, os.path.dirname(fn))
                    if not os.path.isdir(subdntmp):
                        os.makedirs(subdntmp)
                # Extract the file manually.
                with open(os.path.join(dntmp, fn), 'wb') as fout:
                    fout.write(fin.read())
        env = dict(os.environ)
        root_dir = os.getcwd()
        env['PYTHONPATH'] = root_dir + ':' + env.get('PYTHONPATH', '')
        path_script = os.path.join(dntmp, fn_script)
        os.chmod(path_script, os.stat(path_script).st_mode | stat.S_IXUSR)
        command = ["python", fn_script]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                cwd=dntmp, env=env)
        outdata, errdata = proc.communicate()
        if proc.returncode != 0:
            lines = [
                'Command faild', str(command), 'Standard output', '+'*80, outdata.decode('utf-8'),
                '+'*80, 'Standard error', '+'*80, errdata.decode('utf-8'), '+'*80]
            raise AssertionError('\n'.join(lines))


def test_example_000_overview():
    check_example('000_overview', 'simulation.py', ['parameters.txt', 'system.chk'])


def test_example_001_tutorial_bks():
    check_example('001_tutorial_bks', 'runall.py', [
        'bks.pot', 'init/mksystem.py', 'init/rvecs.txt', 'init/struct.xyz',
        'nvt/analysis.py', 'nvt/simulation.py', 'opt/analysis.py', 'opt/simulation.py'])


def test_example_002_external_trajectory():
    check_example('002_external_trajectory', 'rdf.py', ['trajectory.xyz'])


def test_example_003_water_thermostat():
    check_example('003_water_thermostat', 'md.py', ['parameters.txt', 'system.chk'])


def test_example_004_tailcorrections():
    check_example('004_tailcorrections', 'runall.py', ['methane_trappe/common.py',
        'methane_trappe/sp.py','methane_trappe/md.py'])


def test_example_999_back_propagation():
    check_example('999_back_propagation', 'bp.py', [])
