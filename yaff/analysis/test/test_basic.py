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

import shutil, os

from yaff import *
from yaff.analysis.test.common import run_nve_water32, run_nvt_water32, run_opt_water32


def test_plot_energies_nve():
    with run_nve_water32(__name__, 'test_plot_energies_nve') as (dn_tmp, nve, f):
        fn_png = '%s/energies1.png' % dn_tmp
        plot_energies(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_energies_opt():
    with run_opt_water32(__name__, 'test_plot_energies_opt') as (dn_tmp, opt, f):
        fn_png = '%s/energies1.png' % dn_tmp
        plot_energies(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_temperature():
    with run_nve_water32(__name__, 'test_plot_temperature') as (dn_tmp, nve, f):
        fn_png = '%s/temperature1.png' % dn_tmp
        plot_temperature(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_pressure():
    with run_nvt_water32(__name__, 'test_plot_pressure') as (dn_tmp, nvt, f):
        fn_png = '%s/pressure1.png' % dn_tmp
        plot_pressure(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_temp_dist():
    with run_nve_water32(__name__, 'test_plot_temp_dist') as (dn_tmp, nve, f):
        fn_png = '%s/temp_dist1.png' % dn_tmp
        plot_temp_dist(f, fn_png)
        assert os.path.isfile(fn_png)
        fn_png = '%s/temp_dist2.png' % dn_tmp
        plot_temp_dist(f, fn_png, select=[0, 1, 2, 6, 7, 8])
        assert os.path.isfile(fn_png)


def test_plot_density():
    with run_nve_water32(__name__, 'test_plot_density') as (dn_tmp, nve, f):
        fn_png = '%s/density1.png' % dn_tmp
        plot_density(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_cell_pars():
    with run_opt_water32(__name__, 'test_plot_cell_pars') as (dn_tmp, opt, f):
        fn_png = '%s/cell_pars1.png' % dn_tmp
        plot_cell_pars(f, fn_png)
        assert os.path.isfile(fn_png)


def test_plot_epot_contribs_nve():
    with run_nve_water32(__name__, 'test_plot_epot_contribs_nve') as (dn_tmp, nve, f):
        fn_png = '%s/epot_contribs.png' % dn_tmp
        plot_epot_contribs(f, fn_png)
        assert os.path.isfile(fn_png)
