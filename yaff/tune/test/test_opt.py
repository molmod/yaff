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


import numpy as np

from yaff import *


def test_random_quadratic1():
    N = 3
    x_bot = np.random.uniform(-1,1, N)
    def fn(x):
        return ((x - x_bot)**2).sum()

    x0 = np.zeros(3, float)
    x1 = random_opt(fn, x0)
    assert abs(x1 - x_bot).max() < 1e-4


def test_gaussian_quadratic1():
    N = 3
    x_bot = np.random.uniform(-1,1, N)
    def fn(x):
        return ((x - x_bot)**2).sum()

    x0 = np.zeros(3, float)
    x1 = gauss_opt(fn, x0, 0.01, sigma_threshold=0.5e-4)
    assert abs(x1 - x_bot).max() < 1e-4


def test_gaussian_quadratic2():
    N = 3
    x_bot = np.random.uniform(-1,1, N)
    def fn(x):
        tmp = x - x_bot
        return (tmp**2).sum() - 1e-8*np.cos(tmp*800).prod()

    x0 = np.zeros(3, float)
    x1 = gauss_opt(fn, x0, 0.1, sigma_threshold=0.5e-4)
    assert abs(x1 - x_bot).max() < 1e-4


def test_gaussian_quadratic3():
    N = 3
    x_bot = np.random.uniform(-1,1, N)
    def fn(x):
        tmp = x - x_bot
        return (tmp**2).sum() + np.random.normal(0, 1e-8)

    x0 = np.zeros(3, float)
    x1 = gauss_opt(fn, x0, 0.1, sigma_threshold=0.5e-4)
    assert abs(x1 - x_bot).max() < 1e-4
