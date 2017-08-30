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
# --


# Needed for python2 backward compatibility
from __future__ import print_function

# Matplotlib is used for making plots. It is similar to the plotting interface
# in matlab. More information: http://matplotlib.org/
import matplotlib.pyplot as pt

# Usual imports:
import numpy as np
import h5py as h5
from yaff import *

# Just to give you an idea, kT at room temperature in electronvolt.
print('kT [eV]', boltzmann*300/electronvolt)

# Fill in the data here. Unit for energy: eV, unit for pressure: GPa.
data = [
    # (pressure, volume, energy),
    (-1.0, 0.0, 1.0), # fake, please remove.
    ( 0.0, 2.0, 0.0), # fake, please remove.
    ( 1.0, 1.0, 2.0), # fake, please remove.
]
# No further changes are needed.

# units
v_unit = angstrom**3
p_unit = 1e9*pascal
e_unit = electronvolt

# Transform your data into arrays:
p, v, e = np.array(data).T
p *= p_unit
v *= v_unit
e *= e_unit

# Select the reference volume.
v_ref = v[abs(p).argmin()]

# Compute the dimensionless volume.
x = v/v_ref

# Define an auxiliary array of dimensionless volumes to facilite the plots of
# the models
x_aux = np.linspace(x.min(), x.max(), 100)


def fit_linear_pv():
    print('Fit linear model to P(V/V_0)')

    # fit p(v), dm=design matrix, ev=expected values
    dm = np.array([x, np.ones(len(v))]).T
    ev = p
    a, b = np.linalg.lstsq(dm, ev)[0]
    print('    Bulk modulus [GPa]', -a/p_unit)

    # plot
    pt.clf()
    pt.plot(x, p/p_unit, 'ko')
    pt.plot(x_aux, (a*x_aux + b)/p_unit, 'r-')
    pt.xlabel('V/V_0')
    pt.ylabel('Pressure [GPa]')
    pt.savefig('pv.png')


def fit_quadratic_ev():
    print('Fit quadratic model to E(V/V_0)')

    # fit e(v), dm=design matrix, ev=expected values
    dm = np.array([0.5*x**2, x, np.ones(len(v))]).T
    ev = e
    a, b, c = np.linalg.lstsq(dm, ev)[0]
    print('    Bulk modulus [GPa]', a/v_ref/p_unit)
    print('    Energy [eV]', (c - a**2/2/b)/e_unit)

    # plot
    pt.clf()
    pt.plot(x, e/e_unit, 'ko')
    pt.plot(x_aux, (0.5*a*x_aux**2 + b*x_aux + c)/e_unit, 'r-')
    pt.xlabel('V/V_0')
    pt.ylabel('Energy [eV]')
    pt.savefig('ev.png')


fit_linear_pv()
fit_quadratic_ev()
