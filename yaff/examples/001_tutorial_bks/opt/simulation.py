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

# First import the Numpy library
import numpy as np

# Then import the h5py library. h5py is used to access HDF5 files from Python
# scripts. The HDF5 file format is an international standard to efficiently
# store large amounts of array data in a cross-platform file. In Yaff it is
# mainly used to store trajectory data, i.e. the state of the molecular system
# at each step of the an iterative algorithm such as a geometry optimization or
# a molecular dynamics simulation.
import h5py as h5

# Finally load the Yaff library.
from yaff import *

# The amount of output printed on screen can be controlled by uncommenting one
# of the following lines:
#log.set_level(log.debug)
#log.set_level(log.high)
#log.set_level(log.medium) # The default
#log.set_level(log.low)
#log.set_level(log.silent)

# For convenience, the units for pressure and energy are defined here:
p_unit = 1e9*pascal
e_unit = electronvolt

# Load the initial state from the file created by the script init/mksystem.py
system = System.from_file('../init/init.chk')

# Construct a force-field model for the quartz system using the BKS parameters.
# In addition to the force-field parameters, several additional technical
# parameters may be provided to tune the accuracy and efficiency of the
# evaluation of the force-field energy and its partial derivatives, namely the
# Cartesian forces and the (static contribution to) the virial tensor.
#
# rcut
#   The real-space cutoff for the non-bonding interactions. (Both Van der Waals
#   and short-range electrostatics.)
#
# alpha_scale
#   A dimensionless parameter that controls alpha in the Ewald summation method:
#   alpha = alpha_scale/rcut.
#
# gcut_scale
#   A dimensionless parameter to control the cutoff for the reciprocal terms
#   in the Ewald summation: gcut = gcut_scale*alpha.
#
# smooth_ei
#   If True, the real-space term in the Ewald sum is smoothened in the same way
#   is the other non-bonding interactions. This may reduce numerical artifacts
#   during the optimization.
#
# skin
#   The skim parameter
#
# reci_ewald
#   Method for the computation of the reciprocal term in the Ewald sum. For now,
#   only 'ewald' and 'ignore' are supported.
ff = ForceField.generate(system, '../bks.pot', rcut=20*angstrom, alpha_scale=4.0,
                         gcut_scale=2.0, smooth_ei=True, reci_ei='ewald')

# Add an external constant pressure to the force field. The last argument is
# the constant pressure.
ff.add_part(ForcePartPressure(system, 0*p_unit))

# Define the optimization algorithm.
#
# CGOptimizer
#   A conjugate gradient optimizer
#
# StrainCellDOF
#   A definition of the degrees of freedom (DOF) that are subject to
#   optimization. Strain refers to the fact that cell rotations are excluded.
#
# gpos_rms=1e-6
#   The convergence condition for the nuclear positions. Only with the RMS value
#   of the nuclear gradient vector drops below this threshold, the optimization
#   may be converged. (Several other criteria are also implicitly used.)
#
# grvecs_rms=1e-6
#   Similar to previous, but now for the derivative of the energy towards the
#   components of the cell vectors.
#
# hooks
#   A list of extra 'things' that need to be done during the opmization. In this
#   case, just the generation of trajectory files.

# Some code to make sure the output files are written. (No changes needed.)
xyz = XYZWriter('traj.xyz', step=1)
with h5.File('traj.h5', 'w') as f:
    hdf5 = HDF5Writer(f)

    opt = CGOptimizer(StrainCellDOF(ff, gpos_rms=1e-6, grvecs_rms=1e-6), hooks=[xyz, hdf5])

    # Run the optimizer for at most 500 steps. (This should be enough.)
    opt.run(500)

    # Make some nice plots
    plot_epot_contribs(f, 'opt_energy.png')
    plot_cell_pars(f, 'opt_cell.png')

# Write the final state of the system to a file that is used as the initial
# state for the NVT simulations:
system.to_file('opt.chk')

# Print the relevant results on screen
print('#'*80)
print()
print('Volume [A^3]:      ', system.cell.volume/angstrom**3)
print('Pressure [GPa]:    ', ff.part_press.pext/p_unit)
print('Total energy [eV]: ', ff.energy/e_unit)
print('Contributions [eV]:')
for part in ff.parts:
    print('%15s: %s' % (part.name, part.energy/e_unit))
print()
print('#'*80)
