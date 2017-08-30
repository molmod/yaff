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

# The sys library (from system) is needed for command-line parsing.
import sys

# The usual import lines
import numpy as np
import h5py as h5
from yaff import *

# Parse the command line arguments. There is no need to change these lines.
args = sys.argv[1:]
assert len(args) == 3
temp = int(args[0])
nstep = int(args[1])
nskip = int(args[2])
# the suffix is used to select the right trajectory file.
suffix = '%04i_%06i.h5' % (temp, nstep)


# Define the pressure unit
p_unit = 1e9*pascal

# Open the trajectory file for post-processing the MD simulation
with h5.File('traj_%s.h5' % suffix) as f:
    # Get the isotropic pressure. This is the trace of the time-dependent virial
    # stress divided by three.
    press = np.array(f['trajectory/press'][nskip:])

    # The average pressure in the selected unit
    print('Average pressure [GPa]:', press.mean()/p_unit)

    # Block-averaging method to compute the error on the average.
    error = blav(press, fn_png='blav_%s.png' % suffix)[0]
    print('Error on average [GPa]:', error/p_unit)

    # Compute the time auto-correlation of the time-dependent pressure.
    # This is done with a fast-Fourier transform (FFT), which is implemented as an
    # additional result of a spectral analysis. The bsize argument is set such that
    # the spectrum is computed on the signal as a whole, without dividing it into
    # blocks.
    spectrum = Spectrum(f, start=nskip, path='trajectory/press', bsize=len(press)-nskip)
    spectrum.plot_ac('ac_%s.png' % suffix)
