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


nsteps=100

for supercell in 1.1.1 1.2.1 2.2.1 2.2.2 2.3.2 3.3.2 3.3.3 3.4.3 4.4.3 4.4.4
do
    program=yaff
    nproc=1
    python md.py ${program} ${supercell} ${nsteps} > ${program}_${nproc}_${nsteps}_${supercell}.log

    program=lammps
    nproc=8
    mpirun -np ${nproc} python md.py ${program} ${supercell} ${nsteps} > ${program}_${nproc}_${nsteps}_${supercell}.log
    nproc=1
    mpirun -np ${nproc} python md.py ${program} ${supercell} ${nsteps} > ${program}_${nproc}_${nsteps}_${supercell}.log
done
