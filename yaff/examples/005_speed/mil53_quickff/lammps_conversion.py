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
from __future__ import print_function

import os

from yaff.external.lammpsio import write_lammps_table, ff2lammps, get_lammps_ffatypes
from yaff import System, ForceField
from molmod.units import angstrom, kcalmol

def main():
    rcut = 15.0*angstrom
    for nx, ny, nz in [(1,1,1),(1,2,1),(2,2,1),(2,2,2),(2,3,2),
            (3,3,2),(3,3,3),(3,4,3),(4,4,3),(4,4,4)][:]:
        # Generate supercell system
        system = System.from_file('system.chk').supercell(nx,ny,nz)
        dn = 'lammps_%s'%('.'.join("%d"%n for n in [nx,ny,nz]))
        if not os.path.isdir(dn): os.makedirs(dn)
        # Tabulate vdW interactions
        if not os.path.isfile('lammps.table'):
            ff = ForceField.generate(system, ['pars.txt'], rcut=rcut)
            write_lammps_table(ff, fn='lammps.table', rmin=0.50*angstrom,
                nrows=2500, unit_style='real')
        # Write the LAMMPS input files
        ff2lammps(system, 'pars.txt', dn, rcut=15.0*angstrom, tailcorrections=False,
            tabulated=True, unit_style='real')
        # Adapt the sampling options, which are defined in the last 5 lines
        # of lammps.in
        with open(os.path.join(dn,'lammps.in'),'r') as f:
            lines = f.readlines()
        with open(os.path.join(dn,'lammps.in'),'w') as f:
            for line in lines[:-5]:
                f.write(line)
            f.write("timestep 1.0 # in time units\n")
            f.write("velocity all create 600.0 5 # initial temperature in Kelvin and random seed\n")
            f.write("fix 1 all nvt temp 300.0 300.0 100.0\n")
            f.write("fix_modify 1 energy yes # Add thermo/barostat contributions to energy\n")
            f.write("run 100\n")


if __name__=='__main__':
    main()
